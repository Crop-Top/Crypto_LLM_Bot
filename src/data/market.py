import time
from typing import Optional

import ccxt
import pandas as pd
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.config import Settings

OHLCV_COLS = ["timestamp", "open", "high", "low", "close", "volume"]
RETRY_EXCEPTIONS = (
    ccxt.NetworkError,
    ccxt.ExchangeError,
    ccxt.BadRequest,
    ccxt.RateLimitExceeded,
)


class DataHandler:
    def __init__(self, settings: Settings):
        self.settings = settings
        exchange_class = getattr(ccxt, settings.exchange_id)
        config = settings.public_exchange_config
        self.exchange: ccxt.Exchange = exchange_class(config)
        # Load only public market metadata (skip private currency list)
        try:
            # Bybit's fetch_markets is public; fetch_currencies is private.
            # We set has['fetchCurrencies']=False to skip the private call.
            self.exchange.has['fetchCurrencies'] = False
            self.exchange.load_markets()
        except Exception as e:
            logger.debug(f"Market metadata load skipped (non-fatal): {e}")
        if not hasattr(self.exchange, 'rateLimit') or not self.exchange.rateLimit:
            self.exchange.rateLimit = 1000
        self.symbol = self._resolve_symbol()
        self.timeframe = settings.timeframe  # mutable override for UI changes

    def _resolve_symbol(self, raw: Optional[str] = None) -> str:
        """Resolve user symbol (e.g. BTC/USDT) to linear perpetual (BTC/USDT:USDT)."""
        if raw is None:
            raw = self.settings.symbol
        if raw in self.exchange.markets:
            m = self.exchange.markets[raw]
            if m.get("swap") or m.get("linear") or m.get("inverse"):
                return raw
            # If spot, try the swap variant
        if ":" not in raw and "/" in raw:
            base, quote = raw.split("/")
            candidate = f"{base}/{quote}:{quote}"
            if candidate in self.exchange.markets:
                m = self.exchange.markets[candidate]
                if m.get("swap") or m.get("linear"):
                    logger.info(f"Auto-resolved symbol {raw} -> {candidate}")
                    return candidate
        logger.warning(f"Symbol {raw} may not be a perpetual market; OI/funding data may be unavailable.")
        return raw

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(RETRY_EXCEPTIONS),
        before_sleep=lambda _: logger.warning("Retrying fetch after error..."),
    )
    def fetch_ohlcv(self, since: Optional[int] = None, limit: int = 200) -> pd.DataFrame:
        raw = self.exchange.fetch_ohlcv(
            self.symbol,
                timeframe=self.timeframe,
            since=since,
            limit=limit,
        )
        df = pd.DataFrame(raw, columns=OHLCV_COLS)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        return df

    def fetch_open_interest(self, since: Optional[int] = None, limit: int = 200) -> pd.DataFrame:
        try:
            raw = self.exchange.fetch_open_interest_history(
                self.symbol,
                timeframe=self.timeframe,
                since=since,
                limit=limit,
            )
        except Exception as e:
            logger.debug(f"Open interest not available (non-fatal): {e}")
            return pd.DataFrame()
        if not raw:
            return pd.DataFrame()
        df = pd.DataFrame(raw)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        # Bybit returns openInterestAmount for linear perps, openInterestValue for spot
        candidates = ["openInterestAmount", "openInterestValue", "openInterest"]
        col = next((c for c in candidates if c in df.columns and df[c].notna().any()), None)
        if col is None:
            logger.debug("No valid open interest column found.")
            return pd.DataFrame()
        return df[[col]].rename(columns={col: "open_interest"})

    def fetch_funding_rate(self, since: Optional[int] = None, limit: int = 200) -> pd.DataFrame:
        try:
            raw = self.exchange.fetch_funding_rate_history(
                self.symbol,
                since=since,
                limit=limit,
            )
        except Exception as e:
            logger.debug(f"Funding rate not available (non-fatal): {e}")
            return pd.DataFrame()
        if not raw:
            return pd.DataFrame()
        df = pd.DataFrame(raw)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        if "fundingRate" in df.columns:
            return df[["fundingRate"]].rename(columns={"fundingRate": "funding_rate"})
        return pd.DataFrame()

    def fetch_historical(self, days: int = 60) -> pd.DataFrame:
        logger.info(f"Fetching {days}d of historical data for {self.symbol}...")
        now_ms = self.exchange.milliseconds()
        since_ms = now_ms - days * 86_400_000
        all_bars: list[pd.DataFrame] = []
        try:
            while since_ms < now_ms:
                df = self.fetch_ohlcv(since=since_ms, limit=200)
                if df.empty:
                    break
                all_bars.append(df)
                since_ms = int(df.index[-1].timestamp() * 1000) + 1
                time.sleep(self.exchange.rateLimit / 1000)
        except Exception as e:
            logger.error(f"OHLCV fetch failed at timestamp {since_ms}: {type(e).__name__}: {e}")
            raise
        if not all_bars:
            logger.error("No OHLCV data returned at all.")
            return pd.DataFrame()
        merged = pd.concat(all_bars)
        merged = merged[~merged.index.duplicated(keep="first")].sort_index()
        # Attach OI and funding (best-effort)
        start_ms = int(merged.index[0].timestamp() * 1000)
        try:
            oi = self.fetch_open_interest(since=start_ms)
            if not oi.empty:
                merged = merged.join(oi, how="left")
        except Exception as e:
            logger.debug(f"Failed to attach open interest: {e}")
        try:
            fr = self.fetch_funding_rate(since=start_ms)
            if not fr.empty:
                merged = merged.join(fr, how="left")
        except Exception as e:
            logger.debug(f"Failed to attach funding rate: {e}")
        merged.ffill(inplace=True)
        merged.bfill(inplace=True)
        logger.info(f"Historical data shape: {merged.shape}")
        return merged

    def fetch_latest(self) -> pd.DataFrame:
        df = self.fetch_ohlcv(limit=5)
        latest_ts = int(df.index[-1].timestamp() * 1000)
        oi = self.fetch_open_interest(since=latest_ts - 120_000, limit=1)
        fr = self.fetch_funding_rate(since=latest_ts - 120_000, limit=1)
        if not oi.empty:
            df = df.join(oi, how="left")
        if not fr.empty:
            df = df.join(fr, how="left")
        df.ffill(inplace=True)
        return df.tail(1)
