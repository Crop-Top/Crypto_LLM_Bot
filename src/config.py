import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


@dataclass(frozen=True)
class Settings:
    # API
    bybit_api_key: str = field(default_factory=lambda: os.getenv("BYBIT_API_KEY", ""))
    bybit_api_secret: str = field(default_factory=lambda: os.getenv("BYBIT_API_SECRET", ""))

    # Trading
    symbol: str = field(default_factory=lambda: os.getenv("SYMBOL", "BTC/USDT:USDT"))
    timeframe: str = field(default_factory=lambda: os.getenv("TIMEFRAME", "1h"))
    leverage: int = field(default_factory=lambda: int(os.getenv("LEVERAGE", "5")))
    position_size_usdt: float = field(default_factory=lambda: float(os.getenv("POSITION_SIZE_USDT", "100.0")))
    max_position_size_usdt: float = field(default_factory=lambda: float(os.getenv("MAX_POSITION_SIZE_USDT", "1000.0")))
    stop_loss_pct: float = field(default_factory=lambda: float(os.getenv("STOP_LOSS_PCT", "0.02")))
    take_profit_pct: float = field(default_factory=lambda: float(os.getenv("TAKE_PROFIT_PCT", "0.05")))
    max_daily_loss_usdt: float = field(default_factory=lambda: float(os.getenv("MAX_DAILY_LOSS_USDT", "500.0")))

    # Bot
    use_testnet: bool = field(default_factory=lambda: os.getenv("USE_TESTNET", "True").lower() == "true")
    max_retries: int = field(default_factory=lambda: int(os.getenv("MAX_RETRIES", "3")))
    retry_delay: int = field(default_factory=lambda: int(os.getenv("RETRY_DELAY", "5")))

    @property
    def exchange_config(self) -> dict:
        return {
            "apiKey": self.bybit_api_key,
            "secret": self.bybit_api_secret,
            "enableRateLimit": True,
            "options": {
                "defaultType": "linear",
                "testnet": self.use_testnet,
                "fetchCurrencies": False,
            },
        }

    @property
    def public_exchange_config(self) -> dict:
        """No API keys needed — public market data only.
        Fetches from mainnet regardless of USE_TESTNET (public data is identical).
        """
        return {
            "enableRateLimit": True,
            "options": {
                "defaultType": "linear",
                "fetchCurrencies": False,
            },
        }

    @property
    def test_exchange_config(self) -> dict:
        """Minimal config for connectivity diagnostics."""
        return {"enableRateLimit": True, "options": {"defaultType": "linear", "fetchCurrencies": False}}

    @property
    def exchange_id(self) -> str:
        return "bybit"

    def validate(self) -> None:
        missing = []
        if not self.bybit_api_key:
            missing.append("BYBIT_API_KEY")
        if not self.bybit_api_secret:
            missing.append("BYBIT_API_SECRET")
        if missing:
            raise ValueError(
                f"Missing required env vars: {', '.join(missing)}. "
                "Fill in .env or set environment variables."
            )
