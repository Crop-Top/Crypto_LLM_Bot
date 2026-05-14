import warnings
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

warnings.filterwarnings("ignore")
import sklearn  # noqa: E402
sklearn.set_config(assume_finite=True)  # suppress parallel warnings

_IMPORT_ERROR = None
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
except ImportError as e:
    _IMPORT_ERROR = e
    RandomForestClassifier = None  # type: ignore
    StandardScaler = None  # type: ignore


class FeatureEngineer:
    """Computes a rich set of features from OHLCV + optional OI/funding data."""

    FEATURE_COLUMNS: list[str] = []

    @classmethod
    def compute_features(cls, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # Returns
        for p in [1, 3, 5, 10]:
            df[f"ret_{p}"] = close.pct_change(p)

        # Moving averages
        df["sma_7"] = close.rolling(7).mean()
        df["sma_25"] = close.rolling(25).mean()
        df["sma_50"] = close.rolling(50).mean()
        df["ema_12"] = close.ewm(span=12, adjust=False).mean()
        df["ema_26"] = close.ewm(span=26, adjust=False).mean()

        # MACD
        df["macd"] = df["ema_12"] - df["ema_26"]
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # RSI
        df["rsi_14"] = cls._rsi(close, 14)

        # Bollinger Bands
        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        df["bb_upper"] = bb_mid + 2 * bb_std
        df["bb_lower"] = bb_mid - 2 * bb_std
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / bb_mid
        df["bb_pct"] = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)

        # ATR
        df["atr_14"] = cls._atr(high, low, close, 14)

        # Volume
        df["volume_sma_7"] = volume.rolling(7).mean()
        df["volume_ratio"] = volume / df["volume_sma_7"].replace(0, np.nan)

        # Price range
        df["high_low_pct"] = (high - low) / close

        # Volatility
        df["volatility_7"] = close.pct_change().rolling(7).std()
        df["volatility_14"] = close.pct_change().rolling(14).std()

        # Optional: OI and funding
        if "open_interest" in df.columns:
            df["oi_change"] = df["open_interest"].pct_change()
            df["oi_sma_7"] = df["open_interest"].rolling(7).mean()
            df["oi_ratio"] = df["open_interest"] / df["oi_sma_7"].replace(0, np.nan)

        if "funding_rate" in df.columns:
            df["fr_value"] = df["funding_rate"]
            df["fr_abs"] = df["funding_rate"].abs()
            df["fr_sma_7"] = df["funding_rate"].rolling(7).mean()

        # Time features
        idx = df.index
        df["hour"] = idx.hour
        df["day_of_week"] = idx.dayofweek

        # Drop NaN rows introduced by feature computation
        before = len(df)
        df.dropna(inplace=True)
        after = len(df)
        cls.FEATURE_COLUMNS = [c for c in df.columns if c not in ("open", "high", "low", "close", "volume")]
        if after < before:
            logger.debug(f"Feature engineering: {before} → {after} rows")
        return df

    @staticmethod
    def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        tr = pd.concat(
            [
                high - low,
                (high - close.shift()).abs(),
                (low - close.shift()).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return tr.rolling(period).mean()


class TradePredictor:
    """ML model that predicts Buy / Sell / Hold from engineered features."""

    def __init__(self, model: Optional[object] = None):
        if _IMPORT_ERROR is not None:
            raise ImportError("scikit-learn is required.") from _IMPORT_ERROR
        self.model = model or RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_leaf=15,
            random_state=42,
            class_weight="balanced",
            n_jobs=1,
        )
        self.scaler = StandardScaler()
        self.feature_columns: list[str] = []
        self._trained = False
        self._label_map: dict[int, str] = {0: "Sell", 2: "Buy"}

    def _make_labels(self, df: pd.DataFrame, horizon: int = 3) -> pd.Series:
        future_ret = df["close"].shift(-horizon) / df["close"] - 1
        # Binary: Buy if positive return, Sell if negative (always balanced)
        labels = future_ret.apply(lambda x: 2 if x > 0 else (0 if x < 0 else -1))
        return labels.astype(int)

    def train(self, df: pd.DataFrame) -> None:
        logger.info("Training TradePredictor...")
        fe = FeatureEngineer()
        featured = fe.compute_features(df)
        self.feature_columns = FeatureEngineer.FEATURE_COLUMNS

        if featured.empty or not self.feature_columns:
            logger.warning("No valid features could be computed. Skipping training.")
            return

        # Add labels and drop any rows where label is NaN (end of series)
        featured["_label"] = self._make_labels(featured)
        featured.dropna(subset=["_label"], inplace=True)
        featured = featured[featured["_label"] != -1]

        if len(featured) < 100:
            logger.warning(f"Too few samples ({len(featured)}) to train; skipping.")
            logger.debug(f"Feature columns: {self.feature_columns}")
            logger.debug(f"DataFrame shape before filter: {featured.shape}")
            return

        X = featured[self.feature_columns].values
        y = featured["_label"].values.astype(int)

        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.model.fit(X_scaled, y)
        self._trained = True
        logger.info(f"Model trained on {len(X)} samples across {len(self.feature_columns)} features.")

    def predict(self, df: pd.DataFrame) -> tuple[str, float]:
        if not self._trained:
            return "Hold", 0.0
        # Skip feature computation if features are already present
        if self.feature_columns and all(c in df.columns for c in self.feature_columns[:3]):
            featured = df
        else:
            fe = FeatureEngineer()
            featured = fe.compute_features(df)
        if featured.empty or len(self.feature_columns) == 0:
            return "Hold", 0.0
        X = featured[self.feature_columns].iloc[[-1]].values
        X_scaled = self.scaler.transform(X)
        pred = int(self.model.predict(X_scaled)[0])
        probs = self.model.predict_proba(X_scaled)[0]
        confidence = float(max(probs))
        return self._label_map[pred], confidence
