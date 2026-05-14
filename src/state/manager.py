import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import pandas as pd

from src.trading.models import BacktestResult, Signal


@dataclass
class BotState:
    """Thread-safe shared state between the bot loop and the dashboard."""

    mode: str = "idle"  # idle | backtest | demo | live
    current_price: float = 0.0
    current_signal: Signal = Signal.HOLD
    signal_confidence: float = 0.0
    balance_usdt: float = 0.0
    position_size: float = 0.0
    unrealized_pnl: float = 0.0
    daily_pnl: float = 0.0
    total_trades_today: int = 0
    last_update: Optional[datetime] = None
    is_running: bool = False
    error: Optional[str] = None

    # Backtest results (populated after run)
    backtest_result: Optional[BacktestResult] = None

    # Recent data for charting
    price_history: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    oi_history: list[float] = field(default_factory=list)
    fr_history: list[float] = field(default_factory=list)

    _lock: threading.Lock = field(default_factory=threading.Lock)

    def update(self, **kwargs) -> None:
        with self._lock:
            for k, v in kwargs.items():
                if hasattr(self, k):
                    setattr(self, k, v)
            self.last_update = datetime.now()

    def snapshot(self) -> "BotState":
        with self._lock:
            df_copy = self.price_history.copy() if not self.price_history.empty else pd.DataFrame()
            return BotState(
                mode=self.mode,
                current_price=self.current_price,
                current_signal=self.current_signal,
                signal_confidence=self.signal_confidence,
                balance_usdt=self.balance_usdt,
                position_size=self.position_size,
                unrealized_pnl=self.unrealized_pnl,
                daily_pnl=self.daily_pnl,
                total_trades_today=self.total_trades_today,
                last_update=self.last_update,
                is_running=self.is_running,
                error=self.error,
                backtest_result=self.backtest_result,
                price_history=df_copy,
                oi_history=list(self.oi_history),
                fr_history=list(self.fr_history),
            )


# Global singleton
state = BotState()
