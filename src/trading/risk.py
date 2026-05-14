from datetime import date, datetime, timezone
from typing import Optional

from loguru import logger

from src.config import Settings


class RiskManager:
    """Enforces position sizing, stop-loss, take-profit, and daily loss limits."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._daily_pnl: float = 0.0
        self._current_date: Optional[date] = None
        self._position_value: float = 0.0
        self._entry_price: Optional[float] = None
        self._position_side: Optional[str] = None  # "long" or "short"

    def reset_daily(self) -> None:
        today = datetime.now(timezone.utc).date()
        if self._current_date != today:
            self._daily_pnl = 0.0
            self._current_date = today
            logger.info(f"Daily PnL reset for {today}")

    def check_daily_loss(self, current_pnl: float) -> bool:
        self.reset_daily()
        self._daily_pnl += current_pnl
        if self._daily_pnl <= -self.settings.max_daily_loss_usdt:
            logger.warning(f"Daily loss limit hit: {self._daily_pnl:.2f} USDT")
            return False
        return True

    def check_stop_loss(self, current_price: float) -> bool:
        if self._entry_price is None or self._position_side is None:
            return False
        if self._position_side == "long":
            loss_pct = (self._entry_price - current_price) / self._entry_price
            if loss_pct >= self.settings.stop_loss_pct:
                logger.warning(f"Stop-loss triggered: {loss_pct:.2%}")
                return True
        elif self._position_side == "short":
            loss_pct = (current_price - self._entry_price) / self._entry_price
            if loss_pct >= self.settings.stop_loss_pct:
                logger.warning(f"Stop-loss triggered: {loss_pct:.2%}")
                return True
        return False

    def check_take_profit(self, current_price: float) -> bool:
        if self._entry_price is None or self._position_side is None:
            return False
        if self._position_side == "long":
            profit_pct = (current_price - self._entry_price) / self._entry_price
            if profit_pct >= self.settings.take_profit_pct:
                logger.info(f"Take-profit triggered: {profit_pct:.2%}")
                return True
        elif self._position_side == "short":
            profit_pct = (self._entry_price - current_price) / self._entry_price
            if profit_pct >= self.settings.take_profit_pct:
                logger.info(f"Take-profit triggered: {profit_pct:.2%}")
                return True
        return False

    def compute_position_size(self, balance_usdt: float, current_price: float) -> float:
        raw_size = self.settings.position_size_usdt / current_price
        max_size = self.settings.max_position_size_usdt / current_price
        safe_size = min(raw_size, max_size, balance_usdt * 0.95 / current_price)
        return max(safe_size, 0.0)

    def open_position(self, side: str, entry_price: float, size: float) -> None:
        self._position_side = side
        self._entry_price = entry_price
        self._position_value = size * entry_price
        logger.info(f"Opened {side} position: {size:.6f} @ {entry_price:.2f}")

    def close_position(self) -> None:
        self._position_side = None
        self._entry_price = None
        self._position_value = 0.0
