from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class Signal(str, Enum):
    BUY = "Buy"
    SELL = "Sell"
    HOLD = "Hold"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class TradeStatus(str, Enum):
    OPEN = "open"
    CLOSED = "closed"
    CANCELLED = "cancelled"


@dataclass
class Trade:
    id: str
    symbol: str
    side: OrderSide
    entry_price: float
    entry_time: datetime
    quantity: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    status: TradeStatus = TradeStatus.OPEN
    pnl: float = 0.0
    pnl_pct: float = 0.0
    exit_reason: str = ""
    entry_reason: str = ""  # "signal_buy", "signal_sell"


@dataclass
class OrderResult:
    order_id: str
    symbol: str
    side: OrderSide
    type: OrderType
    price: float
    amount: float
    filled: float
    status: str
    timestamp: datetime
    error: Optional[str] = None


@dataclass
class BacktestResult:
    total_return: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    total_fees: float = 0.0
    net_profit: float = 0.0
    initial_capital: float = 0.0
    final_capital: float = 0.0
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    timestamps: list[datetime] = field(default_factory=list)
    signals: list[Signal] = field(default_factory=list)
