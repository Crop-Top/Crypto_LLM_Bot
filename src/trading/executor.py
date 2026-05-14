import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

import ccxt
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.config import Settings
from src.trading.models import OrderResult, OrderSide, OrderType

_RETRY_EXC = (ccxt.NetworkError, ccxt.ExchangeError, ccxt.BadRequest, ccxt.RateLimitExceeded)


class BaseExecutor(ABC):
    @abstractmethod
    def get_balance(self, currency: str = "USDT") -> float:
        ...

    @abstractmethod
    def place_order(
        self,
        side: OrderSide,
        order_type: OrderType = OrderType.MARKET,
        amount: float = 0.0,
        price: Optional[float] = None,
    ) -> OrderResult:
        ...

    @abstractmethod
    def cancel_all_orders(self) -> None:
        ...


class PaperExecutor(BaseExecutor):
    """Simulated paper-trading executor. Tracks virtual balance and positions."""

    def __init__(self, settings: Settings, initial_balance: float = 10_000.0):
        self.settings = settings
        self._balance: dict[str, float] = {"USDT": initial_balance, "BTC": 0.0}
        self._positions: dict[str, float] = {}
        self._trade_log: list[OrderResult] = []
        self._fee_rate = 0.0006  # 0.06% simulated taker fee

    def get_balance(self, currency: str = "USDT") -> float:
        return self._balance.get(currency, 0.0)

    def place_order(
        self,
        side: OrderSide,
        order_type: OrderType = OrderType.MARKET,
        amount: float = 0.0,
        price: Optional[float] = None,
    ) -> OrderResult:
        exchange = ccxt.bybit(self.settings.public_exchange_config)
        ticker = exchange.fetch_ticker(self.settings.symbol)
        exec_price = ticker["last"] if price is None else price
        base_currency = self.settings.symbol.split("/")[0]

        if amount <= 0:
            amount = self.settings.position_size_usdt / exec_price

        cost = amount * exec_price
        fee = cost * self._fee_rate

        if side == OrderSide.BUY:
            if self._balance["USDT"] < cost + fee:
                return OrderResult(
                    order_id="", symbol=self.settings.symbol, side=side,
                    type=order_type, price=exec_price, amount=amount,
                    filled=0.0, status="rejected",
                    timestamp=datetime.now(timezone.utc),
                    error="Insufficient virtual USDT balance",
                )
            self._balance["USDT"] -= cost + fee
            self._balance[base_currency] = self._balance.get(base_currency, 0.0) + amount
        else:  # SELL
            held = self._balance.get(base_currency, 0.0)
            if held < amount:
                amount = held
                cost = amount * exec_price
                if amount <= 0:
                    return OrderResult(
                        order_id="", symbol=self.settings.symbol, side=side,
                        type=order_type, price=exec_price, amount=0.0,
                        filled=0.0, status="rejected",
                        timestamp=datetime.now(timezone.utc),
                        error="No virtual position to sell",
                    )
            self._balance[base_currency] = self._balance.get(base_currency, 0.0) - amount
            self._balance["USDT"] += cost - fee

        order_id = f"paper_{uuid.uuid4().hex[:12]}"
        result = OrderResult(
            order_id=order_id, symbol=self.settings.symbol, side=side,
            type=order_type, price=exec_price, amount=amount,
            filled=amount, status="closed",
            timestamp=datetime.now(timezone.utc),
        )
        self._trade_log.append(result)
        logger.info(f"[Paper] {side.value.upper()} {amount:.6f} @ {exec_price:.2f} | USDT={self._balance['USDT']:.2f}")
        return result

    def cancel_all_orders(self) -> None:
        logger.info("[Paper] Cancel all orders (no-op for paper).")


class LiveExecutor(BaseExecutor):
    """Real exchange executor using CCXT."""

    def __init__(self, settings: Settings):
        self.settings = settings
        exchange_class = getattr(ccxt, settings.exchange_id)
        self.exchange: ccxt.Exchange = exchange_class(settings.exchange_config)
        try:
            self.exchange.load_markets()
        except ccxt.AuthenticationError:
            logger.warning("LiveExecutor: markets loaded without currency details.")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(_RETRY_EXC),
    )
    def get_balance(self, currency: str = "USDT") -> float:
        balance = self.exchange.fetch_balance()
        return float(balance.get("free", {}).get(currency, 0.0))

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(_RETRY_EXC),
    )
    def place_order(
        self,
        side: OrderSide,
        order_type: OrderType = OrderType.MARKET,
        amount: float = 0.0,
        price: Optional[float] = None,
    ) -> OrderResult:
        market = self.exchange.market(self.settings.symbol)
        ticker = self.exchange.fetch_ticker(self.settings.symbol)
        current_price = ticker["last"]

        if amount <= 0:
            amount = (self.settings.position_size_usdt * self.settings.leverage) / current_price

        amount = float(self.exchange.amount_to_precision(self.settings.symbol, amount))
        limit_price = None
        if order_type == OrderType.LIMIT and price is not None:
            limit_price = float(self.exchange.price_to_precision(self.settings.symbol, price))

        try:
            order = self.exchange.create_order(
                symbol=self.settings.symbol,
                type=order_type.value,
                side=side.value,
                amount=amount,
                price=limit_price,
            )
            logger.success(f"[Live] {side.value.upper()} {amount} @ {limit_price or 'market'}")
            return OrderResult(
                order_id=order.get("id", "unknown"),
                symbol=self.settings.symbol,
                side=side,
                type=order_type,
                price=float(order.get("price", current_price)),
                amount=amount,
                filled=float(order.get("filled", 0)),
                status=order.get("status", "unknown"),
                timestamp=datetime.now(timezone.utc),
            )
        except ccxt.InsufficientFunds as e:
            logger.error(f"Insufficient funds: {e}")
            return OrderResult(
                order_id="", symbol=self.settings.symbol, side=side,
                type=order_type, price=current_price, amount=amount,
                filled=0.0, status="rejected",
                timestamp=datetime.now(timezone.utc), error=str(e),
            )

    def cancel_all_orders(self) -> None:
        try:
            self.exchange.cancel_all_orders(self.settings.symbol)
            logger.info("[Live] All orders cancelled.")
        except Exception as e:
            logger.warning(f"Cancel orders failed: {e}")
