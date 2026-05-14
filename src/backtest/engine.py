from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.trading.models import BacktestResult, OrderSide, Signal, Trade, TradeStatus
from src.trading.predictor import FeatureEngineer
from src.trading.strategy import MLStrategy


def _compute_htf_trend(htf_data: pd.DataFrame, lookback: int = 50) -> pd.Series:
    """Return True (uptrend) where close > SMA(lookback) on HTF data (naive index)."""
    df = htf_data.copy()
    if hasattr(df.index, 'tz') and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    sma = df["close"].rolling(lookback).mean()
    trend = df["close"] > sma
    return trend.ffill()


class BacktestEngine:
    """Full-featured backtester: long/short, SL/TP exits, multi-timeframe trend filter."""

    FEE_RATE = 0.0006

    def __init__(
        self,
        strategy: MLStrategy,
        initial_capital: float = 10_000.0,
        slippage: float = 0.0005,
        risk_pct: float = 0.01,
        account_pct: float = 0.10,
        leverage: float = 5.0,
    ):
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.slippage = slippage
        self.risk_pct = risk_pct
        self.account_pct = account_pct
        self.leverage = leverage

    def run(
        self,
        data: pd.DataFrame,
        htf_data: Optional[pd.DataFrame] = None,
    ) -> BacktestResult:
        logger.info(f"Running backtest on {len(data)} bars...")

        featured = FeatureEngineer.compute_features(data.copy())
        if featured.empty:
            return BacktestResult()

        feat_cols = FeatureEngineer.FEATURE_COLUMNS
        if self.strategy.predictor._trained:
            self.strategy.predictor.feature_columns = feat_cols

        # HTF trend filter
        htf_trend: Optional[pd.Series] = None
        if htf_data is not None and not htf_data.empty:
            htf_trend = _compute_htf_trend(htf_data)
            logger.info(f"Multi-timeframe filter active (HTF trend available from {htf_trend.index[0].date()})")

        # Vectorized predictions
        X_all = featured[feat_cols].values
        X_all_scaled = self.strategy.predictor.scaler.transform(X_all)
        raw_preds = self.strategy.predictor.model.predict(X_all_scaled)
        raw_probs = self.strategy.predictor.model.predict_proba(X_all_scaled)
        raw_conf = raw_probs.max(axis=1)
        label_map = self.strategy.predictor._label_map

        capital = self.initial_capital
        position: float = 0.0
        entry_price: float = 0.0
        entry_sl: float = 0.0
        entry_tp: float = 0.0
        entry_confidence: float = 0.0
        locked_margin: float = 0.0
        trades: list[Trade] = []
        equity_curve: list[float] = []
        timestamps: list[datetime] = []
        signals: list[Signal] = []
        trade_id = 0
        sig_buy = sig_sell = sig_hold = 0

        for idx, (timestamp, row) in enumerate(featured.iterrows()):
            h = row["high"]
            l = row["low"]
            c = row["close"]

            pred_class = int(raw_preds[idx])
            confidence = float(raw_conf[idx])
            signal_str = label_map.get(pred_class, "Hold")
            signal = Signal(signal_str) if confidence >= 0.50 else Signal.HOLD
            signals.append(signal)
            if signal == Signal.BUY: sig_buy += 1
            elif signal == Signal.SELL: sig_sell += 1
            else: sig_hold += 1

            # HTF trend filter
            trend_up = True
            trend_down = True
            if htf_trend is not None:
                ts_key = timestamp.tz_localize(None) if hasattr(timestamp, 'tz') and timestamp.tz else timestamp
                valid = htf_trend.index[htf_trend.index <= ts_key]
                if len(valid) > 0:
                    trend_up = bool(htf_trend.loc[valid[-1]])
                    trend_down = not trend_up

            # --- EXIT CHECKS ---
            if position > 0:  # Long
                exit_reason = None
                exit_px = None
                if l <= entry_sl:
                    exit_px = entry_sl * (1 - self.slippage)
                    exit_reason = "stop_loss"
                elif h >= entry_tp:
                    exit_px = entry_tp * (1 - self.slippage)
                    exit_reason = "take_profit"
                elif signal == Signal.SELL and confidence > entry_confidence:
                    exit_px = c * (1 - self.slippage)
                    exit_reason = "signal"
                if exit_reason:
                    margin = position * entry_price / self.leverage
                    gross = position * exit_px
                    fee = gross * self.FEE_RATE
                    pnl = gross - fee - (position * entry_price)
                    capital += margin + pnl
                    pnl_pct = (exit_px - entry_price) / entry_price
                    trades[-1].exit_price = exit_px
                    trades[-1].exit_time = timestamp.to_pydatetime()
                    trades[-1].status = TradeStatus.CLOSED
                    trades[-1].pnl = pnl
                    trades[-1].pnl_pct = pnl_pct
                    trades[-1].exit_reason = exit_reason
                    position = 0.0
                    entry_price = entry_sl = entry_tp = 0.0
                    entry_confidence = 0.0
                    locked_margin = 0.0

            elif position < 0:  # Short
                exit_reason = None
                exit_px = None
                if h >= entry_sl:
                    exit_px = entry_sl * (1 + self.slippage)
                    exit_reason = "stop_loss"
                elif l <= entry_tp:
                    exit_px = entry_tp * (1 + self.slippage)
                    exit_reason = "take_profit"
                elif signal == Signal.BUY and confidence > entry_confidence:
                    exit_px = c * (1 + self.slippage)
                    exit_reason = "signal"
                if exit_reason:
                    qty = abs(position)
                    margin = qty * entry_price / self.leverage
                    buy_cost = qty * exit_px
                    fee = buy_cost * self.FEE_RATE
                    pnl = qty * (entry_price - exit_px) - fee
                    capital += margin + pnl
                    pnl_pct = (entry_price - exit_px) / entry_price
                    trades[-1].exit_price = exit_px
                    trades[-1].exit_time = timestamp.to_pydatetime()
                    trades[-1].status = TradeStatus.CLOSED
                    trades[-1].pnl = pnl
                    trades[-1].pnl_pct = pnl_pct
                    trades[-1].exit_reason = exit_reason
                    position = 0.0
                    entry_price = entry_sl = entry_tp = 0.0
                    entry_confidence = 0.0
                    locked_margin = 0.0

            # --- ENTRY ---
            if position == 0:
                if signal == Signal.BUY and trend_up:
                    if confidence >= 0.60:
                        rr = 2.5
                    elif confidence >= 0.55:
                        rr = 2.0
                    else:
                        rr = 1.5
                    margin = capital * self.account_pct
                    pos_value = margin * self.leverage
                    exec_px = c * (1 + self.slippage)
                    qty = pos_value / exec_px
                    fee = qty * exec_px * self.FEE_RATE
                    if margin + fee <= capital:
                        capital -= margin + fee
                        locked_margin = margin
                        position = qty
                        entry_price = exec_px
                        entry_confidence = confidence
                        entry_sl = exec_px * (1 - self.risk_pct)
                        entry_tp = exec_px * (1 + self.risk_pct * rr)
                        trade_id += 1
                        trades.append(Trade(
                            id=f"bt_{trade_id}", symbol="", side=OrderSide.BUY,
                            entry_price=exec_px, entry_time=timestamp.to_pydatetime(),
                            quantity=qty, stop_loss=entry_sl, take_profit=entry_tp,
                            entry_reason="signal_buy",
                        ))
                elif signal == Signal.SELL and trend_down:
                    if confidence >= 0.60:
                        rr = 2.5
                    elif confidence >= 0.55:
                        rr = 2.0
                    else:
                        rr = 1.5
                    margin = capital * self.account_pct
                    pos_value = margin * self.leverage
                    exec_px = c * (1 - self.slippage)
                    qty = pos_value / exec_px
                    fee = qty * exec_px * self.FEE_RATE
                    if margin + fee <= capital:
                        capital -= margin + fee
                        locked_margin = margin
                        position = -qty
                        entry_price = exec_px
                        entry_confidence = confidence
                        entry_sl = exec_px * (1 + self.risk_pct)
                        entry_tp = exec_px * (1 - self.risk_pct * rr)
                        trade_id += 1
                        trades.append(Trade(
                            id=f"bt_{trade_id}", symbol="", side=OrderSide.SELL,
                            entry_price=exec_px, entry_time=timestamp.to_pydatetime(),
                            quantity=qty, stop_loss=entry_sl, take_profit=entry_tp,
                            entry_reason="signal_sell",
                        ))

            # Mark-to-market (free capital + locked margin + unrealized PnL)
            if position > 0:
                unrealized = position * (c - entry_price)
                equity = capital + locked_margin + unrealized
            elif position < 0:
                qty = abs(position)
                unrealized = qty * (entry_price - c)
                equity = capital + locked_margin + unrealized
            else:
                equity = capital
            equity_curve.append(equity)
            timestamps.append(timestamp.to_pydatetime())

        # Force close at end
        if position > 0:
            margin = position * entry_price / self.leverage
            exit_px = featured["close"].iloc[-1] * (1 - self.slippage)
            gross = position * exit_px
            fee = gross * self.FEE_RATE
            pnl = gross - fee - (position * entry_price)
            capital += margin + pnl
            trades[-1].exit_price = exit_px
            trades[-1].exit_time = timestamps[-1]
            trades[-1].status = TradeStatus.CLOSED
            trades[-1].pnl = pnl
            trades[-1].pnl_pct = (exit_px - entry_price) / entry_price
            trades[-1].exit_reason = "end_of_data"
            equity_curve[-1] = capital
        elif position < 0:
            qty = abs(position)
            margin = qty * entry_price / self.leverage
            exit_px = featured["close"].iloc[-1] * (1 + self.slippage)
            buy_cost = qty * exit_px
            fee = buy_cost * self.FEE_RATE
            pnl = qty * (entry_price - exit_px) - fee
            capital += margin + pnl
            trades[-1].exit_price = exit_px
            trades[-1].exit_time = timestamps[-1]
            trades[-1].status = TradeStatus.CLOSED
            trades[-1].pnl = pnl
            trades[-1].pnl_pct = (entry_price - exit_px) / entry_price
            trades[-1].exit_reason = "end_of_data"
            equity_curve[-1] = capital

        # --- Metrics ---
        closed = [t for t in trades if t.status == TradeStatus.CLOSED]
        total = len(closed)
        winning = [t for t in closed if t.pnl > 0]
        losing = [t for t in closed if t.pnl <= 0]
        win_rate = len(winning) / total if total > 0 else 0.0
        equity_arr = np.array(equity_curve)
        returns = np.diff(equity_arr) / equity_arr[:-1] if len(equity_arr) > 1 else np.array([0.0])
        sharpe = float(np.sqrt(24 * 365) * returns.mean() / returns.std()) if returns.std() > 0 else 0.0
        peak = np.maximum.accumulate(equity_arr)
        drawdowns = (peak - equity_arr) / peak
        max_dd_pct = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0
        max_dd_val = float(np.max(peak - equity_arr)) if len(peak) > 0 else 0.0
        total_fees = sum(
            t.quantity * t.entry_price * self.FEE_RATE
            + (t.quantity * (t.exit_price or 0) * self.FEE_RATE)
            for t in closed
        )

        logger.info(f"Signal distribution: {sig_buy}B / {sig_sell}S / {sig_hold}H (after {len(featured)} bars)")

        result = BacktestResult(
            total_return=(capital - self.initial_capital) / self.initial_capital,
            total_trades=total, winning_trades=len(winning), losing_trades=len(losing),
            win_rate=win_rate, sharpe_ratio=sharpe,
            max_drawdown=max_dd_val, max_drawdown_pct=max_dd_pct,
            total_fees=total_fees, net_profit=capital - self.initial_capital,
            initial_capital=self.initial_capital, final_capital=capital,
            trades=closed, equity_curve=equity_curve,
            timestamps=timestamps, signals=signals,
        )
        logger.info(
            f"Backtest: {total} trades ({len([t for t in closed if t.side == OrderSide.BUY])}L/"
            f"{len([t for t in closed if t.side == OrderSide.SELL])}S), "
            f"return {result.total_return:.2%}, Sharpe {result.sharpe_ratio:.2f}, "
            f"max DD {result.max_drawdown_pct:.2%}"
        )
        return result
