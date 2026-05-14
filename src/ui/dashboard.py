import threading
import time
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.backtest.engine import BacktestEngine
from src.config import Settings
from src.data.market import DataHandler
from loguru import logger

from src.state.manager import BotState, state
from src.trading.executor import LiveExecutor, PaperExecutor
from src.trading.predictor import TradePredictor
from src.trading.risk import RiskManager
from src.trading.strategy import MLStrategy, Signal

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Crypto Trading Bot",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session-level caching of heavy objects
# ---------------------------------------------------------------------------
if "settings" not in st.session_state:
    st.session_state.settings = Settings()
if "data_handler" not in st.session_state:
    st.session_state.data_handler = DataHandler(st.session_state.settings)
if "predictor" not in st.session_state:
    st.session_state.predictor = TradePredictor()
if "risk_manager" not in st.session_state:
    st.session_state.risk_manager = RiskManager(st.session_state.settings)


# ---------------------------------------------------------------------------
# Sidebar — mode selection & global config
# ---------------------------------------------------------------------------
def render_sidebar() -> str:
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/cryptocurrency.png", width=48)
        st.title("Crypto Bot")

        mode = st.radio(
            "Mode",
            ["Backtest", "Demo", "Live"],
            index=0,
            help="Backtest: run on historical data | Demo: paper trade with live data | Live: real orders",
        )

        st.divider()
        st.subheader("Configuration")

        settings = st.session_state.settings
        symbol = st.text_input("Symbol", value=settings.symbol)
        timeframe = st.selectbox(
            "Timeframe",
            ["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
            index=["1m", "5m", "15m", "30m", "1h", "4h", "1d"].index(settings.timeframe),
        )
        pos_size = st.number_input("Position Size (USDT)", min_value=10.0, value=settings.position_size_usdt, step=10.0)
        leverage = st.number_input("Leverage", min_value=1, max_value=125, value=settings.leverage)
        confidence = st.slider("Signal Confidence Threshold", 0.5, 0.95, 0.55, 0.05)

        use_htf = st.checkbox("Multi-Timeframe Filter", value=False, help="Use higher timeframe trend to filter trade direction")
        htf_timeframe = "4h"
        if use_htf:
            htf_timeframe = st.selectbox(
                "Higher Timeframe", ["1h", "4h", "1d"],
                index=1,
                help="Trades only taken in direction of HTF trend (close > SMA50)",
            )

        st.session_state["ui_symbol"] = symbol
        st.session_state["ui_timeframe"] = timeframe
        st.session_state["ui_pos_size"] = pos_size
        st.session_state["ui_leverage"] = leverage
        st.session_state["ui_confidence"] = confidence
        st.session_state["ui_use_htf"] = use_htf
        st.session_state["ui_htf_tf"] = htf_timeframe

        # Propagate to DataHandler so backtest uses sidebar values
        dh = st.session_state.get("data_handler")
        if dh is not None:
            dh.timeframe = timeframe
            dh.symbol = dh._resolve_symbol(raw=symbol)

        st.divider()
        st.subheader("Strategy")
        tf = st.session_state.get("ui_timeframe", "1h")
        htf_info = f" + {st.session_state.get('ui_htf_tf', '4h')} trend filter" if st.session_state.get("ui_use_htf", False) else ""
        st.markdown(
            f"**ML Scalper** on **{tf}**{htf_info}\n\n"
            f"RandomForest → binary Buy/Sell → 2.5:1 RR scalping with SL/TP exits\n\n"
            f"33 features: RSI, MACD, BB, ATR, volume, OI, funding rate"
        )

        st.divider()
        st.caption(f"Bot status: **{state.mode.upper()}**")
        if state.is_running:
            st.success("● Running")
        else:
            st.warning("○ Stopped")

        if state.error:
            st.error(state.error)

        return mode


# ---------------------------------------------------------------------------
# Backtest page
# ---------------------------------------------------------------------------
def render_backtest() -> None:
    st.header("Backtest Strategy")

    col1, col2, col3 = st.columns(3)
    with col1:
        days = st.number_input("Days of historical data", min_value=7, max_value=1825, value=60, step=30)
    with col2:
        initial_capital = st.number_input("Initial Capital (USDT)", min_value=100.0, value=10_000.0, step=500.0)
    with col3:
        confidence = st.slider("Confidence Threshold", 0.50, 0.95, 0.55, 0.05, key="bt_conf")

    use_htf = st.session_state.get("ui_use_htf", False)
    htf_tf = st.session_state.get("ui_htf_tf", "4h")
    run_btn = st.button("▶ Run Backtest", type="primary", use_container_width=True)

    if run_btn:
        state.update(mode="backtest", is_running=True, error=None)
        with st.spinner("Fetching data and running backtest..."):
            try:
                dh = st.session_state.data_handler
                data = dh.fetch_historical(days=int(days))
                if data.empty:
                    st.error("No data returned. Check symbol and exchange.")
                    state.update(is_running=False, error="Empty data")
                    return

                htf_data = None
                if use_htf and htf_tf != st.session_state.get("ui_timeframe", "1h"):
                    logger.info(f"Fetching {htf_tf} data for multi-timeframe filter...")
                    raw = dh.exchange.fetch_ohlcv(dh.symbol, timeframe=htf_tf, limit=500)
                    htf_data = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
                    htf_data["timestamp"] = pd.to_datetime(htf_data["timestamp"], unit="ms", utc=True)
                    htf_data.set_index("timestamp", inplace=True)

                predictor = TradePredictor()
                predictor.train(data)
                strategy = MLStrategy(predictor, confidence_threshold=confidence)
                engine = BacktestEngine(
                    strategy,
                    initial_capital=float(initial_capital),
                    leverage=float(st.session_state.get("ui_leverage", 5)),
                )
                result = engine.run(data, htf_data=htf_data)

                st.session_state["bt_data"] = data
                state.update(backtest_result=result, is_running=False)
            except Exception as e:
                st.error(f"Backtest failed: {e}")
                state.update(is_running=False, error=str(e))

    # Display results if available
    if state.backtest_result is not None:
        result = state.backtest_result

        st.subheader("Performance Metrics")
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Total Return", f"{result.total_return:.2%}")
        m2.metric("Net Profit", f"${result.net_profit:,.2f}")
        m3.metric("Sharpe Ratio", f"{result.sharpe_ratio:.2f}")
        m4.metric("Max Drawdown", f"{result.max_drawdown_pct:.2%}")
        m5.metric("Win Rate", f"{result.win_rate:.1%}")
        m6.metric("Total Trades", str(result.total_trades))

        # Equity curve
        fig_equity = go.Figure()
        fig_equity.add_trace(go.Scatter(
            x=result.timestamps,
            y=result.equity_curve,
            mode="lines",
            name="Equity",
            line=dict(color="#00BFA5"),
            fill="tozeroy",
        ))
        fig_equity.add_hline(y=result.initial_capital, line_dash="dash", line_color="gray", annotation_text="Initial")
        fig_equity.update_layout(
            title="Equity Curve",
            height=350,
            template="plotly_dark",
            yaxis_title="Capital (USDT)",
        )
        st.plotly_chart(fig_equity, use_container_width=True)

        # Drawdown chart
        equity_arr = np.array(result.equity_curve)
        peak = np.maximum.accumulate(equity_arr)
        dd = (peak - equity_arr) / peak * 100
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=result.timestamps,
            y=dd,
            mode="lines",
            name="Drawdown %",
            line=dict(color="#FF5252"),
            fill="tozeroy",
        ))
        fig_dd.update_layout(
            title="Drawdown",
            height=200,
            template="plotly_dark",
            yaxis_title="Drawdown (%)",
        )
        st.plotly_chart(fig_dd, use_container_width=True)

        # Trade Explorer — price chart with SL/TP markers
        if result.trades and "bt_data" in st.session_state:
            bt_data = st.session_state["bt_data"]
            st.subheader("Trade Explorer")

            fig_trades = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.05)

            # Candlestick chart
            fig_trades.add_trace(go.Candlestick(
                x=bt_data.index, open=bt_data["open"], high=bt_data["high"],
                low=bt_data["low"], close=bt_data["close"], name="Price",
            ), row=1, col=1)

            for t in result.trades:
                color = "#00C853" if t.pnl > 0 else "#FF1744"
                entry_x = t.entry_time
                exit_x = t.exit_time or entry_x
                is_long = t.side.value == "buy"

                # Entry marker
                symbol = "triangle-up" if is_long else "triangle-down"
                fig_trades.add_trace(go.Scatter(
                    x=[entry_x], y=[t.entry_price],
                    mode="markers", marker=dict(symbol=symbol, size=11, color=color, line=dict(width=1, color="white")),
                    name=f"Entry {t.id}", showlegend=False,
                ), row=1, col=1)

                # Exit marker
                exit_sym = "x" if t.exit_reason == "stop_loss" else "circle"
                if t.exit_price:
                    fig_trades.add_trace(go.Scatter(
                        x=[exit_x], y=[t.exit_price],
                        mode="markers", marker=dict(symbol=exit_sym, size=10, color=color),
                        name=f"Exit {t.id}", showlegend=False,
                    ), row=1, col=1)

                # SL/TP lines
                if t.stop_loss:
                    sl_color = "red"
                    fig_trades.add_trace(go.Scatter(
                        x=[entry_x, exit_x], y=[t.stop_loss, t.stop_loss],
                        mode="lines", line=dict(color=sl_color, width=1, dash="dash"),
                        name=f"SL {t.id}", showlegend=False,
                    ), row=1, col=1)
                if t.take_profit:
                    tp_color = "lime"
                    fig_trades.add_trace(go.Scatter(
                        x=[entry_x, exit_x], y=[t.take_profit, t.take_profit],
                        mode="lines", line=dict(color=tp_color, width=1, dash="dash"),
                        name=f"TP {t.id}", showlegend=False,
                    ), row=1, col=1)

            # Equity curve on second row
            fig_trades.add_trace(go.Scatter(
                x=result.timestamps, y=result.equity_curve,
                mode="lines", name="Equity", line=dict(color="#00BFA5"), fill="tozeroy",
            ), row=2, col=1)
            fig_trades.add_hline(y=result.initial_capital, line_dash="dash", line_color="gray", row=2, col=1)

            fig_trades.update_layout(
                height=600, template="plotly_dark", hovermode="x unified",
                xaxis_rangeslider_visible=False,
            )
            st.plotly_chart(fig_trades, use_container_width=True)

        # Trades table
        if result.trades:
            st.subheader("Trade Log")
            trades_df = pd.DataFrame([
                {
                    "Entry": t.entry_time.strftime("%m/%d %H:%M"),
                    "Exit": t.exit_time.strftime("%m/%d %H:%M") if t.exit_time else "",
                    "Side": t.side.value,
                    "Entry $": round(t.entry_price, 1),
                    "Exit $": round(t.exit_price or 0, 1),
                    "SL $": round(t.stop_loss, 1) if t.stop_loss else "",
                    "TP $": round(t.take_profit, 1) if t.take_profit else "",
                    "PnL $": round(t.pnl, 2),
                    "PnL %": f"{t.pnl_pct * 100:.1f}%" if t.pnl_pct else "",
                    "Exit Reason": t.exit_reason,
                }
                for t in result.trades
            ])
            st.dataframe(trades_df, use_container_width=True, hide_index=True)

            csv = trades_df.to_csv(index=False)
            st.download_button("📥 Download CSV", csv, "backtest_trades.csv", "text/csv")


# ---------------------------------------------------------------------------
# Demo / Live page
# ---------------------------------------------------------------------------
def run_bot_loop(mode: str, dh: DataHandler, predictor: TradePredictor, strategy: MLStrategy) -> None:
    """Background thread that runs the trading loop."""
    settings = st.session_state.settings
    if mode == "demo":
        executor = PaperExecutor(settings, initial_balance=10_000.0)
    else:
        executor = LiveExecutor(settings)

    risk = st.session_state.risk_manager
    state.update(mode=mode, is_running=True, error=None)

    # Train on recent data
    try:
        historical = dh.fetch_historical(days=30)
        if not historical.empty:
            predictor.train(historical)
    except Exception as e:
        state.update(error=f"Training failed: {e}")

    data_buffer = pd.DataFrame()

    while state.is_running:
        try:
            latest = dh.fetch_latest()
            if latest.empty:
                time.sleep(30)
                continue

            price = float(latest["close"].iloc[-1])
            data_buffer = pd.concat([data_buffer, latest]).tail(500) if not data_buffer.empty else latest
            if len(data_buffer) < 50:
                time.sleep(30)
                continue

            signal = strategy.generate_signal(data_buffer)
            _, raw_confidence = predictor.predict(data_buffer)

            # Update shared state
            balance = executor.get_balance("USDT") if hasattr(executor, "get_balance") else 0.0
            oi_v = float(latest.get("open_interest", pd.Series([0.0])).iloc[-1])
            fr_v = float(latest.get("funding_rate", pd.Series([0.0])).iloc[-1])
            snap = state.snapshot()
            oi_list = list(snap.oi_history) + ([oi_v] if oi_v else [])
            fr_list = list(snap.fr_history) + ([fr_v] if fr_v else [])

            state.update(
                current_price=price,
                current_signal=signal,
                signal_confidence=raw_confidence,
                balance_usdt=balance,
                price_history=data_buffer,
                oi_history=oi_list[-500:],
                fr_history=fr_list[-500:],
            )

            # Execute signal
            if signal in (Signal.BUY, Signal.SELL):
                side = signal.value.lower()
                if mode == "demo" or risk.check_daily_loss(state.unrealized_pnl):
                    executor.place_order(side=side, amount=0)

            time.sleep(60)

        except Exception as e:
            state.update(error=str(e))
            time.sleep(30)


def render_trading(mode: str) -> None:
    st.header(f"{mode} Trading")

    dh = st.session_state.data_handler
    predictor = st.session_state.predictor
    strategy = MLStrategy(predictor)

    col1, col2 = st.columns([3, 1])
    with col2:
        start_btn = st.button(
            f"▶ Start {mode}" if not state.is_running else "■ Stop",
            type="primary" if not state.is_running else "secondary",
            use_container_width=True,
        )

        if start_btn:
            if state.is_running:
                state.update(is_running=False, mode="idle")
            else:
                thread = threading.Thread(
                    target=run_bot_loop,
                    args=(mode.lower(), dh, predictor, strategy),
                    daemon=True,
                )
                thread.start()

    with col1:
        snap = state.snapshot()
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Price", f"${snap.current_price:,.2f}")
        m2.metric("Signal", snap.current_signal.value)
        m3.metric("Confidence", f"{snap.signal_confidence:.0%}")
        m4.metric("Balance", f"${snap.balance_usdt:,.2f}")
        m5.metric("Daily PnL", f"${snap.daily_pnl:,.2f}")

    snap = state.snapshot()
    if not snap.price_history.empty:
        df = snap.price_history
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            row_heights=[0.5, 0.25, 0.25],
        )

        fig.add_trace(
            go.Candlestick(
                x=df.index[-100:], open=df["open"][-100:], high=df["high"][-100:],
                low=df["low"][-100:], close=df["close"][-100:], name="Price",
            ),
            row=1, col=1,
        )

        if snap.oi_history:
            oi_idx = df.index[-len(snap.oi_history):] if len(df.index) >= len(snap.oi_history) else df.index
            fig.add_trace(
                go.Scatter(x=oi_idx, y=snap.oi_history[-len(oi_idx):], mode="lines", name="Open Interest", line=dict(color="orange")),
                row=2, col=1,
            )

        if snap.fr_history:
            fr_idx = df.index[-len(snap.fr_history):] if len(df.index) >= len(snap.fr_history) else df.index
            colors = ["green" if v >= 0 else "red" for v in snap.fr_history[-len(fr_idx):]]
            fig.add_trace(
                go.Bar(x=fr_idx, y=snap.fr_history[-len(fr_idx):], name="Funding Rate", marker_color=colors),
                row=3, col=1,
            )

        fig.update_layout(height=650, template="plotly_dark", hovermode="x unified", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    # Active trades / positions (simplified view)
    with st.expander("Active Trades", expanded=False):
        st.info(f"No active {mode.lower()} positions." if not state.is_running else "Fetching positions...")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    mode = render_sidebar()

    if mode == "Backtest":
        render_backtest()
    elif mode in ("Demo", "Live"):
        render_trading(mode)

    # Footer
    st.divider()
    snap = state.snapshot()
    st.caption(
        f"Last update: {snap.last_update.strftime('%Y-%m-%d %H:%M:%S') if snap.last_update else 'N/A'} | "
        f"Mode: {mode}"
    )


if __name__ == "__main__":
    main()
