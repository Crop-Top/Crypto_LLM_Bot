"""
CLI headless entry point. Runs the bot in a specified mode without the dashboard.

Usage:
    python run.py --mode backtest --days 60
    python run.py --mode demo
    python run.py --mode live
"""

import argparse
import sys
import time

from loguru import logger

from src.backtest.engine import BacktestEngine
from src.config import Settings
from src.data.market import DataHandler
from src.trading.executor import LiveExecutor, PaperExecutor
from src.trading.predictor import TradePredictor
from src.trading.strategy import MLStrategy, Signal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crypto Trading Bot — CLI")
    parser.add_argument("--mode", choices=["backtest", "demo", "live"], default="backtest", help="Operating mode")
    parser.add_argument("--days", type=int, default=60, help="Days of historical data for backtest")
    parser.add_argument("--capital", type=float, default=10_000.0, help="Initial capital for backtest/demo")
    parser.add_argument("--confidence", type=float, default=0.55, help="Signal confidence threshold")
    return parser.parse_args()


def run_backtest(args: argparse.Namespace) -> None:
    logger.info("=== BACKTEST MODE ===")
    settings = Settings()
    dh = DataHandler(settings)
    data = dh.fetch_historical(days=args.days)
    if data.empty:
        logger.error("No data fetched. Exiting.")
        return

    predictor = TradePredictor()
    predictor.train(data)
    strategy = MLStrategy(predictor, confidence_threshold=args.confidence)
    engine = BacktestEngine(strategy, initial_capital=args.capital)
    result = engine.run(data)

    print("\n" + "=" * 50)
    print("BACKTEST RESULTS")
    print("=" * 50)
    print(f"  Total Return:    {result.total_return:>8.2%}")
    print(f"  Net Profit:      ${result.net_profit:>8,.2f}")
    print(f"  Sharpe Ratio:    {result.sharpe_ratio:>8.2f}")
    print(f"  Max Drawdown:    {result.max_drawdown_pct:>8.2%}")
    print(f"  Win Rate:        {result.win_rate:>8.1%}")
    print(f"  Total Trades:    {result.total_trades:>8}")
    print(f"  Final Capital:   ${result.final_capital:>8,.2f}")
    print("=" * 50)


def run_live(args: argparse.Namespace, is_demo: bool) -> None:
    mode = "DEMO" if is_demo else "LIVE"
    logger.info(f"=== {mode} MODE ===")
    settings = Settings()
    dh = DataHandler(settings)
    executor = PaperExecutor(settings, initial_balance=args.capital) if is_demo else LiveExecutor(settings)

    predictor = TradePredictor()
    logger.info("Training on recent data...")
    historical = dh.fetch_historical(days=30)
    if not historical.empty:
        predictor.train(historical)

    strategy = MLStrategy(predictor, confidence_threshold=args.confidence)
    logger.info(f"Starting {mode} loop for {settings.symbol} @ {settings.timeframe}")

    try:
        while True:
            latest = dh.fetch_latest()
            if latest.empty:
                time.sleep(30)
                continue

            price = float(latest["close"].iloc[-1])
            signal = strategy.generate_signal(latest)
            balance = executor.get_balance("USDT")
            logger.info(f"[{mode}] Price={price:.2f} | Signal={signal.value} | Balance={balance:.2f}")

            if signal in (Signal.BUY, Signal.SELL):
                executor.place_order(side=signal.value.lower())

            time.sleep(60)
    except KeyboardInterrupt:
        logger.info("Shutdown by user.")


def main() -> None:
    args = parse_args()
    from src.utils.logger import setup_logger
    setup_logger("cli")

    if args.mode == "backtest":
        run_backtest(args)
    elif args.mode == "demo":
        run_live(args, is_demo=True)
    elif args.mode == "live":
        run_live(args, is_demo=False)


if __name__ == "__main__":
    main()
