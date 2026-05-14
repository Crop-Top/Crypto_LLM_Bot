# Crypto Trading Bot — Project Handover

## Goal
Build a modular, professional-grade cryptocurrency trading bot with:
- ML-based trade prediction (RandomForest on 33 technical features)
- Scalping strategy with 2.5:1 risk/reward (1% risk, 2.5% target per trade)
- Both long and short positions, with SL/TP exits
- Optional multi-timeframe trend filter (e.g., 4h SMA50)
- Web dashboard (Streamlit) with mode switching: Backtest / Demo / Live

## Current State (May 2026)

### What Works
- **Data Layer**: OHLCV + Open Interest + Funding Rate fetching from Bybit (CCXT), pagination, retry logic, symbol auto-resolution (`BTC/USDT` → `BTC/USDT:USDT`)
- **ML Predictor**: RandomForest classifier, binary median-split labels (always balanced Buy/Sell), 33 features (RSI, MACD, BB, ATR, volume, OI, funding rate, time features), StandardScaler
- **Strategy**: Binary Buy/Sell with configurable confidence threshold (default 0.55 → signals below become Hold)
- **Backtest Engine**: Vectorized predictions (fast — 8760 bars in ~2s), leverage/margin accounting, SL/TP exits per bar, long+short with position flipping, multi-timeframe trend filter, per-trade PnL tracking
- **Executors**: PaperExecutor (virtual balance) and LiveExecutor (CCXT real orders)
- **Dashboard**: 3 modes via sidebar radio, trade explorer chart (entry/exit markers, SL/TP lines, long vs short triangles), equity curve, trade table with exit reasons, strategy info panel
- **Config**: Frozen dataclass from `.env`, sidebar overrides propagate to DataHandler at runtime

### Key Metrics (30d 1h backtest, $10k, 5x lev, 20% pos)
| Leverage | Return | Max DD | Trades | Win Rate |
|----------|--------|--------|--------|----------|
| 1x | ~8% | ~20% | ~70 | ~60% |
| 5x | ~46% | ~57% | ~70 | ~60% |
| 10x | ~111% | ~73% | ~70 | ~60% |

### Known Issues / Gotchas
- Equity curve MTM uses `capital + locked_margin + unrealized_pnl` (not `capital + position * price`)
- Short MTM formula was `2 * entry - c` (double-count bug), now `entry - c`
- Labels were fixed thresholds (`±0.3%`), now binary median split (no look-ahead bias, always balanced)
- OI+failing endpoints return empty DataFrame silently (non-fatal)
- Windows: `n_jobs=1` required (process spawning overhead with `n_jobs=-1`)

## Architecture

```
app.py                          # streamlit run app.py
run.py                          # python run.py --mode backtest|demo|live
src/
├── config.py                   # Frozen dataclass from .env
├── data/market.py              # DataHandler: fetch OHLCV+OI+funding via CCXT
├── trading/
│   ├── models.py               # Signal, Trade, OrderResult, BacktestResult
│   ├── predictor.py            # FeatureEngineer + TradePredictor (RandomForest)
│   ├── strategy.py             # BaseStrategy ABC + MLStrategy
│   ├── executor.py             # PaperExecutor + LiveExecutor
│   └── risk.py                 # RiskManager (stop-loss, daily loss cap)
├── backtest/engine.py          # BacktestEngine: vectorized, leverage, SL/TP, HTF
├── state/manager.py            # BotState singleton (thread-safe)
├── ui/dashboard.py             # Streamlit app with 3 modes
└── utils/logger.py             # Loguru setup
```

## Key Files to Know

| File | Purpose |
|------|---------|
| `.env` | API keys + all trading params |
| `src/backtest/engine.py` | Core backtest logic — SL/TP, lev, long/short, HTF filter |
| `src/trading/predictor.py` | Feature engineering + model training + prediction |
| `src/data/market.py` | Bybit data fetching with pagination |
| `src/ui/dashboard.py` | Streamlit UI — sidebar, backtest page, trade explorer |

## `.env` Configuration

```
BYBIT_API_KEY=...
BYBIT_API_SECRET=...
SYMBOL=BTC/USDT:USDT           # :USDT suffix for linear perpetuals
TIMEFRAME=1h
LEVERAGE=5
POSITION_SIZE_USDT=100
STOP_LOSS_PCT=0.02
TAKE_PROFIT_PCT=0.05
USE_TESTNET=True
```

## What's Next / Todo

1. **Live trading**: Demo mode works (PaperExecutor). Live mode needs testing with real keys.
2. **Model persistence**: Save/load trained model so it doesn't retrain every dashboard restart.
3. **Slippage modeling**: Current fixed 0.05% — could use order book or volume-based slippage.
4. **Walk-forward optimization**: Currently trains on whole backtest period, which is look-ahead biased.
5. **Position sizing**: `position_pct` is per-trade % of capital — could be dynamic (Kelly, volatility-adjusted).
6. **Take profit logic**: The current engine checks `low <= TP` for shorts and `high >= TP` for longs. This works but could be more sophisticated (trailing stops, partial TP).
7. **Risk per trade**: Current `risk_pct=0.01` uses fixed % — should this be configurable from sidebar?
8. **Portfolio mode**: Multiple symbols traded simultaneously.

## Commands

```powershell
pip install -r requirements.txt
streamlit run app.py                              # Dashboard
python run.py --mode backtest --days 60           # CLI backtest
python run.py --mode demo --days 30               # Paper trading loop
```

## Common Pitfalls

- **Symbol format**: Must end with `:USDT` for linear perpetuals (e.g., `BTC/USDT:USDT`), not `BTC/USDT`
- **API keys**: Backtest/Demo don't need keys (uses public endpoints). Live mode requires valid Bybit keys.
- **Testnet vs Mainnet**: Public data always fetches from mainnet. Set `USE_TESTNET=True` in `.env` for paper/live on testnet.
- **Backtest speed**: Data fetching is the bottleneck (~20s per 365d of 1h data). The engine itself runs 8760 bars in ~2s.
- **Enums**: `Signal.BUY = "Buy"`, `Signal.SELL = "Sell"`, `Signal.HOLD = "Hold"`. Trading uses `OrderSide.BUY = "buy"`, `OrderSide.SELL = "sell"`.
