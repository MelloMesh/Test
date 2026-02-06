# Crypto Perpetuals Trading Agent

Automated trading agent for Binance USDT-M perpetual futures.
Strategy: RSI Divergence entries at Fibonacci Golden Pocket levels.

---

## Strategy Summary

**Where:** Price must be in the Fibonacci golden pocket (0.618 - 0.886 retracement)
**When:** 15m/30m RSI divergence forming after overbought/oversold peak
**Entry:** Anticipatory — enter as divergence develops, before full confirmation
**Stop:** Below 0.886 fib level (max 5% from entry). Moves to breakeven on confirmation candle close.
**Take Profit:** Tiered exits at 0.5, 0.382, 0.236 fib levels and extensions. Allocation optimized by backtest.

---

## Risk Rules

| Rule | Value |
|------|-------|
| Default risk per trade | 1% of equity |
| Max risk per trade | 2% (full confluence only) |
| Max total exposure | 5% at any time |
| Max concurrent positions | 3 |
| Daily loss halt | 3% → auto-stop 24h |
| Weekly loss halt | 7% → manual review required |
| Consecutive loss response | 3 losses → reduce to 0.5% |
| Margin mode | Isolated only |
| Leverage | 3-5x preferred, 10x absolute max |
| Stop loss | Required on every trade, no exceptions |

---

## Project Structure

```
crypto_agent/
├── src/
│   ├── __init__.py
│   ├── config.py              # All settings, env vars, risk params
│   ├── exchange.py            # Binance connection (testnet/live)
│   ├── data/
│   │   ├── __init__.py
│   │   ├── fetcher.py         # Fetch OHLCV, funding rates
│   │   └── storage.py         # SQLite storage for candles
│   ├── indicators/
│   │   ├── __init__.py
│   │   ├── rsi.py             # RSI calculation + peak/divergence detection
│   │   ├── fibonacci.py       # Swing detection + fib level calculation
│   │   └── volume.py          # OBV, volume confirmation
│   ├── strategy/
│   │   ├── __init__.py
│   │   ├── signals.py         # Signal dataclass
│   │   └── golden_pocket.py   # Core strategy: fib + RSI divergence
│   ├── risk/
│   │   ├── __init__.py
│   │   ├── position_sizer.py  # Size from risk% and stop distance
│   │   ├── risk_manager.py    # Enforce limits, circuit breakers
│   │   └── portfolio.py       # Track equity, open positions, P&L
│   ├── backtest/
│   │   ├── __init__.py
│   │   ├── engine.py          # Core backtest loop
│   │   ├── costs.py           # Fees, slippage, funding rate simulation
│   │   ├── stop_manager.py    # Simulate stop-to-breakeven logic
│   │   ├── tp_optimizer.py    # Test TP allocation combinations
│   │   └── report.py          # Performance metrics + output
│   ├── execution/
│   │   ├── __init__.py
│   │   ├── order_manager.py   # Place/cancel/modify orders via ccxt
│   │   ├── position_tracker.py# Sync positions with exchange
│   │   ├── stop_handler.py    # Monitor for confirmation candle → move stop
│   │   └── paper_mode.py      # Mock execution layer
│   ├── agent/
│   │   ├── __init__.py
│   │   └── trading_agent.py   # Main trading loop
│   ├── monitoring/
│   │   ├── __init__.py
│   │   └── dashboard.py       # Terminal status display
│   └── utils/
│       ├── __init__.py
│       └── logger.py          # Structured logging
├── tests/
│   ├── test_exchange.py
│   ├── test_fetcher.py
│   ├── test_storage.py
│   ├── test_rsi.py
│   ├── test_fibonacci.py
│   ├── test_strategy.py
│   ├── test_position_sizer.py
│   ├── test_risk_manager.py
│   └── test_backtest.py
├── data/                      # SQLite DB (gitignored)
├── logs/                      # Log files (gitignored)
├── .env.example               # Template for API keys
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Setup

### Prerequisites
- Python 3.11+
- Binance Futures Testnet account (for paper trading)

### Install
```bash
cd crypto_agent
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your testnet API keys
```

### Get Testnet API Keys
1. Go to https://testnet.binancefuture.com
2. Log in with GitHub
3. API Management → Generate key + secret
4. Paste into .env

### Run Tests
```bash
python3 -m pytest tests/ -v
```

---

## Usage

### Fetch and store candle data
```python
from src.data.fetcher import fetch_candles, fetch_candles_bulk
from src.data.storage import store_candles, load_candles, count_candles

# Fetch recent candles
candles = fetch_candles("BTC/USDT:USDT", "15m", limit=200)
store_candles(candles, "BTC/USDT:USDT", "15m")
print(f"Stored {count_candles()} candles")

# Bulk fetch historical data
candles = fetch_candles_bulk("BTC/USDT:USDT", "15m", days=30)
store_candles(candles, "BTC/USDT:USDT", "15m")
```

### Generate signals (Phase 2+)
```python
from src.strategy.golden_pocket import scan_for_signals
signals = scan_for_signals("BTC/USDT:USDT", days=7)
for s in signals:
    print(f"{s.direction} | {s.reason} | risk={s.risk_pct}")
```

### Run backtest (Phase 4+)
```bash
python3 -m src.backtest.engine \
    --symbol BTC/USDT:USDT \
    --timeframes 15m,30m \
    --days 60 \
    --initial-equity 10000 \
    --optimize-tp
```

### Paper trading (Phase 5+)
```bash
python3 -m src.agent.trading_agent --mode paper
```

---

## Build Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Data Pipeline + Scaffold | ✅ Complete |
| 2 | Indicators + Signals | ✅ Complete |
| 3 | Risk Manager | ✅ Complete |
| 4 | Backtesting | ✅ Complete |
| 5 | Paper Trading | ✅ Complete |
| 6 | Live Trading | ☐ Not started |

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `binanceusdm requires apiKey and secret` | Only for private endpoints. Public data doesn't need keys. |
| Connection timeout | Add `timeout: 30000` to exchange config. Testnet can be slow. |
| `insufficient margin` | Position too large for testnet balance (~1000 USDT). |
| `rate limit exceeded` | Enable `exchange.enableRateLimit = True`. |
| ImportError | `pip install -r requirements.txt` |
| Swing detection finds too many/too few swings | Adjust `min_bars` parameter in config. |
| No signals generated for days | Expected — strategy is selective. Check fib levels and RSI peaks in logs. |

---

## Live Trading Checklist

- [ ] Paper trading profitable 3+ consecutive months
- [ ] Max drawdown under 10%
- [ ] Win rate > 50%, profit factor > 1.3
- [ ] Understand every line of code
- [ ] Risk limits and circuit breakers tested
- [ ] Only risking money you can afford to lose

---

## Disclaimer

Educational project. Crypto derivatives carry substantial risk. Leveraged trading can result in losses exceeding initial investment. Past performance does not guarantee future results.
