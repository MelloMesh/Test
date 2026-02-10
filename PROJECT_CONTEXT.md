# PROJECT CONTEXT — Agentic Crypto Trading System

> **READ THIS FILE EVERY SESSION.** It defines the bot's architecture, purpose, current
> state, and conventions. Update it as the system evolves.

---

## 1. Purpose

Build a **self-improving, agentic crypto trading bot** that:

- Connects to crypto exchanges (Binance initially, extensible via CCXT)
- Executes trades autonomously based on configurable strategies
- Uses **recursive learning**: reviews its own trade history, identifies what worked
  and what didn't, and adjusts strategy parameters over time
- Manages risk with hard limits, position sizing, and drawdown protection
- Logs everything for auditability and learning

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                  Agent Orchestrator                  │
│           (main loop / decision engine)              │
├──────────┬──────────┬───────────┬───────────────────┤
│ Exchange │ Strategy │   Risk    │    Recursive      │
│Connector │  Engine  │  Manager  │    Learner        │
├──────────┼──────────┼───────────┼───────────────────┤
│  CCXT    │ Signals  │ Position  │  Trade Journal    │
│  Binance │ Indicat. │ Sizing    │  Performance      │
│  WebSock │ Scoring  │ Stop-loss │  Parameter Tuning │
│  REST    │ Filters  │ Drawdown  │  Feedback Loop    │
├──────────┴──────────┴───────────┴───────────────────┤
│              Shared: Config · Logger · Models        │
└─────────────────────────────────────────────────────┘
```

---

## 3. Directory Structure

```
Test/
├── PROJECT_CONTEXT.md          # THIS FILE — read every session
├── requirements.txt            # Python dependencies
├── .env.example                # Template for secrets (never commit .env)
├── .gitignore
├── config/
│   └── settings.yaml           # Runtime configuration (pairs, intervals, limits)
├── src/
│   ├── __init__.py
│   ├── main.py                 # Entry point — starts the orchestrator
│   ├── config.py               # Loads settings.yaml + env vars
│   ├── logger.py               # Structured logging setup
│   ├── models.py               # Data classes: Candle, Signal, Trade, Position
│   ├── exchange/
│   │   ├── __init__.py
│   │   ├── connector.py        # CCXT wrapper — fetch candles, place orders
│   │   └── websocket.py        # (future) live price streaming
│   ├── strategy/
│   │   ├── __init__.py
│   │   ├── engine.py           # Evaluates strategies, produces Signals
│   │   ├── indicators.py       # TA functions: RSI, EMA, MACD, Bollinger, etc.
│   │   └── filters.py          # Volume, spread, volatility filters
│   ├── risk/
│   │   ├── __init__.py
│   │   ├── manager.py          # Position sizing, max drawdown, exposure caps
│   │   └── portfolio.py        # Tracks open positions, PnL, equity curve
│   ├── learning/
│   │   ├── __init__.py
│   │   ├── journal.py          # Persists every trade with full context
│   │   ├── analyzer.py         # Reviews journal, computes win rate, Sharpe, etc.
│   │   └── tuner.py            # Adjusts strategy params based on analysis
│   └── orchestrator/
│       ├── __init__.py
│       └── agent.py            # Main loop: fetch → analyze → decide → execute → learn
├── data/
│   ├── trades/                 # Trade journal JSON files
│   └── snapshots/              # Strategy parameter snapshots after tuning
└── tests/
    ├── __init__.py
    ├── test_indicators.py
    ├── test_risk.py
    └── test_learning.py
```

---

## 4. Module Responsibilities

### 4.1 Exchange Connector (`src/exchange/connector.py`)
- Wraps CCXT for exchange-agnostic API calls
- Methods: `fetch_candles()`, `place_order()`, `get_balance()`, `cancel_order()`
- Handles rate limiting, retries, error normalization
- Supports paper trading mode (no real orders)

### 4.2 Strategy Engine (`src/strategy/engine.py`)
- Accepts candle data, runs indicators, applies filters
- Produces `Signal` objects (BUY / SELL / HOLD) with confidence scores
- Strategies are composable — multiple indicator signals get aggregated
- Default strategy: EMA crossover + RSI confirmation + volume filter

### 4.3 Risk Manager (`src/risk/manager.py`)
- Validates signals before execution
- Enforces: max position size, max open positions, max daily loss, max drawdown
- Calculates position size using Kelly criterion or fixed-fraction
- Attaches stop-loss and take-profit to every trade

### 4.4 Portfolio Tracker (`src/risk/portfolio.py`)
- Tracks all open/closed positions
- Calculates real-time PnL, equity curve, unrealized gains
- Provides portfolio state to orchestrator and risk manager

### 4.5 Recursive Learner (`src/learning/`)
- **Journal** (`journal.py`): Logs every trade — entry/exit prices, strategy params,
  indicators at time of signal, outcome, slippage
- **Analyzer** (`analyzer.py`): Periodically reviews N most recent trades. Computes
  win rate, profit factor, Sharpe ratio, max drawdown, avg hold time.
  Identifies which strategy params correlate with winners vs losers.
- **Tuner** (`tuner.py`): Adjusts strategy parameters (e.g., RSI thresholds,
  EMA periods) within safe bounds. Saves parameter snapshots. Uses simple
  hill-climbing initially, extensible to Bayesian optimization later.

### 4.6 Agent Orchestrator (`src/orchestrator/agent.py`)
- Main event loop: runs on configurable interval (e.g., every 5 minutes)
- Each tick: fetch data → run strategy → check risk → execute (or skip) → log
- After N trades, triggers the learning cycle (analyze → tune → snapshot)
- Handles graceful shutdown, error recovery, state persistence

---

## 5. Key Data Models (`src/models.py`)

| Model      | Fields                                                                 |
|------------|------------------------------------------------------------------------|
| `Candle`   | timestamp, open, high, low, close, volume                              |
| `Signal`   | symbol, direction (BUY/SELL/HOLD), confidence, strategy_name, metadata |
| `Trade`    | id, symbol, side, entry_price, exit_price, quantity, pnl, timestamps   |
| `Position` | symbol, side, entry_price, quantity, stop_loss, take_profit, status     |

---

## 6. Configuration (`config/settings.yaml`)

```yaml
exchange:
  name: binance
  paper_mode: true            # Start in paper mode — no real money
  api_key_env: EXCHANGE_API_KEY
  api_secret_env: EXCHANGE_API_SECRET

trading:
  symbols: ["BTC/USDT", "ETH/USDT"]
  timeframe: "5m"
  tick_interval_seconds: 300

strategy:
  ema_fast: 12
  ema_slow: 26
  rsi_period: 14
  rsi_overbought: 70
  rsi_oversold: 30
  min_volume_usdt: 100000
  signal_threshold: 0.6

risk:
  max_position_pct: 0.05     # 5% of portfolio per trade
  max_open_positions: 3
  max_daily_loss_pct: 0.03   # 3% daily loss → halt trading
  max_drawdown_pct: 0.10     # 10% drawdown → halt trading
  stop_loss_pct: 0.02        # 2% stop loss
  take_profit_pct: 0.04      # 4% take profit

learning:
  review_after_n_trades: 20
  lookback_trades: 50
  param_adjustment_step: 0.05
  max_param_deviation: 0.30  # Max 30% drift from defaults
  snapshot_dir: "data/snapshots"
```

---

## 7. Tech Stack

- **Language**: Python 3.11+
- **Exchange**: CCXT (Binance default)
- **Indicators**: pandas + ta-lib (or custom)
- **Data**: pandas DataFrames, JSON persistence
- **Logging**: Python `logging` with structured JSON output
- **Config**: PyYAML + python-dotenv
- **Testing**: pytest

---

## 8. Development Conventions

- All code in `src/`, tests in `tests/`
- Type hints on all function signatures
- Docstrings on public functions
- No secrets in code — use `.env` + `settings.yaml`
- Paper mode ON by default — explicit flag to go live
- Every trade gets journaled, no exceptions
- Strategy parameter changes are snapshot'd before/after

---

## 9. Current Status

| Component            | Status      | Notes                           |
|----------------------|-------------|---------------------------------|
| Project scaffold     | IN PROGRESS | Building directory structure    |
| Config system        | PENDING     |                                 |
| Exchange connector   | PENDING     |                                 |
| Strategy engine      | PENDING     |                                 |
| Risk manager         | PENDING     |                                 |
| Portfolio tracker    | PENDING     |                                 |
| Recursive learner    | PENDING     |                                 |
| Orchestrator         | PENDING     |                                 |
| Tests                | PENDING     |                                 |

---

## 10. Session Checklist

Every time we start a new session:

1. Read `PROJECT_CONTEXT.md` (this file)
2. Check git status and current branch
3. Review what was last worked on (Section 9)
4. Continue from where we left off
5. Update Section 9 before ending the session
