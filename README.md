# Autonomous Crypto USDT Perpetual Futures Trading System
## Multi-Timeframe Signal Discovery, Backtesting & Live Trading

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 System Overview

This is an **autonomous, self-learning trading system** specialized for cryptocurrency USDT perpetual futures (BTCUSDT.P, ETHUSDT.P, etc.) across **multiple timeframes (2m, 5m, 15m, 30m)**.

### Core Philosophy

**We don't trust "best practices" — we TEST everything.**

If a signal wins 55%+ with positive profit factor in backtesting on crypto perpetuals, it earns the right to trade live. If it wins 48%, we kill it and try something else.

### Key Features

- ✅ **Multi-Timeframe Analysis**: 2m, 5m, 15m, 30m simultaneous trading
- ✅ **Autonomous Signal Discovery**: Generates 30-50+ hypotheses across all timeframes
- ✅ **Rigorous Backtesting**: 6+ months historical data with realistic slippage & funding costs
- ✅ **Cross-Timeframe Validation**: Higher timeframe confirmation for higher win rates
- ✅ **Funding Rate Integration**: Exploit perpetual futures funding extremes (70%+ reversal probability)
- ✅ **Regime Adaptation**: Automatically adjusts to trending/ranging/choppy markets
- ✅ **US-Accessible Exchanges**: Kraken (primary), Gemini (backup)

---

## 📊 What Makes This Different

### Traditional Approach
- Uses pre-built indicators blindly
- Single timeframe trading
- No funding rate consideration
- Generic strategies for all assets

### This System
- **Tests everything** on crypto perpetuals specifically
- **Multi-timeframe confluence** (30m trend → 15m confirm → 5m entry → 2m execute)
- **Funding rate exploitation** (extreme rates = mean reversion signals)
- **Learned system** that evolves with data
- **Timeframe-specific thresholds** (2m needs 2.0x PF, 30m needs 1.3x PF)

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone <your-repo-url>
cd Test

# Install dependencies
pip install -r requirements.txt

# Set up environment variables (optional)
cp .env.example .env
# Edit .env with your API keys
```

### 2. Configuration

Edit `config.py` to customize:

```python
# Instruments to trade
INSTRUMENTS = ["BTCUSDT.P", "ETHUSDT.P", "BNBUSDT.P"]

# Timeframes to test
TIMEFRAMES = ["2m", "5m", "15m", "30m"]

# Risk parameters
POSITION_SIZE_PCT = 0.02  # 2% per trade
MAX_CONCURRENT = 2        # Max 2 positions
DAILY_LOSS_LIMIT = -0.03  # -3% daily circuit breaker
```

### 3. Run System

#### Full Workflow (Discovery → Data → Backtest → Analysis)
```bash
python main.py --mode full
```

#### Quick Mode (Test Subset)
```bash
python main.py --mode full --quick
```

#### Individual Phases
```bash
# Signal Discovery Only
python main.py --mode discovery

# Backtesting Only (requires existing data)
python main.py --mode backtest --no-fetch

# Edge Analysis Only
python main.py --mode analysis
```

---

## 📁 Project Structure

```
Test/
├── config.py                          # System configuration
├── main.py                            # Main orchestration script
├── requirements.txt                   # Python dependencies
├── README.md                          # This file
│
├── src/
│   ├── data/
│   │   └── kraken_fetcher.py         # Kraken API data fetching
│   ├── signals/
│   │   ├── signal_discovery.py       # Generate 30-50+ hypotheses
│   │   └── cross_timeframe.py        # Multi-TF validation logic
│   ├── backtest/
│   │   └── backtest_engine.py        # Backtest with realistic costs
│   ├── analysis/
│   │   └── edge_analyzer.py          # Filter signals with edge
│   └── trading/                       # (Future: live trading)
│
├── data/
│   ├── raw/                           # Historical OHLC data
│   ├── processed/                     # Processed data
│   └── funding_rates/                 # Funding rate history
│
├── results/
│   ├── discovery/                     # Generated hypotheses
│   ├── backtests/                     # Backtest results
│   ├── analysis/                      # Edge analysis reports
│   └── trades/                        # Trade logs
│
├── notebooks/                         # Jupyter analysis notebooks
└── tests/                             # Unit tests
```

---

## 🔄 Execution Workflow

### Phase 1: Signal Discovery
Generates 30-50+ trading hypotheses across all timeframes:

**Examples:**
- **2m**: Momentum_Spike_Reversal_2m (62% expected accuracy)
- **5m**: RSI_Oversold_5m_Reversal (65% expected accuracy)
- **15m**: Breakout_Confluence_15m (58% expected accuracy)
- **30m**: Major_Level_Reversal_30m (55% expected accuracy)

**Output:** `results/discovery/hypotheses_multiframe.json`

### Phase 2: Data Fetching
Downloads historical data from Kraken:
- OHLC candles for all timeframes (6+ months)
- Funding rate history
- Handles rate limiting with exponential backoff

**Output:** `data/raw/*.csv`

### Phase 3: Backtesting
Tests every signal hypothesis with realistic simulation:
- Entry/exit slippage (1.0-4.0 pips depending on timeframe)
- Funding rate costs (8-hourly)
- Stop loss & target logic
- Regime detection

**Output:** `results/backtests/backtest_results_multiframe.json`

### Phase 4: Edge Analysis
Filters signals meeting timeframe-specific thresholds:

| Timeframe | Min Win Rate | Min Profit Factor | Min Sharpe |
|-----------|--------------|-------------------|------------|
| 2m        | 55%          | 2.0x              | 1.0        |
| 5m        | 55%          | 1.5x              | 1.0        |
| 15m       | 55%          | 1.4x              | 1.0        |
| 30m       | 55%          | 1.3x              | 1.0        |

**Output:** `results/analysis/edge_analysis_multiframe.json`

### Phase 5: Cross-Timeframe Validation
Validates signals across multiple timeframes:
- Does 5m entry align with 15m/30m trend?
- Does multi-TF confirmation improve win rate?

**Output:** Enhanced validation metrics

---

## 📈 Multi-Timeframe Strategy

### Timeframe Hierarchy

```
30m (Macro Trend)
  ↓ Confirms overall direction (bull/bear/range)

15m (Confirmation)
  ↓ Identifies pullbacks and setups

5m (Entry Trigger)
  ↓ Generates specific entry signals

2m (Execution)
  ↓ Fine-tunes entry point and manages exits
```

### Example Trade Flow

1. **30m**: Uptrend (price > 50-SMA, ADX > 25)
2. **15m**: Pullback near support (RSI < 40)
3. **5m**: Reversal signal (RSI oversold, OBV recovery)
4. **2m**: Entry confirmation (momentum spike)
5. **→ Execute long trade**

---

## 💰 Funding Rate Exploitation

Perpetual futures have unique funding rates that create edge:

| Funding Rate     | Interpretation              | Action                        |
|------------------|-----------------------------|-------------------------------|
| < 0.0001         | Neutral                     | Trade normally                |
| 0.0001 - 0.0002  | Moderate bullish            | Caution on longs              |
| > 0.0003         | **Extreme bullish**         | **Expect reversal (70%+ prob)** |
| < -0.0002        | Extreme bearish             | Expect bounce                 |

**Key Insight:** When funding > 0.0003, everyone is overly long → mean reversion opportunity.

---

## 🎯 Expected Performance

### After 4 Weeks of Live Trading

**By Timeframe:**
```
2m scalps:   Win 58%, PF 2.0x, Hold 2-5 min,     Trades/day 20-30
5m scalps:   Win 62%, PF 1.7x, Hold 5-15 min,    Trades/day 10-15
15m setups:  Win 56%, PF 1.4x, Hold 15-45 min,   Trades/day 3-5
30m moves:   Win 55%, PF 1.3x, Hold 30m-2h,      Trades/day 1-2
```

**Combined Performance:**
```
Overall Win Rate:        57-62%
Overall Profit Factor:   1.6x+
Monthly Return:          4-8%
Max Drawdown:            <15%
Total Trades/Month:      200-500
```

---

## ⚙️ Configuration Parameters

### Risk Management
```python
LEVERAGE = 1.0                    # NO LEVERAGE (1:1)
POSITION_SIZE_PCT = 0.02          # 2% per trade
MAX_CONCURRENT = 2                # Max 2 open positions
DAILY_LOSS_LIMIT = -0.03          # -3% daily → pause
MAX_DRAWDOWN = 0.15               # 15% max account drawdown
```

### Slippage Assumptions (Realistic)
```python
SLIPPAGE = {
    "2m": 1.0,    # pips (tight spreads)
    "5m": 1.5,    # pips
    "15m": 2.5,   # pips
    "30m": 4.0    # pips (wider spreads)
}
```

### Minimum Risk:Reward by Timeframe
```python
MIN_RISK_REWARD = {
    "2m": 2.0,    # Very tight, slippage matters
    "5m": 1.5,    # Standard
    "15m": 1.3,   # Wider targets
    "30m": 1.2    # Largest targets
}
```

---

## 🛡️ Risk Management

### Circuit Breakers
- **Daily Loss Limit**: -3% → Pause all trading
- **Max Drawdown**: 15% → Review and adjust
- **Max Concurrent Positions**: 2 trades only

### Position Sizing
- Fixed 2% per trade
- Dynamic weighting by market regime:
  - **Trending**: 40% on 30m, 30% on 15m, 20% on 5m, 10% on 2m
  - **Choppy**: 40% on 2m, 35% on 5m, 20% on 15m, 5% on 30m

---

## 🔧 Advanced Usage

### Custom Signal Creation

Add your own signals to `src/signals/signal_discovery.py`:

```python
signals.append(SignalHypothesis(
    id=self._generate_id("5m", "custom"),
    name="My_Custom_Signal_5m",
    timeframe="5m",
    description="Your signal description",
    entry_conditions=[
        "RSI(14) < 30",
        "Price > 50-SMA",
        "Volume > 1.5x average"
    ],
    stop_loss="1 ATR below entry",
    target="+30 pips",
    typical_hold_minutes=(5, 15),
    slippage_pips=1.5,
    expected_accuracy=0.60,
    signal_type="mean_reversion",
    indicators_used=["RSI", "SMA", "Volume"],
    regime_best="ranging",
    funding_rate_consideration=False
))
```

### Parameter Optimization

Modify indicator ranges in `config.py`:

```python
INDICATOR_RANGES = {
    "RSI": {
        "period": [7, 14, 21],
        "oversold": [20, 25, 30, 35],
        "overbought": [65, 70, 75, 80],
    },
    # Add more...
}
```

### Backtesting Custom Date Ranges

```python
# In your script
from src.backtest.backtest_engine import BacktestEngine
import config

engine = BacktestEngine(config)

# Modify data lookback
config.DATA_LOOKBACK_MONTHS = 12  # 12 months instead of 6

result = engine.backtest_signal(signal, "BTCUSDT.P")
```

---

## 📊 Results & Outputs

### Key Output Files

1. **Signal Hypotheses**: `results/discovery/hypotheses_multiframe.json`
   - All generated signal hypotheses
   - Entry conditions, expected performance

2. **Backtest Results**: `results/backtests/backtest_results_multiframe.json`
   - Win rate, profit factor, Sharpe ratio
   - Trade-by-trade breakdown
   - Regime analysis

3. **Edge Analysis**: `results/analysis/edge_analysis_multiframe.json`
   - Filtered signals with edge
   - Ranked by composite score
   - Cross-timeframe consistency analysis

4. **Final Report**: `results/final_report.json`
   - Complete execution summary
   - Performance metrics
   - Deployment recommendations

---

## 🧪 Testing

### Run Unit Tests
```bash
pytest tests/ -v
```

### Validate Data Fetching
```bash
python src/data/kraken_fetcher.py
```

### Test Signal Discovery
```bash
python src/signals/signal_discovery.py
```

### Test Backtesting Engine
```bash
python src/backtest/backtest_engine.py
```

---

## 🚨 Important Notes

### Exchange Selection for US Traders
```
✅ KRAKEN:        US accessible, perpetual futures, 99.8% uptime
✅ GEMINI:        US accessible, good alternative
⚠️  COINBASE:     Limited perpetual pairs
❌ BINANCE.COM:   US IP BLOCKED
❌ BINANCE.US:    Perpetual futures BLOCKED
```

### Funding Rate Timing
- Funding paid every 8 hours
- Check exact timestamps for your exchange
- Include in cost calculations

### No Leverage
- System uses **1:1 leverage only**
- Safer for scalping strategies
- Avoids liquidation risk

---

## 🔮 Roadmap

### Phase 1: Discovery & Backtesting ✅ (Current)
- [x] Multi-timeframe signal discovery
- [x] Kraken data fetching
- [x] Backtesting engine
- [x] Edge analysis

### Phase 2: Paper Trading (Next)
- [ ] Live paper trading simulation
- [ ] Real-time signal generation
- [ ] Performance tracking vs. backtest

### Phase 3: Live Trading
- [ ] Kraken Futures API integration
- [ ] Order execution engine
- [ ] Real-time monitoring dashboard
- [ ] Automated position management

### Phase 4: Machine Learning Enhancement
- [ ] Pattern recognition
- [ ] Adaptive parameter tuning
- [ ] Ensemble signal combination

---

## 📚 Resources

### Documentation
- [Kraken Futures API Docs](https://docs.futures.kraken.com/)
- [Technical Indicators Guide](https://www.investopedia.com/technical-analysis-4689657)
- [Multi-Timeframe Analysis](https://www.babypips.com/learn/forex/multiple-time-frame-analysis)

### Further Reading
- Perpetual futures mechanics
- Funding rate arbitrage strategies
- Multi-timeframe trading psychology

---

## ⚖️ License

MIT License - See LICENSE file for details

---

## ⚠️ Disclaimer

**This system is for educational and research purposes.**

- Cryptocurrency trading involves substantial risk
- Past performance does not guarantee future results
- Only trade with capital you can afford to lose
- Backtest results may not reflect live trading performance
- Always start with paper trading before risking real capital

**The system is provided "as-is" without warranties of any kind.**

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

---

## 📞 Support

For questions or issues:
- Open a GitHub issue
- Check existing documentation
- Review backtest outputs

---

## 🎓 Credits

Built with:
- Python 3.8+
- pandas, numpy (data processing)
- Kraken Futures API
- TA-Lib (technical indicators)

---

**Happy Trading! 🚀📈**

Remember: This system learns. Feed it data, test rigorously, and let the backtests guide your decisions.

**"If it doesn't work in backtesting, it won't work live."**
