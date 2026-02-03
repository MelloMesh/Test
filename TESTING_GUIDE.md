# HTF Testing Guide

Complete guide for backtesting and forward testing (paper trading) your HTF trading system.

## Table of Contents

1. [Backtesting](#backtesting) - Test strategies on historical data
2. [Paper Trading](#paper-trading) - Test strategies on live data (no real money)
3. [Going Live](#going-live) - Deploy with real capital

---

## Backtesting

**What it does:** Tests your HTF signals on historical data to see how they would have performed.

### Quick Start

```bash
# Run HTF backtest on BTC
python3 run_backtest_htf.py
```

### Sample Output

```
================================================================================
                        📈 BACKTEST SUMMARY
================================================================================

  📊 Total Trades                                           127
  ✅ Win Rate                                             62.3%
  ✅ Profit Factor                                        1.85x
  ✅ Total Return                                        +18.4%
  ✅ Max Drawdown                                          8.2%
  ✅ Sharpe Ratio                                          1.45

================================================================================
                      🎯 HTF ALIGNMENT IMPACT
================================================================================

  HTF-Aligned Trades:
    ✅   Win Rate                                         65.2%
    ✅   Profit Factor                                    2.10x
    📊   Total Trades                                      98

  Trades Blocked by HTF Filter:
    ✅   Would-be Losers Avoided                           29

================================================================================
                    📊 PERFORMANCE BY SIGNAL TYPE
================================================================================

  Signal                              Trades   Win Rate  Profit Factor
  ------------------------------------------------------------------------------
  🟢 HTF_Bearish_Rejection_30m_Short      24      70.8%          2.45x
  🟢 HTF_Bullish_Pullback_30m_Long        18      66.7%          2.12x
  🟢 HTF_Support_Bounce_30m_Long          15      60.0%          1.85x
  🟡 HTF_Resistance_Rejection_30m_Short   12      50.0%          1.42x
```

### Customizing Backtests

Edit `run_backtest_htf.py`:

```python
# Test different symbols
symbols = ['XXBTZUSD', 'XETHZUSD', 'XSOLUSD']

# Change time period
start_date = end_date - timedelta(days=180)  # 6 months
```

### Key Metrics Explained

| Metric | Good | Excellent | What It Means |
|--------|------|-----------|---------------|
| **Win Rate** | >55% | >60% | Percentage of winning trades |
| **Profit Factor** | >1.5 | >2.0 | Total wins ÷ total losses |
| **Total Return** | >10% | >20% | Overall profit percentage |
| **Max Drawdown** | <15% | <10% | Largest peak-to-trough loss |
| **Sharpe Ratio** | >1.0 | >1.5 | Risk-adjusted returns |

---

## Paper Trading

**What it does:** Tracks live signals and simulates trades in real-time without risking real money. Perfect for forward testing!

### Quick Start

```bash
# Start paper trading engine
python3 run_paper_trading.py
```

### How It Works

1. **Listens for signals** from your live scanner
2. **Opens paper trades** automatically when HTF-aligned signals appear
3. **Tracks positions** and checks stop loss / take profit every minute
4. **Closes trades** when SL/TP hit
5. **Sends Telegram updates** for every trade opened/closed
6. **Saves results** to `paper_trading_results.json`

### Sample Output

```
================================================================================
                     🚀 HTF PAPER TRADING ENGINE
================================================================================

  This mode simulates trades from live signals
  No real money is used - perfect for testing strategies!

================================================================================

================================================================================
                     📊 PAPER TRADING PERFORMANCE
================================================================================

  💼 Account:
     Initial Capital:  $   10,000.00
     Current Capital:  $   10,845.00
     Total P&L:        $     +845.00
     Return:                  +8.5%

  📈 Trades:
     Total:                       23
     Open:                         3
     Winners:                     15
     Losers:                       8
     Win Rate:                 65.2%

  🔓 Open Trades:
     BTCUSDT: SHORT @ $105,240.0000
     ETHUSDT: SHORT @ $3,890.5000
     SOLUSDT: LONG @ $142.3500

================================================================================

📡 Listening for signals from live scanner...
   (Signals will be picked up automatically when scanner runs)

⏳ Checking open trades every 60 seconds...
```

### Telegram Notifications

You'll receive notifications for every trade:

**Trade Opened:**
```
🟢 PAPER TRADE OPENED 🟢

💰 TRADE:
   Symbol: SOLUSDT
   Direction: LONG
   Entry: $142.35

🛡️ RISK MANAGEMENT:
   Stop Loss: $139.50 (-2%)
   Take Profit: $148.05 (+4%)
   Position Size: $200.00

📊 HTF CONTEXT:
   Bias: BULLISH
   Alignment: 85%
   Regime: TRENDING

💼 ACCOUNT:
   Current Capital: $10,845.00
```

**Trade Closed:**
```
🟢 PAPER TRADE CLOSED 🟢

💰 RESULT:
   Symbol: SOLUSDT
   Direction: LONG
   Entry: $142.35
   Exit: $148.05

   P&L: +$8.00 (+4.0%)
   Reason: Take Profit

📊 PERFORMANCE:
   Win Rate: 65.2% (15W / 8L)
   Total P&L: +$845.00
   Total Return: +8.5%
   Current Capital: $10,853.00
```

### Running Both Scanner and Paper Trading

**Terminal 1 - Live Scanner:**
```bash
python3 run_htf_live_scanner_bybit.py
```

**Terminal 2 - Paper Trading:**
```bash
python3 run_paper_trading.py
```

The paper trading engine will automatically pick up signals from the live scanner and simulate trades!

### Configuration

Edit `run_paper_trading.py`:

```python
# Change initial capital
engine = PaperTradingEngine(initial_capital=50000)  # $50k

# Adjust risk per trade (default 2%)
position_size_pct = 1.0  # 1% per trade (more conservative)

# Modify risk:reward ratio
risk_pct = 0.01    # 1% risk
reward_pct = 0.03  # 3% reward = 1:3 R:R
```

### View Results

Results are automatically saved to `paper_trading_results.json`. You can analyze:

- Every trade taken
- Win/loss streaks
- Performance by signal type
- HTF alignment impact
- Time-based patterns

---

## Going Live

Once you've validated performance through backtesting and paper trading, you can go live:

### Prerequisites

✅ **Backtest Results:** 60%+ win rate, 1.5+ profit factor
✅ **Paper Trading:** 2+ weeks forward testing with positive results
✅ **Risk Management:** Clear stop losses, position sizing rules
✅ **Exchange Account:** Funded account on Binance/Bybit

### Live Trading Steps

1. **Start with small capital** (10% of your account)
2. **Monitor closely** for first 20 trades
3. **Scale gradually** as confidence builds
4. **Keep paper trading running** to compare results

### Risk Warning

⚠️ **IMPORTANT**: Trading involves substantial risk of loss. Always:
- Trade with money you can afford to lose
- Use proper position sizing (1-2% per trade)
- Set stop losses on every trade
- Never override your system's signals emotionally

---

## Comparison: Backtest vs Paper Trading

| Aspect | Backtesting | Paper Trading |
|--------|-------------|---------------|
| **Data** | Historical | Live |
| **Speed** | Fast (minutes) | Real-time (days/weeks) |
| **Purpose** | Validate strategy | Test execution |
| **Strengths** | Large sample size | Real market conditions |
| **Weaknesses** | Look-ahead bias risk | Slower feedback |
| **When to Use** | Initial strategy testing | Pre-live validation |

### Recommended Workflow

```
1. Develop Strategy
   ↓
2. Backtest (run_backtest_htf.py)
   ↓ (If results good)
3. Paper Trade (run_paper_trading.py)
   ↓ (2-4 weeks, if consistent)
4. Go Live (small size)
   ↓ (Monitor & scale)
5. Full Deployment
```

---

## FAQ

**Q: How long should I paper trade?**
A: Minimum 2 weeks, ideally 4-6 weeks to see performance across different market conditions.

**Q: What win rate should I expect?**
A: HTF-aligned signals typically achieve 60-65% win rate. Below 55% may indicate issues.

**Q: Can I run paper trading 24/7?**
A: Yes! It automatically checks positions and handles signals. Run it in a `screen` or `tmux` session.

**Q: Do I need the live scanner running for paper trading?**
A: Not required, but recommended for real-time signal testing. Paper trading can also work with manual signal input.

**Q: What's the difference between paper trading and forward testing?**
A: Same thing! Paper trading = forward testing = simulated live trading without real money.

---

## Support

Having issues?

1. Check logs in `logs/` directory
2. Review `paper_trading_results.json` for trade details
3. Ensure live scanner is running for real-time signals
4. Verify Telegram bot is configured

Happy testing! 🚀
