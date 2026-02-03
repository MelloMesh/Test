# Live Trading System Guide

Complete guide to running the live trading signal scanner with Telegram notifications.

## 🎯 System Overview

The live trading system consists of:

1. **Real-time Data Stream** - WebSocket connection to Kraken Futures
2. **Signal Scanner** - Scans for validated strategies on every candle close
3. **Telegram Bot** - Sends instant mobile notifications for signals
4. **Risk Management** - Automatic position sizing based on account risk

## 📋 Prerequisites

### 1. Complete Backtesting First

Before running live signals, you need validated strategies from backtesting:

```bash
# Run signal discovery and backtesting
python3 run.sh --mode full

# Or step by step:
python3 run.sh --mode discovery    # Find signal hypotheses
python3 run.sh --mode backtest     # Test all signals
python3 run.sh --mode analyze      # Analyze and rank results
```

This creates `results/analysis/edge_analysis_multiframe.json` with validated signals.

### 2. Set Up Telegram Bot

Follow the instructions in `TELEGRAM_SETUP.md`:

1. Create bot with @BotFather on Telegram
2. Get your chat ID from @userinfobot
3. Create `.telegram_config.json`:

```json
{
  "token": "your-bot-token-here",
  "chat_id": "your-chat-id-here"
}
```

### 3. Install Dependencies

```bash
pip3 install -r requirements.txt
```

Key dependencies:
- `websockets` - Real-time data streaming
- `aiohttp` - Async HTTP requests
- `python-telegram-bot` - Telegram notifications

## 🚀 Running the Live Scanner

### Basic Usage

```bash
python3 run_live_scanner.py
```

This will:
- ✅ Load validated signals from backtesting results
- ✅ Connect to Kraken WebSocket for real-time data
- ✅ Monitor all instruments and timeframes (BTC, ETH, SOL across 2m, 5m, 15m, 30m)
- ✅ Send Telegram notifications when signals trigger
- ✅ Print status updates every 5 minutes

### What You'll See

**On startup:**
```
================================================================================
🚀 STARTING LIVE TRADING SCANNER 🚀
================================================================================
Instruments: BTCUSDT.P, ETHUSDT.P, SOLUSDT.P
Timeframes: 2m, 5m, 15m, 30m
Telegram: ✅ Enabled
================================================================================

✓ Loaded 12 validated signals for live trading
🔌 Connecting to Kraken WebSocket...
✅ Connected to real-time data feed
```

**Every candle close:**
```
Candle complete: BTCUSDT.P 5m | Close: 45123.50 | Volume: 1234567.00
```

**When a signal triggers:**
```
================================================================================
🎯 NEW TRADING SIGNAL 🎯
================================================================================
Signal: RSI_Mean_Reversion_5m
Instrument: BTCUSDT.P (5m)
Direction: LONG
Entry: $45,123.50
Stop Loss: $44,800.00
Take Profit: $45,770.00
R:R Ratio: 1:2.0
Position Size: $1,000.00
Risk: $100.00
Backtest WR: 62.0%
================================================================================

✅ Signal sent to Telegram
```

**Status updates (every 5 minutes):**
```
📊 STATUS UPDATE
Runtime: 15.2 minutes
Candles processed: 48
Signals sent: 2
Processing rate: 3.2 candles/min
```

### Stopping the Scanner

Press `Ctrl+C` to gracefully shutdown:

```
⚠️  Keyboard interrupt received
Stopping scanner...
Runtime: 3600s
Candles processed: 1440
Signals sent: 15
================================================================================

✅ Scanner stopped by user
```

## 📱 Telegram Notifications

When the scanner is running, you'll receive Telegram messages for:

### 1. Startup Notification
```
🚀 Live Scanner Started

Monitoring 3 instruments across 4 timeframes.

You will receive signals when opportunities are detected.
```

### 2. Trading Signals
```
🟢 NEW TRADING SIGNAL 🟢

📊 Signal: RSI_Mean_Reversion_5m
💰 Instrument: BTCUSDT.P (5m)
📈 Direction: LONG

💵 ENTRY: $45,123.50

🛡️ RISK MANAGEMENT:
   Stop Loss: $44,800.00
   Take Profit: $45,770.00
   R:R Ratio: 1:2.0

💰 POSITION:
   Size: $1,000.00
   Risk: $100.00

📊 BACKTEST PERFORMANCE:
   Win Rate: 62.0%
   Profit Factor: 1.8x
   Confidence: 65%

🌐 CONTEXT:
   Macro Trend: Bullish
   Higher TF: ✓ Aligned
   Regime: Trending

💡 REASON:
RSI oversold on trending market with volume confirmation
```

### 3. Shutdown Notification
```
🛑 Live Scanner Stopped

Scanner stopped after 15 signals sent.
```

## ⚙️ Configuration

### Modify Signal Thresholds

Edit `config.py` to adjust which signals qualify for live trading:

```python
# Lenient thresholds based on R:R math
LIVE_THRESHOLDS = {
    "2m": {
        "min_win_rate": 0.38,  # For 2:1 R:R
        "min_profit_factor": 1.2,
        "min_trades": 50,
        "required_rr": 2.0,
    },
    # ... adjust for each timeframe
}
```

### Account Settings

```python
# In config.py
INITIAL_CAPITAL = 10000  # Starting balance ($10,000)
POSITION_SIZE_PCT = 0.02  # Risk 2% per trade
```

### Risk:Reward Ratios

```python
# Minimum R:R by timeframe
MIN_RISK_REWARD = {
    "2m": 2.0,   # 2:1 R:R for 2-minute trades
    "5m": 2.0,   # 2:1 R:R
    "15m": 2.5,  # 2.5:1 R:R for longer timeframes
    "30m": 3.0,  # 3:1 R:R
}
```

## 🔍 Monitoring and Debugging

### View Logs in Real-Time

The scanner outputs detailed logs. To save logs to a file:

```bash
python3 run_live_scanner.py 2>&1 | tee live_scanner.log
```

### Check Telegram Bot Without Scanner

Test if your Telegram bot is working:

```bash
python3 check_telegram_bot.py
```

Expected output:
```
✅ Bot is VALID!
   Username: @your_bot_name
   Name: Your Bot
   Bot ID: 123456789
```

### Verify Backtesting Results Exist

```bash
ls -lh results/analysis/
```

Should show:
```
edge_analysis_multiframe.json  # Required for live scanner
```

If missing, run backtesting first:
```bash
python3 run.sh --mode full
```

## 🎓 Understanding Signal Generation

### Signal Flow

1. **Candle Closes** → WebSocket receives new price data
2. **Scanner Triggers** → Check if any validated strategies match current conditions
3. **Risk Calculation** → Calculate position size, stop loss, take profit
4. **Multi-TF Validation** → Verify higher timeframes align
5. **Send Signal** → Log locally + Send to Telegram

### Signal Quality Filters

Signals must pass:

✅ **Win Rate** - Above break-even for their R:R ratio
✅ **Profit Factor** - Minimum 1.2x (makes $1.20 for every $1 risked)
✅ **Sample Size** - Minimum number of backtest trades
✅ **R:R Ratio** - Meets minimum reward:risk threshold
✅ **Current Conditions** - Entry criteria match live market data

### Position Sizing Math

For a $10,000 account with 2% risk:

```
Risk per trade = $10,000 × 0.02 = $200
Entry price = $45,000
Stop loss = $44,500
Price risk = $45,000 - $44,500 = $500

Position size = $200 / $500 = 0.4 units
Position value = 0.4 × $45,000 = $18,000

If stopped out: Lose $200 (2%)
If target hit (2:1 R:R): Win $400 (4%)
```

## 🚧 Next Steps

After running the live scanner successfully:

### Phase 3: Paper Trading Engine

Build simulated trading to track performance:

- Execute signals in paper trading mode
- Track open positions and P&L
- Compare live results vs backtest expectations
- Build confidence before using real capital

### Phase 4: Risk Management Dashboard

- Monitor all active positions
- Track daily/weekly performance
- Alert if signals underperform backtests
- Automatic pause if drawdown exceeds threshold

### Phase 5: Live Trading (Real Capital)

Only after paper trading proves system works:

- Connect to real exchange API
- Start with very small position sizes
- Gradually scale up as confidence builds

## 🛡️ Safety Reminders

⚠️ **This is paper trading / signal generation ONLY**
⚠️ **No real money is being traded**
⚠️ **Always validate signals manually before placing real trades**
⚠️ **Start small when transitioning to live trading**
⚠️ **Never risk more than you can afford to lose**

## 🐛 Troubleshooting

### "No module named 'telegram'"

```bash
pip3 install python-telegram-bot websockets aiohttp
```

### "403 Forbidden" from Telegram

- Verify bot token is correct
- Make sure you've sent `/start` to your bot in Telegram
- Check your chat ID is correct
- Try from a different network (might be firewall)

### "No edge analysis found"

Run backtesting first:
```bash
python3 run.sh --mode full
```

### "WebSocket connection failed"

- Check internet connection
- Kraken may be down (check status.kraken.com)
- Firewall blocking WebSocket connections

### No Signals Being Generated

This is normal! Signals only trigger when:
- Market conditions match strategy entry criteria
- Price action confirms the setup
- Higher timeframes align

You may go hours without a signal if markets are choppy or ranging.

## 📚 Additional Resources

- **ROADMAP_TO_LIVE_TRADING.md** - Complete implementation plan
- **TELEGRAM_SETUP.md** - Detailed Telegram bot setup
- **README.md** - Full system documentation
- **config.py** - All configuration parameters

## 💡 Tips for Success

1. **Be Patient** - Quality over quantity. Wait for high-confidence setups.
2. **Respect Risk Limits** - Never exceed 2% risk per trade.
3. **Track Performance** - Compare live results to backtest expectations.
4. **Stay Disciplined** - Follow the signals, don't second-guess the system.
5. **Continuous Learning** - Review winning and losing trades to improve.

---

**Ready to start? Run the scanner and let the system find opportunities for you!**

```bash
python3 run_live_scanner.py
```

Good luck! 🚀
