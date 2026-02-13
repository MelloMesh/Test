# Long Elite Squeeze (LES) Strategy - Quick Start

## 🚀 Quick Installation

### Step 1: Copy the Strategy
1. Open `Long_Elite_Squeeze_LES_Strategy.pine`
2. Copy all the code (Ctrl+A, Ctrl+C)

### Step 2: Add to TradingView
1. Go to [TradingView](https://www.tradingview.com/)
2. Open the Pine Editor (bottom of chart)
3. Click "New" → "Blank indicator"
4. Paste the code
5. Click "Add to Chart"

### Step 3: Configure
- For **Stocks**: Use default settings
- For **Crypto**: Enable "Use Crypto Mode" in inputs
- Adjust risk % as needed (start with 0.5-1.0%)

---

## 📊 What This Strategy Does

The LES strategy identifies high-probability long entries by combining:

1. **Squeeze Momentum Pattern** - Waits for 6+ consecutive strong bullish bars, then enters on first sign of slowing
2. **WaveTrend Alignment** - Confirms bullish momentum with oscillator
3. **High Volume** - Requires RVOL ≥ 1.5x average
4. **RSI Confirmation** - Ensures healthy trend (50-70 range)
5. **Session Filter** - Only trades during active hours (optional)

---

## 🎯 Entry Criteria (ALL must be TRUE)

- ✅ 6+ dark green squeeze bars → light green flip
- ✅ WaveTrend bullish (wt1 > wt2) with recent cross
- ✅ Volume 1.5x+ above average
- ✅ RSI between 50-70 (not overextended)
- ✅ Within trading session (if enabled)

---

## 💰 Trade Management

### Position Sizing
- **Risk per trade**: 1% of account (default)
- **Stop loss**: Entry - (1.5 × ATR)
- Calculated automatically based on volatility

### Profit Targets
- **TP1 (50%)**: Entry + (2.0 × ATR)
- **TP2 (50%)**: Entry + (3.0 × ATR)

### Auto Exits
- **Exhaustion**: After 2+ strong bars, exits on first weakness
- **Trend Reversal**: WaveTrend crosses down
- **Time**: Closes after 50 bars if still open

---

## ⚙️ Quick Settings Presets

### Conservative (Safer, Fewer Trades)
```
Risk %: 0.5
Min Dark Green Bars: 8
Min RVOL: 2.0
ATR Mult SL: 2.0
```

### Balanced (Default)
```
Risk %: 1.0
Min Dark Green Bars: 6
Min RVOL: 1.5
ATR Mult SL: 1.5
```

### Aggressive (More Trades, Higher Risk)
```
Risk %: 2.0
Min Dark Green Bars: 4
Min RVOL: 1.2
ATR Mult SL: 1.0
```

---

## 📈 Best Timeframes

| Timeframe | Trading Style | Signals/Week |
|-----------|--------------|--------------|
| 5-min | Scalping | 10-30 |
| 15-min | Day Trading | 5-15 |
| 1-hour | Swing (Short) | 3-8 |
| 4-hour | Swing (Medium) | 1-4 |
| Daily | Position | 1-3 |

---

## 📊 HUD & Stats

### Real-Time HUD (Top Right)
Shows current status of all filters:
- Position status (IN TRADE / WAITING)
- Each filter: ✓ PASS or ✗ FAIL
- Current RVOL and RSI values

### Backtest Stats (Bottom Right)
- Total trades
- Win rate %
- Profit factor
- Max drawdown %
- Average R multiple

---

## ⚠️ Important Notes

### Non-Repainting
- All signals use confirmed bars only
- What you see in backtest = what you get in real-time
- No lookahead bias or future data

### Risk Management
- **Never risk more than 1-2% per trade**
- Start with 0.5% while learning
- Use proper position sizing
- Respect max drawdown limits

### Backtesting Checklist
Before going live:
- [ ] Test on 1+ year of data
- [ ] Include 0.1% commission
- [ ] Add 2-5 ticks slippage
- [ ] Verify profit factor > 1.5
- [ ] Check max drawdown acceptable
- [ ] Paper trade for 2-4 weeks

---

## 🔧 Common Adjustments

### Too Many Signals → Reduce by:
- Increase "Min Dark Green Bars" (6 → 8)
- Raise "Min RVOL" (1.5 → 2.0)
- Tighten RSI range

### Too Few Signals → Increase by:
- Decrease "Min Dark Green Bars" (6 → 4)
- Lower "Min RVOL" (1.5 → 1.2)
- Disable session filter
- Enable crypto mode (wider thresholds)

### Stops Too Tight → Fix by:
- Increase "ATR Mult SL" (1.5 → 2.0)
- Use higher timeframe
- Adjust risk % to maintain position size

### Targets Not Hit → Adjust by:
- Reduce "ATR Mult TP1" (2.0 → 1.5)
- Reduce "ATR Mult TP2" (3.0 → 2.0)
- Disable auto-exits temporarily

---

## 📚 Full Documentation

For complete details, see:
- **LES_STRATEGY_GUIDE.md** - Comprehensive strategy documentation
- **Long_Elite_Squeeze_LES_Strategy.pine** - Full source code with comments

---

## 🎓 Learning Path

1. **Week 1**: Backtest with default settings on your favorite symbol
2. **Week 2**: Optimize parameters for your market (don't overfit!)
3. **Week 3**: Paper trade and watch signals in real-time
4. **Week 4**: Start with small real positions (0.5% risk)
5. **Month 2+**: Scale up as you gain confidence

---

## ✅ Pre-Launch Checklist

Before risking real money:
- [ ] Strategy logic fully understood
- [ ] Backtested on your market (1+ year)
- [ ] Profit factor > 1.5, Win rate > 50%
- [ ] Max drawdown acceptable for risk tolerance
- [ ] Paper traded for 2+ weeks
- [ ] All inputs optimized (not overfitted)
- [ ] Risk per trade ≤ 1% of account
- [ ] Exit plan defined for all scenarios
- [ ] Broker commission/slippage accounted for
- [ ] Emergency plan if strategy fails

---

## 💡 Pro Tips

1. **Context Matters**: Check higher timeframe trend before entries
2. **Volume Is King**: Higher RVOL = higher probability setups
3. **Patience Pays**: Wait for all filters to align
4. **Respect Exits**: Don't second-guess automated exits
5. **Track Performance**: Log every trade and review weekly
6. **Adapt**: Markets change; be ready to adjust parameters
7. **Psychology**: Follow the system even during drawdowns
8. **Diversify**: Don't trade just one symbol or timeframe

---

## 🚨 Red Flags

Stop trading the strategy if:
- Consecutive losses exceed 5 trades
- Drawdown exceeds tested maximum by 50%
- Win rate drops below 40% for 20+ trades
- Market conditions fundamentally change
- You stop trusting the signals (emotional trading)

Take a break, review what changed, and adjust or move on.

---

**Remember**:
- Backtesting ≠ Future performance
- Every trade is independent
- Focus on process, not outcomes
- Protect your capital first

**Good luck and trade safe! 📊✨**
