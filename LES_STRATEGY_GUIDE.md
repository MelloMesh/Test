# Long Elite Squeeze (LES) Strategy - Complete Guide

## Overview

This is a professional-grade TradingView Pine Script v6 strategy implementing the **Long Elite Squeeze (LES)** concept. The strategy combines multiple technical indicators with strict pattern recognition to identify high-probability long entries with minimal repainting risk.

## Strategy Components

### 1. Squeeze Momentum (LazyBear Style)

The core of this strategy is the Squeeze Momentum indicator, which detects periods of consolidation (squeeze) followed by explosive moves.

**How it works:**
- **Bollinger Bands** (length 20, mult 2.0) measure volatility around price
- **Keltner Channels** (length 20, ATR mult 1.5) measure volatility based on ATR
- **Squeeze Detection**: When BB contracts inside KC, a "squeeze" is on
- **Momentum Histogram**: Linear regression-based momentum calculation

**Color Classification:**
- **Dark Green**: Bullish momentum increasing (strong uptrend)
- **Light Green**: Bullish momentum decreasing (slowing, potential reversal)
- **Dark Red**: Bearish momentum increasing (strong downtrend)
- **Light Red**: Bearish momentum decreasing (slowing, potential reversal)

**LES Pattern Requirement:**
The strategy requires **at least 6 consecutive dark green bars** followed by a flip to **light green**. This signals:
1. Strong bullish momentum has built up (6+ dark green bars)
2. Momentum is now slowing (light green flip), indicating potential exhaustion
3. A reversal or pullback may be imminent for entry

### 2. WaveTrend Oscillator (LazyBear Style)

WaveTrend is a momentum oscillator that helps identify overbought/oversold conditions and trend direction.

**Calculation:**
```
ap  = hlc3
esa = EMA(ap, 10)
d   = EMA(|ap - esa|, 10)
ci  = (ap - esa) / (0.015 * d)
wt1 = EMA(ci, 21)
wt2 = SMA(wt1, 4)
```

**Entry Requirements:**
- wt1 must be above wt2 (bullish alignment)
- A bullish crossover must have occurred within the last 5 bars (configurable)
- Overbought/Oversold levels:
  - **Standard**: OB = +53, OS = -53
  - **Crypto Mode**: OB = +60, OS = -60 (wider thresholds)

**Exit Signal:**
- When wt1 crosses below wt2 while in a trade, exit immediately (bearish reversal)

### 3. Relative Volume (RVOL)

RVOL measures current volume against its 20-period average to ensure sufficient trading activity.

**Calculation:**
```
RVOL = Current Volume / SMA(Volume, 20)
```

**Entry Requirement:**
- RVOL must be ≥ 1.5 (default, configurable)
- This ensures the setup occurs with above-average participation

### 4. RSI Alignment

RSI provides trend confirmation and prevents entries during extreme conditions.

**Entry Requirements:**
- **Standard Mode**: 50 < RSI < 70
- **Crypto Mode**: 45 < RSI < 75 (slightly relaxed)

**Logic:**
- RSI > 50 confirms uptrend
- RSI < 70 prevents entries when already overextended

### 5. Session Filter

The session filter restricts entries to specific trading hours, ideal for day traders.

**Default Session:** 0930-1600 (US Regular Trading Hours)

**Behavior:**
- When enabled: New entries only during session hours
- When disabled: Entries allowed anytime
- Exits and stops: Always active, regardless of session

## Entry Logic

A long entry is triggered when **ALL** of the following conditions are met:

1. ✅ **Squeeze Pattern**: 6+ consecutive dark green bars followed by light green flip
2. ✅ **WaveTrend**: wt1 > wt2 AND bullish cross within last 5 bars
3. ✅ **RVOL**: Current RVOL ≥ 1.5
4. ✅ **RSI**: Within acceptable range (50-70 standard, 45-75 crypto)
5. ✅ **Session**: Inside session hours (if filter enabled)
6. ✅ **Confirmation**: Bar is confirmed (no repainting)

## Trade Management

### Position Sizing

Position size is calculated based on **risk percentage** and **ATR-based stop loss**:

```
Risk Amount = Account Equity × (Risk % / 100)
Stop Distance = ATR(14) × ATR Multiplier (default 1.5)
Position Size = Risk Amount / Stop Distance
```

**Default**: 1% risk per trade (configurable 0.1% - 10%)

### Stop Loss & Take Profits

**ATR-Based Levels:**
- **Stop Loss**: Entry - (1.5 × ATR)
- **Take Profit 1**: Entry + (2.0 × ATR) — Close 50% of position
- **Take Profit 2**: Entry + (3.0 × ATR) — Close remaining 50%

All multipliers are fully configurable.

### Automated Exit Conditions

The strategy includes three intelligent exit mechanisms:

#### 1. Exhaustion Exit
**Logic:**
- After entry, if we see 2+ consecutive dark green bars (momentum building)
- Then the histogram flips to light green or red (momentum exhaustion)
- **Action**: Close entire position at market

**Rationale**: This catches momentum exhaustion before it fully reverses.

#### 2. WaveTrend Reversal Exit
**Logic:**
- While in a long position, if wt1 crosses under wt2
- **Action**: Close entire position at market

**Rationale**: WaveTrend bearish cross signals trend reversal.

#### 3. Time-Based Exit
**Logic:**
- If trade has been open for more than 50 bars (default, configurable)
- **Action**: Close entire position at market

**Rationale**: Prevents holding stale positions that aren't moving.

## Anti-Repainting Measures

This strategy implements **strict non-repainting logic**:

1. ✅ **Confirmed Bars Only**: All entry logic uses `barstate.isconfirmed`
2. ✅ **[1] Indexing**: Pattern detection looks at previous confirmed bars
3. ✅ **Process Orders on Close**: `process_orders_on_close=true`
4. ✅ **No Lookahead Bias**: No `request.security()` with lookahead enabled
5. ✅ **Historical Consistency**: Signals appear the same in real-time and historical backtests

## Visual Elements

### Price Chart Overlays
- **EMA 21** (Blue): Short-term trend
- **EMA 50** (Orange): Medium-term trend
- **Setup Signal**: Triangle up below bars when setup is valid
- **Background Highlight**: Green background on setup bars

### Squeeze Momentum Pane
- **Histogram**: 4-color momentum bars (dark/light green/red)
- **Zero Line**: Reference line
- **Pattern Marker**: Yellow triangle when 6+ dark green pattern completes

### WaveTrend Pane
- **WT1** (Aqua): Primary WaveTrend line
- **WT2** (Red): Signal line
- **OB/OS Lines**: Dotted horizontal levels

### HUD (Top Right)
Real-time filter status showing:
- **Position Status**: IN TRADE or WAITING
- **Squeeze**: ✓ PASS / ✗ FAIL
- **WaveTrend**: ✓ PASS / ✗ FAIL
- **RVOL**: ✓ PASS / ✗ FAIL (with current value)
- **RSI**: ✓ PASS / ✗ FAIL (with current value)
- **Session**: ✓ IN SESSION / ✗ OUT

### Backtest Stats (Bottom Right)
- **Total Trades**: Number of closed trades
- **Win Rate %**: Percentage of winning trades
- **Profit Factor**: Gross profit / gross loss
- **Max DD %**: Maximum drawdown percentage
- **Avg R Multiple**: Average reward-to-risk ratio

## Configuration & Usage

### Basic Setup

1. Copy the script code from `Long_Elite_Squeeze_LES_Strategy.pine`
2. Open TradingView and create a new Pine Script indicator
3. Paste the code and click "Add to Chart"
4. The strategy will load with default settings

### Recommended Settings

#### For Stock Trading (Day Trading)
```
- Use Crypto Mode: FALSE
- Use Session Filter: TRUE
- Session Time: 0930-1600
- Risk % per Trade: 1.0
- Min Dark Green Bars: 6
- Min RVOL: 1.5
- RSI Length: 14
```

#### For Crypto Trading (24/7)
```
- Use Crypto Mode: TRUE
- Use Session Filter: FALSE
- Risk % per Trade: 1.0
- Min Dark Green Bars: 6
- Min RVOL: 2.0 (higher due to crypto volatility)
- WT OB/OS: ±60 (auto-adjusted when crypto mode enabled)
```

#### For Conservative Trading
```
- Risk % per Trade: 0.5
- Min Dark Green Bars: 8 (stricter pattern)
- Min RVOL: 2.0 (ensure high volume)
- ATR Mult SL: 2.0 (wider stop)
- Max Bars in Trade: 30 (exit faster)
```

#### For Aggressive Trading
```
- Risk % per Trade: 2.0
- Min Dark Green Bars: 4 (looser pattern)
- Min RVOL: 1.2 (lower volume threshold)
- ATR Mult TP1: 1.5 (take profit earlier)
- ATR Mult TP2: 2.5
```

### Input Parameters Reference

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| **General** |
| Use Crypto Mode | false | bool | Widens OB/OS and RSI thresholds |
| Risk % per Trade | 1.0 | 0.1-10.0 | Percentage of equity to risk |
| **Session** |
| Use Session Filter | true | bool | Restrict entries to session |
| Session Time | 0930-1600 | session | Trading session hours |
| **Squeeze Momentum** |
| BB Length | 20 | 1+ | Bollinger Bands period |
| BB Multiplier | 2.0 | 0.1+ | Bollinger Bands standard deviation |
| KC Length | 20 | 1+ | Keltner Channel period |
| KC ATR Multiplier | 1.5 | 0.1+ | Keltner Channel ATR multiplier |
| Momentum Length | 20 | 1+ | Linear regression period |
| Min Dark Green Bars | 6 | 3-20 | Pattern requirement |
| **WaveTrend** |
| WT Channel Length | 10 | 1+ | WaveTrend channel period |
| WT Average Length | 21 | 1+ | WaveTrend smoothing period |
| WT Cross Lookback | 5 | 1-20 | Bars to look back for cross |
| **RVOL** |
| RVOL Period | 20 | 1+ | Volume SMA period |
| Min RVOL | 1.5 | 0.5+ | Minimum relative volume |
| **RSI** |
| RSI Length | 14 | 2-100 | RSI calculation period |
| **Trade Management** |
| ATR Length | 14 | 1+ | ATR calculation period |
| ATR Mult SL | 1.5 | 0.5+ | Stop loss distance |
| ATR Mult TP1 | 2.0 | 0.5+ | First target (50%) |
| ATR Mult TP2 | 3.0 | 0.5+ | Second target (50%) |
| Max Bars in Trade | 50 | 0+ | Time exit (0=disabled) |

## Optimization Tips

### 1. Optimize for Your Market
- **Stocks**: Standard settings work well
- **Forex**: Consider reducing Min RVOL to 1.2
- **Crypto**: Enable Crypto Mode, increase RVOL to 2.0

### 2. Timeframe Recommendations
- **5-min**: Aggressive, many signals
- **15-min**: Balanced, good for day trading
- **1-hour**: Fewer but higher quality setups
- **4-hour / Daily**: Swing trading

### 3. Risk Management
- **Never risk more than 1-2% per trade**
- Start with 0.5% while learning the strategy
- Consider portfolio-level risk if running multiple strategies

### 4. Backtesting Best Practices
- Test on at least 1+ years of data
- Include commission (0.1%) and slippage (2-5 ticks)
- Verify profit factor > 1.5 and win rate > 50%
- Check max drawdown is acceptable for your risk tolerance
- Forward test on demo account before live trading

## Performance Expectations

### Realistic Benchmarks
- **Win Rate**: 50-65% (quality over quantity)
- **Profit Factor**: 1.5-2.5 (good systems)
- **Average R**: 0.5-1.5R (positive expectancy)
- **Max Drawdown**: 10-20% (depends on risk settings)
- **Trades per Month**: 5-20 (depends on timeframe and market)

### Warning Signs
- ⚠️ Win rate > 80%: Likely curve-fitting or overfitting
- ⚠️ Profit factor > 4.0: Unrealistic, check for lookahead bias
- ⚠️ Max DD < 5%: Not enough trades or unrealistic backtest
- ⚠️ Too many trades: Pattern requirements may be too loose

## Common Issues & Solutions

### Issue: No Signals Generated
**Solutions:**
- Reduce `Min Dark Green Bars` from 6 to 4
- Lower `Min RVOL` from 1.5 to 1.2
- Widen RSI range (enable Crypto Mode)
- Disable Session Filter
- Check if WaveTrend lookback is too strict

### Issue: Too Many False Signals
**Solutions:**
- Increase `Min Dark Green Bars` from 6 to 8
- Raise `Min RVOL` from 1.5 to 2.0
- Tighten RSI range
- Enable Session Filter to avoid low-liquidity periods

### Issue: Stops Too Tight
**Solutions:**
- Increase `ATR Mult SL` from 1.5 to 2.0
- Reduce `Risk %` to maintain position sizing
- Consider higher timeframe for less noise

### Issue: Targets Not Reached
**Solutions:**
- Reduce `ATR Mult TP1` from 2.0 to 1.5
- Reduce `ATR Mult TP2` from 3.0 to 2.0
- Increase `Max Bars in Trade` to allow more time
- Disable automated exits if too aggressive

## Advanced Customization

### Adding Additional Filters

You can enhance the strategy by adding:

1. **Trend Filter**: Require close > EMA(200)
2. **ADX Filter**: Require ADX > 25 for trending markets
3. **Volume Profile**: Entry near VPOC or high-volume nodes
4. **Support/Resistance**: Entries near key levels

### Modifying Exit Logic

Current exits can be customized:

1. **Trailing Stop**: Replace fixed stop with trailing stop based on ATR
2. **Time-of-Day Exit**: Close all trades at end of session
3. **Indicator-Based**: Exit when RSI > 70 or other conditions
4. **Profit Protection**: Move stop to breakeven after TP1 hit

### Multi-Timeframe Analysis

For higher confidence:

1. Check higher timeframe squeeze alignment
2. Verify WaveTrend on multiple timeframes
3. Use daily chart for overall trend direction

## Disclaimer

This strategy is provided for **educational purposes only**.

- Past performance does not guarantee future results
- Always backtest thoroughly before live trading
- Start with paper trading or small position sizes
- Markets change, and strategies may need adjustment
- Consult with a financial advisor before risking capital

## Version History

### v1.0 (2024)
- Initial release
- Complete LES implementation with all 6 components
- Anti-repainting logic implemented
- HUD and stats tables added
- Comprehensive trade management

## Support & Contribution

For questions, issues, or improvements:
1. Review this documentation thoroughly
2. Check TradingView Pine Script v6 documentation
3. Test changes on historical data before live use
4. Share findings and improvements with the community

---

**Happy Trading! 📈**

*Remember: The best strategy is one you understand, trust, and can execute consistently.*
