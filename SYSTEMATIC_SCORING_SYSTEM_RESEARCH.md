# Systematic Confluence Scoring System for Crypto Futures
## Comprehensive Research on Professional Quant Signal Combination Methods

---

## TABLE OF CONTENTS

1. [Foundational Frameworks](#1-foundational-frameworks)
2. [The Four-Pillar Architecture](#2-the-four-pillar-architecture)
3. [Individual Indicator Scoring Methods](#3-individual-indicator-scoring-methods)
4. [Multi-Timeframe Conflict Resolution](#4-multi-timeframe-conflict-resolution)
5. [Crypto-Specific Indicator Weighting](#5-crypto-specific-indicator-weighting)
6. [Bayesian & Probabilistic Signal Combination](#6-bayesian--probabilistic-signal-combination)
7. [Academic Optimal Combination Methods](#7-academic-optimal-combination-methods)
8. [Classic Methodology Adaptations](#8-classic-methodology-adaptations)
9. [Composite Score Formula & Weights](#9-composite-score-formula--weights)
10. [Kelly Criterion Position Sizing](#10-kelly-criterion-position-sizing)
11. [Implementation Guidelines](#11-implementation-guidelines)

---

## 1. FOUNDATIONAL FRAMEWORKS

### 1.1 The Fundamental Law of Active Management (Grinold-Kahn)

The academic foundation for combining signals comes from Richard Grinold's Fundamental Law:

```
IR ≈ IC × √BR × TC
```

Where:
- **IR** (Information Ratio) = Risk-adjusted excess return
- **IC** (Information Coefficient) = Correlation between signal and actual returns (-1 to +1)
- **BR** (Breadth) = Number of independent forecasts per year
- **TC** (Transfer Coefficient) = Implementation efficiency (0 to 1)

**Key Implications for Signal Combination:**
- A weak signal (IC = 0.01) applied thousands of times can be more valuable than a strong signal (IC = 0.2) applied rarely
- Signals MUST be independent — correlated signals do not add breadth
- Implementation friction (slippage, fees) directly reduces alpha
- The formula is additive in squared IR, meaning independent signal sources compound

### 1.2 Confluence Scoring Philosophy

Professional systematic traders use "weight of the evidence" frameworks that:
- Assign normalized scores (-1 to +1 or 0 to 10) to each indicator
- Weight indicators by their empirical predictive power (IC)
- Require minimum confluence thresholds before entry
- Scale position size proportionally to composite score strength

---

## 2. THE FOUR-PILLAR ARCHITECTURE

Professional quant firms organize indicators into four orthogonal pillars. This structure ensures signals provide genuinely independent information:

### Pillar 1: TREND (Direction & Structure)
Indicators: Multi-TF RSI trend, CHOCH, Weinstein Stage, MA alignment
Purpose: Establish directional bias
Typical Weight: 30% of composite score

### Pillar 2: MOMENTUM (Speed & Exhaustion)
Indicators: RSI divergences, RSI levels, Bollinger squeeze
Purpose: Measure move intensity and reversal probability
Typical Weight: 25% of composite score

### Pillar 3: VOLUME & FLOW (Participation & Conviction)
Indicators: CVD trend/divergence, Relative Volume, OI delta
Purpose: Validate price moves with real participation
Typical Weight: 25% of composite score

### Pillar 4: SENTIMENT & POSITIONING (Crowd Behavior)
Indicators: Funding rate + velocity, Long/Short ratio, 24h range position
Purpose: Identify overcrowding, contrarian opportunities
Typical Weight: 20% of composite score

**Critical Rule:** The strongest signals occur when ALL four pillars align. When pillars conflict, reduce position size or stand aside.

---

## 3. INDIVIDUAL INDICATOR SCORING METHODS

### 3.1 Multi-Timeframe RSI (15m, 30m, 4h, Daily)

**Scoring Method:**
Each timeframe RSI is normalized to a -1 to +1 scale:

```
RSI_Score(tf) =
  if RSI > 70: map(RSI, 70, 100) → (-0.3 to -1.0)  [overbought = bearish lean]
  if RSI > 50: map(RSI, 50, 70) → (+0.3 to +1.0)    [bullish momentum]
  if RSI > 30: map(RSI, 30, 50) → (-0.3 to -1.0)     [bearish momentum]
  if RSI < 30: map(RSI, 0, 30) → (+0.3 to +1.0)      [oversold = bullish lean]
```

**Multi-TF Composite (weighted by timeframe importance):**
```
RSI_Composite = 0.10 × RSI_Score(15m)
              + 0.15 × RSI_Score(30m)
              + 0.35 × RSI_Score(4h)
              + 0.40 × RSI_Score(daily)
```

Higher timeframes receive greater weight per Elder's principle. The daily and 4h readings dominate with 75% of the weight, while shorter timeframes refine timing.

**Context Switch:** In trending markets, RSI > 50 is bullish continuation (don't fade). In ranging markets, extremes (>70, <30) are mean-reversion signals.

---

### 3.2 Funding Rate + Funding Velocity

**Funding Rate Score:**
```
FR_Score =
  if FR > +0.05%/8h:  -1.0  (extreme bullish crowding → contrarian bearish)
  if FR > +0.03%/8h:  -0.5  (elevated bullish → mild bearish lean)
  if FR > +0.01%/8h:  +0.3  (moderate bullish → trend confirmation)
  if FR ~ 0:           0.0  (neutral)
  if FR < -0.01%/8h:  -0.3  (moderate bearish → trend confirmation)
  if FR < -0.03%/8h:  +0.5  (elevated bearish → mild bullish lean)
  if FR < -0.05%/8h:  +1.0  (extreme bearish crowding → contrarian bullish)
```

**Funding Velocity (Rate of Change):**
```
FV = (FR_current - FR_8h_ago) / abs(FR_8h_ago)
```

Funding velocity measures the acceleration of positioning. Rapidly rising funding
is more dangerous than elevated-but-stable funding.

```
FV_Score =
  if FV > +50% (funding accelerating positive): -0.5 (overheating)
  if FV < -50% (funding collapsing from positive): +0.5 (unwind → bounce)
  else: 0
```

**Combined Funding Score:**
```
Funding_Score = 0.65 × FR_Score + 0.35 × FV_Score
```

**Important caveat from research:** In bull markets, naturally higher funding rates are a "cost of doing business" — only extreme outliers (>0.05% per 8h sustained) reliably signal reversals. Context matters enormously.

---

### 3.3 Open Interest Delta (Short-Term & 4h)

**OI Delta Score:**
```
OI_Delta_ST = (OI_now - OI_1h_ago) / OI_1h_ago × 100    [short-term]
OI_Delta_4h = (OI_now - OI_4h_ago) / OI_4h_ago × 100    [medium-term]
```

**Scoring Logic (context-dependent):**

| Price Direction | OI Change | Interpretation | Score |
|----------------|-----------|----------------|-------|
| Price ↑ | OI ↑ | New longs entering (trend confirmation) | +0.8 |
| Price ↑ | OI ↓ | Short covering (weak rally) | +0.3 |
| Price ↓ | OI ↑ | New shorts entering (trend confirmation) | -0.8 |
| Price ↓ | OI ↓ | Long liquidation (capitulation → potential bottom) | +0.5 |

**OI Normalized Score:**
```
OI_Score = 0.40 × OI_Delta_ST_Score + 0.60 × OI_Delta_4h_Score
```

The 4h delta gets more weight because short-term OI spikes can be noisy.

**Key insight from research:** Large OI decreases paired with liquidation cascades often represent capitulation — "buying when others have to sell" provides liquidity and is a classic institutional strategy.

---

### 3.4 CVD (Cumulative Volume Delta) Trend & Divergences

**CVD Formula:**
```
CVD = Σ (Buy Volume at Ask - Sell Volume at Bid)
```

**CVD Trend Score:**
```
CVD_Trend =
  if CVD making higher highs AND higher lows (4h): +1.0 (strong buying)
  if CVD rising but decelerating:                   +0.5
  if CVD flat:                                       0.0
  if CVD falling but decelerating:                  -0.5
  if CVD making lower highs AND lower lows (4h):   -1.0 (strong selling)
```

**CVD Divergence Score (critical — highest signal value):**
```
CVD_Divergence =
  Price HH + CVD LH (bearish divergence): -1.0
  Price LL + CVD HL (bullish divergence):  +1.0
  No divergence:                            0.0
```

**Spot vs. Perpetual CVD Comparison:**
```
If Spot_CVD > Perp_CVD while price rising: +0.5 (real buying, not leverage)
If Spot_CVD < Perp_CVD while price rising: -0.3 (leverage-driven, fragile)
```

**Combined CVD Score:**
```
CVD_Score = 0.40 × CVD_Trend + 0.40 × CVD_Divergence + 0.20 × Spot_vs_Perp
```

---

### 3.5 Long/Short Ratio

**Data Sources:**
- Binance Top Trader L/S Ratio (top 20% by margin)
- OKX Top Trader L/S Ratio (top 5% by position value)
- Taker Buy/Sell Ratio (short-term momentum)

**Scoring (contrarian when extreme, confirming when moderate):**
```
LS_Score =
  if L/S > 2.0:   -1.0  (extreme long crowding → bearish)
  if L/S > 1.5:   -0.5  (elevated long bias)
  if L/S 1.0-1.5:  0.0  to +0.3 (mild bullish confirmation)
  if L/S 0.7-1.0: -0.3  to 0.0  (mild bearish confirmation)
  if L/S < 0.7:   +0.5  (elevated short bias)
  if L/S < 0.5:   +1.0  (extreme short crowding → bullish)
```

**Cross-Exchange Divergence Bonus:**
```
If Binance and OKX ratios diverge significantly: reduce confidence by 0.3
If Binance and OKX ratios align at extremes: increase confidence by 0.2
```

---

### 3.6 Bollinger Band Squeeze (Squeeze Percentile)

**Bollinger BandWidth Percentile (BBWP):**
```
BBW = (Upper Band - Lower Band) / Middle Band × 100
BBWP = Percentile_Rank(BBW, lookback=252)  [where in the past year does current BBW sit]
```

**Scoring (squeeze is a CONDITION, not directional):**
```
Squeeze_Score =
  if BBWP < 5%:   1.0  (extreme squeeze → imminent expansion, high conviction)
  if BBWP < 20%:  0.7  (moderate squeeze → likely expansion soon)
  if BBWP 20-80%: 0.0  (normal volatility, no edge)
  if BBWP > 80%: -0.3  (expanded → volatility likely to contract)
  if BBWP > 95%: -0.7  (extreme expansion → exhaustion likely)
```

**Direction Determination (requires other pillars):**
```
Squeeze_Direction = sign(CVD_Score + RSI_Composite + Funding_Score)
Squeeze_Contribution = Squeeze_Score × Squeeze_Direction
```

The squeeze acts as a CONVICTION MULTIPLIER, not a directional signal. It amplifies the composite score when volatility is compressed and due for expansion.

---

### 3.7 Change of Character (CHOCH) Across Timeframes

**Detection Algorithm:**
```
1. Identify swing highs/lows using pivot detection (lookback L/R = 5 bars)
2. In uptrend: CHOCH = price closes below the swing low that produced the last BOS
3. In downtrend: CHOCH = price closes above the swing high that produced the last BOS
4. Require candle BODY close beyond the level (not just wick)
```

**Multi-Timeframe CHOCH Scoring:**
```
CHOCH_Score =
  Daily CHOCH:  ±1.0 × direction  (highest significance)
  4h CHOCH:     ±0.7 × direction
  1h CHOCH:     ±0.4 × direction
  15m CHOCH:    ±0.2 × direction

  Multi-TF_CHOCH = Σ(CHOCH_Score per TF) / count(TFs with CHOCH)
```

**Hierarchy Rule:** Only trade lower-TF CHOCH if it aligns with higher-TF bias. A 15m bullish CHOCH against a daily bearish CHOCH is IGNORED.

**Validation Checklist (each item = +0.2):**
- [ ] HTF bias alignment
- [ ] Key level (support/resistance/FVG)
- [ ] Volume confirmation (RVOL > 1.5)
- [ ] Clean structure (not inside minor sub-waves)
- [ ] R:R >= 1:2

---

### 3.8 RSI Divergences

**Types and Scores:**
```
Regular Bullish Divergence:  Price LL + RSI HL → +1.0 (reversal signal)
Regular Bearish Divergence:  Price HH + RSI LH → -1.0 (reversal signal)
Hidden Bullish Divergence:   Price HL + RSI LL → +0.7 (trend continuation)
Hidden Bearish Divergence:   Price LH + RSI HH → -0.7 (trend continuation)
```

**Timeframe Weighting:**
```
Divergence signals on higher timeframes are more reliable:
  Daily divergence:   weight = 1.0
  4h divergence:      weight = 0.8
  1h divergence:      weight = 0.5
  15m divergence:     weight = 0.3
```

**Divergence Strength Scoring (additional modifiers):**
```
+ 0.2 if divergence spans > 10 bars (more significant)
+ 0.2 if confirmed by volume divergence (CVD also diverging)
+ 0.1 if at key support/resistance level
- 0.3 if against higher-TF trend (counter-trend divergence is weaker)
```

---

### 3.9 Relative Volume (RVOL)

**Formula:**
```
RVOL = Current_Volume / Average_Volume(same_time_of_day, lookback=20)
```

**Scoring:**
```
RVOL_Score =
  if RVOL > 3.0:  1.0  (extreme participation → high conviction)
  if RVOL > 2.0:  0.8  ("in play" — strong breakout confirmation)
  if RVOL > 1.5:  0.5  (elevated — moderate confirmation)
  if RVOL 0.8-1.5: 0.0 (normal — no edge from volume)
  if RVOL < 0.8: -0.5  (low participation — moves are suspect)
  if RVOL < 0.5: -0.8  (very low — likely to be noise/false signal)
```

**Exhaustion Warning:**
```
if RVOL > 5.0: flag as potential exhaustion/climactic volume
  → reduce directional score by 0.3 (climactic moves often reverse)
```

RVOL acts as a CONFIDENCE MULTIPLIER for other signals. Breakouts with RVOL > 2.0 show 40% greater follow-through (NYSE research).

---

### 3.10 24h Range Position

**Formula:**
```
Range_Position = (Current_Price - Low_24h) / (High_24h - Low_24h)
```
Result: 0.0 (at the low) to 1.0 (at the high)

**Scoring (context-dependent on trend):**
```
In Uptrend:
  Range_Pos > 0.80: +0.5 (strong, near highs — continuation likely)
  Range_Pos 0.40-0.80: +0.3 (healthy pullback in trend)
  Range_Pos < 0.20: +0.7 (deep pullback at support — buy the dip)

In Downtrend:
  Range_Pos < 0.20: -0.5 (weak, near lows — continuation likely)
  Range_Pos 0.20-0.60: -0.3 (healthy bounce in downtrend)
  Range_Pos > 0.80: -0.7 (extended bounce at resistance — sell the rip)

In Range/Neutral:
  Range_Pos < 0.20: +0.5 (mean reversion long)
  Range_Pos > 0.80: -0.5 (mean reversion short)
```

**ADR% Comparison:**
```
Current_Range = High_24h - Low_24h
ADR_avg = Average(Daily_Range, 14)
ADR_Ratio = Current_Range / ADR_avg

if ADR_Ratio > 1.5: move has likely exhausted daily range → mean reversion bias
if ADR_Ratio < 0.5: range compression → breakout anticipation (similar to BB squeeze)
```

---

## 4. MULTI-TIMEFRAME CONFLICT RESOLUTION

### 4.1 Elder Triple Screen Adaptation

Alexander Elder's Triple Screen system established the foundational principle: use THREE timeframes with a ratio of 3-5x between each.

**Modern Crypto Adaptation:**
```
Screen 1 (The Tide):    Daily chart → Establish directional bias
Screen 2 (The Wave):    4h chart   → Identify entry zones (pullbacks/rallies)
Screen 3 (The Ripple):  15m-30m    → Precise entry timing
```

**Rules:**
- Screen 1 determines if you go LONG or SHORT (never trade against it)
- Screen 2 must show a setup that aligns with Screen 1
- Screen 3 provides the exact entry trigger

### 4.2 Hierarchical Weighting System

When signals conflict across timeframes, apply this hierarchy:

```
Timeframe Priority Weights:
  Daily:  0.40 (dominant — overrides all others)
  4h:     0.30 (primary — strong influence)
  1h:     0.15 (secondary — refinement)
  15m:    0.10 (execution — timing only)
  5m:     0.05 (noise — entry trigger only)
```

**Conflict Resolution Rules:**
1. If Daily and 4h AGREE: trade in that direction with full size
2. If Daily and 4h CONFLICT: reduce size by 50% or stand aside
3. If Daily is neutral but 4h has clear signal: trade at 60% size
4. Lower timeframes NEVER override higher timeframes for direction
5. Lower timeframes CAN override higher timeframes for TIMING (delay entry)

### 4.3 Alignment Score

```
Alignment_Score = Σ(TF_Direction × TF_Weight) for all timeframes

Where TF_Direction:
  +1 = bullish on that timeframe
   0 = neutral
  -1 = bearish on that timeframe

Possible range: -1.0 to +1.0

Thresholds:
  |Alignment| > 0.7: Strong alignment → full position
  |Alignment| 0.4-0.7: Moderate alignment → 50-75% position
  |Alignment| < 0.4: Weak/conflicting → stand aside or 25% max
```

---

## 5. CRYPTO-SPECIFIC INDICATOR WEIGHTING

### 5.1 Why Crypto Indicators Differ from Traditional

Crypto markets have unique structural characteristics that require different weighting:

1. **24/7 Trading**: No overnight gaps, continuous price discovery
2. **Funding Rate Mechanism**: Unique to perpetual futures, no traditional equivalent
3. **Leverage Transparency**: OI and liquidation data are publicly visible
4. **Retail Dominance**: L/S ratios reflect a different participant mix than equities
5. **Younger Asset Class**: Less historical data, regime changes more frequent
6. **Higher Volatility**: Traditional RSI/BB parameters may need adjustment

### 5.2 Recommended Weighting Split

```
For Crypto Futures Specifically:

Traditional Technical Indicators:     45% of total weight
  - Multi-TF RSI:           12%
  - RSI Divergences:         8%
  - Bollinger Squeeze:       8%
  - CHOCH/Market Structure:  10%
  - RVOL:                    7%

Crypto-Native Indicators:             55% of total weight
  - Funding Rate + Velocity: 15%
  - Open Interest Delta:     12%
  - CVD Trend + Divergences: 15%
  - Long/Short Ratio:        8%
  - 24h Range Position:      5%
```

**Rationale:** Crypto-native indicators get 55% because they capture information UNIQUE to crypto derivatives markets that traditional indicators cannot access. Funding rates, OI, and CVD provide direct visibility into leveraged positioning — the primary driver of crypto's largest moves (liquidation cascades, short squeezes, funding arbitrage).

### 5.3 Regime-Adaptive Weighting

Professional firms adjust weights based on market regime:

```
HIGH VOLATILITY / TRENDING:
  Increase: Trend (CHOCH, RSI trend) → +5%
  Increase: Flow (CVD, OI) → +5%
  Decrease: Mean-reversion (BB squeeze, range position) → -5%
  Decrease: Sentiment (contrarian L/S, funding) → -5%

LOW VOLATILITY / RANGING:
  Increase: Squeeze (BB percentile) → +5%
  Increase: Sentiment (funding extremes, L/S) → +5%
  Decrease: Trend (CHOCH, RSI trend) → -5%
  Decrease: Flow (CVD directional) → -5%

LIQUIDATION CASCADE / EXTREME:
  Increase: OI delta (capitulation detection) → +10%
  Increase: Funding velocity (unwind detection) → +10%
  Decrease: Traditional technicals → -10%
  Decrease: L/S ratio → -10%
```

---

## 6. BAYESIAN & PROBABILISTIC SIGNAL COMBINATION

### 6.1 Naive Bayes Signal Combination

The simplest probabilistic approach treats each indicator as independent evidence:

```
P(Long | Indicators) ∝ P(Long) × Π P(Indicator_i | Long)

Where:
  P(Long) = base rate probability (e.g., 0.50 in neutral market)
  P(Indicator_i | Long) = probability of seeing this indicator value given a long setup
```

**Example:**
```
P(Long | RSI_bullish, CVD_rising, Funding_negative, CHOCH_bullish)
  = P(Long)
  × P(RSI_bullish | Long)
  × P(CVD_rising | Long)
  × P(Funding_neg | Long)
  × P(CHOCH_bull | Long)
  / P(Evidence)

Each conditional probability is estimated from backtesting data.
```

### 6.2 Log-Odds Bayesian Scoring

A more practical implementation uses log-odds:

```
Log_Odds_Prior = log(P_base / (1 - P_base))  [e.g., log(0.5/0.5) = 0]

For each signal i:
  Evidence_i = log(P(signal_i | Long) / P(signal_i | Short))

Composite_Log_Odds = Log_Odds_Prior + Σ Evidence_i

Final_Probability = 1 / (1 + exp(-Composite_Log_Odds))
```

**Advantages:**
- Naturally handles different signal strengths
- Each signal is additive in log-odds space
- Outputs a probability (0-1) that can be directly used for position sizing
- Can be updated incrementally as new signals arrive

### 6.3 Bayesian Updating in Practice

```
Start with prior: P(trend_up) = 0.50

Observe Signal 1 (Daily RSI > 60, bullish):
  Likelihood ratio: 2.0 (this signal is 2x more likely in uptrends)
  Posterior: 0.50 × 2.0 / (0.50 × 2.0 + 0.50 × 1.0) = 0.667

Observe Signal 2 (CVD divergence bearish):
  Likelihood ratio: 0.5 (this signal is 2x more likely in downtrends)
  Posterior: 0.667 × 0.5 / (0.667 × 0.5 + 0.333 × 1.0) = 0.50

Observe Signal 3 (Funding rate extreme positive):
  Likelihood ratio: 0.4 (bearish lean — contrarian)
  Posterior: 0.50 × 0.4 / (0.50 × 0.4 + 0.50 × 1.0) = 0.286

Result: 28.6% probability of continued uptrend → lean short
```

### 6.4 Calibrating Likelihood Ratios from Backtest Data

```
For each indicator signal value:
  1. Collect all historical instances of this signal
  2. Count outcomes: wins (direction correct) vs. losses
  3. LR = (hits / total_long_signals) / (hits / total_short_signals)

  Typical calibrated likelihood ratios:
    Strong signal (rare but reliable):  LR = 2.5 - 4.0
    Moderate signal:                    LR = 1.3 - 2.0
    Weak signal:                        LR = 1.1 - 1.3
    Noise:                              LR ≈ 1.0
```

---

## 7. ACADEMIC OPTIMAL COMBINATION METHODS

### 7.1 Information Coefficient (IC)-Based Weighting

The most academically rigorous approach weights each indicator by its measured IC:

```
Weight_i = IC_i / Σ IC_j  (normalized IC weighting)

Where IC_i = Spearman_Rank_Correlation(Signal_i_t, Return_t+1)
  measured over rolling lookback window (e.g., 60-120 days)
```

**Dynamic IC Weighting (from arXiv 2025 research):**
Recalculate IC every rebalancing period and dynamically adjust weights.
This outperforms static weights because signal quality varies over time.

### 7.2 PCA-Based Combination

```
1. Compute indicators matrix X = [RSI, Funding, OI, CVD, LS, ...]
2. Standardize each column (z-score)
3. Apply PCA to extract orthogonal components
4. Use top K components (explaining >80% variance) as composite factors
5. Weight components by eigenvalue proportion
```

**Advantage:** Eliminates multicollinearity between correlated indicators (e.g., RSI and BB overlap).

### 7.3 Ensemble Machine Learning Approach

From academic research (2024-2025):

```
Model 1: Ridge Regression on indicator scores
Model 2: Random Forest on indicator scores
Model 3: XGBoost on indicator scores
Model 4: LSTM on raw indicator time series

Ensemble_Score = w1 × M1 + w2 × M2 + w3 × M3 + w4 × M4

Where weights w_i are optimized via:
  - IC-Mean weighting (best empirical performance)
  - Inverse variance weighting
  - Bayesian model averaging
```

**Key Finding:** IC_Mean-based dynamic weighting of ensemble components outperforms static equal-weighting in both backtested returns and predictive performance.

### 7.4 Mixture Design Optimization

From Financial Innovation (2020):

```
1. Define weight space: w1 + w2 + ... + wn = 1, each w_i ≥ 0
2. Generate systematic weight combinations via mixture design
3. Backtest each combination
4. Fit polynomial regression: Performance = f(w1, w2, ..., wn)
5. Find optimal w* that maximizes performance
6. Validate on out-of-sample data
```

**Advantage:** Captures interaction effects between indicators (e.g., RSI + CVD together may be more predictive than sum of individual contributions).

---

## 8. CLASSIC METHODOLOGY ADAPTATIONS

### 8.1 Elder Triple Screen (Modern Crypto Adaptation)

**Original (1986):** Weekly MACD → Daily Stochastic → Intraday trailing stop

**Crypto Adaptation:**
```
Screen 1 - Daily Chart (The Tide):
  - 21 EMA slope direction
  - Daily RSI > 50 (bull) or < 50 (bear)
  - Daily CHOCH status
  → Determines: LONG ONLY or SHORT ONLY

Screen 2 - 4h Chart (The Wave):
  - RSI pullback into 40-50 zone (in uptrend) or 50-60 (in downtrend)
  - CVD divergence at support/resistance
  - Funding rate not at extremes
  - OI delta showing fresh positioning
  → Determines: ENTRY ZONE

Screen 3 - 15m/30m Chart (The Ripple):
  - CHOCH in direction of Screen 1
  - RVOL > 1.5 confirming participation
  - BB squeeze releasing in direction of Screen 1
  → Determines: EXACT ENTRY TRIGGER
```

### 8.2 Mark Minervini SEPA (Crypto Adaptation)

**Original 8-Point Trend Template:**
1. Price > 50 MA, 150 MA, 200 MA
2. 150 MA > 200 MA
3. 200 MA trending up for 1+ month
4. 50 MA > 150 MA and 200 MA
5. Price > 52-week low by at least 25%
6. Price within 25% of 52-week high
7. Relative Strength rating > 70
8. Price trading above 50 MA with increasing volume

**Crypto Adaptation:**
```
1. Price > 21 EMA, 50 EMA, 200 EMA                    [Pass/Fail: +1/0]
2. 50 EMA > 200 EMA (Golden Cross)                     [Pass/Fail: +1/0]
3. 200 EMA slope positive (20+ days)                    [Pass/Fail: +1/0]
4. 21 EMA > 50 EMA > 200 EMA (full alignment)         [Pass/Fail: +1/0]
5. Price > 90-day low by at least 30%                   [Pass/Fail: +1/0]
6. Price within 25% of 90-day high                      [Pass/Fail: +1/0]
7. CVD trend positive (proxy for RS rating)             [Pass/Fail: +1/0]
8. RVOL > 1.5 on recent advance                        [Pass/Fail: +1/0]

SEPA_Score = Sum / 8   (0.0 to 1.0)
Trade only when SEPA_Score >= 0.75 (6/8 criteria met)
```

**VCP Adaptation:** Look for decreasing ranges in consolidation patterns with declining volume, then breakout with RVOL > 2.0.

### 8.3 Stan Weinstein Stage Analysis (Crypto Adaptation)

**The SATA Score (0-10):**
```
10 Technical Attributes (each scored 0 or 1):
1. Price above 30-week MA                               [+1]
2. 30-week MA slope positive                            [+1]
3. Price making higher highs (weekly)                   [+1]
4. Price making higher lows (weekly)                    [+1]
5. Volume increasing on up weeks                        [+1]
6. Volume decreasing on down weeks                      [+1]
7. Mansfield Relative Strength > 0                      [+1]
8. No overhead resistance within 10%                    [+1]
9. Price recently broke out of base on volume           [+1]
10. 10-week MA > 30-week MA                             [+1]

Stage Determination:
  SATA 8-10:  Stage 2 (Advancing)  → LONG candidates
  SATA 4-7:   Stage 1/3 (Base/Distribution) → NEUTRAL
  SATA 0-3:   Stage 4 (Declining) → SHORT candidates
```

**Crypto Adaptation:** Replace 30-week MA with 200-period MA on daily chart. Add OI and funding as additional scoring criteria (+1 each) for crypto-specific confirmation.

### 8.4 Larry Williams COT-Based Scoring (Crypto Adaptation)

**Original WillCo Index:**
```
WillCo = (Current_Net_Commercial - Min_Net_Commercial_26w)
       / (Max_Net_Commercial_26w - Min_Net_Commercial_26w) × 100
```

**Crypto Adaptation (using Funding + OI as proxy for "Commercials"):**
```
Since crypto doesn't have CFTC reporting, the closest proxies are:
- Funding Rate: Proxy for speculative positioning (like Small Specs)
- Top Trader L/S Ratio: Proxy for "informed" money (like Commercials)
- Exchange Whale Wallet Flows: Proxy for institutional positioning

Crypto_COT_Score:
  When Top Traders are net long AND small specs (funding) are bearish: +1.0
  When Top Traders are net short AND small specs (funding) are bullish: -1.0
  When both align: 0.0 (no informational edge)
```

---

## 9. COMPOSITE SCORE FORMULA & WEIGHTS

### 9.1 The Master Composite Score

```
COMPOSITE_SCORE = (
    // PILLAR 1: TREND (30%)
    0.12 × RSI_Composite           +
    0.10 × CHOCH_Multi_TF          +
    0.08 × Trend_Alignment         +  // (EMA alignment, Weinstein stage)

    // PILLAR 2: MOMENTUM (25%)
    0.10 × RSI_Divergence_Score    +
    0.08 × BB_Squeeze_Contribution +
    0.07 × Range_Position_Score    +

    // PILLAR 3: VOLUME & FLOW (25%)
    0.10 × CVD_Score               +
    0.08 × OI_Score                +
    0.07 × RVOL_Score              +

    // PILLAR 4: SENTIMENT (20%)
    0.10 × Funding_Score           +
    0.06 × LS_Ratio_Score          +
    0.04 × Cross_Exchange_Sentiment
)
```

**Score Range:** -1.0 to +1.0
- Positive = Long bias
- Negative = Short bias
- Magnitude = Conviction level

### 9.2 Signal Thresholds

```
| Composite Score | Action | Position Size |
|----------------|--------|---------------|
| +0.70 to +1.00 | Strong Long | 80-100% of max |
| +0.40 to +0.69 | Moderate Long | 40-70% of max |
| +0.20 to +0.39 | Weak Long | 20-40% of max |
| -0.19 to +0.19 | No Trade / Neutral | 0% |
| -0.20 to -0.39 | Weak Short | 20-40% of max |
| -0.40 to -0.69 | Moderate Short | 40-70% of max |
| -0.70 to -1.00 | Strong Short | 80-100% of max |
```

### 9.3 Confluence Gate (Minimum Pillar Requirement)

**Before taking a trade, verify PILLAR CONFLUENCE:**
```
Pillar_1_Direction = sign(RSI_Composite + CHOCH_Score + Trend_Alignment)
Pillar_2_Direction = sign(RSI_Divergence + BB_Squeeze + Range_Position)
Pillar_3_Direction = sign(CVD_Score + OI_Score + RVOL_Score)
Pillar_4_Direction = sign(Funding_Score + LS_Ratio_Score)

Pillars_Aligned = count of pillars matching Composite_Score direction

Rules:
  4/4 pillars aligned: Full position (HIGH CONVICTION)
  3/4 pillars aligned: 75% position (GOOD CONVICTION)
  2/4 pillars aligned: 40% position (LOW CONVICTION, tight stops)
  1/4 or 0/4 aligned: NO TRADE (insufficient confluence)
```

### 9.4 Conviction Multipliers

```
Final_Conviction = Base_Score × TF_Alignment_Mult × Volatility_Mult × Regime_Mult

Where:
  TF_Alignment_Mult = 0.5 + (0.5 × |Alignment_Score|)     [0.5 to 1.0]
  Volatility_Mult   = 1.0 if BB_Squeeze active, else 0.85
  Regime_Mult       = 1.0 if regime clear, 0.7 if transitional
```

---

## 10. KELLY CRITERION POSITION SIZING

### 10.1 Core Formula

```
Kelly% = (W × R - L) / R

Where:
  W = Win rate (from backtest for this signal strength range)
  L = Loss rate (1 - W)
  R = Average Win / Average Loss (reward:risk ratio)
```

### 10.2 Signal-Strength Adjusted Position Sizing

```
Step 1: Calculate base Kelly%
  Example: W = 0.58, R = 1.8
  Kelly% = (0.58 × 1.8 - 0.42) / 1.8 = 0.347 = 34.7%

Step 2: Apply Fractional Kelly (risk reduction)
  Professional standard: Half Kelly = 17.35%
  Conservative: Quarter Kelly = 8.67%

Step 3: Scale by Composite Score (signal strength)
  Adjusted_Size = Fractional_Kelly × |Composite_Score| × Pillar_Gate_Factor

Step 4: Apply hard caps
  Max single position: 5% of portfolio (regardless of Kelly)
  Max correlated exposure: 15% of portfolio
  Max total exposure: 50% of portfolio
```

### 10.3 Tiered Position Sizing Table

```
| Composite Score | Pillar Agreement | Kelly Fraction | Typical Size |
|----------------|-----------------|----------------|-------------|
| |Score| > 0.80 | 4/4 pillars | Half Kelly | 3-5% |
| |Score| > 0.60 | 3-4/4 pillars | Third Kelly | 2-3% |
| |Score| > 0.40 | 3/4 pillars | Quarter Kelly | 1-2% |
| |Score| > 0.20 | 2/4 pillars | Eighth Kelly | 0.5-1% |
| |Score| < 0.20 | Any | Zero | 0% |
```

### 10.4 Bayesian Kelly (Dynamic)

```
Instead of fixed W and R, update them based on recent performance:

W_bayesian = (prior_wins + recent_wins) / (prior_total + recent_total)
R_bayesian = (prior_avg_win/loss + recent_avg_win/loss) / 2

This creates a "Bayesian shrinkage" effect that prevents overfitting
to short recent streaks while adapting to genuine regime changes.

Typical prior: 100 "virtual" trades at baseline W=0.50, R=1.5
```

---

## 11. IMPLEMENTATION GUIDELINES

### 11.1 Backtesting Requirements

1. **Minimum 1000 trades** for statistical significance
2. **Walk-forward optimization** (never optimize on full dataset)
3. **Out-of-sample testing** on at least 30% of data
4. **Monte Carlo simulation** (10,000 randomized runs) to assess robustness
5. **Transaction cost modeling** (0.04% taker fee minimum for crypto)
6. **Slippage modeling** (0.01-0.05% depending on liquidity)
7. **Regime-specific performance** (bull, bear, sideways separately)

### 11.2 Weight Optimization Schedule

```
Frequency: Monthly recalibration
Method: Rolling IC calculation (120-day window)

For each indicator:
  1. Calculate trailing IC (Spearman correlation with forward returns)
  2. If IC < 0.01: zero the weight (indicator is noise)
  3. If IC has decayed >50% from historical average: halve the weight
  4. Normalize remaining weights to sum to 1.0
  5. Compare optimized weights vs. equal weights vs. prior weights
  6. Use weighted average of optimized and prior (shrinkage toward prior)
```

### 11.3 Risk Management Overrides

```
HARD RULES (override any composite score):

1. Max drawdown circuit breaker:
   If portfolio DD > 10%: reduce all positions by 50%
   If portfolio DD > 15%: close all positions, paper trade only

2. Volatility scaling:
   Position_Size = Base_Size × (Target_Vol / Realized_Vol)
   Target_Vol = 15% annualized (typical)

3. Correlation check:
   If taking multiple positions, check cross-correlations
   If corr > 0.7: treat as single position for sizing purposes

4. Event risk:
   Before known events (FOMC, CPI, halving), reduce size by 50%
   During extreme VIX/MOVE index readings, reduce size by 30%

5. Time-based stops:
   If trade hasn't moved in expected direction within 2× expected
   holding period, exit regardless of score
```

### 11.4 Score Decay & Staleness

```
Signals are not permanent. Apply time decay to scores:

Score_Effective = Score_Original × decay_factor^(bars_since_signal / halflife)

Halflife by indicator type:
  CHOCH:           24 bars on that timeframe
  RSI Divergence:  12 bars on that timeframe
  Funding:          8 hours (1 funding period)
  OI Delta:         4 hours
  CVD Trend:       12 bars on reference timeframe
  BB Squeeze:      36 bars (squeeze can persist)
  L/S Ratio:       24 hours
  RVOL:             4 bars (volume is very time-sensitive)
  Range Position:  Recalculated continuously (no decay)
```

---

## KEY SOURCES & REFERENCES

### Academic & Institutional
- Grinold, R. (1989). "The Fundamental Law of Active Management"
- Clarke, de Silva, Thorley (2002). "Portfolio Constraints and the Fundamental Law"
- arXiv (2025). "Combined Machine Learning with Dynamic Weighting for Stock Selection"
- Financial Innovation (2020). "Optimal Weights in Weighted-Scoring Stock-Picking Models"

### Classic Trading Methodologies
- Elder, A. (1986). "Triple Screen Trading System" — Futures Magazine
- Minervini, M. (2013). "Trade Like a Stock Market Wizard" — SEPA methodology
- Weinstein, S. (1988). "Secrets for Profiting in Bull and Bear Markets"
- Williams, L. (2005). "Trade Stocks and Commodities with the Insiders: Secrets of the COT Report"
- Bollinger, J. (2001). "Bollinger on Bollinger Bands"
- Kelly, J.L. (1956). "A New Interpretation of Information Rate" — Bell System Technical Journal

### Crypto-Specific
- CryptoCred. "Comprehensive Guide to Crypto Futures Indicators"
- CoinGlass. Long/Short Ratio, Funding Rate, Open Interest data
- CryptoQuant. On-chain and derivatives analytics
- TradingRiot. "Understanding Cryptocurrency Derivatives Data"

### Tools & Platforms
- TradingView: VMDM, BBWP, nOI+Funding+CVD strategy scripts
- SentimenTrader: Absolute and Relative Trend Score framework
- smart-money-concepts (Python): Algorithmic BOS/CHOCH detection
