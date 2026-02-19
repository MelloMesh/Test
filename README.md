# Crypto Screener — Python CLI (Binance)

An autonomous, agentic cryptocurrency screener that surfaces high-probability trading
setups in real time using Binance public market data. Pure technical signal system —
no ML, no execution, screener only.

---

## Architecture

```
crypto_screener/
├── config.py                 # thresholds, weights, symbol count, timeframes
├── data/
│   └── binance_client.py     # OHLCV + 24h ticker via Binance public REST
├── signals/
│   ├── rsi.py                # RSI overbought/oversold, divergence, mid-line cross
│   ├── bollinger.py          # Mean reversion, BB squeeze breakout, %B extremes
│   └── volume.py             # Volume spike, OBV trend, MFI(14)
├── screener/
│   ├── scorer.py             # Multi-factor signal combiner + confidence scoring
│   └── runner.py             # Async scan loop, rich terminal table output
├── backtest/
│   └── engine.py             # Vectorized backtester, ATR-based stops, performance metrics
└── tests/
    └── test_signals.py       # 37 independent unit tests (one per signal function)
```

---

## Requirements

```
Python 3.11+
pandas >= 2.0
numpy >= 1.26
scipy >= 1.11
requests >= 2.31
rich >= 13.0
pytest >= 7.4
```

Install:
```bash
pip install -r crypto_screener/requirements.txt
```

---

## How to Run

### Live Screener

```bash
# Scan top 30 symbols across 15m, 1h, 4h (defaults)
python -m crypto_screener.screener.runner

# Scan top 50 symbols, only 1h, with lower threshold
python -m crypto_screener.screener.runner --symbols 50 --tf 1h --threshold 2.5
```

**CLI flags:**

| Flag | Default | Description |
|---|---|---|
| `--symbols N` | 30 | Top-N symbols by Binance 24h USDT volume |
| `--tf TF [TF...]` | 15m 1h 4h | Timeframes to scan |
| `--threshold T` | 3.0 | Minimum \|composite score\| to surface a setup |

### Backtester

```bash
# Backtest BTC/USDT 1h with default settings (1000 candles ≈ 6 weeks on 1h)
python -m crypto_screener.backtest.engine

# Backtest ETH/USDT 4h with more history
python -m crypto_screener.backtest.engine --symbol ETHUSDT --tf 4h --limit 2000

# Lower threshold to generate more signals for backtest analysis
python -m crypto_screener.backtest.engine --symbol BTCUSDT --tf 1h --limit 1500 --threshold 2.5
```

### Unit Tests

```bash
python -m pytest crypto_screener/tests/test_signals.py -v
```

---

## How to Interpret Output

```
                    Crypto Screener — High-Probability Setups
 SYMBOL      TF   DIR    SCORE   CONF   SIGNALS FIRED
 BTCUSDT     1h   LONG   +4.50   0.82   RSI_BULL_DIV, BB_SQUEEZE_BULL, OBV_BULL_DIV, VOL_SPIKE
 ETHUSDT     4h   SHORT  -3.50   0.64   RSI_OB, BB_UPPER, MFI_OB
```

| Column | Meaning |
|---|---|
| **DIR** | LONG or SHORT — determined by composite score sign |
| **SCORE** | Sum of all active signal weights (positive = bullish, negative = bearish) |
| **CONF** | `abs(score) / max_possible_score` — normalized 0–1 |
| **SIGNALS FIRED** | All signals that contributed to the score |

A setup is only surfaced when **all three gates** pass:
1. `|score| >= threshold` (default 3.0)
2. Signals came from **≥ 2 distinct categories** (RSI, BB, VOLUME)
3. Symbol 24h volume is above 20th percentile of universe

---

## Signal Weights

| Signal | Weight | Trigger |
|---|---|---|
| RSI_BULL_DIV | +2.0 | Price lower low, RSI higher low |
| RSI_BEAR_DIV | −2.0 | Price higher high, RSI lower high |
| BB_SQUEEZE_BULL | +2.0 | Post-squeeze breakout above upper band + vol confirm |
| BB_SQUEEZE_BEAR | −2.0 | Post-squeeze breakout below lower band + vol confirm |
| MFI_OS | +1.5 | MFI(14) < 20 |
| MFI_OB | −1.5 | MFI(14) > 80 |
| OBV_BULL_DIV | +1.5 | OBV slope up while price slope down |
| OBV_BEAR_DIV | −1.5 | OBV slope down while price slope up |
| RSI_OS | +1.0 | RSI(14) < 30 |
| RSI_OB | −1.0 | RSI(14) > 70 |
| BB_LOWER | +1.0 | Close below lower Bollinger Band |
| BB_UPPER | −1.0 | Close above upper Bollinger Band |
| BB_PCT_B_LOW | +1.0 | %B < 0.05 (extreme oversold) |
| BB_PCT_B_HIGH | −1.0 | %B > 0.95 (extreme overbought) |
| VOL_SPIKE | +1.0 | Volume > 2.5x 20-period average (direction-neutral) |
| OBV_CONFIRM | +0.5 | OBV and price slope in same direction |
| RSI_MID_BULL | +0.5 | RSI crosses 50 from below |
| RSI_MID_BEAR | −0.5 | RSI crosses 50 from above |

---

## How to Configure

Edit `crypto_screener/config.py` to change any threshold, weight, or parameter.
Key settings:

```python
TOP_N_SYMBOLS = 30        # symbols to scan
TIMEFRAMES = ["15m","1h","4h"]
COMPOSITE_THRESHOLD = 3.0 # min score to surface
WEIGHTS = { ... }         # signal weights dict
ATR_PERIOD = 14
STOP_ATR_MULT = 1.5       # stop = entry ± 1.5 ATR
TAKE_PROFIT_ATR_MULT = 2.5
```

---

## Backtester Interpretation

```
Backtest Report — BTCUSDT / 1h (1000 candles)
  Trades       │ 47
  Win Rate     │ 58.3%
  Avg R:R      │ 1.82
  Sharpe       │ 1.24
  Max Drawdown │ -8.4%
  Total Return │ 34.2%
  Status       │ [VALIDATED]
```

Strategy is flagged **[UNVALIDATED]** if:
- Win rate < 52%, or
- Annualized Sharpe < 0.8

**Note:** More candles = more reliable backtest. Use `--limit 1500` on 1h for ~2 months.

---

# Bybit Futures Screener

A live browser-based scanner for every USDT perpetual on Bybit. No install, no login.

**Link:** https://mellomesh.github.io/Test/

---

## How to Use (Step by Step)

### 1. Open & Wait
Go to the link above. First load takes ~2 minutes while it pulls data for all contracts. After that, it caches and loads instantly.

### 2. Scan the Table
Rows are sorted by **Score** (highest first). Top rows = most signals firing at once.

### 3. Filter What You Care About
Click filter buttons to narrow results (combine multiple):

| Button | Shows |
|---|---|
| **RSI ≤ 30** | Oversold coins (bounce candidates) |
| **RSI ≥ 70** | Overbought coins (pullback candidates) |
| **Divergence** | Price vs RSI disagree (reversal signal) |
| **CHOCH** | Market structure shift on any timeframe |
| **CHOCH Aligned** | Structure shift on 2+ timeframes same direction |
| **Rel Vol > 2x** | Volume is 2x+ above 20-day average |

### 4. Search a Specific Coin
Type in the search bar to filter by symbol name.

### 5. Sort by Any Column
- **Click** a column header to sort by that column (ascending/descending toggle)
- **Shift+Click** additional columns to add secondary sort levels (e.g. sort by CHOCH first, then by Score within ties)
- Numbered badges show your sort priority when multi-sorting

### 6. Confirm on Chart
Use the data as a starting point — always check the actual chart before trading.

---

## Reading the Columns

| Column | What It Shows |
|---|---|
| **Symbol** | Trading pair |
| **Price** | Current price |
| **24H Chg%** | 24h price change (green = up, red = down) |
| **24H Vol** | 24h dollar volume |
| **Rel Vol** | Volume vs 20-day average (green 2.0x = double normal) |
| **RSI 15/30m** | 15-min and 30-min RSI values + divergence dots |
| **RSI HTF** | Most notable RSI from 4H or Daily chart |
| **CHOCH** | Change of Character — e.g. "BUL 3/4" = bullish on 3 of 4 timeframes |
| **Score** | Combined signal strength (higher = more confluence) |
| **Funding** | Funding rate (green = shorts pay longs, red = longs pay shorts) |
| **OI** | Open interest in dollars |

---

## RSI Column Quick Reference

**Number color = RSI level:**
- Green = oversold (≤ 30)
- White = neutral (30-60)
- Orange = warming up (60-70)
- Red = overbought (≥ 70)

**Dot next to number = divergence:**
- Green dot = bullish divergence (price lower low, RSI higher low — selling weakening)
- Red dot = bearish divergence (price higher high, RSI lower high — buying weakening)
- No dot = no divergence

---

## Score Breakdown

| Component | Max Points |
|---|---|
| RSI extremeness (any timeframe) | 25 |
| Divergence (15m + 30m) | 30 |
| CHOCH signals + alignment bonus | 50 |
| Relative volume > 2x | 15 |

---

## Good to Know

- Auto-refreshes every 60 seconds
- Data is cached — revisits load instantly
- High score = "look at this chart", not "trade this"
- Best setups: extreme RSI + divergence + CHOCH aligned + high volume
