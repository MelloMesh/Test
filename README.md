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
Click a column header to sort ascending/descending.

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
