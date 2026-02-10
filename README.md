# Bybit Futures Screener — Quick Guide

**What is this?**
A live scanner that checks every USDT perpetual contract on Bybit and highlights the ones worth looking at. It runs in your browser — no install, no login.

**Link:** https://mellomesh.github.io/Test/

---

## Columns explained

| Column | What it means |
|---|---|
| **Symbol** | The trading pair |
| **Price** | Current price |
| **24H Chg%** | How much it moved in 24 hours. Green = up, red = down |
| **24H Vol** | Total dollar volume traded in 24 hours |
| **Rel Vol** | How today's volume compares to the 20-day average. Green 2.0x = double normal volume |
| **RSI 15/30m** | Two RSI values (15-minute / 30-minute) — see details below |
| **RSI HTF** | The most notable RSI from the 4-hour or daily chart, with a label showing which one |
| **CHOCH** | Change of Character — market structure is shifting. "BUL 3/4" = bullish shift on 3 of 4 timeframes. Blue highlight = all timeframes agree |
| **Score** | A single number combining all signals. Higher = more reasons to look at this chart |
| **Funding** | Funding rate. Green = shorts paying longs. Red = longs paying shorts |
| **OI** | Open interest in dollar terms |

---

## Reading the RSI 15/30m column

This column packs two pieces of information: **RSI level** and **divergence**.

**The number color tells you the RSI level:**
- **Green number** = oversold (RSI ≤ 30) — price may be due for a bounce
- **Orange number** = getting overbought (RSI 60-70)
- **Red number** = overbought (RSI ≥ 70) — price may be due for a pullback
- **White number** = neutral (RSI 30-60)

**The colored dot next to a number means divergence is detected:**
- **Green dot** = bullish divergence — price is making lower lows but RSI is making higher lows. This is a reversal signal suggesting the selling is weakening and price may bounce up
- **Red dot** = bearish divergence — price is making higher highs but RSI is making lower highs. This is a reversal signal suggesting the buying is weakening and price may drop
- **No dot** = no divergence detected

**Example:** `40.4🟢 / 37.8` means:
- 15m RSI is 40.4 (neutral) **with bullish divergence** (green dot)
- 30m RSI is 37.8 (neutral) with no divergence

The dots and number colors are independent — you can have a neutral RSI number with a divergence dot, or an extreme RSI number with no dot.

---

## How to use it

1. **Open the link and wait** — first load takes a couple minutes while it scans all contracts
2. **Look at the top rows** — sorted by Score, so the most interesting setups are at the top
3. **Use the filter buttons** to narrow down:
   - **RSI ≤ 30** — oversold coins (long candidates)
   - **RSI ≥ 70** — overbought coins (short candidates)
   - **Divergence** — price and RSI disagree (reversal signal)
   - **CHOCH Aligned** — structure shifting on multiple timeframes
   - **Rel Vol > 2x** — unusual volume activity
4. **Search** for a specific coin using the search bar
5. **Click any column header** to sort by that column
6. **Open the chart** on TradingView or Bybit to confirm before trading

---

## Tips

- The page auto-refreshes every 60 seconds
- Next time you visit, it loads instantly from cache
- A high Score doesn't mean "trade this" — it means "look at this chart"
- Best setups usually combine: extreme RSI + divergence + CHOCH + high volume
