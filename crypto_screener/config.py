"""
config.py — Global configuration for the crypto screener.
Single source of truth for all thresholds, weights, and parameters.
"""

# ── Universe ──────────────────────────────────────────────────────────────────
TOP_N_SYMBOLS: int = 100         # number of top-volume USDT pairs to scan
QUOTE_ASSET: str = "USDT"        # only pairs quoted in this asset

# Absolute minimum 24h USD volume — hard floor before any relative filtering.
# At a hedge fund level, anything below $20M/day has unacceptable market impact
# for even modest position sizes and is susceptible to wash trading / manipulation.
MIN_VOLUME_USD: float = 20_000_000.0  # $20M minimum 24h quote volume

# Base assets that are stablecoins or pegged — skip entirely (not tradeable setups)
EXCLUDED_BASE_ASSETS: set[str] = {
    "USDC", "FDUSD", "BUSD", "TUSD", "DAI", "USDP",  # stablecoins
    "PAXG", "XAUT",                                     # gold-pegged
    "EUR", "GBP", "AUD",                               # fiat-pegged
    "USD1", "USD0",                                     # synthetic dollars
}

# ── Timeframes ────────────────────────────────────────────────────────────────
TIMEFRAMES: list[str] = ["15m", "1h", "4h"]
DEFAULT_TIMEFRAME: str = "1h"
KLINE_LIMIT: int = 300           # candles fetched per timeframe (≥ 120 for BB squeeze percentile)
BACKTEST_KLINE_LIMIT: int = 1000 # ~6 months on 1h; ~10 months on 4h

# ── Concurrency ───────────────────────────────────────────────────────────────
MAX_WORKERS: int = 15            # parallel fetch threads; 15 keeps us well under Binance rate limits

# ── Watch mode ────────────────────────────────────────────────────────────────
WATCH_INTERVAL_SECONDS: int = 60  # seconds between scan cycles in --watch mode

# ── Binance REST ──────────────────────────────────────────────────────────────
BINANCE_BASE_URL: str = "https://api.binance.com"
REQUEST_TIMEOUT: int = 6         # seconds; fail fast so timed-out symbols don't stall the pool
MAX_RETRIES: int = 1             # one retry is enough; bad symbols are skipped, not retried forever
CACHE_TTL_SECONDS: int = 55      # slightly under WATCH_INTERVAL so cache expires between cycles

# ── RSI Parameters ───────────────────────────────────────────────────────────
RSI_PERIOD: int = 14
RSI_OVERSOLD: float = 30.0
RSI_OVERBOUGHT: float = 70.0
RSI_MIDLINE: float = 50.0
RSI_DIV_MIN_WINDOW: int = 5
RSI_DIV_MAX_WINDOW: int = 20
RSI_DIV_ORDER: int = 5          # scipy argrelextrema order; higher = stricter, fewer false positives on noisy alts

# ── Bollinger Band Parameters ────────────────────────────────────────────────
BB_PERIOD: int = 20
BB_STD: float = 2.0
BB_SQUEEZE_PERCENTILE: float = 20.0   # bandwidth < 20th pct of trailing window = squeeze
BB_SQUEEZE_WINDOW: int = 120          # trailing candles for percentile calc
BB_BREAKOUT_VOL_MULT: float = 1.5     # vol must be > 1.5x avg to confirm breakout
BB_PERCENT_B_LOW: float = 0.05        # extreme oversold
BB_PERCENT_B_HIGH: float = 0.95       # extreme overbought

# ── Volume Parameters ────────────────────────────────────────────────────────
VOL_SPIKE_MULT: float = 2.5           # current vol > 2.5x 20-period avg
VOL_SPIKE_WINDOW: int = 20
OBV_SLOPE_WINDOW: int = 14            # linear regression period for OBV trend
VOL_ILLIQUID_PERCENTILE: float = 20.0 # ignore symbols below 20th pct volume (relative gate, after absolute floor)
MFI_PERIOD: int = 14
MFI_OVERSOLD: float = 20.0
MFI_OVERBOUGHT: float = 80.0

# ── Trend / Regime Parameters ─────────────────────────────────────────────────
# EMA-200 defines structural trend. In trending markets (ADX > threshold) we
# only accept signals that align with the trend direction — avoids falling knives
# and shorting rockets. In ranging markets (ADX < threshold) mean-reversion is
# valid in either direction.
EMA_TREND_PERIOD: int = 200
EMA_TREND_BUFFER: float = 0.005      # 0.5% buffer around EMA to avoid noise at midpoint
ADX_PERIOD: int = 14
ADX_TRENDING_THRESHOLD: float = 20.0 # ADX > 20 = trending; ADX ≤ 20 = ranging
REQUIRE_TREND_ALIGNMENT: bool = True  # gate: block counter-trend setups in trending markets
MIN_CANDLES: int = 220               # minimum candles required before scoring (covers EMA-200 warmup)

# ── Scoring & Thresholds ─────────────────────────────────────────────────────
COMPOSITE_THRESHOLD: float = 3.5      # raised from 3.0; tighter filter given fixed signal inflation
MIN_SIGNAL_CATEGORIES: int = 2        # must fire signals from ≥2 distinct categories

# Signal weights — calibrated against SYSTEMATIC_SCORING_SYSTEM_RESEARCH.md
WEIGHTS: dict[str, float] = {
    # RSI
    "RSI_OS": 1.0,          # oversold (bullish)
    "RSI_OB": -1.0,         # overbought (bearish)
    "RSI_BULL_DIV": 2.0,    # bullish divergence — highest conviction RSI signal
    "RSI_BEAR_DIV": -2.0,   # bearish divergence
    "RSI_MID_BULL": 0.5,    # mid-line cross from below
    "RSI_MID_BEAR": -0.5,   # mid-line cross from above

    # Bollinger Bands
    "BB_LOWER": 1.0,        # close < lower band (mean reversion long)
    "BB_UPPER": -1.0,       # close > upper band (mean reversion short)
    "BB_SQUEEZE_BULL": 2.0, # post-squeeze bullish breakout
    "BB_SQUEEZE_BEAR": -2.0,# post-squeeze bearish breakout
    # BB_PCT_B fires with BB_LOWER/BB_UPPER ~95% of the time (both fire when %B < 0).
    # Reduced to 0.3 / -0.3 so co-firing adds marginal confirmation rather than
    # doubling the band-touch score. The rare case where %B is 0.01–0.05 (close
    # just inside the lower band) still gets a small independent signal.
    "BB_PCT_B_LOW": 0.3,    # %B < 0.05 — slight extra confirmation when below/near lower band
    "BB_PCT_B_HIGH": -0.3,  # %B > 0.95 — slight extra confirmation when above/near upper band

    # Volume
    # VOL_SPIKE is direction-neutral. It now only adds positive weight, meaning it
    # contributes to LONG score regardless of price direction. This is an acceptable
    # simplification — the trend gate prevents it from pushing counter-trend setups
    # over threshold in strongly trending markets.
    "VOL_SPIKE": 0.75,      # direction-neutral momentum confirmation (reduced from 1.0)
    "OBV_BULL_DIV": 1.5,    # OBV diverging bullish from price
    "OBV_BEAR_DIV": -1.5,   # OBV diverging bearish from price
    # OBV_CONFIRM only fires when OBV ↑ and price ↑ (bullish confirmation).
    # Bearish OBV confirmation (both ↓) currently fires nothing — keeping this
    # asymmetry intentional but reducing weight since it's a weak independent signal.
    "OBV_CONFIRM": 0.3,     # OBV confirming bullish direction (reduced from 0.5)
    "MFI_OS": 1.5,          # MFI oversold (price + volume)
    "MFI_OB": -1.5,         # MFI overbought
}

# Max possible score (sum of all positive weights) for normalization
MAX_POSSIBLE_SCORE: float = sum(w for w in WEIGHTS.values() if w > 0)

# ── Backtester Parameters ────────────────────────────────────────────────────
ATR_PERIOD: int = 14
STOP_ATR_MULT: float = 1.5          # stop = entry ± 1.5 * ATR
TAKE_PROFIT_ATR_MULT: float = 2.5   # TP = entry ± 2.5 * ATR
MIN_WIN_RATE: float = 0.52          # flag strategy if below this
MIN_SHARPE: float = 0.8             # flag strategy if below this

# ── Output ────────────────────────────────────────────────────────────────────
RICH_TABLE_TITLE: str = "Crypto Screener — High-Probability Setups"
