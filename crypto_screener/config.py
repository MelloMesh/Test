"""
config.py — Global configuration for the crypto screener.
Single source of truth for all thresholds, weights, and parameters.
"""

# ── Universe ──────────────────────────────────────────────────────────────────
TOP_N_SYMBOLS: int = 30          # number of top-volume USDT pairs to scan
QUOTE_ASSET: str = "USDT"        # only pairs quoted in this asset

# ── Timeframes ────────────────────────────────────────────────────────────────
TIMEFRAMES: list[str] = ["15m", "1h", "4h"]
DEFAULT_TIMEFRAME: str = "1h"
KLINE_LIMIT: int = 300           # candles fetched per timeframe (≥ 120 for BB squeeze percentile)
BACKTEST_KLINE_LIMIT: int = 1000 # ~6 months on 1h; ~10 months on 4h

# ── Binance REST ──────────────────────────────────────────────────────────────
BINANCE_BASE_URL: str = "https://api.binance.com"
REQUEST_TIMEOUT: int = 10        # seconds
MAX_RETRIES: int = 4             # exponential backoff retries
CACHE_TTL_SECONDS: int = 60      # in-memory cache lifetime

# ── RSI Parameters ───────────────────────────────────────────────────────────
RSI_PERIOD: int = 14
RSI_OVERSOLD: float = 30.0
RSI_OVERBOUGHT: float = 70.0
RSI_MIDLINE: float = 50.0
RSI_DIV_MIN_WINDOW: int = 5
RSI_DIV_MAX_WINDOW: int = 20
RSI_DIV_ORDER: int = 3          # scipy argrelextrema order (local extrema sensitivity)

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
VOL_ILLIQUID_PERCENTILE: float = 20.0 # ignore symbols below 20th pct volume
MFI_PERIOD: int = 14
MFI_OVERSOLD: float = 20.0
MFI_OVERBOUGHT: float = 80.0

# ── Scoring & Thresholds ─────────────────────────────────────────────────────
COMPOSITE_THRESHOLD: float = 3.0      # minimum |score| to surface a setup
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
    "BB_PCT_B_LOW": 1.0,    # %B < 0.05 extreme oversold
    "BB_PCT_B_HIGH": -1.0,  # %B > 0.95 extreme overbought

    # Volume
    "VOL_SPIKE": 1.0,       # direction-neutral momentum confirmation
    "OBV_BULL_DIV": 1.5,    # OBV diverging bullish from price
    "OBV_BEAR_DIV": -1.5,   # OBV diverging bearish from price
    "OBV_CONFIRM": 0.5,     # OBV confirming direction
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
