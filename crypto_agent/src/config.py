"""
Configuration for the crypto trading agent.
All settings, environment variables, and risk parameters.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

DB_PATH = DATA_DIR / "candles.db"

# ---------------------------------------------------------------------------
# Exchange
# ---------------------------------------------------------------------------
BINANCE_API_KEY = os.environ.get("BINANCE_API_KEY", "")
BINANCE_SECRET = os.environ.get("BINANCE_SECRET", "")

LIVE_TRADING_ENABLED = os.environ.get("LIVE_TRADING_ENABLED", "false").lower() == "true"

EXCHANGE_TIMEOUT = 30_000  # ms

# ---------------------------------------------------------------------------
# Symbols & Timeframes
# ---------------------------------------------------------------------------
DEFAULT_SYMBOLS = ["BTC/USDT:USDT", "ETH/USDT:USDT"]
SUPPORTED_TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "4h"]
STRATEGY_TIMEFRAMES = ["15m", "30m"]  # Primary execution timeframes

# ---------------------------------------------------------------------------
# Risk Management (NON-NEGOTIABLE)
# ---------------------------------------------------------------------------
DEFAULT_RISK_PCT = 0.01        # 1% of equity per trade
MAX_RISK_PCT = 0.02            # 2% max (full confluence)
MAX_TOTAL_EXPOSURE = 0.05      # 5% of portfolio at any time
MAX_CONCURRENT_POSITIONS = 3
DAILY_LOSS_LIMIT = 0.03        # 3% → halt 24 hours
WEEKLY_LOSS_LIMIT = 0.07       # 7% → halt until manual review
CONSECUTIVE_LOSS_THRESHOLD = 3  # After 3 losses → reduce to 0.5%
REDUCED_RISK_PCT = 0.005       # 0.5% after consecutive losses
MAX_STOP_DISTANCE_PCT = 0.05   # 5% max stop distance from entry

# ---------------------------------------------------------------------------
# Leverage
# ---------------------------------------------------------------------------
DEFAULT_LEVERAGE = 3
MAX_LEVERAGE = 10
MARGIN_MODE = "isolated"       # Never cross margin

# ---------------------------------------------------------------------------
# Fibonacci Levels
# ---------------------------------------------------------------------------
FIB_RETRACEMENT_LEVELS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 0.886, 1.0]
FIB_EXTENSION_LEVELS = [1.272, 1.618, 2.0]
GOLDEN_POCKET_START = 0.618
GOLDEN_POCKET_END = 0.886

# ---------------------------------------------------------------------------
# RSI Configuration
# ---------------------------------------------------------------------------
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
RSI_PEAK_MIN_CANDLES = 2  # Minimum candles of reversal to confirm peak

# ---------------------------------------------------------------------------
# Swing Detection
# ---------------------------------------------------------------------------
SWING_LOOKBACK = 100           # Candles to analyze
SWING_MIN_BARS = 5             # Min candles on each side of swing point

# ---------------------------------------------------------------------------
# Trend & Regime Filters
# ---------------------------------------------------------------------------
HTF_EMA_PERIOD = 50            # EMA period for higher-timeframe trend
ADX_PERIOD = 14                # ADX calculation period
ADX_STRONG_TREND = 40          # ADX above this = strong trend → skip divergence
ADX_WEAK_TREND = 20            # ADX below this = ranging → ideal for divergence
ATR_PERIOD = 14                # ATR calculation period
ATR_STOP_MIN_MULTIPLIER = 1.0  # Stop must be at least 1.0 ATR from entry
ATR_STOP_MAX_MULTIPLIER = 3.0  # Stop must not exceed 3.0 ATR from entry

# Map each execution TF to a higher TF for trend confirmation
HTF_MAP = {
    "1m": "15m",
    "5m": "1h",
    "15m": "1h",
    "30m": "4h",
    "1h": "4h",
    "4h": "1d",
}

# ---------------------------------------------------------------------------
# Take Profit Defaults (refined by backtesting)
# Research: 3 levels is sweet spot. Trail final portion for max R.
# ---------------------------------------------------------------------------
DEFAULT_TP_CONFIG = [
    {"level": 0.5, "pct": 0.40},    # TP1: 40% at 0.5 fib (secure profit early)
    {"level": 0.236, "pct": 0.30},   # TP2: 30% at 0.236 fib (deeper recovery)
    {"level": "trail", "pct": 0.30}, # TP3: 30% trailed — no fixed exit, let winners run
]

# Trailing portion config
TRAIL_ACTIVATION_R = 1.5             # Start trailing after 1.5R profit
TRAIL_ATR_MULTIPLIER = 2.0           # Trail distance: 2.0 × ATR

# ---------------------------------------------------------------------------
# Costs (Binance Futures)
# ---------------------------------------------------------------------------
MAKER_FEE = 0.0002    # 0.02%
TAKER_FEE = 0.0004    # 0.04%
SLIPPAGE_EST = 0.0003  # 0.03% estimated slippage

# Flat round-trip cost estimate for backtesting
# (taker fee + slippage) × 2 sides = (0.04% + 0.03%) × 2 = 0.14%
ROUND_TRIP_COST_PCT = 0.0014

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
