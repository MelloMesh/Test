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
# Take Profit Defaults (refined by backtesting)
# ---------------------------------------------------------------------------
DEFAULT_TP_CONFIG = [
    {"level": 0.5, "pct": 0.30},    # TP1: 30% at 0.5 fib
    {"level": 0.382, "pct": 0.30},   # TP2: 30% at 0.382 fib
    {"level": 0.236, "pct": 0.20},   # TP3: 20% at 0.236 fib
    {"level": 1.272, "pct": 0.20},   # TP4: 20% at extension
]

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
