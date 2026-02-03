"""
Crypto USDT Perpetual Futures Multi-Timeframe Trading System Configuration
===========================================================================

This configuration file contains all parameters for the autonomous trading system
specialized in cryptocurrency perpetual futures across multiple timeframes.
"""

from typing import Dict, List
from dataclasses import dataclass

# ===== CRYPTO PERPETUAL FUTURES MULTI-TIMEFRAME CONFIGURATION =====

# INSTRUMENTS (CRYPTO USDT PERPETUAL FUTURES ONLY)
INSTRUMENTS = [
    "BTCUSDT.P",      # Bitcoin Perpetual (largest, most liquid)
    "ETHUSDT.P",      # Ethereum Perpetual (2nd largest)
    "BNBUSDT.P",      # Binance Coin Perpetual
    "ADAUSDT.P",      # Cardano Perpetual
    "DOGEUSDT.P",     # Dogecoin Perpetual
    "XRPUSDT.P",      # Ripple Perpetual
    "SOLUSDT.P",      # Solana Perpetual
    "AVAXUSDT.P",     # Avalanche Perpetual
]

# ===== HIGHER TIMEFRAME (HTF) TOP-DOWN ANALYSIS =====
# HTF provides market context and directional bias
HTF_TIMEFRAMES = ["1w", "1d", "4h"]  # Weekly, Daily, 4-Hour for context
HTF_ENABLED = True  # Enable HTF filtering (recommended)

# LTF executes signals aligned with HTF bias
LTF_TIMEFRAMES = ["30m", "15m", "5m"]  # Lower timeframes for execution

# All timeframes (for legacy compatibility)
TIMEFRAMES = ["2m", "5m", "15m", "30m"]  # Will be replaced by LTF_TIMEFRAMES
PRIMARY_TIMEFRAME = "5m"  # Default for discovery
TEST_CROSS_TIMEFRAME_CONFIRMATION = True  # Validate across TFs

# HTF Analysis Settings
HTF_ALIGNMENT_REQUIRED = True  # Only trade signals aligned with HTF
HTF_MIN_ALIGNMENT_SCORE = 50  # Minimum alignment score (0-100)
HTF_MIN_BIAS_STRENGTH = 40  # Minimum bias strength to trade (0-100)
HTF_ALLOW_COUNTERTREND = False  # Allow counter-trend trades in ranging markets

# Timeframe characteristics
TIMEFRAME_CHARACTERISTICS = {
    # Higher Timeframes (HTF) - Context/Bias
    "1w": {
        "duration_minutes": 10080,  # 7 days
        "role": "primary_context",
        "purpose": "Major trend direction, key macro levels"
    },
    "1d": {
        "duration_minutes": 1440,  # 24 hours
        "role": "secondary_context",
        "purpose": "Daily trend, swing levels, momentum"
    },
    "4h": {
        "duration_minutes": 240,
        "role": "tertiary_context",
        "purpose": "Intraday trend, immediate bias"
    },
    # Lower Timeframes (LTF) - Execution
    "30m": {
        "duration_minutes": 30,
        "typical_hold_minutes": (30, 120),
        "target_pips": (50, 200),
        "spread_pips": (1.5, 3.0),
        "slippage_pips": (1.5, 2.0),
        "best_for": "Strong signals only",
        "setup_type": "Major support/resistance, large moves",
        "role": "execution"
    },
    "15m": {
        "duration_minutes": 15,
        "typical_hold_minutes": (15, 45),
        "target_pips": (30, 100),
        "spread_pips": (1.0, 2.0),
        "slippage_pips": (1.0, 1.5),
        "best_for": "Institutional-grade signals",
        "setup_type": "Technical breakdowns, confluence",
        "role": "execution"
    },
    "5m": {
        "duration_minutes": 5,
        "typical_hold_minutes": (5, 15),
        "target_pips": (10, 30),
        "spread_pips": (0.5, 1.0),
        "slippage_pips": (0.5, 1.0),
        "best_for": "Balanced risk/reward",
        "setup_type": "Mean reversion, breakouts",
        "role": "execution"
    },
    # Legacy (will be phased out)
    "2m": {
        "duration_minutes": 2,
        "typical_hold_minutes": (2, 5),
        "target_pips": (5, 20),
        "spread_pips": (0.3, 0.5),
        "slippage_pips": (0.3, 0.7),
        "best_for": "High-frequency, minimal risk",
        "setup_type": "Momentum, microstructure",
        "role": "execution"
    },
}

DATA_LOOKBACK_MONTHS = 6                    # 6 months per timeframe
ASSET_CLASS = "crypto_perpetual_futures"
QUOTE_CURRENCY = "USDT"

# EXCHANGE CONFIGURATION
PRIMARY_EXCHANGE = "Kraken"
BACKUP_EXCHANGE = "Gemini"
EXCLUDE_EXCHANGES = ["Binance.com", "OKX", "Huobi"]

# Exchange-specific configurations
KRAKEN_CONFIG = {
    "api_endpoint": "https://api.kraken.com",
    "futures_endpoint": "https://futures.kraken.com",
    "uptime": 0.998,  # 99.8% verified
    "rate_limit_calls_per_second": 1,
    "rate_limit_calls_per_minute": 15,
}

GEMINI_CONFIG = {
    "api_endpoint": "https://api.gemini.com",
    "uptime": 0.997,
    "rate_limit_calls_per_second": 1,
}

# BACKTESTING THRESHOLDS (Timeframe-Specific) - STRICT for validation
THRESHOLDS = {
    "2m": {
        "min_win_rate": 0.55,
        "min_profit_factor": 2.0,
        "min_sharpe": 1.0,
        "min_trades": 100,  # Need more trades for statistical significance
    },
    "5m": {
        "min_win_rate": 0.55,
        "min_profit_factor": 1.5,
        "min_sharpe": 1.0,
        "min_trades": 80,
    },
    "15m": {
        "min_win_rate": 0.55,
        "min_profit_factor": 1.4,
        "min_sharpe": 1.0,
        "min_trades": 60,
    },
    "30m": {
        "min_win_rate": 0.55,
        "min_profit_factor": 1.3,
        "min_sharpe": 1.0,
        "min_trades": 40,
    }
}

# LIVE TRADING THRESHOLDS (More lenient, R:R based)
# Formula: Required WR = 1 / (1 + R:R) + 5% buffer
# 2:1 R:R needs 33% + 5% = 38% WR
# 3:1 R:R needs 25% + 5% = 30% WR
LIVE_THRESHOLDS = {
    "2m": {
        "min_win_rate": 0.38,  # For 2:1 R:R (33% break-even + 5% buffer)
        "min_profit_factor": 1.2,
        "min_trades": 50,
        "required_rr": 2.0,
    },
    "5m": {
        "min_win_rate": 0.38,  # For 2:1 R:R
        "min_profit_factor": 1.2,
        "min_trades": 40,
        "required_rr": 2.0,
    },
    "15m": {
        "min_win_rate": 0.38,  # For 2:1 R:R
        "min_profit_factor": 1.2,
        "min_trades": 30,
        "required_rr": 2.0,
    },
    "30m": {
        "min_win_rate": 0.30,  # For 3:1 R:R (25% break-even + 5% buffer)
        "min_profit_factor": 1.2,
        "min_trades": 20,
        "required_rr": 3.0,
    }
}

# SLIPPAGE ASSUMPTIONS (Timeframe-Specific, in pips)
SLIPPAGE = {
    "2m": 1.0,    # Total cost: spread + slippage ~1.0 pip
    "5m": 1.5,    # Total cost: ~1.5 pips
    "15m": 2.5,   # Total cost: ~2.5 pips
    "30m": 4.0    # Total cost: ~4.0 pips
}

# MINIMUM RISK:REWARD (Timeframe-Specific)
MIN_RISK_REWARD = {
    "2m": 2.0,     # Very tight, slippage matters more
    "5m": 1.5,     # Standard
    "15m": 1.3,    # Wider targets
    "30m": 1.2     # Largest targets
}

# RISK MANAGEMENT
LEVERAGE = 1.0                              # NO LEVERAGE
POSITION_SIZE_PCT = 0.02                    # 2% per trade
MAX_CONCURRENT = 2                          # Max 2 open positions (all TFs)
DAILY_LOSS_LIMIT = -0.03                    # -3% daily → pause
MAX_DRAWDOWN = 0.15                         # 15% max account
STOP_LOSS_ATR_MULTIPLIER = 1.0             # Stop loss = 1 ATR below entry
VOLUME_FILTER_MIN_DAILY_USD = 100_000_000  # $100M minimum daily volume

# MULTI-TIMEFRAME RULES
USE_HIGHER_TF_CONFIRMATION = True           # Confirm entries on higher TF
TREND_TF = "30m"                            # Macro trend timeframe
CONFIRMATION_TF = "15m"                     # Confirmation timeframe
ENTRY_TF = "5m"                             # Primary entry timeframe
EXECUTION_TF = "2m"                         # Fine-tune execution

# Multi-timeframe weight allocation
TIMEFRAME_WEIGHTS = {
    "trending_market": {
        "30m": 0.40,  # Bigger positions on longer TF
        "15m": 0.30,
        "5m": 0.20,
        "2m": 0.10,
    },
    "choppy_market": {
        "2m": 0.40,   # Favor tighter TF, quicker exits
        "5m": 0.35,
        "15m": 0.20,
        "30m": 0.05,
    },
    "default": {
        "2m": 0.20,
        "5m": 0.35,
        "15m": 0.30,
        "30m": 0.15,
    }
}

# FUNDING RATE CONFIGURATION
FUNDING_RATE_THRESHOLDS = {
    "neutral": 0.0001,           # Below this = neutral
    "moderate_bullish": 0.0002,  # Above this = caution on longs
    "extreme_bullish": 0.0003,   # Above this = expect reversal (70%+ probability)
    "extreme_bearish": -0.0002,  # Below this = expect bounce
}

FUNDING_RATE_INTERVAL_HOURS = 8  # Paid every 8 hours

# LIVE TRADING
PAPER_TRADING_DAYS = 14
PERFORMANCE_REVIEW_PERIOD = 50      # Trades
ACCURACY_VARIANCE_THRESHOLD = 0.05  # 5% drift from backtest
STARTING_CAPITAL_USDT = 1000
INITIAL_CAPITAL = STARTING_CAPITAL_USDT  # Alias for consistency

# SIGNAL FILTERING
EDGE_SIGNALS_TO_DEPLOY = 5          # Top signals per TF
KILL_BELOW_WIN_RATE = 0.50
KILL_BELOW_PROFIT_FACTOR = 1.2
ROLLING_WINDOW_TRADES = 50          # For performance evaluation

# SIGNAL DISCOVERY
MIN_HYPOTHESES_PER_TIMEFRAME = 8    # Generate at least 8 hypotheses per TF
MAX_HYPOTHESES_PER_TIMEFRAME = 12   # Maximum 12 per TF
TOTAL_HYPOTHESES_TARGET = 40        # Target 40 total hypotheses

# INDICATOR PARAMETERS (Default ranges for optimization)
INDICATOR_RANGES = {
    "RSI": {
        "period": [7, 14, 21],
        "oversold": [20, 25, 30, 35],
        "overbought": [65, 70, 75, 80],
    },
    "MACD": {
        "fast": [8, 12, 16],
        "slow": [21, 26, 30],
        "signal": [7, 9, 11],
    },
    "SMA": {
        "periods": [10, 20, 50, 100, 200],
    },
    "EMA": {
        "periods": [9, 12, 20, 26, 50, 100],
    },
    "ATR": {
        "period": [10, 14, 20],
    },
    "Bollinger_Bands": {
        "period": [15, 20, 25],
        "std_dev": [1.5, 2.0, 2.5],
    },
    "Stochastic": {
        "k_period": [8, 14, 21],
        "d_period": [3, 5, 7],
    }
}

# MARKET REGIME DETECTION
REGIME_TYPES = ["trending", "ranging", "choppy", "volatile"]
REGIME_DETECTION_LOOKBACK = 100  # Candles to analyze for regime

# ADX thresholds for regime detection
ADX_TRENDING_THRESHOLD = 25
ADX_STRONG_TREND_THRESHOLD = 40

# ATR thresholds for volatility (as % of price)
ATR_LOW_VOLATILITY = 0.01    # <1% = low volatility
ATR_HIGH_VOLATILITY = 0.03   # >3% = high volatility

# LOGGING
LOG_LEVEL = "INFO"
LOG_FILE = "logs/trading_system.log"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# DATA PATHS
DATA_DIR = "data"
RAW_DATA_DIR = f"{DATA_DIR}/raw"
PROCESSED_DATA_DIR = f"{DATA_DIR}/processed"
FUNDING_RATES_DIR = f"{DATA_DIR}/funding_rates"

# RESULTS PATHS
RESULTS_DIR = "results"
DISCOVERY_DIR = f"{RESULTS_DIR}/discovery"
BACKTESTS_DIR = f"{RESULTS_DIR}/backtests"
ANALYSIS_DIR = f"{RESULTS_DIR}/analysis"
TRADES_DIR = f"{RESULTS_DIR}/trades"

# API KEYS (Load from environment variables)
import os
KRAKEN_API_KEY = os.getenv("KRAKEN_API_KEY", "")
KRAKEN_API_SECRET = os.getenv("KRAKEN_API_SECRET", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_API_SECRET = os.getenv("GEMINI_API_SECRET", "")


@dataclass
class TimeframeConfig:
    """Configuration for a specific timeframe"""
    name: str
    duration_minutes: int
    typical_hold_range: tuple
    target_pips_range: tuple
    spread_range: tuple
    slippage_range: tuple
    min_win_rate: float
    min_profit_factor: float
    min_sharpe: float
    total_cost_pips: float
    min_risk_reward: float


def get_timeframe_config(tf: str) -> TimeframeConfig:
    """Get configuration for a specific timeframe"""
    char = TIMEFRAME_CHARACTERISTICS[tf]
    thresh = THRESHOLDS[tf]

    return TimeframeConfig(
        name=tf,
        duration_minutes=char["duration_minutes"],
        typical_hold_range=char["typical_hold_minutes"],
        target_pips_range=char["target_pips"],
        spread_range=char["spread_pips"],
        slippage_range=char["slippage_pips"],
        min_win_rate=thresh["min_win_rate"],
        min_profit_factor=thresh["min_profit_factor"],
        min_sharpe=thresh["min_sharpe"],
        total_cost_pips=SLIPPAGE[tf],
        min_risk_reward=MIN_RISK_REWARD[tf],
    )


# Prior agent knowledge (pre-loaded findings)
PRIOR_AGENT_KNOWLEDGE = {
    "crypto_market_analyzer": {
        "exchange_access": {
            "us_accessible_perpetual_exchanges": ["Kraken", "Gemini", "Coinbase"],
            "us_restricted_perpetual": ["Binance", "Binance.US (limited)", "OKX"],
            "best_for_scalping_perpetuals": ["Kraken", "Gemini"],
            "best_for_api_reliability": ["Kraken", "Coinbase"],
            "multi_timeframe_support": ["Kraken", "Gemini", "Coinbase"],
            "last_verified": "2025-01-20"
        },
        "perpetual_specific": {
            "funding_rates_matter": True,
            "funding_rate_lag": "8 hours",
            "funding_rate_extreme_threshold": 0.0003,
            "funding_extreme_predicts_reversal": 0.80
        }
    }
}


# System settings
SYSTEM_MODE = "discovery"  # discovery, backtest, paper_trade, live_trade
ENABLE_LOGGING = True
ENABLE_PERFORMANCE_TRACKING = True
ENABLE_CROSS_TIMEFRAME_VALIDATION = True


if __name__ == "__main__":
    # Print configuration summary
    print("=== Crypto Perpetual Futures Multi-Timeframe Trading System Configuration ===")
    print(f"\nInstruments: {', '.join(INSTRUMENTS)}")
    print(f"Timeframes: {', '.join(TIMEFRAMES)}")
    print(f"Primary Exchange: {PRIMARY_EXCHANGE}")
    print(f"Data Lookback: {DATA_LOOKBACK_MONTHS} months")
    print(f"\nRisk Management:")
    print(f"  - Leverage: {LEVERAGE}x")
    print(f"  - Position Size: {POSITION_SIZE_PCT * 100}%")
    print(f"  - Max Concurrent: {MAX_CONCURRENT}")
    print(f"  - Daily Loss Limit: {DAILY_LOSS_LIMIT * 100}%")
    print(f"  - Max Drawdown: {MAX_DRAWDOWN * 100}%")
    print(f"\nTimeframe Thresholds:")
    for tf in TIMEFRAMES:
        thresh = THRESHOLDS[tf]
        print(f"  {tf}: Win Rate >{thresh['min_win_rate']*100}%, PF >{thresh['min_profit_factor']}x")
