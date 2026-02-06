"""
Binance Futures exchange connection management.
Supports testnet (default) and live mode.

Public data (OHLCV, tickers, funding rates) always uses the live API
since it requires no authentication. Private endpoints (orders, balance)
use testnet by default, live only when LIVE_TRADING_ENABLED=true.
"""

import ccxt

from src.config import (
    BINANCE_API_KEY,
    BINANCE_SECRET,
    BINANCE_TESTNET_API_KEY,
    BINANCE_TESTNET_SECRET,
    EXCHANGE_TIMEOUT,
    LIVE_TRADING_ENABLED,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_exchange(private: bool = False) -> ccxt.binanceusdm:
    """
    Create a Binance USDT-M Futures exchange instance.

    Args:
        private: If True, include API keys for authenticated endpoints
                 (orders, balance). Uses testnet unless LIVE_TRADING_ENABLED.
                 If False, public-only — always uses live API (no keys needed).

    Returns:
        Configured ccxt.binanceusdm instance.
    """
    config: dict = {
        "options": {"defaultType": "future"},
        "enableRateLimit": True,
        "timeout": EXCHANGE_TIMEOUT,
    }

    if private:
        if LIVE_TRADING_ENABLED:
            if not BINANCE_API_KEY or not BINANCE_SECRET:
                raise ValueError(
                    "BINANCE_API_KEY and BINANCE_SECRET must be set for live trading"
                )
            config["apiKey"] = BINANCE_API_KEY
            config["secret"] = BINANCE_SECRET
            exchange = ccxt.binanceusdm(config)
            logger.warning("LIVE TRADING MODE — using real API keys")
        else:
            if not BINANCE_TESTNET_API_KEY or not BINANCE_TESTNET_SECRET:
                raise ValueError(
                    "BINANCE_TESTNET_API_KEY and BINANCE_TESTNET_SECRET must be set. "
                    "Get keys at https://testnet.binancefuture.com"
                )
            config["apiKey"] = BINANCE_TESTNET_API_KEY
            config["secret"] = BINANCE_TESTNET_SECRET
            exchange = ccxt.binanceusdm(config)
            exchange.set_sandbox_mode(True)
            logger.info("Connected to Binance Futures TESTNET (private)")
    else:
        # Public data: always use live API (no keys needed, no testnet)
        exchange = ccxt.binanceusdm(config)
        logger.info("Connected to Binance Futures (public data)")

    return exchange


def get_public_exchange() -> ccxt.binanceusdm:
    """
    Convenience function: get a public-only exchange instance.
    Always uses live API — no API keys needed.
    For fetching OHLCV, tickers, funding rates.
    """
    return get_exchange(private=False)
