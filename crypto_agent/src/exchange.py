"""
Binance Futures exchange connection management.

Three modes:
- Public (default): Live mainnet, no API keys — for OHLCV, tickers, funding rates.
- Paper trading: Live mainnet for data, all orders mocked locally — no testnet needed.
- Live trading: Live mainnet with real API keys — only when LIVE_TRADING_ENABLED=true.

Testnet is NOT used. Paper mode gets real market data from mainnet and
simulates all order execution locally through the paper_mode module.
"""

import ccxt

from src.config import (
    BINANCE_API_KEY,
    BINANCE_SECRET,
    EXCHANGE_TIMEOUT,
    LIVE_TRADING_ENABLED,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_exchange(private: bool = False) -> ccxt.binanceusdm:
    """
    Create a Binance USDT-M Futures exchange instance.

    Always uses live mainnet. Paper trading does not need private keys
    because all orders are simulated locally.

    Args:
        private: If True, include API keys for authenticated endpoints.
                 Only needed for LIVE trading (placing real orders).
                 Paper trading mocks everything — no keys required.

    Returns:
        Configured ccxt.binanceusdm instance.
    """
    config: dict = {
        "options": {"defaultType": "future"},
        "enableRateLimit": True,
        "timeout": EXCHANGE_TIMEOUT,
    }

    if private and LIVE_TRADING_ENABLED:
        if not BINANCE_API_KEY or not BINANCE_SECRET:
            raise ValueError(
                "BINANCE_API_KEY and BINANCE_SECRET must be set for live trading"
            )
        config["apiKey"] = BINANCE_API_KEY
        config["secret"] = BINANCE_SECRET
        exchange = ccxt.binanceusdm(config)
        logger.warning("LIVE TRADING MODE — using real API keys on mainnet")
    else:
        # Public data or paper mode: live mainnet, no keys
        exchange = ccxt.binanceusdm(config)
        if private:
            logger.info("Connected to Binance Futures mainnet (PAPER mode — orders mocked locally)")
        else:
            logger.info("Connected to Binance Futures mainnet (public data)")

    return exchange


def get_public_exchange() -> ccxt.binanceusdm:
    """
    Convenience function: get a public-only exchange instance.
    Uses live mainnet — no API keys needed.
    For fetching OHLCV, tickers, funding rates.
    """
    return get_exchange(private=False)
