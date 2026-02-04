"""
Exchange adapter factory for creating exchange instances.
"""

from typing import Optional
import logging

from .base import BaseExchange
from .bybit import BybitExchange
from .coinbase import CoinbaseExchange
from .binance import BinanceExchange
from ..config import ExchangeConfig


class ExchangeFactory:
    """
    Factory for creating exchange adapter instances.

    Allows easy swapping between different exchanges without changing agent code.
    """

    @staticmethod
    def create(config: ExchangeConfig) -> BaseExchange:
        """
        Create an exchange adapter instance.

        Args:
            config: Exchange configuration

        Returns:
            Exchange adapter instance

        Raises:
            ValueError: If exchange name is not supported
        """
        logger = logging.getLogger("ExchangeFactory")

        exchange_name = config.name.lower()

        if exchange_name == "bybit":
            logger.warning(
                "Bybit restricts US users. For production use in the US, "
                "consider using Coinbase, Kraken, Gemini, or Binance.US instead. "
                "To swap exchanges, simply change the 'name' in ExchangeConfig "
                "and implement the corresponding adapter."
            )

            return BybitExchange(
                api_key=config.api_key,
                api_secret=config.api_secret,
                testnet=config.testnet,
                rate_limit=config.rate_limit_per_second,
                max_retries=config.max_retries,
                timeout=config.timeout
            )

        elif exchange_name == "coinbase":
            logger.info(
                "Using Coinbase Advanced Trade - fully US-compliant and regulated."
            )

            return CoinbaseExchange(
                api_key=config.api_key,
                api_secret=config.api_secret,
                rate_limit=config.rate_limit_per_second,
                max_retries=config.max_retries,
                timeout=config.timeout
            )

        elif exchange_name == "binance":
            use_futures = config.testnet  # Use config.testnet to indicate futures
            logger.info(
                f"Using Binance {'Futures' if use_futures else 'Spot'} API - "
                "Globally accessible"
            )

            return BinanceExchange(
                api_key=config.api_key,
                api_secret=config.api_secret,
                use_futures=use_futures,
                testnet=False,  # Testnet handled separately
                rate_limit=config.rate_limit_per_second,
                max_retries=config.max_retries,
                timeout=config.timeout
            )

        # Add more exchanges here as needed:
        # elif exchange_name == "kraken":
        #     return KrakenExchange(...)
        # elif exchange_name == "gemini":
        #     return GeminiExchange(...)

        else:
            raise ValueError(
                f"Unsupported exchange: {config.name}. "
                f"To add support, implement a new adapter inheriting from BaseExchange."
            )


def get_us_compliant_exchanges() -> dict:
    """
    Get information about US-compliant exchanges.

    Returns:
        Dictionary of exchange information
    """
    return {
        "binance": {
            "name": "Binance Futures",
            "api_docs": "https://binance-docs.github.io/apidocs/futures/en/",
            "us_accessible": True,
            "notes": "Futures API globally accessible (check local regulations)"
        },
        "coinbase": {
            "name": "Coinbase Advanced Trade",
            "api_docs": "https://docs.cloud.coinbase.com/advanced-trade-api/docs",
            "us_accessible": True,
            "notes": "Fully US-compliant, regulated by FinCEN"
        },
        "kraken": {
            "name": "Kraken",
            "api_docs": "https://docs.kraken.com/rest/",
            "us_accessible": True,
            "notes": "Available in most US states"
        },
        "gemini": {
            "name": "Gemini",
            "api_docs": "https://docs.gemini.com/rest-api/",
            "us_accessible": True,
            "notes": "New York-based, fully regulated"
        },
        "binance_us": {
            "name": "Binance.US",
            "api_docs": "https://docs.binance.us/",
            "us_accessible": True,
            "notes": "US-specific version of Binance"
        }
    }
