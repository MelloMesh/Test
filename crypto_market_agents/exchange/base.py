"""
Abstract base interface for exchange adapters.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime


class BaseExchange(ABC):
    """
    Abstract base class for exchange adapters.

    This interface allows easy swapping between different exchanges
    without changing agent logic.
    """

    @abstractmethod
    async def connect(self) -> bool:
        """
        Establish connection to the exchange.

        Returns:
            True if connection successful, False otherwise
        """
        pass

    @abstractmethod
    async def disconnect(self):
        """Close connection to the exchange."""
        pass

    @abstractmethod
    async def get_trading_symbols(self, quote_currency: str = "USDT") -> List[str]:
        """
        Get list of available trading symbols.

        Args:
            quote_currency: Quote currency filter (e.g., "USDT")

        Returns:
            List of trading symbol strings
        """
        pass

    @abstractmethod
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Get current ticker data for a symbol.

        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")

        Returns:
            Dictionary containing:
                - symbol: str
                - last_price: float
                - bid: float
                - ask: float
                - volume_24h: float
                - price_change_24h_pct: float
                - high_24h: float
                - low_24h: float
                - timestamp: datetime
        """
        pass

    @abstractmethod
    async def get_tickers(self, symbols: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Get ticker data for multiple symbols.

        Args:
            symbols: List of symbols (None for all symbols)

        Returns:
            List of ticker dictionaries
        """
        pass

    @abstractmethod
    async def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 100,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Get candlestick/kline data.

        Args:
            symbol: Trading symbol
            interval: Timeframe (e.g., "1", "5", "15", "60" for minutes)
            limit: Number of candles to retrieve
            start_time: Start time (optional)
            end_time: End time (optional)

        Returns:
            List of candle dictionaries containing:
                - timestamp: datetime
                - open: float
                - high: float
                - low: float
                - close: float
                - volume: float
        """
        pass

    @abstractmethod
    async def get_order_book(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        """
        Get order book depth.

        Args:
            symbol: Trading symbol
            limit: Depth limit

        Returns:
            Dictionary containing:
                - bids: List[List[float, float]] (price, quantity)
                - asks: List[List[float, float]] (price, quantity)
                - timestamp: datetime
        """
        pass

    @abstractmethod
    async def get_recent_trades(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent trades.

        Args:
            symbol: Trading symbol
            limit: Number of trades

        Returns:
            List of trade dictionaries
        """
        pass

    @abstractmethod
    def check_us_accessibility(self) -> Dict[str, Any]:
        """
        Check if the exchange is accessible from US IPs.

        Returns:
            Dictionary containing:
                - accessible: bool
                - restrictions: List[str]
                - notes: str
        """
        pass
