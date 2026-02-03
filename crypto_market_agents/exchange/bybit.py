"""
Bybit exchange adapter implementation.

IMPORTANT: Bybit has restricted US users from trading on their platform.
While some market data endpoints may be accessible, users should verify
compliance with local regulations and Bybit's terms of service.

This adapter is designed to be easily swappable with other US-compliant exchanges.
"""

import aiohttp
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import logging

from .base import BaseExchange
from ..utils.rate_limiter import RateLimiter, ExponentialBackoff


class BybitExchange(BaseExchange):
    """
    Bybit exchange adapter.

    Uses public API endpoints for market data.
    Note: Bybit restricts US users - this adapter should be swapped
    for a US-compliant exchange in production.
    """

    BASE_URL = "https://api.bybit.com"

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: bool = False,
        rate_limit: int = 10,
        max_retries: int = 3,
        timeout: int = 30
    ):
        """
        Initialize Bybit adapter.

        Args:
            api_key: API key (optional for public endpoints)
            api_secret: API secret (optional for public endpoints)
            testnet: Use testnet
            rate_limit: Max requests per second
            max_retries: Maximum retry attempts
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.timeout = timeout
        self.max_retries = max_retries

        if testnet:
            self.BASE_URL = "https://api-testnet.bybit.com"

        self.session: Optional[aiohttp.ClientSession] = None
        self.rate_limiter = RateLimiter(max_requests=rate_limit, time_window=1.0)
        self.logger = logging.getLogger(self.__class__.__name__)

    async def connect(self) -> bool:
        """Establish connection to Bybit."""
        try:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )

            # Test connection
            result = await self._request("GET", "/v5/market/time")
            if result:
                self.logger.info("Successfully connected to Bybit")
                return True
            return False

        except Exception as e:
            self.logger.error(f"Failed to connect to Bybit: {e}")
            return False

    async def disconnect(self):
        """Close connection to Bybit."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.logger.info("Disconnected from Bybit")

    def check_us_accessibility(self) -> Dict[str, Any]:
        """
        Check US accessibility for Bybit.

        Returns:
            Accessibility information
        """
        return {
            "accessible": False,
            "restrictions": [
                "Bybit restricts US users from trading",
                "Some market data endpoints may be accessible",
                "Check Bybit's terms of service and local regulations"
            ],
            "notes": (
                "Consider using US-compliant exchanges like Coinbase, Kraken, "
                "Gemini, or Binance.US. This adapter can be swapped via the "
                "exchange factory."
            ),
            "recommended_alternatives": [
                "Coinbase Advanced Trade",
                "Kraken",
                "Gemini",
                "Binance.US"
            ]
        }

    async def get_trading_symbols(self, quote_currency: str = "USDT") -> List[str]:
        """Get list of trading symbols."""
        try:
            result = await self._request(
                "GET",
                "/v5/market/instruments-info",
                params={"category": "spot"}
            )

            if not result or "result" not in result:
                return []

            symbols = []
            for item in result["result"].get("list", []):
                symbol = item.get("symbol", "")
                if symbol.endswith(quote_currency) and item.get("status") == "Trading":
                    symbols.append(symbol)

            self.logger.info(f"Found {len(symbols)} trading symbols")
            return sorted(symbols)

        except Exception as e:
            self.logger.error(f"Failed to get trading symbols: {e}")
            return []

    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get ticker data for a symbol."""
        try:
            result = await self._request(
                "GET",
                "/v5/market/tickers",
                params={"category": "spot", "symbol": symbol}
            )

            if not result or "result" not in result:
                return {}

            items = result["result"].get("list", [])
            if not items:
                return {}

            data = items[0]
            return {
                "symbol": data.get("symbol"),
                "last_price": float(data.get("lastPrice", 0)),
                "bid": float(data.get("bid1Price", 0)),
                "ask": float(data.get("ask1Price", 0)),
                "volume_24h": float(data.get("volume24h", 0)),
                "price_change_24h_pct": float(data.get("price24hPcnt", 0)) * 100,
                "high_24h": float(data.get("highPrice24h", 0)),
                "low_24h": float(data.get("lowPrice24h", 0)),
                "timestamp": datetime.now(timezone.utc)
            }

        except Exception as e:
            self.logger.error(f"Failed to get ticker for {symbol}: {e}")
            return {}

    async def get_tickers(self, symbols: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Get ticker data for multiple symbols."""
        try:
            result = await self._request(
                "GET",
                "/v5/market/tickers",
                params={"category": "spot"}
            )

            if not result or "result" not in result:
                return []

            tickers = []
            for data in result["result"].get("list", []):
                symbol = data.get("symbol")

                if symbols and symbol not in symbols:
                    continue

                try:
                    ticker = {
                        "symbol": symbol,
                        "last_price": float(data.get("lastPrice", 0)),
                        "bid": float(data.get("bid1Price", 0)),
                        "ask": float(data.get("ask1Price", 0)),
                        "volume_24h": float(data.get("volume24h", 0)),
                        "price_change_24h_pct": float(data.get("price24hPcnt", 0)) * 100,
                        "high_24h": float(data.get("highPrice24h", 0)),
                        "low_24h": float(data.get("lowPrice24h", 0)),
                        "timestamp": datetime.now(timezone.utc)
                    }
                    tickers.append(ticker)
                except (ValueError, TypeError) as e:
                    self.logger.warning(f"Invalid data for {symbol}: {e}")
                    continue

            return tickers

        except Exception as e:
            self.logger.error(f"Failed to get tickers: {e}")
            return []

    async def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 100,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Get candlestick data."""
        try:
            params = {
                "category": "spot",
                "symbol": symbol,
                "interval": interval,
                "limit": min(limit, 1000)
            }

            if start_time:
                params["start"] = int(start_time.timestamp() * 1000)
            if end_time:
                params["end"] = int(end_time.timestamp() * 1000)

            result = await self._request("GET", "/v5/market/kline", params=params)

            if not result or "result" not in result:
                return []

            klines = []
            for item in result["result"].get("list", []):
                klines.append({
                    "timestamp": datetime.fromtimestamp(int(item[0]) / 1000, tz=timezone.utc),
                    "open": float(item[1]),
                    "high": float(item[2]),
                    "low": float(item[3]),
                    "close": float(item[4]),
                    "volume": float(item[5])
                })

            # Bybit returns newest first, reverse to get chronological order
            return list(reversed(klines))

        except Exception as e:
            self.logger.error(f"Failed to get klines for {symbol}: {e}")
            return []

    async def get_order_book(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        """Get order book depth."""
        try:
            result = await self._request(
                "GET",
                "/v5/market/orderbook",
                params={"category": "spot", "symbol": symbol, "limit": limit}
            )

            if not result or "result" not in result:
                return {}

            data = result["result"]
            return {
                "bids": [[float(p), float(q)] for p, q in data.get("b", [])],
                "asks": [[float(p), float(q)] for p, q in data.get("a", [])],
                "timestamp": datetime.now(timezone.utc)
            }

        except Exception as e:
            self.logger.error(f"Failed to get order book for {symbol}: {e}")
            return {}

    async def get_recent_trades(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent trades."""
        try:
            result = await self._request(
                "GET",
                "/v5/market/recent-trade",
                params={"category": "spot", "symbol": symbol, "limit": min(limit, 1000)}
            )

            if not result or "result" not in result:
                return []

            trades = []
            for item in result["result"].get("list", []):
                trades.append({
                    "price": float(item.get("price", 0)),
                    "quantity": float(item.get("size", 0)),
                    "timestamp": datetime.fromtimestamp(
                        int(item.get("time", 0)) / 1000, tz=timezone.utc
                    ),
                    "side": item.get("side", "").lower()
                })

            return trades

        except Exception as e:
            self.logger.error(f"Failed to get recent trades for {symbol}: {e}")
            return []

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Make HTTP request to Bybit API.

        Args:
            method: HTTP method
            endpoint: API endpoint
            params: Query parameters

        Returns:
            Response data or None
        """
        if not self.session:
            raise RuntimeError("Session not initialized. Call connect() first.")

        url = f"{self.BASE_URL}{endpoint}"
        backoff = ExponentialBackoff(initial_delay=1.0, max_delay=30.0)

        for attempt in range(self.max_retries):
            try:
                await self.rate_limiter.acquire()

                async with self.session.request(method, url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("retCode") == 0:
                            backoff.reset()
                            return data
                        else:
                            self.logger.warning(
                                f"API error: {data.get('retMsg', 'Unknown error')}"
                            )
                            return None

                    elif response.status == 429:
                        self.logger.warning("Rate limit exceeded, backing off...")
                        await backoff.wait()

                    elif response.status in [403, 451]:
                        self.logger.error(
                            f"Access forbidden (status {response.status}). "
                            "This may indicate geo-restrictions for US users."
                        )
                        return None

                    else:
                        self.logger.warning(
                            f"HTTP {response.status}: {await response.text()}"
                        )
                        if attempt < self.max_retries - 1:
                            await backoff.wait()

            except asyncio.TimeoutError:
                self.logger.warning(f"Request timeout (attempt {attempt + 1})")
                if attempt < self.max_retries - 1:
                    await backoff.wait()

            except Exception as e:
                self.logger.error(f"Request failed: {e}")
                if attempt < self.max_retries - 1:
                    await backoff.wait()

        return None
