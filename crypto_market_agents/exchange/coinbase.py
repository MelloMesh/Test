"""
Coinbase Advanced Trade API adapter implementation.

Coinbase is fully US-compliant and regulated by FinCEN.
Public API endpoints are available without authentication.

API Documentation: https://docs.cloud.coinbase.com/advanced-trade-api/docs
"""

import aiohttp
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import logging

from .base import BaseExchange
from ..utils.rate_limiter import RateLimiter, ExponentialBackoff


class CoinbaseExchange(BaseExchange):
    """
    Coinbase Advanced Trade API adapter.

    Fully US-compliant exchange with public market data endpoints.
    """

    # Use Coinbase Exchange (Pro) API for public market data
    BASE_URL = "https://api.exchange.coinbase.com"

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
        Initialize Coinbase adapter.

        Args:
            api_key: API key (optional for public endpoints)
            api_secret: API secret (optional for public endpoints)
            testnet: Not applicable for Coinbase
            rate_limit: Max requests per second
            max_retries: Maximum retry attempts
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.timeout = timeout
        self.max_retries = max_retries

        self.session: Optional[aiohttp.ClientSession] = None
        self.rate_limiter = RateLimiter(max_requests=rate_limit, time_window=1.0)
        self.logger = logging.getLogger(self.__class__.__name__)

    async def connect(self) -> bool:
        """Establish connection to Coinbase."""
        try:
            headers = {
                'Content-Type': 'application/json',
                'User-Agent': 'CryptoMarketAgents/1.0'
            }

            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                headers=headers
            )

            # Test connection with products endpoint
            result = await self._request("GET", "/products")
            if result and isinstance(result, list):
                self.logger.info("Successfully connected to Coinbase Exchange API")
                return True
            return False

        except Exception as e:
            self.logger.error(f"Failed to connect to Coinbase: {e}")
            return False

    async def disconnect(self):
        """Close connection to Coinbase."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.logger.info("Disconnected from Coinbase")

    def check_us_accessibility(self) -> Dict[str, Any]:
        """
        Check US accessibility for Coinbase.

        Returns:
            Accessibility information
        """
        return {
            "accessible": True,
            "restrictions": [],
            "notes": (
                "Coinbase is fully US-compliant and regulated by FinCEN. "
                "Available in all 50 US states. Public market data is freely accessible."
            ),
            "recommended_alternatives": []
        }

    async def get_trading_symbols(self, quote_currency: str = "USDT") -> List[str]:
        """
        Get list of trading symbols.

        Note: Coinbase uses USD, USDC, and other quote currencies.
        For crypto-to-crypto pairs, we'll look for matches.
        """
        try:
            result = await self._request("GET", "/products")

            if not result or not isinstance(result, list):
                return []

            symbols = []

            # Coinbase uses different quote currencies
            quote_variants = ["USD", "USDT", "USDC"] if quote_currency == "USDT" else [quote_currency]

            for product in result:
                product_id = product.get('id', '')
                # Status check - only trading products
                status = product.get('status', '')
                trading_disabled = product.get('trading_disabled', False)

                if status == 'online' and not trading_disabled:
                    # Check if it ends with any of our quote variants
                    for quote in quote_variants:
                        if product_id.endswith(f"-{quote}"):
                            # Convert to unified format (e.g., BTC-USD -> BTCUSD)
                            symbol = product_id.replace('-', '')
                            symbols.append(symbol)
                            break

            self.logger.info(f"Found {len(symbols)} trading symbols on Coinbase")
            return sorted(symbols)

        except Exception as e:
            self.logger.error(f"Failed to get trading symbols: {e}")
            return []

    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Get ticker data for a symbol.

        Args:
            symbol: Trading symbol (e.g., "BTCUSD")
        """
        try:
            # Convert symbol format (BTCUSD -> BTC-USD)
            product_id = self._format_symbol(symbol)

            # Get product ticker
            ticker = await self._request("GET", f"/products/{product_id}/ticker")

            if not ticker:
                return {}

            # Get 24h stats
            stats = await self._request("GET", f"/products/{product_id}/stats")

            last_price = float(ticker.get('price', 0))
            bid = float(ticker.get('bid', last_price * 0.999))
            ask = float(ticker.get('ask', last_price * 1.001))

            # Extract stats
            volume_24h = float(stats.get('volume', 0)) if stats else 0
            high_24h = float(stats.get('high', last_price)) if stats else last_price
            low_24h = float(stats.get('low', last_price)) if stats else last_price
            open_24h = float(stats.get('open', last_price)) if stats else last_price

            # Calculate 24h change
            price_change_24h_pct = 0
            if open_24h > 0:
                price_change_24h_pct = ((last_price - open_24h) / open_24h) * 100

            return {
                "symbol": symbol,
                "last_price": last_price,
                "bid": bid,
                "ask": ask,
                "volume_24h": volume_24h,
                "price_change_24h_pct": price_change_24h_pct,
                "high_24h": high_24h,
                "low_24h": low_24h,
                "timestamp": datetime.now(timezone.utc)
            }

        except Exception as e:
            self.logger.error(f"Failed to get ticker for {symbol}: {e}")
            return {}

    async def get_tickers(self, symbols: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Get ticker data for multiple symbols."""
        try:
            # Get all products
            products = await self._request("GET", "/products")

            if not products or not isinstance(products, list):
                return []

            tickers = []

            # Process each product
            for product in products:
                product_id = product.get('id', '')
                symbol = product_id.replace('-', '')

                # Filter by symbols if provided
                if symbols and symbol not in symbols:
                    continue

                # Only online products
                if product.get('status') != 'online' or product.get('trading_disabled', False):
                    continue

                try:
                    # Get ticker for this product
                    ticker = await self.get_ticker(symbol)
                    if ticker:
                        tickers.append(ticker)

                    # Rate limit - small delay between requests
                    await asyncio.sleep(0.15)

                except Exception as e:
                    self.logger.warning(f"Failed to get ticker for {symbol}: {e}")
                    continue

                # Limit to avoid too many requests
                if len(tickers) >= 50:
                    break

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
        """
        Get candlestick data.

        Args:
            symbol: Trading symbol
            interval: Timeframe in minutes (e.g., "1", "5", "15", "60")
            limit: Number of candles
            start_time: Start time
            end_time: End time
        """
        try:
            product_id = self._format_symbol(symbol)

            # Convert interval to Coinbase granularity (in seconds)
            granularity_map = {
                "1": 60,       # 1 minute
                "5": 300,      # 5 minutes
                "15": 900,     # 15 minutes
                "30": 1800,    # 30 minutes
                "60": 3600,    # 1 hour
                "360": 21600,  # 6 hours
                "1440": 86400  # 1 day
            }

            granularity = granularity_map.get(str(interval), 300)

            # Build params
            params = {
                "granularity": granularity
            }

            if start_time:
                params["start"] = start_time.isoformat()
            if end_time:
                params["end"] = end_time.isoformat()

            result = await self._request(
                "GET",
                f"/products/{product_id}/candles",
                params=params
            )

            if not result or not isinstance(result, list):
                return []

            klines = []
            # Coinbase returns: [timestamp, low, high, open, close, volume]
            for candle in result[:limit]:
                klines.append({
                    "timestamp": datetime.fromtimestamp(int(candle[0]), tz=timezone.utc),
                    "open": float(candle[3]),
                    "high": float(candle[2]),
                    "low": float(candle[1]),
                    "close": float(candle[4]),
                    "volume": float(candle[5])
                })

            # Sort chronologically
            klines.sort(key=lambda x: x['timestamp'])

            return klines

        except Exception as e:
            self.logger.error(f"Failed to get klines for {symbol}: {e}")
            return []

    async def get_order_book(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        """Get order book depth."""
        try:
            product_id = self._format_symbol(symbol)

            # Coinbase order book levels: 1, 2, or 3
            # Level 2 gives us top 50 bids and asks
            result = await self._request(
                "GET",
                f"/products/{product_id}/book",
                params={"level": 2}
            )

            if not result:
                return {}

            return {
                "bids": [[float(b[0]), float(b[1])] for b in result.get('bids', [])[:limit]],
                "asks": [[float(a[0]), float(a[1])] for a in result.get('asks', [])[:limit]],
                "timestamp": datetime.now(timezone.utc)
            }

        except Exception as e:
            self.logger.error(f"Failed to get order book for {symbol}: {e}")
            return {}

    async def get_recent_trades(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent trades."""
        try:
            product_id = self._format_symbol(symbol)

            result = await self._request(
                "GET",
                f"/products/{product_id}/trades"
            )

            if not result or not isinstance(result, list):
                return []

            trades = []
            for trade in result[:limit]:
                trades.append({
                    "price": float(trade.get('price', 0)),
                    "quantity": float(trade.get('size', 0)),
                    "timestamp": datetime.fromisoformat(trade['time'].replace('Z', '+00:00')),
                    "side": trade.get('side', 'unknown').lower()
                })

            return trades

        except Exception as e:
            self.logger.error(f"Failed to get recent trades for {symbol}: {e}")
            return []

    def _format_symbol(self, symbol: str) -> str:
        """
        Convert symbol to Coinbase product ID format.

        Examples:
            BTCUSD -> BTC-USD
            ETHUSD -> ETH-USD
            BTCUSDT -> BTC-USDT (if exists, else BTC-USD)
        """
        # Common mappings
        if symbol.endswith("USDT"):
            base = symbol[:-4]
            # Try USDT first, fall back to USD
            return f"{base}-USDT"
        elif symbol.endswith("USDC"):
            base = symbol[:-4]
            return f"{base}-USDC"
        elif symbol.endswith("USD"):
            base = symbol[:-3]
            return f"{base}-USD"
        else:
            # Assume last 3-4 chars are quote currency
            if len(symbol) > 6:
                base = symbol[:-4]
                quote = symbol[-4:]
            else:
                base = symbol[:-3]
                quote = symbol[-3:]
            return f"{base}-{quote}"

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Make HTTP request to Coinbase API.

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
                        backoff.reset()
                        return data

                    elif response.status == 429:
                        self.logger.warning("Rate limit exceeded, backing off...")
                        await backoff.wait()

                    else:
                        text = await response.text()
                        self.logger.warning(f"HTTP {response.status}: {text}")
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
