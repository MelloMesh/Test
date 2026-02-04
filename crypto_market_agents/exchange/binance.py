"""
Binance exchange adapter implementation.

Supports both Spot and Futures markets. Futures API bypasses geo-restrictions
that affect some regions.

API Documentation:
- Spot: https://binance-docs.github.io/apidocs/spot/en/
- Futures: https://binance-docs.github.io/apidocs/futures/en/
"""

import aiohttp
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import logging

from .base import BaseExchange
from ..utils.rate_limiter import RateLimiter, ExponentialBackoff


class BinanceExchange(BaseExchange):
    """
    Binance exchange adapter for both Spot and Futures markets.

    Futures API is accessible globally without geo-restrictions.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        use_futures: bool = True,
        testnet: bool = False,
        rate_limit: int = 20,
        max_retries: int = 3,
        timeout: int = 30
    ):
        """
        Initialize Binance adapter.

        Args:
            api_key: API key (optional for public endpoints)
            api_secret: API secret (optional for public endpoints)
            use_futures: Use Futures API (True) or Spot API (False)
            testnet: Use testnet
            rate_limit: Max requests per second
            max_retries: Maximum retry attempts
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.use_futures = use_futures
        self.testnet = testnet
        self.timeout = timeout
        self.max_retries = max_retries

        # Set base URL based on market type
        if use_futures:
            if testnet:
                self.BASE_URL = "https://testnet.binancefuture.com"
            else:
                self.BASE_URL = "https://fapi.binance.com"
            self.exchange_info_endpoint = "/fapi/v1/exchangeInfo"
            self.klines_endpoint = "/fapi/v1/klines"
            self.ticker_endpoint = "/fapi/v1/ticker/24hr"
            self.ticker_price_endpoint = "/fapi/v1/ticker/price"
        else:
            if testnet:
                self.BASE_URL = "https://testnet.binance.vision"
            else:
                self.BASE_URL = "https://api.binance.com"
            self.exchange_info_endpoint = "/api/v3/exchangeInfo"
            self.klines_endpoint = "/api/v3/klines"
            self.ticker_endpoint = "/api/v3/ticker/24hr"
            self.ticker_price_endpoint = "/api/v3/ticker/price"

        self.session: Optional[aiohttp.ClientSession] = None
        self.rate_limiter = RateLimiter(max_requests=rate_limit, time_window=1.0)
        self.logger = logging.getLogger(self.__class__.__name__)

    async def connect(self) -> bool:
        """Establish connection to Binance."""
        try:
            headers = {
                'Content-Type': 'application/json',
                'User-Agent': 'CryptoMarketAgents/2.0'
            }

            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                headers=headers
            )

            # Test connection
            result = await self._request("GET", self.exchange_info_endpoint)
            if result and 'symbols' in result:
                market_type = "Futures" if self.use_futures else "Spot"
                self.logger.info(f"Successfully connected to Binance {market_type} API")
                return True
            return False

        except Exception as e:
            self.logger.error(f"Failed to connect to Binance: {e}")
            return False

    async def disconnect(self):
        """Close connection to Binance."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.logger.info("Disconnected from Binance")

    def check_us_accessibility(self) -> Dict[str, Any]:
        """
        Check US accessibility for Binance.

        Returns:
            Accessibility information
        """
        if self.use_futures:
            return {
                "accessible": True,
                "restrictions": [],
                "notes": (
                    "Binance Futures API is globally accessible without geo-restrictions. "
                    "However, check local regulations before trading."
                ),
                "recommended_alternatives": []
            }
        else:
            return {
                "accessible": False,
                "restrictions": ["Binance.com restricted for US users"],
                "notes": (
                    "Use Binance.US for spot trading or Binance Futures API for global access."
                ),
                "recommended_alternatives": ["Binance.US", "Binance Futures"]
            }

    async def get_trading_symbols(self, quote_currency: str = "USDT") -> List[str]:
        """Get list of trading symbols."""
        try:
            result = await self._request("GET", self.exchange_info_endpoint)

            if not result or 'symbols' not in result:
                return []

            symbols = []

            for symbol_info in result['symbols']:
                symbol = symbol_info['symbol']

                # For futures, check contract type
                if self.use_futures:
                    if (symbol.endswith(quote_currency) and
                        symbol_info.get('contractType') == 'PERPETUAL' and
                        symbol_info['status'] == 'TRADING'):
                        symbols.append(symbol)
                else:
                    if (symbol.endswith(quote_currency) and
                        symbol_info['status'] == 'TRADING'):
                        symbols.append(symbol)

            self.logger.info(f"Found {len(symbols)} trading symbols on Binance")
            return sorted(symbols)

        except Exception as e:
            self.logger.error(f"Failed to get trading symbols: {e}")
            return []

    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get ticker data for a symbol."""
        try:
            result = await self._request(
                "GET",
                self.ticker_endpoint,
                params={"symbol": symbol}
            )

            if not result:
                return {}

            # Handle both single ticker and list response
            if isinstance(result, list):
                if not result:
                    return {}
                ticker_data = result[0]
            else:
                ticker_data = result

            last_price = float(ticker_data.get('lastPrice', 0))

            return {
                "symbol": ticker_data.get('symbol'),
                "last_price": last_price,
                "bid": float(ticker_data.get('bidPrice', last_price * 0.999)),
                "ask": float(ticker_data.get('askPrice', last_price * 1.001)),
                "volume_24h": float(ticker_data.get('volume', 0)),
                "price_change_24h_pct": float(ticker_data.get('priceChangePercent', 0)),
                "high_24h": float(ticker_data.get('highPrice', 0)),
                "low_24h": float(ticker_data.get('lowPrice', 0)),
                "timestamp": datetime.now(timezone.utc)
            }

        except Exception as e:
            self.logger.error(f"Failed to get ticker for {symbol}: {e}")
            return {}

    async def get_tickers(self, symbols: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Get ticker data for multiple symbols."""
        try:
            # Get all tickers at once (more efficient than individual requests)
            result = await self._request("GET", self.ticker_endpoint)

            if not result or not isinstance(result, list):
                return []

            tickers = []

            for ticker_data in result:
                symbol = ticker_data.get('symbol')

                # Filter by symbols if provided
                if symbols and symbol not in symbols:
                    continue

                try:
                    last_price = float(ticker_data.get('lastPrice', 0))

                    ticker = {
                        "symbol": symbol,
                        "last_price": last_price,
                        "bid": float(ticker_data.get('bidPrice', last_price * 0.999)),
                        "ask": float(ticker_data.get('askPrice', last_price * 1.001)),
                        "volume_24h": float(ticker_data.get('volume', 0)),
                        "price_change_24h_pct": float(ticker_data.get('priceChangePercent', 0)),
                        "high_24h": float(ticker_data.get('highPrice', 0)),
                        "low_24h": float(ticker_data.get('lowPrice', 0)),
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
        """
        Get candlestick data.

        Args:
            symbol: Trading symbol
            interval: Timeframe (1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w, 1M)
            limit: Number of candles
            start_time: Start time
            end_time: End time
        """
        try:
            # Map our interval format to Binance format
            interval_map = {
                "1": "1m",
                "5": "5m",
                "15": "15m",
                "30": "30m",
                "60": "1h",
                "240": "4h",
                "360": "6h",
                "1440": "1d",
                "10080": "1w"
            }

            binance_interval = interval_map.get(str(interval), interval)

            # Ensure interval format is correct (append 'm' if just a number)
            if binance_interval.isdigit():
                binance_interval = f"{binance_interval}m"

            params = {
                "symbol": symbol,
                "interval": binance_interval,
                "limit": min(limit, 1000)  # Binance max is 1000
            }

            if start_time:
                params["startTime"] = int(start_time.timestamp() * 1000)
            if end_time:
                params["endTime"] = int(end_time.timestamp() * 1000)

            result = await self._request("GET", self.klines_endpoint, params=params)

            if not result or not isinstance(result, list):
                return []

            klines = []
            # Binance returns: [timestamp, open, high, low, close, volume, ...]
            for candle in result:
                klines.append({
                    "timestamp": datetime.fromtimestamp(int(candle[0]) / 1000, tz=timezone.utc),
                    "open": float(candle[1]),
                    "high": float(candle[2]),
                    "low": float(candle[3]),
                    "close": float(candle[4]),
                    "volume": float(candle[5])
                })

            return klines

        except Exception as e:
            self.logger.error(f"Failed to get klines for {symbol}: {e}")
            return []

    async def get_order_book(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        """Get order book depth."""
        try:
            endpoint = "/fapi/v1/depth" if self.use_futures else "/api/v3/depth"

            result = await self._request(
                "GET",
                endpoint,
                params={"symbol": symbol, "limit": min(limit, 1000)}
            )

            if not result:
                return {}

            return {
                "bids": [[float(b[0]), float(b[1])] for b in result.get('bids', [])],
                "asks": [[float(a[0]), float(a[1])] for a in result.get('asks', [])],
                "timestamp": datetime.now(timezone.utc)
            }

        except Exception as e:
            self.logger.error(f"Failed to get order book for {symbol}: {e}")
            return {}

    async def get_recent_trades(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent trades."""
        try:
            endpoint = "/fapi/v1/trades" if self.use_futures else "/api/v3/trades"

            result = await self._request(
                "GET",
                endpoint,
                params={"symbol": symbol, "limit": min(limit, 1000)}
            )

            if not result or not isinstance(result, list):
                return []

            trades = []
            for trade in result:
                trades.append({
                    "price": float(trade.get('price', 0)),
                    "quantity": float(trade.get('qty', 0)),
                    "timestamp": datetime.fromtimestamp(
                        int(trade.get('time', 0)) / 1000, tz=timezone.utc
                    ),
                    "side": "buy" if trade.get('isBuyerMaker') else "sell"
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
        Make HTTP request to Binance API.

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

                    elif response.status in [418, 451]:
                        # 418: IP banned, 451: geo-restricted
                        self.logger.error(
                            f"Access issue (status {response.status}). "
                            "Consider using Binance Futures API for global access."
                        )
                        return None

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
