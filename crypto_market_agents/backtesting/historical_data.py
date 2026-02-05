"""
Historical Data Fetcher - Downloads and caches historical market data.
"""

import asyncio
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta, timezone
from pathlib import Path
import logging


class HistoricalDataFetcher:
    """
    Fetches and caches historical OHLCV data for backtesting.
    """

    def __init__(
        self,
        exchange,
        cache_dir: str = "backtest_data"
    ):
        """
        Initialize historical data fetcher.

        Args:
            exchange: Exchange adapter instance
            cache_dir: Directory for caching downloaded data
        """
        self.exchange = exchange
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        self.logger = logging.getLogger(self.__class__.__name__)

    def _get_cache_filename(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime
    ) -> Path:
        """
        Generate cache filename for historical data.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe (e.g., '1h', '4h', '1d')
            start_date: Start date
            end_date: End date

        Returns:
            Path to cache file
        """
        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")
        safe_symbol = symbol.replace("/", "_")
        filename = f"{safe_symbol}_{timeframe}_{start_str}_{end_str}.json"
        return self.cache_dir / filename

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Fetch historical OHLCV data for a symbol.

        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            timeframe: Candle timeframe ('1m', '5m', '15m', '1h', '4h', '1d')
            start_date: Start date (default: 30 days ago)
            end_date: End date (default: now)
            use_cache: Use cached data if available

        Returns:
            List of OHLCV candles with format:
            [{
                'timestamp': datetime,
                'open': float,
                'high': float,
                'low': float,
                'close': float,
                'volume': float
            }, ...]
        """
        # Default dates
        if end_date is None:
            end_date = datetime.now(timezone.utc)
        if start_date is None:
            start_date = end_date - timedelta(days=30)

        # Check cache
        cache_file = self._get_cache_filename(symbol, timeframe, start_date, end_date)

        if use_cache and cache_file.exists():
            self.logger.info(f"Loading cached data: {cache_file.name}")
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
                # Convert timestamps back to datetime
                return [
                    {
                        **candle,
                        'timestamp': datetime.fromisoformat(candle['timestamp'])
                    }
                    for candle in cached_data
                ]

        # Fetch from exchange
        self.logger.info(
            f"Fetching historical data for {symbol} ({timeframe}) "
            f"from {start_date.date()} to {end_date.date()}"
        )

        all_candles = []
        current_date = start_date

        # Fetch in chunks (exchange typically limits to 1000 candles per request)
        chunk_size = self._get_chunk_size(timeframe)

        while current_date < end_date:
            try:
                # Calculate chunk end
                chunk_end = min(current_date + chunk_size, end_date)

                # Fetch chunk from exchange
                candles = await self._fetch_chunk(
                    symbol,
                    timeframe,
                    current_date,
                    chunk_end
                )

                if candles:
                    all_candles.extend(candles)
                    self.logger.debug(
                        f"Fetched {len(candles)} candles for {symbol} "
                        f"({current_date.date()} to {chunk_end.date()})"
                    )

                current_date = chunk_end

                # Rate limiting
                await asyncio.sleep(0.2)

            except Exception as e:
                self.logger.error(f"Error fetching chunk for {symbol}: {e}")
                break

        self.logger.info(
            f"Fetched {len(all_candles)} total candles for {symbol} ({timeframe})"
        )

        # Cache the data
        if all_candles:
            cache_data = [
                {
                    **candle,
                    'timestamp': candle['timestamp'].isoformat()
                }
                for candle in all_candles
            ]
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            self.logger.info(f"Cached data to {cache_file.name}")

        return all_candles

    async def _fetch_chunk(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """
        Fetch a chunk of OHLCV data from exchange.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            start_date: Chunk start
            end_date: Chunk end

        Returns:
            List of candles
        """
        limit = 1000  # Most exchanges limit to 1000 candles

        try:
            # Fetch from exchange using BaseExchange interface
            candles = await self.exchange.get_klines(
                symbol=symbol,
                interval=timeframe,
                limit=limit,
                start_time=start_date,
                end_time=end_date
            )

            if not candles:
                return []

            # Filter candles within requested range (some exchanges may return more)
            filtered_candles = [
                c for c in candles
                if start_date <= c['timestamp'] <= end_date
            ]

            return filtered_candles

        except Exception as e:
            self.logger.error(f"Error in _fetch_chunk: {e}")
            return []

    def _get_chunk_size(self, timeframe: str) -> timedelta:
        """
        Get appropriate chunk size based on timeframe.

        Args:
            timeframe: Candle timeframe

        Returns:
            Chunk duration as timedelta
        """
        # Map timeframe to chunk size (in days)
        # Aim for ~1000 candles per chunk
        chunk_sizes = {
            '1m': timedelta(hours=16),     # 1000 minutes = ~16 hours
            '5m': timedelta(days=3),       # 1000 * 5 min = ~3.5 days
            '15m': timedelta(days=10),     # 1000 * 15 min = ~10 days
            '1h': timedelta(days=41),      # 1000 hours = ~41 days
            '4h': timedelta(days=166),     # 1000 * 4 hours = ~166 days
            '1d': timedelta(days=1000),    # 1000 days
        }

        return chunk_sizes.get(timeframe, timedelta(days=30))

    async def fetch_multiple_symbols(
        self,
        symbols: List[str],
        timeframe: str = "1h",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        use_cache: bool = True
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Fetch historical data for multiple symbols.

        Args:
            symbols: List of trading symbols
            timeframe: Candle timeframe
            start_date: Start date
            end_date: End date
            use_cache: Use cached data

        Returns:
            Dictionary mapping symbol -> OHLCV data
        """
        tasks = [
            self.fetch_ohlcv(symbol, timeframe, start_date, end_date, use_cache)
            for symbol in symbols
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        data = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                self.logger.error(f"Failed to fetch {symbol}: {result}")
                data[symbol] = []
            else:
                data[symbol] = result

        return data

    async def get_available_symbols(self, quote_currency: str = "USDT") -> List[str]:
        """
        Get available symbols for a quote currency.

        Args:
            quote_currency: Quote currency (e.g., 'USDT')

        Returns:
            List of available symbols
        """
        try:
            markets = await self.exchange.fetch_markets()
            symbols = [
                market['symbol']
                for market in markets
                if market['quote'] == quote_currency and market.get('active', True)
            ]
            self.logger.info(f"Found {len(symbols)} {quote_currency} pairs")
            return symbols

        except Exception as e:
            self.logger.error(f"Error fetching markets: {e}")
            return []

    def get_price_at_time(
        self,
        candles: List[Dict[str, Any]],
        target_time: datetime
    ) -> Optional[float]:
        """
        Get price at a specific time from candle data.

        Args:
            candles: List of OHLCV candles (must be sorted by timestamp)
            target_time: Target datetime

        Returns:
            Close price at target time or None
        """
        if not candles:
            return None

        # Binary search for closest candle
        left, right = 0, len(candles) - 1
        closest_idx = 0
        min_diff = float('inf')

        while left <= right:
            mid = (left + right) // 2
            candle_time = candles[mid]['timestamp']
            diff = abs((candle_time - target_time).total_seconds())

            if diff < min_diff:
                min_diff = diff
                closest_idx = mid

            if candle_time < target_time:
                left = mid + 1
            elif candle_time > target_time:
                right = mid - 1
            else:
                return candles[mid]['close']

        return candles[closest_idx]['close']
