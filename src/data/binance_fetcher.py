"""
Binance API Data Fetcher (Alternative to Bybit)
Fetches OHLCV data for ALL USDT perpetual pairs
Binance has a more permissive public API - no 403 errors
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class BinanceFetcher:
    """
    Fetches data from Binance for all USDT perpetual futures
    More reliable than Bybit for public data access
    """

    def __init__(self, config=None):
        self.config = config
        self.base_url = "https://fapi.binance.com"  # Futures API
        self.session = requests.Session()

        # Keep it simple - no custom headers (they can trigger blocking)
        # Rate limiting
        self.requests_per_second = 10
        self.last_request_time = 0

    def _rate_limit(self):
        """Enforce rate limiting"""
        now = time.time()
        time_since_last = now - self.last_request_time
        min_interval = 1.0 / self.requests_per_second

        if time_since_last < min_interval:
            time.sleep(min_interval - time_since_last)

        self.last_request_time = time.time()

    def get_all_usdt_perpetuals(self) -> List[str]:
        """
        Fetch ALL USDT perpetual pairs from Binance

        Returns:
            List of symbols like ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', ...]
        """
        self._rate_limit()

        try:
            url = f"{self.base_url}/fapi/v1/exchangeInfo"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            # Extract USDT perpetuals
            instruments = []
            for symbol_info in data['symbols']:
                symbol = symbol_info['symbol']
                # Only USDT perpetuals (not BUSD or coin-margined)
                if symbol.endswith('USDT') and symbol_info['status'] == 'TRADING' and symbol_info['contractType'] == 'PERPETUAL':
                    instruments.append(symbol)

            logger.info(f"Found {len(instruments)} USDT perpetual pairs on Binance")
            return sorted(instruments)

        except Exception as e:
            logger.error(f"Error fetching Binance instruments: {e}")
            return []

    def get_ohlcv(
        self,
        symbol: str,
        interval: str,
        start_time: datetime,
        end_time: datetime,
        limit: int = 1500
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from Binance

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            interval: Timeframe ('1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w')
            start_time: Start datetime
            end_time: End datetime
            limit: Max candles per request (max 1500)

        Returns:
            DataFrame with OHLCV data
        """
        self._rate_limit()

        try:
            url = f"{self.base_url}/fapi/v1/klines"
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": int(start_time.timestamp() * 1000),
                "endTime": int(end_time.timestamp() * 1000),
                "limit": limit
            }

            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if not data:
                return pd.DataFrame()

            # Parse candles
            # Binance returns: [timestamp, open, high, low, close, volume, close_time, quote_volume, trades, taker_buy_base, taker_buy_quote, ignore]
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
            ])

            # Convert types
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)

            # Keep only needed columns
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

            return df

        except Exception as e:
            logger.error(f"Error fetching {symbol} {interval} data: {e}")
            return pd.DataFrame()

    def fetch_historical_data(
        self,
        symbol: str,
        interval: str,
        start_date: datetime,
        end_date: datetime,
        save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch historical data with pagination

        Args:
            symbol: Trading pair
            interval: Timeframe
            start_date: Start date
            end_date: End date
            save_path: Optional path to save CSV

        Returns:
            Complete DataFrame with all historical data
        """
        logger.info(f"Fetching {symbol} {interval} from {start_date} to {end_date}")

        all_data = []
        current_start = start_date
        batch_size = 1500  # Binance max limit

        while current_start < end_date:
            # Calculate batch end
            interval_minutes = self._interval_to_minutes(interval)
            current_end = min(
                current_start + timedelta(minutes=interval_minutes * batch_size),
                end_date
            )

            # Fetch batch
            df = self.get_ohlcv(symbol, interval, current_start, current_end, limit=batch_size)

            if df.empty:
                break

            all_data.append(df)
            logger.debug(f"Fetched {len(df)} candles")

            # Move to next batch
            current_start = df['timestamp'].iloc[-1] + timedelta(minutes=interval_minutes)

            # Rate limiting
            time.sleep(0.1)

        if not all_data:
            logger.warning(f"No data fetched for {symbol} {interval}")
            return pd.DataFrame()

        # Combine all batches
        combined = pd.concat(all_data, ignore_index=True)
        combined = combined.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)

        logger.info(f"✓ {symbol} {interval}: {len(combined)} candles")

        # Save to CSV if path provided
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            combined.to_csv(save_path, index=False)
            logger.info(f"Saved to {save_path}")

        return combined

    def _interval_to_minutes(self, interval: str) -> int:
        """Convert interval string to minutes"""
        if interval.endswith('m'):
            return int(interval[:-1])
        elif interval.endswith('h'):
            return int(interval[:-1]) * 60
        elif interval == '1d':
            return 1440
        elif interval == '1w':
            return 10080
        else:
            return 5

    def get_top_volume_pairs(self, top_n: int = 50) -> List[str]:
        """
        Get top N pairs by 24h volume

        Returns:
            List of top symbols by volume
        """
        self._rate_limit()

        try:
            url = f"{self.base_url}/fapi/v1/ticker/24hr"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            # Extract volume data for USDT perpetuals
            tickers = []
            for item in data:
                symbol = item['symbol']
                if symbol.endswith('USDT'):
                    volume_24h = float(item.get('quoteVolume', 0))
                    tickers.append((symbol, volume_24h))

            # Sort by volume descending
            tickers.sort(key=lambda x: x[1], reverse=True)

            top_symbols = [sym for sym, vol in tickers[:top_n]]

            logger.info(f"Top {top_n} pairs by volume: {', '.join(top_symbols[:10])}...")
            return top_symbols

        except Exception as e:
            logger.error(f"Error fetching top volume pairs: {e}")
            return []

    def fetch_all_instruments_htf_data(
        self,
        timeframes: List[str] = ['1w', '1d', '4h'],
        lookback_days: int = 180,
        max_instruments: Optional[int] = None,
        output_dir: str = "data/binance_htf"
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Fetch HTF data for ALL Binance USDT perpetuals

        Args:
            timeframes: List of HTF timeframes to fetch
            lookback_days: Days of history to fetch
            max_instruments: Limit number of instruments (None = all)
            output_dir: Directory to save data

        Returns:
            Dict: {symbol: {timeframe: DataFrame}}
        """
        # Get all instruments
        all_instruments = self.get_all_usdt_perpetuals()

        if max_instruments:
            all_instruments = all_instruments[:max_instruments]

        logger.info(f"Fetching HTF data for {len(all_instruments)} instruments...")

        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=lookback_days)

        all_data = {}

        for i, symbol in enumerate(all_instruments, 1):
            logger.info(f"[{i}/{len(all_instruments)}] Processing {symbol}...")

            symbol_data = {}

            for tf in timeframes:
                # Fetch data
                df = self.fetch_historical_data(
                    symbol=symbol,
                    interval=tf,
                    start_date=start_date,
                    end_date=end_date,
                    save_path=f"{output_dir}/{symbol}_{tf}.csv"
                )

                if not df.empty:
                    symbol_data[tf] = df

                # Rate limiting
                time.sleep(0.1)

            if symbol_data:
                all_data[symbol] = symbol_data

            # Progress update
            if i % 20 == 0:
                logger.info(f"Progress: {i}/{len(all_instruments)} instruments processed")

        logger.info(f"✓ Fetched HTF data for {len(all_data)} instruments")
        return all_data


def main():
    """Test Binance fetcher"""
    logging.basicConfig(level=logging.INFO)

    fetcher = BinanceFetcher()

    # Test 1: Get all USDT perpetuals
    print("\n" + "="*80)
    print("TEST 1: Fetch all USDT perpetual pairs from Binance")
    print("="*80)

    instruments = fetcher.get_all_usdt_perpetuals()
    print(f"\nFound {len(instruments)} USDT perpetuals")
    print(f"Sample (first 20): {instruments[:20]}")

    # Test 2: Get top volume pairs
    print("\n" + "="*80)
    print("TEST 2: Get top 20 pairs by volume")
    print("="*80)

    top_pairs = fetcher.get_top_volume_pairs(top_n=20)
    print(f"\nTop 20 by 24h volume:")
    for i, pair in enumerate(top_pairs, 1):
        print(f"  {i}. {pair}")

    # Test 3: Fetch sample HTF data for BTC
    print("\n" + "="*80)
    print("TEST 3: Fetch HTF data for BTCUSDT")
    print("="*80)

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=30)

    for tf in ['1w', '1d', '4h']:
        df = fetcher.get_ohlcv('BTCUSDT', tf, start, end)
        print(f"\n{tf}: {len(df)} candles")
        if not df.empty:
            print(df.tail(3))

    print("\n✅ Binance API working successfully!")
    print("Use BinanceFetcher instead of BybitFetcher if you get 403 errors")


if __name__ == "__main__":
    main()
