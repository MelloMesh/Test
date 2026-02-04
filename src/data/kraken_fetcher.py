"""
Kraken API Data Fetcher for Crypto Perpetual Futures
Multi-Timeframe OHLC and Funding Rate Data Collection
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import time
import json
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KrakenFuturesFetcher:
    """
    Fetches OHLC and funding rate data from Kraken for crypto perpetual futures
    across multiple timeframes (2m, 5m, 15m, 30m)
    """

    def __init__(self, config):
        self.config = config
        self.base_url = "https://futures.kraken.com/api/v3"
        self.rate_limit_delay = 1.0 / config.KRAKEN_CONFIG["rate_limit_calls_per_second"]
        self.last_request_time = 0

        # Timeframe mapping (Kraken format)
        self.timeframe_map = {
            "1m": "1m",
            "2m": "2m",  # May need to aggregate from 1m if not available
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "1h",
            "4h": "4h",
            "1d": "1d",
        }

        # Instrument mapping (convert to Kraken format)
        self.instrument_map = {
            "BTCUSDT.P": "PF_XBTUSD",  # Kraken uses XBT for Bitcoin
            "ETHUSDT.P": "PF_ETHUSD",
            "BNBUSDT.P": "PF_BNBUSD",
            "ADAUSDT.P": "PF_ADAUSD",
            "DOGEUSDT.P": "PF_DOGEUSD",
            "XRPUSDT.P": "PF_XRPUSD",
            "SOLUSDT.P": "PF_SOLUSD",
            "AVAXUSDT.P": "PF_AVAXUSD",
        }

    def _rate_limit(self):
        """Enforce rate limiting"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()

    def _make_request(self, endpoint: str, params: Optional[Dict] = None, retry_count: int = 4) -> Dict:
        """
        Make HTTP request with exponential backoff retry logic

        Args:
            endpoint: API endpoint
            params: Query parameters
            retry_count: Number of retries on failure

        Returns:
            JSON response as dictionary
        """
        url = f"{self.base_url}/{endpoint}"

        for attempt in range(retry_count):
            try:
                self._rate_limit()
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                return response.json()

            except requests.exceptions.RequestException as e:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s, 8s
                logger.warning(f"Request failed (attempt {attempt + 1}/{retry_count}): {e}")

                if attempt < retry_count - 1:
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed after {retry_count} attempts")
                    raise

    def get_historical_ohlc(
        self,
        instrument: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Fetch historical OHLC data for a specific instrument and timeframe

        Args:
            instrument: Trading instrument (e.g., "BTCUSDT.P")
            timeframe: Candle timeframe (e.g., "5m")
            start_date: Start date for historical data
            end_date: End date for historical data

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        # Convert instrument to Kraken format
        kraken_symbol = self.instrument_map.get(instrument)
        if not kraken_symbol:
            logger.warning(f"Instrument {instrument} not mapped, using as-is")
            kraken_symbol = instrument

        # Convert timeframe
        kraken_timeframe = self.timeframe_map.get(timeframe, timeframe)

        logger.info(f"Fetching {instrument} ({kraken_symbol}) {timeframe} data from {start_date} to {end_date}")

        # Kraken API uses tick_type for timeframes
        tick_type_map = {
            "1m": "1m",
            "2m": "2m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "1h",
            "4h": "4h",
            "1d": "1d",
        }

        tick_type = tick_type_map.get(timeframe, "5m")

        # Build request parameters
        params = {
            "symbol": kraken_symbol,
            "tick_type": tick_type,
            "from": int(start_date.timestamp()),
            "to": int(end_date.timestamp()),
        }

        try:
            # Fetch data from API
            response = self._make_request("history", params)

            if "candles" not in response:
                logger.error(f"No candles data in response: {response}")
                return pd.DataFrame()

            candles = response["candles"]

            # Parse candles into DataFrame
            df = pd.DataFrame(candles)

            if df.empty:
                logger.warning(f"No data returned for {instrument} {timeframe}")
                return df

            # Rename columns to standard format
            df.rename(columns={
                "time": "timestamp",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
            }, inplace=True)

            # Convert timestamp to datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

            # Ensure numeric types
            for col in ["open", "high", "low", "close", "volume"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            # Sort by timestamp
            df.sort_values("timestamp", inplace=True)
            df.reset_index(drop=True, inplace=True)

            logger.info(f"Fetched {len(df)} candles for {instrument} {timeframe}")
            return df

        except Exception as e:
            logger.error(f"Error fetching OHLC data: {e}")
            return pd.DataFrame()

    def get_funding_rates(
        self,
        instrument: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Fetch historical funding rates for a perpetual contract

        Args:
            instrument: Trading instrument (e.g., "BTCUSDT.P")
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with columns: timestamp, funding_rate
        """
        kraken_symbol = self.instrument_map.get(instrument, instrument)

        logger.info(f"Fetching funding rates for {instrument} from {start_date} to {end_date}")

        try:
            # Kraken funding rates endpoint
            params = {
                "symbol": kraken_symbol,
            }

            response = self._make_request("fundingrates", params)

            if "rates" not in response:
                logger.warning(f"No funding rates in response: {response}")
                return pd.DataFrame()

            rates = response["rates"]
            df = pd.DataFrame(rates)

            if df.empty:
                return df

            # Convert timestamp
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            elif "time" in df.columns:
                df["timestamp"] = pd.to_datetime(df["time"], unit="ms")

            # Filter by date range
            df = df[(df["timestamp"] >= start_date) & (df["timestamp"] <= end_date)]

            # Ensure funding_rate column exists
            if "fundingRate" in df.columns:
                df.rename(columns={"fundingRate": "funding_rate"}, inplace=True)

            df.sort_values("timestamp", inplace=True)
            df.reset_index(drop=True, inplace=True)

            logger.info(f"Fetched {len(df)} funding rate records")
            return df[["timestamp", "funding_rate"]]

        except Exception as e:
            logger.error(f"Error fetching funding rates: {e}")
            return pd.DataFrame()

    def fetch_all_timeframes(
        self,
        instrument: str,
        timeframes: List[str],
        lookback_months: int = 6
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLC data for all specified timeframes

        Args:
            instrument: Trading instrument
            timeframes: List of timeframes to fetch
            lookback_months: Number of months to look back

        Returns:
            Dictionary mapping timeframe -> DataFrame
        """
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=lookback_months * 30)

        data = {}

        for tf in timeframes:
            logger.info(f"Fetching {instrument} {tf} data...")
            df = self.get_historical_ohlc(instrument, tf, start_date, end_date)

            if not df.empty:
                data[tf] = df
                logger.info(f"✓ {instrument} {tf}: {len(df)} candles")
            else:
                logger.warning(f"✗ {instrument} {tf}: No data")

        return data

    def save_data(self, data: Dict[str, pd.DataFrame], instrument: str, output_dir: str = "data/raw"):
        """
        Save fetched data to CSV files

        Args:
            data: Dictionary mapping timeframe -> DataFrame
            instrument: Instrument name
            output_dir: Output directory path
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for timeframe, df in data.items():
            if df.empty:
                continue

            filename = f"{instrument}_{timeframe}.csv"
            filepath = output_path / filename

            df.to_csv(filepath, index=False)
            logger.info(f"Saved {filepath}")

    def load_data(self, instrument: str, timeframe: str, data_dir: str = "data/raw") -> pd.DataFrame:
        """
        Load previously saved data from CSV

        Args:
            instrument: Instrument name
            timeframe: Timeframe
            data_dir: Data directory path

        Returns:
            DataFrame with OHLC data
        """
        filepath = Path(data_dir) / f"{instrument}_{timeframe}.csv"

        if not filepath.exists():
            logger.warning(f"File not found: {filepath}")
            return pd.DataFrame()

        df = pd.read_csv(filepath)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        return df


def main():
    """Example usage"""
    import sys
    sys.path.append("../..")
    import config

    fetcher = KrakenFuturesFetcher(config)

    # Fetch data for Bitcoin across all timeframes
    instrument = "BTCUSDT.P"
    timeframes = config.TIMEFRAMES

    logger.info(f"Fetching data for {instrument} across {len(timeframes)} timeframes")

    data = fetcher.fetch_all_timeframes(
        instrument=instrument,
        timeframes=timeframes,
        lookback_months=config.DATA_LOOKBACK_MONTHS
    )

    # Save to disk
    fetcher.save_data(data, instrument, config.RAW_DATA_DIR)

    # Fetch funding rates
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=config.DATA_LOOKBACK_MONTHS * 30)

    funding_df = fetcher.get_funding_rates(instrument, start_date, end_date)

    if not funding_df.empty:
        funding_path = Path(config.FUNDING_RATES_DIR) / f"{instrument}_funding.csv"
        funding_path.parent.mkdir(parents=True, exist_ok=True)
        funding_df.to_csv(funding_path, index=False)
        logger.info(f"Saved funding rates to {funding_path}")

    logger.info("Data fetching complete!")


if __name__ == "__main__":
    main()
