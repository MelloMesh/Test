"""
Data fetching from Binance Futures.
Handles OHLCV candles, funding rates, and orderbook data.
"""

import time
from datetime import datetime, timezone

import pandas as pd

from src.exchange import get_public_exchange
from src.utils.logger import get_logger

logger = get_logger(__name__)


def fetch_candles(
    symbol: str = "BTC/USDT:USDT",
    timeframe: str = "15m",
    limit: int = 200,
    since: int | None = None,
    exchange: object | None = None,
) -> pd.DataFrame:
    """
    Fetch OHLCV candles from Binance Futures.

    Args:
        symbol: Trading pair (e.g., "BTC/USDT:USDT").
        timeframe: Candle timeframe (1m, 5m, 15m, 30m, 1h, 4h).
        limit: Number of candles to fetch (max 1500 per request).
        since: Start timestamp in milliseconds. If None, fetches most recent.
        exchange: Optional exchange instance (creates one if not provided).

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    if exchange is None:
        exchange = get_public_exchange()

    logger.debug(f"Fetching {limit} {timeframe} candles for {symbol}")

    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)

    if not ohlcv:
        logger.warning(f"No candles returned for {symbol} {timeframe}")
        return pd.DataFrame(
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )

    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

    logger.info(
        f"Fetched {len(df)} {timeframe} candles for {symbol} "
        f"({df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]})"
    )
    return df


def fetch_candles_bulk(
    symbol: str = "BTC/USDT:USDT",
    timeframe: str = "15m",
    days: int = 30,
    exchange: object | None = None,
) -> pd.DataFrame:
    """
    Fetch a large amount of historical candles by paginating requests.

    Args:
        symbol: Trading pair.
        timeframe: Candle timeframe.
        days: Number of days of history to fetch.
        exchange: Optional exchange instance.

    Returns:
        DataFrame with all fetched candles, deduplicated.
    """
    if exchange is None:
        exchange = get_public_exchange()

    tf_minutes = _timeframe_to_minutes(timeframe)
    total_candles = (days * 24 * 60) // tf_minutes
    batch_size = 1000  # Binance max per request is 1500, use 1000 for safety

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = now_ms - (days * 24 * 60 * 60 * 1000)

    all_candles: list[pd.DataFrame] = []
    since = start_ms
    fetched = 0

    logger.info(
        f"Bulk fetching ~{total_candles} {timeframe} candles for {symbol} ({days} days)"
    )

    while since < now_ms and fetched < total_candles + batch_size:
        df = fetch_candles(
            symbol=symbol,
            timeframe=timeframe,
            limit=batch_size,
            since=since,
            exchange=exchange,
        )

        if df.empty:
            break

        all_candles.append(df)
        fetched += len(df)

        # Advance since to the last candle's timestamp + 1 interval
        last_ts = int(df["timestamp"].iloc[-1].timestamp() * 1000)
        since = last_ts + (tf_minutes * 60 * 1000)

        # Rate limit courtesy
        time.sleep(0.2)

        if len(df) < batch_size:
            break  # No more data available

    if not all_candles:
        return pd.DataFrame(
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )

    result = pd.concat(all_candles, ignore_index=True)
    result = result.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    logger.info(f"Bulk fetch complete: {len(result)} {timeframe} candles for {symbol}")
    return result


def fetch_funding_rate(
    symbol: str = "BTC/USDT:USDT",
    exchange: object | None = None,
) -> dict:
    """
    Fetch the current funding rate for a perpetual contract.

    Returns:
        Dict with 'symbol', 'funding_rate', 'funding_timestamp', 'datetime'.
    """
    if exchange is None:
        exchange = get_public_exchange()

    logger.debug(f"Fetching funding rate for {symbol}")
    funding = exchange.fetch_funding_rate(symbol)
    logger.info(f"Funding rate for {symbol}: {funding.get('fundingRate', 'N/A')}")
    return funding


def fetch_funding_rate_history(
    symbol: str = "BTC/USDT:USDT",
    since: int | None = None,
    limit: int = 100,
    exchange: object | None = None,
) -> pd.DataFrame:
    """
    Fetch historical funding rates.

    Args:
        symbol: Trading pair.
        since: Start timestamp in ms.
        limit: Number of records.
        exchange: Optional exchange instance.

    Returns:
        DataFrame with columns: timestamp, funding_rate, symbol.
    """
    if exchange is None:
        exchange = get_public_exchange()

    logger.debug(f"Fetching funding rate history for {symbol}")

    rates = exchange.fetch_funding_rate_history(symbol, since=since, limit=limit)

    if not rates:
        return pd.DataFrame(columns=["timestamp", "funding_rate", "symbol"])

    df = pd.DataFrame(rates)
    # Normalize columns
    result = pd.DataFrame({
        "timestamp": pd.to_datetime(df["timestamp"], unit="ms", utc=True),
        "funding_rate": df["fundingRate"],
        "symbol": symbol,
    })

    logger.info(f"Fetched {len(result)} funding rate records for {symbol}")
    return result


def _timeframe_to_minutes(timeframe: str) -> int:
    """Convert timeframe string to minutes."""
    multipliers = {"m": 1, "h": 60, "d": 1440, "w": 10080}
    unit = timeframe[-1]
    value = int(timeframe[:-1])
    return value * multipliers.get(unit, 1)
