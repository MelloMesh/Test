"""
SQLite storage for OHLCV candles and funding rates.
Handles save, load, and gap detection.
"""

import sqlite3
from datetime import datetime, timezone

import pandas as pd

from src.config import DB_PATH
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
_CANDLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS candles (
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL NOT NULL,
    PRIMARY KEY (symbol, timeframe, timestamp)
);
CREATE INDEX IF NOT EXISTS idx_candles_lookup
    ON candles (symbol, timeframe, timestamp);
"""

_FUNDING_SCHEMA = """
CREATE TABLE IF NOT EXISTS funding_rates (
    symbol TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    funding_rate REAL NOT NULL,
    PRIMARY KEY (symbol, timestamp)
);
CREATE INDEX IF NOT EXISTS idx_funding_lookup
    ON funding_rates (symbol, timestamp);
"""


def _get_connection(db_path: str | None = None) -> sqlite3.Connection:
    """Get a SQLite connection, creating tables if needed."""
    path = db_path or str(DB_PATH)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_CANDLE_SCHEMA)
    conn.executescript(_FUNDING_SCHEMA)
    return conn


def store_candles(
    df: pd.DataFrame,
    symbol: str = "BTC/USDT:USDT",
    timeframe: str = "15m",
    db_path: str | None = None,
) -> int:
    """
    Store OHLCV candles in SQLite. Uses INSERT OR REPLACE for idempotency.

    Args:
        df: DataFrame with columns: timestamp, open, high, low, close, volume.
        symbol: Trading pair.
        timeframe: Candle timeframe.
        db_path: Optional custom DB path (for testing).

    Returns:
        Number of rows inserted/replaced.
    """
    if df.empty:
        return 0

    conn = _get_connection(db_path)
    try:
        rows = []
        for _, row in df.iterrows():
            ts = row["timestamp"]
            if isinstance(ts, pd.Timestamp):
                ts = ts.isoformat()
            elif isinstance(ts, datetime):
                ts = ts.isoformat()
            rows.append((
                symbol, timeframe, str(ts),
                float(row["open"]), float(row["high"]),
                float(row["low"]), float(row["close"]),
                float(row["volume"]),
            ))

        conn.executemany(
            "INSERT OR REPLACE INTO candles "
            "(symbol, timeframe, timestamp, open, high, low, close, volume) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        conn.commit()
        logger.info(f"Stored {len(rows)} {timeframe} candles for {symbol}")
        return len(rows)
    finally:
        conn.close()


def load_candles(
    symbol: str = "BTC/USDT:USDT",
    timeframe: str = "15m",
    days: int | None = None,
    limit: int | None = None,
    db_path: str | None = None,
) -> pd.DataFrame:
    """
    Load candles from SQLite.

    Args:
        symbol: Trading pair.
        timeframe: Candle timeframe.
        days: If set, only load candles from the last N days.
        limit: If set, only load the last N candles.
        db_path: Optional custom DB path.

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume.
    """
    conn = _get_connection(db_path)
    try:
        query = (
            "SELECT timestamp, open, high, low, close, volume "
            "FROM candles WHERE symbol = ? AND timeframe = ?"
        )
        params: list = [symbol, timeframe]

        if days is not None:
            cutoff = datetime.now(timezone.utc).isoformat()
            # Approximate: we filter in Python for simplicity
            query += " ORDER BY timestamp ASC"
        elif limit is not None:
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
        else:
            query += " ORDER BY timestamp ASC"

        df = pd.read_sql_query(query, conn, params=params)

        if df.empty:
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        if days is not None:
            cutoff_dt = datetime.now(timezone.utc) - pd.Timedelta(days=days)
            df = df[df["timestamp"] >= cutoff_dt]

        if limit is not None:
            df = df.sort_values("timestamp").reset_index(drop=True)

        return df.sort_values("timestamp").reset_index(drop=True)
    finally:
        conn.close()


def store_funding_rates(
    df: pd.DataFrame,
    symbol: str = "BTC/USDT:USDT",
    db_path: str | None = None,
) -> int:
    """Store funding rate records in SQLite."""
    if df.empty:
        return 0

    conn = _get_connection(db_path)
    try:
        rows = []
        for _, row in df.iterrows():
            ts = row["timestamp"]
            if isinstance(ts, pd.Timestamp):
                ts = ts.isoformat()
            rows.append((symbol, str(ts), float(row["funding_rate"])))

        conn.executemany(
            "INSERT OR REPLACE INTO funding_rates (symbol, timestamp, funding_rate) "
            "VALUES (?, ?, ?)",
            rows,
        )
        conn.commit()
        logger.info(f"Stored {len(rows)} funding rates for {symbol}")
        return len(rows)
    finally:
        conn.close()


def count_candles(
    symbol: str | None = None,
    timeframe: str | None = None,
    db_path: str | None = None,
) -> int:
    """Count candles in the database, optionally filtered."""
    conn = _get_connection(db_path)
    try:
        query = "SELECT COUNT(*) FROM candles"
        params: list = []
        conditions: list[str] = []

        if symbol:
            conditions.append("symbol = ?")
            params.append(symbol)
        if timeframe:
            conditions.append("timeframe = ?")
            params.append(timeframe)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        cursor = conn.execute(query, params)
        return cursor.fetchone()[0]
    finally:
        conn.close()


def detect_gaps(
    symbol: str = "BTC/USDT:USDT",
    timeframe: str = "15m",
    db_path: str | None = None,
) -> list[dict]:
    """
    Detect gaps in stored candle data.

    Returns:
        List of dicts with 'start', 'end', 'missing_candles' for each gap.
    """
    df = load_candles(symbol=symbol, timeframe=timeframe, db_path=db_path)
    if len(df) < 2:
        return []

    from src.data.fetcher import _timeframe_to_minutes
    expected_delta = pd.Timedelta(minutes=_timeframe_to_minutes(timeframe))

    gaps = []
    for i in range(1, len(df)):
        actual_delta = df["timestamp"].iloc[i] - df["timestamp"].iloc[i - 1]
        if actual_delta > expected_delta * 1.5:  # Allow some tolerance
            missing = int(actual_delta / expected_delta) - 1
            gaps.append({
                "start": df["timestamp"].iloc[i - 1],
                "end": df["timestamp"].iloc[i],
                "missing_candles": missing,
            })

    if gaps:
        logger.warning(
            f"Found {len(gaps)} gaps in {symbol} {timeframe} data "
            f"({sum(g['missing_candles'] for g in gaps)} total missing candles)"
        )

    return gaps
