"""Tests for SQLite storage module."""

import tempfile

import pandas as pd
import pytest

from src.data.storage import (
    count_candles,
    detect_gaps,
    load_candles,
    store_candles,
    store_funding_rates,
)


@pytest.fixture
def tmp_db(tmp_path):
    """Provide a temporary database path for testing."""
    return str(tmp_path / "test_candles.db")


@pytest.fixture
def sample_candles():
    """Create sample OHLCV data."""
    return pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=10, freq="15min", tz="UTC"),
        "open": [100.0 + i for i in range(10)],
        "high": [101.0 + i for i in range(10)],
        "low": [99.0 + i for i in range(10)],
        "close": [100.5 + i for i in range(10)],
        "volume": [1000.0 + i * 100 for i in range(10)],
    })


class TestStoreCandles:
    """Test candle storage operations."""

    def test_store_returns_count(self, sample_candles, tmp_db):
        count = store_candles(sample_candles, "BTC/USDT:USDT", "15m", db_path=tmp_db)
        assert count == 10

    def test_store_empty_dataframe(self, tmp_db):
        df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        count = store_candles(df, "BTC/USDT:USDT", "15m", db_path=tmp_db)
        assert count == 0

    def test_store_idempotent(self, sample_candles, tmp_db):
        """Storing the same data twice should not duplicate rows."""
        store_candles(sample_candles, "BTC/USDT:USDT", "15m", db_path=tmp_db)
        store_candles(sample_candles, "BTC/USDT:USDT", "15m", db_path=tmp_db)
        total = count_candles("BTC/USDT:USDT", "15m", db_path=tmp_db)
        assert total == 10


class TestLoadCandles:
    """Test candle loading operations."""

    def test_load_roundtrip(self, sample_candles, tmp_db):
        store_candles(sample_candles, "BTC/USDT:USDT", "15m", db_path=tmp_db)
        loaded = load_candles("BTC/USDT:USDT", "15m", db_path=tmp_db)
        assert len(loaded) == 10
        assert list(loaded.columns) == ["timestamp", "open", "high", "low", "close", "volume"]

    def test_load_preserves_values(self, sample_candles, tmp_db):
        store_candles(sample_candles, "BTC/USDT:USDT", "15m", db_path=tmp_db)
        loaded = load_candles("BTC/USDT:USDT", "15m", db_path=tmp_db)
        assert loaded["open"].iloc[0] == pytest.approx(100.0)
        assert loaded["close"].iloc[0] == pytest.approx(100.5)

    def test_load_with_limit(self, sample_candles, tmp_db):
        store_candles(sample_candles, "BTC/USDT:USDT", "15m", db_path=tmp_db)
        loaded = load_candles("BTC/USDT:USDT", "15m", limit=5, db_path=tmp_db)
        assert len(loaded) == 5

    def test_load_empty_returns_empty_df(self, tmp_db):
        loaded = load_candles("BTC/USDT:USDT", "15m", db_path=tmp_db)
        assert isinstance(loaded, pd.DataFrame)
        assert len(loaded) == 0

    def test_load_sorted_by_timestamp(self, sample_candles, tmp_db):
        store_candles(sample_candles, "BTC/USDT:USDT", "15m", db_path=tmp_db)
        loaded = load_candles("BTC/USDT:USDT", "15m", db_path=tmp_db)
        timestamps = loaded["timestamp"].tolist()
        assert timestamps == sorted(timestamps)

    def test_load_different_symbols_isolated(self, sample_candles, tmp_db):
        store_candles(sample_candles, "BTC/USDT:USDT", "15m", db_path=tmp_db)
        loaded_eth = load_candles("ETH/USDT:USDT", "15m", db_path=tmp_db)
        assert len(loaded_eth) == 0

    def test_load_different_timeframes_isolated(self, sample_candles, tmp_db):
        store_candles(sample_candles, "BTC/USDT:USDT", "15m", db_path=tmp_db)
        loaded_30m = load_candles("BTC/USDT:USDT", "30m", db_path=tmp_db)
        assert len(loaded_30m) == 0


class TestCountCandles:
    """Test candle counting."""

    def test_count_all(self, sample_candles, tmp_db):
        store_candles(sample_candles, "BTC/USDT:USDT", "15m", db_path=tmp_db)
        assert count_candles(db_path=tmp_db) == 10

    def test_count_by_symbol(self, sample_candles, tmp_db):
        store_candles(sample_candles, "BTC/USDT:USDT", "15m", db_path=tmp_db)
        assert count_candles("BTC/USDT:USDT", db_path=tmp_db) == 10
        assert count_candles("ETH/USDT:USDT", db_path=tmp_db) == 0

    def test_count_by_timeframe(self, sample_candles, tmp_db):
        store_candles(sample_candles, "BTC/USDT:USDT", "15m", db_path=tmp_db)
        assert count_candles(timeframe="15m", db_path=tmp_db) == 10
        assert count_candles(timeframe="30m", db_path=tmp_db) == 0


class TestDetectGaps:
    """Test gap detection in stored data."""

    def test_no_gaps_in_continuous_data(self, sample_candles, tmp_db):
        store_candles(sample_candles, "BTC/USDT:USDT", "15m", db_path=tmp_db)
        gaps = detect_gaps("BTC/USDT:USDT", "15m", db_path=tmp_db)
        assert len(gaps) == 0

    def test_detect_gap(self, tmp_db):
        """Create data with a gap and verify detection."""
        timestamps = list(pd.date_range("2025-01-01", periods=5, freq="15min", tz="UTC"))
        # Add a 1-hour gap
        timestamps.append(timestamps[-1] + pd.Timedelta(hours=2))

        df = pd.DataFrame({
            "timestamp": timestamps,
            "open": [100.0] * 6,
            "high": [101.0] * 6,
            "low": [99.0] * 6,
            "close": [100.5] * 6,
            "volume": [1000.0] * 6,
        })
        store_candles(df, "BTC/USDT:USDT", "15m", db_path=tmp_db)
        gaps = detect_gaps("BTC/USDT:USDT", "15m", db_path=tmp_db)
        assert len(gaps) >= 1
        assert gaps[0]["missing_candles"] > 0


class TestStoreFundingRates:
    """Test funding rate storage."""

    def test_store_funding_rates(self, tmp_db):
        df = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=3, freq="8h", tz="UTC"),
            "funding_rate": [0.0001, 0.00015, 0.0002],
            "symbol": ["BTC/USDT:USDT"] * 3,
        })
        count = store_funding_rates(df, "BTC/USDT:USDT", db_path=tmp_db)
        assert count == 3
