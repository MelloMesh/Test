"""Tests for data fetcher module."""

import pytest
import pandas as pd

from src.data.fetcher import fetch_candles, fetch_funding_rate, _timeframe_to_minutes
from src.exchange import get_public_exchange


def _network_available():
    """Check if we can reach Binance API."""
    try:
        ex = get_public_exchange()
        ex.fetch_ticker("BTC/USDT:USDT")
        return True
    except Exception:
        return False


network = pytest.mark.skipif(
    not _network_available(), reason="Binance API unreachable"
)


@network
class TestFetchCandles:
    """Test OHLCV candle fetching (requires network)."""

    def test_fetch_candles_returns_dataframe(self):
        df = fetch_candles("BTC/USDT:USDT", "15m", limit=10)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_fetch_candles_has_correct_columns(self):
        df = fetch_candles("BTC/USDT:USDT", "15m", limit=10)
        expected_cols = {"timestamp", "open", "high", "low", "close", "volume"}
        assert set(df.columns) == expected_cols

    def test_fetch_candles_timestamp_is_datetime(self):
        df = fetch_candles("BTC/USDT:USDT", "15m", limit=10)
        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])

    def test_fetch_candles_prices_are_positive(self):
        df = fetch_candles("BTC/USDT:USDT", "15m", limit=10)
        assert (df["open"] > 0).all()
        assert (df["high"] > 0).all()
        assert (df["low"] > 0).all()
        assert (df["close"] > 0).all()

    def test_fetch_candles_high_gte_low(self):
        df = fetch_candles("BTC/USDT:USDT", "15m", limit=10)
        assert (df["high"] >= df["low"]).all()

    def test_fetch_candles_sorted_by_timestamp(self):
        df = fetch_candles("BTC/USDT:USDT", "15m", limit=50)
        timestamps = df["timestamp"].tolist()
        assert timestamps == sorted(timestamps)

    def test_fetch_candles_respects_limit(self):
        df = fetch_candles("BTC/USDT:USDT", "15m", limit=20)
        assert len(df) <= 20

    def test_fetch_candles_multiple_timeframes(self):
        for tf in ["5m", "15m", "30m", "1h"]:
            df = fetch_candles("BTC/USDT:USDT", tf, limit=5)
            assert len(df) > 0, f"No data for timeframe {tf}"


@network
class TestFetchFundingRate:
    """Test funding rate fetching (requires network)."""

    def test_fetch_funding_rate_returns_dict(self):
        rate = fetch_funding_rate("BTC/USDT:USDT")
        assert isinstance(rate, dict)

    def test_fetch_funding_rate_has_rate(self):
        rate = fetch_funding_rate("BTC/USDT:USDT")
        assert "fundingRate" in rate
        assert rate["fundingRate"] is not None


class TestTimeframeConversion:
    """Test timeframe string to minutes conversion."""

    def test_minutes(self):
        assert _timeframe_to_minutes("1m") == 1
        assert _timeframe_to_minutes("5m") == 5
        assert _timeframe_to_minutes("15m") == 15
        assert _timeframe_to_minutes("30m") == 30

    def test_hours(self):
        assert _timeframe_to_minutes("1h") == 60
        assert _timeframe_to_minutes("4h") == 240

    def test_days(self):
        assert _timeframe_to_minutes("1d") == 1440
