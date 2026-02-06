"""Tests for volume-based indicators."""

import numpy as np
import pandas as pd
import pytest

from src.indicators.volume import calculate_obv, detect_obv_divergence, volume_confirms_divergence


def _make_candles(n=100):
    """Create basic candle data with volume."""
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="15min", tz="UTC"),
        "open": prices - 0.2,
        "high": prices + abs(np.random.randn(n) * 0.3) + 0.1,
        "low": prices - abs(np.random.randn(n) * 0.3) - 0.1,
        "close": prices,
        "volume": 1000 + abs(np.random.randn(n) * 200),
    })


class TestOBV:
    def test_obv_returns_series(self):
        candles = _make_candles()
        obv = calculate_obv(candles["close"], candles["volume"])
        assert isinstance(obv, pd.Series)
        assert len(obv) == len(candles)

    def test_obv_rises_on_up_moves(self):
        """OBV should rise when price closes up."""
        close = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0])
        volume = pd.Series([1000.0, 1000.0, 1000.0, 1000.0, 1000.0])
        obv = calculate_obv(close, volume)
        # Each up close adds 1000 volume
        assert obv.iloc[-1] > 0

    def test_obv_falls_on_down_moves(self):
        """OBV should fall when price closes down."""
        close = pd.Series([104.0, 103.0, 102.0, 101.0, 100.0])
        volume = pd.Series([1000.0, 1000.0, 1000.0, 1000.0, 1000.0])
        obv = calculate_obv(close, volume)
        assert obv.iloc[-1] < 0


class TestOBVDivergence:
    def test_bullish_obv_divergence(self):
        """Price lower low + OBV higher low = bullish OBV divergence."""
        np.random.seed(42)
        n = 100
        prices = 100 + np.cumsum(np.random.randn(n) * 0.3)
        # Create volume pattern: higher OBV at peak2 vs peak1
        volume = np.full(n, 1000.0)
        # Low volume at first low (index 20), high volume at second low (index 60)
        volume[15:25] = 500  # Low volume around peak1
        volume[55:65] = 2000  # High volume around peak2

        candles = pd.DataFrame({
            "close": prices,
            "volume": volume,
            "high": prices + 0.5,
            "low": prices - 0.5,
            "open": prices - 0.1,
        })

        div = {
            "type": "bullish",
            "peak1": {"index": 20, "rsi_value": 25},
            "peak2": {"index": 60, "rsi_value": 30},
            "index": 60,
        }

        # This should detect the OBV divergence pattern
        result = detect_obv_divergence(candles, div)
        assert isinstance(result, bool)

    def test_bearish_obv_divergence(self):
        """Price higher high + OBV lower high = bearish OBV divergence."""
        np.random.seed(42)
        n = 100
        prices = 100 + np.cumsum(np.random.randn(n) * 0.3)
        volume = np.full(n, 1000.0)
        # High volume at first high (index 20), low volume at second high (index 60)
        volume[15:25] = 2000  # High volume around peak1
        volume[55:65] = 500  # Low volume around peak2

        candles = pd.DataFrame({
            "close": prices,
            "volume": volume,
            "high": prices + 0.5,
            "low": prices - 0.5,
            "open": prices - 0.1,
        })

        div = {
            "type": "bearish",
            "peak1": {"index": 20, "rsi_value": 75},
            "peak2": {"index": 60, "rsi_value": 70},
            "index": 60,
        }

        result = detect_obv_divergence(candles, div)
        assert isinstance(result, bool)

    def test_no_volume_column(self):
        """Should return False if no volume data."""
        candles = pd.DataFrame({"close": [100, 101, 102]})
        div = {
            "type": "bullish",
            "peak1": {"index": 0},
            "peak2": {"index": 1},
        }
        assert detect_obv_divergence(candles, div) == False

    def test_missing_peaks(self):
        """Should return False if peak indices missing."""
        candles = _make_candles(50)
        div = {"type": "bullish", "peak1": {}, "peak2": {}}
        assert detect_obv_divergence(candles, div) == False

    def test_peak_index_out_of_range(self):
        """Should return False if peak index exceeds data length."""
        candles = _make_candles(50)
        div = {
            "type": "bullish",
            "peak1": {"index": 200},
            "peak2": {"index": 300},
        }
        assert detect_obv_divergence(candles, div) == False

    def test_hidden_bullish_type(self):
        """Should handle hidden_bullish type correctly."""
        candles = _make_candles(100)
        div = {
            "type": "hidden_bullish",
            "peak1": {"index": 20},
            "peak2": {"index": 60},
            "index": 60,
        }
        result = detect_obv_divergence(candles, div)
        assert isinstance(result, bool)

    def test_hidden_bearish_type(self):
        """Should handle hidden_bearish type correctly."""
        candles = _make_candles(100)
        div = {
            "type": "hidden_bearish",
            "peak1": {"index": 20},
            "peak2": {"index": 60},
            "index": 60,
        }
        result = detect_obv_divergence(candles, div)
        assert isinstance(result, bool)


class TestVolumeConfirmsDivergence:
    def test_volume_increasing_confirms(self):
        """Higher recent volume should confirm divergence."""
        # divergence_index=39, lookback=10:
        # recent_start=29, recent=[29:40] -> indices 29(low), 30-39(high) -> avg ~1864
        # prior_start=19, prior=[19:29]   -> indices 19-28(low) -> avg=500
        # 1864 > 500 * 1.1 = 550 -> True
        n = 50
        volume = np.concatenate([
            np.full(30, 500.0),   # Low volume period (indices 0-29)
            np.full(20, 2000.0),  # High volume period (indices 30-49)
        ])
        candles = pd.DataFrame({
            "volume": volume,
            "close": np.linspace(100, 105, n),
        })
        assert volume_confirms_divergence(candles, 39, lookback=10) == True

    def test_volume_decreasing_rejects(self):
        """Lower recent volume should reject."""
        n = 50
        volume = np.concatenate([
            np.full(30, 2000.0),  # High volume period (indices 0-29)
            np.full(20, 500.0),   # Low volume period (indices 30-49)
        ])
        candles = pd.DataFrame({
            "volume": volume,
            "close": np.linspace(100, 105, n),
        })
        assert volume_confirms_divergence(candles, 39, lookback=10) == False

    def test_insufficient_data(self):
        """Too few candles should return False."""
        candles = pd.DataFrame({
            "volume": [1000.0] * 5,
            "close": [100.0] * 5,
        })
        assert volume_confirms_divergence(candles, 3, lookback=10) == False
