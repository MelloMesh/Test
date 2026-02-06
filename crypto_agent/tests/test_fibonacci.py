"""Tests for Fibonacci level calculation and swing detection."""

import numpy as np
import pandas as pd
import pytest

from src.indicators.fibonacci import (
    calculate_fib_levels,
    detect_swings,
    get_latest_swing_pair,
    golden_pocket_depth,
    is_in_golden_pocket,
)


class TestDetectSwings:
    """Test swing high/low detection using fractal method."""

    def _make_candles_with_swing(self):
        """Create OHLCV data with a clear swing high and swing low."""
        n = 50
        # Price goes up to 110 at index 15, down to 90 at index 35
        prices = []
        for i in range(n):
            if i < 15:
                p = 100 + i * 0.67  # Rise to ~110
            elif i < 35:
                p = 110 - (i - 15) * 1.0  # Fall to ~90
            else:
                p = 90 + (i - 35) * 0.67  # Rise again

            prices.append(p)

        df = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=n, freq="15min", tz="UTC"),
            "open": [p - 0.5 for p in prices],
            "high": [p + 1.0 for p in prices],
            "low": [p - 1.0 for p in prices],
            "close": prices,
            "volume": [1000.0] * n,
        })
        return df

    def test_detect_swings_returns_list(self):
        df = self._make_candles_with_swing()
        swings = detect_swings(df, min_bars=3)
        assert isinstance(swings, list)

    def test_detect_swings_finds_highs_and_lows(self):
        df = self._make_candles_with_swing()
        swings = detect_swings(df, min_bars=3)
        types = {s["type"] for s in swings}
        # Should find at least one high and one low
        assert len(swings) >= 2
        assert "high" in types or "low" in types

    def test_swing_has_required_fields(self):
        df = self._make_candles_with_swing()
        swings = detect_swings(df, min_bars=3)
        if swings:
            s = swings[0]
            assert "type" in s
            assert "price" in s
            assert "index" in s
            assert "timestamp" in s

    def test_swing_high_price_is_correct(self):
        df = self._make_candles_with_swing()
        swings = detect_swings(df, min_bars=3)
        highs = [s for s in swings if s["type"] == "high"]
        if highs:
            # The swing high should be near the peak of our data (~111)
            assert highs[0]["price"] > 105

    def test_swing_low_price_is_correct(self):
        df = self._make_candles_with_swing()
        swings = detect_swings(df, min_bars=3)
        lows = [s for s in swings if s["type"] == "low"]
        if lows:
            # The swing low should be near the trough (~89)
            assert lows[0]["price"] < 95

    def test_more_min_bars_fewer_swings(self):
        """Larger min_bars should find fewer (more significant) swings."""
        df = self._make_candles_with_swing()
        swings_3 = detect_swings(df, min_bars=3)
        swings_7 = detect_swings(df, min_bars=7)
        assert len(swings_7) <= len(swings_3)

    def test_no_swings_in_flat_data(self):
        """Flat price data should produce no swings."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=50, freq="15min", tz="UTC"),
            "open": [100.0] * 50,
            "high": [100.5] * 50,
            "low": [99.5] * 50,
            "close": [100.0] * 50,
            "volume": [1000.0] * 50,
        })
        swings = detect_swings(df, min_bars=5)
        assert len(swings) == 0

    def test_swings_sorted_by_index(self):
        df = self._make_candles_with_swing()
        swings = detect_swings(df, min_bars=3)
        indices = [s["index"] for s in swings]
        assert indices == sorted(indices)

    def test_insufficient_data(self):
        """Too few candles should return empty list."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=5, freq="15min", tz="UTC"),
            "open": [100.0] * 5,
            "high": [101.0] * 5,
            "low": [99.0] * 5,
            "close": [100.0] * 5,
            "volume": [1000.0] * 5,
        })
        swings = detect_swings(df, min_bars=5)
        assert len(swings) == 0


class TestCalculateFibLevels:
    """Test Fibonacci level calculation."""

    def test_basic_fib_levels(self):
        levels = calculate_fib_levels(swing_high=100.0, swing_low=80.0)
        assert levels is not None
        assert len(levels) > 0

    def test_key_levels_present(self):
        levels = calculate_fib_levels(100.0, 80.0)
        for key in [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 0.886, 1.0]:
            assert key in levels

    def test_extension_levels_present(self):
        levels = calculate_fib_levels(100.0, 80.0)
        assert 1.272 in levels
        assert 1.618 in levels

    def test_level_0_is_swing_high(self):
        levels = calculate_fib_levels(100.0, 80.0)
        assert levels[0.0] == pytest.approx(100.0)

    def test_level_1_is_swing_low(self):
        levels = calculate_fib_levels(100.0, 80.0)
        assert levels[1.0] == pytest.approx(80.0)

    def test_levels_are_ordered(self):
        levels = calculate_fib_levels(100.0, 80.0)
        retracement_levels = [levels[r] for r in [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 0.886, 1.0]]
        # Levels should be decreasing (from high toward low)
        for i in range(len(retracement_levels) - 1):
            assert retracement_levels[i] >= retracement_levels[i + 1]

    def test_golden_pocket_values(self):
        levels = calculate_fib_levels(100.0, 80.0)
        # 0.618 of 20 = 12.36 → 100 - 12.36 = 87.64
        assert levels[0.618] == pytest.approx(87.64)
        # 0.886 of 20 = 17.72 → 100 - 17.72 = 82.28
        assert levels[0.886] == pytest.approx(82.28)

    def test_50_level(self):
        levels = calculate_fib_levels(100.0, 80.0)
        assert levels[0.5] == pytest.approx(90.0)

    def test_invalid_swing(self):
        """Swing high <= swing low should return empty."""
        levels = calculate_fib_levels(80.0, 100.0)
        assert levels == {}

    def test_real_btc_range(self):
        """Test with realistic BTC price range."""
        levels = calculate_fib_levels(105000.0, 95000.0)
        assert levels[0.618] == pytest.approx(105000 - 10000 * 0.618)
        assert levels[0.886] == pytest.approx(105000 - 10000 * 0.886)


class TestIsInGoldenPocket:
    """Test golden pocket detection."""

    def test_price_in_golden_pocket(self):
        levels = calculate_fib_levels(100.0, 80.0)
        # 0.618 level ≈ 87.64, 0.886 level ≈ 82.28
        assert is_in_golden_pocket(85.0, levels) is True

    def test_price_above_golden_pocket(self):
        levels = calculate_fib_levels(100.0, 80.0)
        assert is_in_golden_pocket(95.0, levels) is False

    def test_price_below_golden_pocket(self):
        levels = calculate_fib_levels(100.0, 80.0)
        assert is_in_golden_pocket(80.0, levels) is False

    def test_price_at_gp_boundary(self):
        levels = calculate_fib_levels(100.0, 80.0)
        # At the 0.618 level exactly
        assert is_in_golden_pocket(levels[0.618], levels) is True
        # At the 0.886 level exactly
        assert is_in_golden_pocket(levels[0.886], levels) is True

    def test_empty_levels(self):
        assert is_in_golden_pocket(100.0, {}) is False


class TestGoldenPocketDepth:
    """Test golden pocket depth measurement."""

    def test_depth_at_start(self):
        levels = calculate_fib_levels(100.0, 80.0)
        depth = golden_pocket_depth(levels[0.618], levels)
        assert depth is not None
        assert depth == pytest.approx(0.0, abs=0.05)

    def test_depth_at_end(self):
        levels = calculate_fib_levels(100.0, 80.0)
        depth = golden_pocket_depth(levels[0.886], levels)
        assert depth is not None
        assert depth == pytest.approx(1.0, abs=0.05)

    def test_depth_in_middle(self):
        levels = calculate_fib_levels(100.0, 80.0)
        mid = (levels[0.618] + levels[0.886]) / 2
        depth = golden_pocket_depth(mid, levels)
        assert depth is not None
        assert 0.3 < depth < 0.7

    def test_depth_outside_returns_none(self):
        levels = calculate_fib_levels(100.0, 80.0)
        assert golden_pocket_depth(95.0, levels) is None


class TestGetLatestSwingPair:
    """Test swing pair extraction for fib calculation."""

    def test_returns_pair(self):
        swings = [
            {"type": "low", "price": 90.0, "index": 5, "timestamp": None},
            {"type": "high", "price": 110.0, "index": 15, "timestamp": None},
        ]
        pair = get_latest_swing_pair(swings)
        assert pair is not None
        low_swing, high_swing = pair
        assert low_swing["price"] < high_swing["price"]

    def test_returns_none_insufficient_swings(self):
        swings = [{"type": "high", "price": 100.0, "index": 5, "timestamp": None}]
        assert get_latest_swing_pair(swings) is None

    def test_returns_none_empty(self):
        assert get_latest_swing_pair([]) is None

    def test_handles_same_type_swings(self):
        """Two consecutive highs should still find a pair with a prior low."""
        swings = [
            {"type": "low", "price": 90.0, "index": 5, "timestamp": None},
            {"type": "high", "price": 105.0, "index": 15, "timestamp": None},
            {"type": "high", "price": 110.0, "index": 25, "timestamp": None},
        ]
        pair = get_latest_swing_pair(swings)
        assert pair is not None
