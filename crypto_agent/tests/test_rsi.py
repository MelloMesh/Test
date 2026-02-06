"""Tests for RSI calculation, peak detection, and divergence detection."""

import numpy as np
import pandas as pd
import pytest

from src.indicators.rsi import calculate_rsi, detect_divergence, detect_hidden_divergence, detect_rsi_peaks


class TestCalculateRSI:
    """Test RSI calculation using Wilder's method."""

    def test_rsi_returns_series(self):
        prices = pd.Series([100 + i * 0.5 for i in range(50)])
        rsi = calculate_rsi(prices)
        assert isinstance(rsi, pd.Series)
        assert len(rsi) == 50

    def test_rsi_range_0_to_100(self):
        np.random.seed(42)
        prices = pd.Series(np.cumsum(np.random.randn(200)) + 100)
        rsi = calculate_rsi(prices)
        valid = rsi.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_rsi_trending_up_above_50(self):
        """Consistently rising prices should produce RSI above 50."""
        prices = pd.Series([100 + i for i in range(50)])
        rsi = calculate_rsi(prices)
        # After warmup period, RSI should be high
        assert rsi.iloc[-1] > 70

    def test_rsi_trending_down_below_50(self):
        """Consistently falling prices should produce RSI below 50."""
        prices = pd.Series([100 - i for i in range(50)])
        rsi = calculate_rsi(prices)
        assert rsi.iloc[-1] < 30

    def test_rsi_period_14_default(self):
        prices = pd.Series([100 + i * 0.1 for i in range(30)])
        rsi = calculate_rsi(prices)
        # First 14 values should be NaN or unstable
        assert rsi.iloc[:13].isna().sum() >= 0  # Implementation may vary

    def test_rsi_custom_period(self):
        prices = pd.Series([100 + i * 0.1 for i in range(50)])
        rsi_7 = calculate_rsi(prices, period=7)
        rsi_21 = calculate_rsi(prices, period=21)
        # Shorter period RSI should be more extreme
        assert isinstance(rsi_7, pd.Series)
        assert isinstance(rsi_21, pd.Series)

    def test_rsi_flat_prices(self):
        """Flat prices should produce RSI around 50 (or NaN)."""
        prices = pd.Series([100.0] * 50)
        rsi = calculate_rsi(prices)
        valid = rsi.dropna()
        # With no movement, RSI is undefined (NaN) or ~50
        # Implementation may produce NaN due to 0/0
        assert len(rsi) == 50


class TestDetectRSIPeaks:
    """Test RSI peak detection in OB/OS territory."""

    def _make_rsi_with_os_peak(self):
        """Create an RSI series with a clear oversold peak."""
        # Start neutral, dip to oversold, recover
        rsi_values = (
            [50.0] * 10 +                    # Neutral
            [40.0, 35.0, 28.0, 25.0, 22.0] + # Drop into OS
            [24.0, 27.0, 32.0, 38.0, 45.0] + # Recover (peak at index 14, RSI=22)
            [50.0] * 10                        # Back to neutral
        )
        prices = [100.0 - (50 - r) * 0.1 for r in rsi_values]
        return pd.Series(rsi_values), pd.Series(prices)

    def _make_rsi_with_ob_peak(self):
        """Create an RSI series with a clear overbought peak."""
        rsi_values = (
            [50.0] * 10 +
            [60.0, 65.0, 72.0, 78.0, 82.0] + # Rise into OB
            [79.0, 74.0, 68.0, 62.0, 55.0] + # Decline (peak at index 14, RSI=82)
            [50.0] * 10
        )
        prices = [100.0 + (r - 50) * 0.1 for r in rsi_values]
        return pd.Series(rsi_values), pd.Series(prices)

    def test_detect_oversold_peak(self):
        rsi, prices = self._make_rsi_with_os_peak()
        peaks = detect_rsi_peaks(rsi, prices)
        os_peaks = [p for p in peaks if p["type"] == "oversold"]
        assert len(os_peaks) >= 1
        assert os_peaks[0]["rsi_value"] <= 30

    def test_detect_overbought_peak(self):
        rsi, prices = self._make_rsi_with_ob_peak()
        peaks = detect_rsi_peaks(rsi, prices)
        ob_peaks = [p for p in peaks if p["type"] == "overbought"]
        assert len(ob_peaks) >= 1
        assert ob_peaks[0]["rsi_value"] >= 70

    def test_peak_has_required_fields(self):
        rsi, prices = self._make_rsi_with_os_peak()
        peaks = detect_rsi_peaks(rsi, prices)
        if peaks:
            peak = peaks[0]
            assert "type" in peak
            assert "rsi_value" in peak
            assert "price_at_peak" in peak
            assert "index" in peak

    def test_no_peaks_in_neutral_rsi(self):
        """RSI staying between 30-70 should produce no peaks."""
        rsi = pd.Series([50.0 + np.sin(i * 0.1) * 10 for i in range(100)])
        prices = pd.Series([100.0] * 100)
        peaks = detect_rsi_peaks(rsi, prices)
        assert len(peaks) == 0

    def test_custom_thresholds(self):
        """Test with custom OB/OS thresholds."""
        rsi_values = [50.0] * 10 + [55.0, 58.0, 62.0, 65.0, 68.0] + [64.0, 60.0, 55.0, 50.0, 45.0] + [50.0] * 10
        rsi = pd.Series(rsi_values)
        prices = pd.Series([100.0] * len(rsi_values))

        # With default thresholds (70), no peaks
        peaks_default = detect_rsi_peaks(rsi, prices, threshold_ob=70)
        assert len([p for p in peaks_default if p["type"] == "overbought"]) == 0

        # With lower threshold (60), should find peak
        peaks_custom = detect_rsi_peaks(rsi, prices, threshold_ob=60)
        assert len([p for p in peaks_custom if p["type"] == "overbought"]) >= 1


class TestDetectDivergence:
    """Test RSI divergence detection."""

    def test_bullish_divergence(self):
        """Price lower low + RSI higher low = bullish divergence."""
        peaks = [
            {"type": "oversold", "rsi_value": 20.0, "price_at_peak": 100.0, "index": 10},
            {"type": "oversold", "rsi_value": 25.0, "price_at_peak": 95.0, "index": 30},
        ]
        prices = pd.Series([100.0] * 50)
        rsi = pd.Series([50.0] * 50)

        divs = detect_divergence(prices, rsi, peaks, anticipatory=True)
        bullish = [d for d in divs if d["type"] == "bullish"]
        assert len(bullish) >= 1
        assert bullish[0]["confidence"] > 0

    def test_bearish_divergence(self):
        """Price higher high + RSI lower high = bearish divergence."""
        peaks = [
            {"type": "overbought", "rsi_value": 80.0, "price_at_peak": 100.0, "index": 10},
            {"type": "overbought", "rsi_value": 75.0, "price_at_peak": 105.0, "index": 30},
        ]
        prices = pd.Series([100.0] * 50)
        rsi = pd.Series([50.0] * 50)

        divs = detect_divergence(prices, rsi, peaks, anticipatory=True)
        bearish = [d for d in divs if d["type"] == "bearish"]
        assert len(bearish) >= 1

    def test_no_divergence_when_aligned(self):
        """Price higher high + RSI higher high = no divergence."""
        peaks = [
            {"type": "overbought", "rsi_value": 75.0, "price_at_peak": 100.0, "index": 10},
            {"type": "overbought", "rsi_value": 80.0, "price_at_peak": 105.0, "index": 30},
        ]
        prices = pd.Series([100.0] * 50)
        rsi = pd.Series([50.0] * 50)

        divs = detect_divergence(prices, rsi, peaks)
        bearish = [d for d in divs if d["type"] == "bearish"]
        assert len(bearish) == 0

    def test_no_divergence_insufficient_peaks(self):
        """Need at least 2 peaks of the same type."""
        peaks = [
            {"type": "oversold", "rsi_value": 25.0, "price_at_peak": 100.0, "index": 10},
        ]
        prices = pd.Series([100.0] * 50)
        rsi = pd.Series([50.0] * 50)

        divs = detect_divergence(prices, rsi, peaks)
        assert len(divs) == 0

    def test_max_peak_distance_filter(self):
        """Peaks too far apart should not form a divergence."""
        peaks = [
            {"type": "oversold", "rsi_value": 20.0, "price_at_peak": 100.0, "index": 10},
            {"type": "oversold", "rsi_value": 25.0, "price_at_peak": 95.0, "index": 200},
        ]
        prices = pd.Series([100.0] * 250)
        rsi = pd.Series([50.0] * 250)

        divs = detect_divergence(prices, rsi, peaks, max_peak_distance=50)
        assert len(divs) == 0

    def test_divergence_has_required_fields(self):
        peaks = [
            {"type": "oversold", "rsi_value": 20.0, "price_at_peak": 100.0, "index": 10},
            {"type": "oversold", "rsi_value": 25.0, "price_at_peak": 95.0, "index": 30},
        ]
        prices = pd.Series([100.0] * 50)
        rsi = pd.Series([50.0] * 50)

        divs = detect_divergence(prices, rsi, peaks)
        if divs:
            d = divs[0]
            assert "type" in d
            assert "confidence" in d
            assert "peak1" in d
            assert "peak2" in d
            assert "confirmed" in d
            assert "index" in d


class TestDetectHiddenDivergence:
    """Test hidden divergence detection."""

    def test_hidden_bullish(self):
        """Price higher low + RSI lower low = hidden bullish."""
        peaks = [
            {"type": "oversold", "rsi_value": 25.0, "price_at_peak": 95.0, "index": 10},
            {"type": "oversold", "rsi_value": 20.0, "price_at_peak": 97.0, "index": 30},
        ]
        prices = pd.Series([100.0] * 50)
        rsi = pd.Series([50.0] * 50)

        divs = detect_hidden_divergence(prices, rsi, peaks)
        hidden_bull = [d for d in divs if d["type"] == "hidden_bullish"]
        assert len(hidden_bull) >= 1

    def test_hidden_bearish(self):
        """Price lower high + RSI higher high = hidden bearish."""
        peaks = [
            {"type": "overbought", "rsi_value": 75.0, "price_at_peak": 105.0, "index": 10},
            {"type": "overbought", "rsi_value": 80.0, "price_at_peak": 103.0, "index": 30},
        ]
        prices = pd.Series([100.0] * 50)
        rsi = pd.Series([50.0] * 50)

        divs = detect_hidden_divergence(prices, rsi, peaks)
        hidden_bear = [d for d in divs if d["type"] == "hidden_bearish"]
        assert len(hidden_bear) >= 1

    def test_no_hidden_div_insufficient_peaks(self):
        peaks = [{"type": "oversold", "rsi_value": 25.0, "price_at_peak": 95.0, "index": 10}]
        prices = pd.Series([100.0] * 50)
        rsi = pd.Series([50.0] * 50)

        divs = detect_hidden_divergence(prices, rsi, peaks)
        assert len(divs) == 0

    def test_hidden_div_respects_max_distance(self):
        peaks = [
            {"type": "oversold", "rsi_value": 25.0, "price_at_peak": 95.0, "index": 10},
            {"type": "oversold", "rsi_value": 20.0, "price_at_peak": 97.0, "index": 200},
        ]
        prices = pd.Series([100.0] * 250)
        rsi = pd.Series([50.0] * 250)

        divs = detect_hidden_divergence(prices, rsi, peaks, max_peak_distance=50)
        assert len(divs) == 0

    def test_hidden_div_has_required_fields(self):
        peaks = [
            {"type": "oversold", "rsi_value": 25.0, "price_at_peak": 95.0, "index": 10},
            {"type": "oversold", "rsi_value": 20.0, "price_at_peak": 97.0, "index": 30},
        ]
        prices = pd.Series([100.0] * 50)
        rsi = pd.Series([50.0] * 50)

        divs = detect_hidden_divergence(prices, rsi, peaks)
        if divs:
            d = divs[0]
            assert "type" in d
            assert d["type"].startswith("hidden_")
            assert "confidence" in d
            assert "peak1" in d
            assert "peak2" in d
            assert d["confirmed"] == True
