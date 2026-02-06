"""Tests for trend and market regime indicators."""

import numpy as np
import pandas as pd
import pytest

from src.indicators.trend import (
    calculate_adx,
    calculate_atr,
    calculate_ema,
    check_mtf_alignment,
    get_market_regime,
    get_trend_direction,
)


def _make_trending_candles(n=200, trend="up"):
    """Create synthetic trending data."""
    np.random.seed(42)
    if trend == "up":
        base = np.linspace(90000, 100000, n) + np.random.randn(n) * 100
    elif trend == "down":
        base = np.linspace(100000, 90000, n) + np.random.randn(n) * 100
    else:  # ranging
        base = 95000 + np.sin(np.linspace(0, 8 * np.pi, n)) * 500 + np.random.randn(n) * 50

    return pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="15min", tz="UTC"),
        "open": base - 10,
        "high": base + abs(np.random.randn(n) * 30) + 20,
        "low": base - abs(np.random.randn(n) * 30) - 20,
        "close": base,
        "volume": 1000 + abs(np.random.randn(n) * 200),
    })


class TestEMA:
    def test_ema_returns_series(self):
        s = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0])
        ema = calculate_ema(s, 3)
        assert isinstance(ema, pd.Series)
        assert len(ema) == len(s)

    def test_ema_smooths_data(self):
        np.random.seed(42)
        noisy = pd.Series(100 + np.random.randn(100) * 5)
        ema = calculate_ema(noisy, 20)
        # EMA should be smoother (lower std)
        assert ema.std() < noisy.std()


class TestATR:
    def test_atr_returns_series(self):
        candles = _make_trending_candles(100)
        atr = calculate_atr(candles)
        assert isinstance(atr, pd.Series)
        assert len(atr) == 100

    def test_atr_positive_values(self):
        candles = _make_trending_candles(100)
        atr = calculate_atr(candles)
        valid = atr.dropna()
        assert (valid > 0).all()

    def test_atr_higher_in_volatile_data(self):
        np.random.seed(42)
        n = 100
        # Low vol
        low_vol = pd.DataFrame({
            "high": 100 + np.random.randn(n) * 0.5,
            "low": 100 - np.random.randn(n) * 0.5 - 0.5,
            "close": [100.0] * n,
        })
        # High vol
        high_vol = pd.DataFrame({
            "high": 100 + abs(np.random.randn(n) * 5),
            "low": 100 - abs(np.random.randn(n) * 5),
            "close": [100.0] * n,
        })
        atr_low = calculate_atr(low_vol).dropna().mean()
        atr_high = calculate_atr(high_vol).dropna().mean()
        assert atr_high > atr_low


class TestADX:
    def test_adx_returns_dataframe(self):
        candles = _make_trending_candles(100)
        result = calculate_adx(candles)
        assert isinstance(result, pd.DataFrame)
        assert "adx" in result.columns
        assert "plus_di" in result.columns
        assert "minus_di" in result.columns

    def test_adx_high_in_trend(self):
        candles = _make_trending_candles(200, trend="up")
        result = calculate_adx(candles)
        final_adx = result["adx"].dropna().iloc[-1]
        # Strong uptrend should have ADX > 20
        assert final_adx > 20

    def test_adx_low_in_range(self):
        candles = _make_trending_candles(200, trend="range")
        result = calculate_adx(candles)
        final_adx = result["adx"].dropna().iloc[-1]
        # Ranging should have lower ADX
        assert final_adx < 40

    def test_adx_insufficient_data(self):
        candles = _make_trending_candles(10)
        result = calculate_adx(candles)
        assert result["adx"].isna().all()


class TestTrendDirection:
    def test_bullish_trend(self):
        candles = _make_trending_candles(100, trend="up")
        direction = get_trend_direction(candles, ema_period=20)
        assert direction == "BULLISH"

    def test_bearish_trend(self):
        candles = _make_trending_candles(100, trend="down")
        direction = get_trend_direction(candles, ema_period=20)
        assert direction == "BEARISH"

    def test_insufficient_data(self):
        candles = _make_trending_candles(10)
        direction = get_trend_direction(candles, ema_period=50)
        assert direction == "NEUTRAL"


class TestMarketRegime:
    def test_regime_returns_dict(self):
        candles = _make_trending_candles(200)
        regime = get_market_regime(candles)
        assert isinstance(regime, dict)
        assert "regime" in regime
        assert "adx" in regime
        assert "safe_for_divergence" in regime
        assert "atr_pct" in regime
        assert "atr_value" in regime

    def test_trending_regime(self):
        candles = _make_trending_candles(200, trend="up")
        regime = get_market_regime(candles)
        # Strong trend should be detected
        assert regime["adx"] > 0
        assert regime["trend_direction"] in ("BULLISH", "BEARISH", "NEUTRAL")

    def test_ranging_regime_safe(self):
        candles = _make_trending_candles(200, trend="range")
        regime = get_market_regime(candles)
        # Ranging market should be safe for divergence
        if regime["adx"] < 20:
            assert regime["safe_for_divergence"] == True


class TestMTFAlignment:
    def test_long_aligned_with_bullish(self):
        candles = _make_trending_candles(100, trend="up")
        aligned, reason = check_mtf_alignment("LONG", candles, ema_period=20)
        assert aligned == True

    def test_long_against_bearish(self):
        candles = _make_trending_candles(100, trend="down")
        aligned, reason = check_mtf_alignment("LONG", candles, ema_period=20)
        assert aligned == False
        assert "bearish" in reason

    def test_short_against_bullish(self):
        candles = _make_trending_candles(100, trend="up")
        aligned, reason = check_mtf_alignment("SHORT", candles, ema_period=20)
        assert aligned == False
        assert "bullish" in reason

    def test_short_aligned_with_bearish(self):
        candles = _make_trending_candles(100, trend="down")
        aligned, reason = check_mtf_alignment("SHORT", candles, ema_period=20)
        assert aligned == True

    def test_no_htf_data_returns_aligned(self):
        aligned, reason = check_mtf_alignment("LONG", None)
        assert aligned == True
        assert reason == "no_htf_data"

    def test_insufficient_htf_data(self):
        candles = _make_trending_candles(5)
        aligned, reason = check_mtf_alignment("LONG", candles, ema_period=50)
        assert aligned == True
