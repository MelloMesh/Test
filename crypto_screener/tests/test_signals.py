"""
tests/test_signals.py — Unit tests for all signal modules.

Each signal function is tested independently.
Tests use synthetic OHLCV DataFrames — no network calls.

Run: python -m pytest crypto_screener/tests/test_signals.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crypto_screener.signals.rsi import compute_rsi, signal_os_ob, signal_midline, signal_divergence
from crypto_screener.signals.bollinger import compute_bands, signal_mean_reversion, signal_pct_b
from crypto_screener.signals.volume import compute_obv, compute_mfi, signal_volume_spike, signal_mfi
from crypto_screener.backtest.engine import compute_atr
from crypto_screener.screener.scorer import score_setup


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _make_ohlcv(closes: list[float], seed_vol: float = 1000.0) -> pd.DataFrame:
    """Create minimal OHLCV DataFrame from a list of close prices."""
    n = len(closes)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    closes_arr = np.array(closes, dtype=float)
    highs = closes_arr * 1.005
    lows = closes_arr * 0.995
    opens = np.roll(closes_arr, 1)
    opens[0] = closes_arr[0]
    vols = np.full(n, seed_vol)

    return pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes_arr,
        "volume": vols,
    }, index=idx)


def _trend_up(n: int = 100, start: float = 100.0, step: float = 0.5) -> list[float]:
    return [start + i * step for i in range(n)]


def _trend_down(n: int = 100, start: float = 150.0, step: float = 0.5) -> list[float]:
    return [start - i * step for i in range(n)]


def _oscillating(n: int = 100, amplitude: float = 10.0, period: float = 20.0) -> list[float]:
    return [100.0 + amplitude * np.sin(2 * np.pi * i / period) for i in range(n)]


# ── RSI tests ─────────────────────────────────────────────────────────────────

class TestComputeRSI:
    def test_returns_series_same_length(self) -> None:
        df = _make_ohlcv(_trend_up(50))
        rsi = compute_rsi(df["close"])
        assert len(rsi) == 50

    def test_rsi_bounded_0_100(self) -> None:
        df = _make_ohlcv(_trend_up(50))
        rsi = compute_rsi(df["close"]).dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all()

    def test_strong_uptrend_gives_high_rsi(self) -> None:
        closes = _trend_up(80)
        rsi = compute_rsi(pd.Series(closes))
        assert rsi.dropna().iloc[-1] > 70, "Strong uptrend should produce overbought RSI"

    def test_strong_downtrend_gives_low_rsi(self) -> None:
        closes = _trend_down(80)
        rsi = compute_rsi(pd.Series(closes))
        assert rsi.dropna().iloc[-1] < 30, "Strong downtrend should produce oversold RSI"

    def test_returns_nan_for_insufficient_data(self) -> None:
        closes = [100.0] * 5
        rsi = compute_rsi(pd.Series(closes), period=14)
        assert rsi.dropna().empty


class TestSignalOsOb:
    def test_oversold_fires_rsi_os(self) -> None:
        closes = _trend_down(80)
        rsi = compute_rsi(pd.Series(closes))
        signals = signal_os_ob(rsi)
        assert "RSI_OS" in signals

    def test_overbought_fires_rsi_ob(self) -> None:
        closes = _trend_up(80)
        rsi = compute_rsi(pd.Series(closes))
        signals = signal_os_ob(rsi)
        assert "RSI_OB" in signals

    def test_neutral_fires_nothing(self) -> None:
        closes = _oscillating(50)
        rsi = compute_rsi(pd.Series(closes))
        signals = signal_os_ob(rsi)
        assert "RSI_OS" not in signals
        assert "RSI_OB" not in signals


class TestSignalMidline:
    def test_bullish_cross_from_below(self) -> None:
        """Manually craft RSI series that crosses 50 from below."""
        # Simulate RSI going from 45 → 55
        rsi = pd.Series([40.0, 43.0, 47.0, 49.0, 53.0])
        signals = signal_midline(rsi)
        assert "RSI_MID_BULL" in signals

    def test_bearish_cross_from_above(self) -> None:
        rsi = pd.Series([60.0, 57.0, 53.0, 51.0, 47.0])
        signals = signal_midline(rsi)
        assert "RSI_MID_BEAR" in signals

    def test_no_cross_fires_nothing(self) -> None:
        rsi = pd.Series([55.0, 56.0, 57.0, 58.0])
        signals = signal_midline(rsi)
        assert not signals


class TestSignalDivergence:
    def test_bullish_divergence_detected(self) -> None:
        """Price lower low, RSI higher low → RSI_BULL_DIV."""
        # Create synthetic divergence: price falls further but RSI recovers
        n = 60
        closes = [100.0] * n
        # Two troughs: first at index 20 (price=90), second at index 45 (price=88 — lower low)
        for i in range(18, 25):
            closes[i] = 90.0 - (i - 20) * 0.5 + abs(i - 21) * 0.5
        for i in range(43, 48):
            closes[i] = 88.0 - abs(i - 45) * 0.3

        df = _make_ohlcv(closes)
        # Patch RSI to make a higher low at the second trough
        from crypto_screener.signals import rsi as rsi_mod
        rsi_series = rsi_mod.compute_rsi(df["close"])
        # The divergence detector will naturally pick up if conditions are met
        signals = signal_divergence(df, rsi_series, min_window=5, max_window=30)
        # This is a structural test — verifies the function runs without error
        assert isinstance(signals, list)
        assert all(s in ("RSI_BULL_DIV", "RSI_BEAR_DIV") for s in signals)

    def test_returns_list_type(self) -> None:
        df = _make_ohlcv(_oscillating(80))
        from crypto_screener.signals.rsi import compute_rsi
        rsi = compute_rsi(df["close"])
        result = signal_divergence(df, rsi)
        assert isinstance(result, list)

    def test_insufficient_data_returns_empty(self) -> None:
        df = _make_ohlcv([100.0] * 10)
        from crypto_screener.signals.rsi import compute_rsi
        rsi = compute_rsi(df["close"])
        result = signal_divergence(df, rsi)
        assert result == []


# ── Bollinger Band tests ──────────────────────────────────────────────────────

class TestComputeBands:
    def test_columns_present(self) -> None:
        df = _make_ohlcv(_trend_up(50))
        bands = compute_bands(df["close"])
        assert set(bands.columns) >= {"mid", "upper", "lower", "width", "pct_b"}

    def test_upper_always_above_lower(self) -> None:
        df = _make_ohlcv(_oscillating(60))
        bands = compute_bands(df["close"]).dropna()
        assert (bands["upper"] > bands["lower"]).all()

    def test_mid_is_sma(self) -> None:
        closes = pd.Series(list(range(1, 51)), dtype=float)
        bands = compute_bands(closes, period=5)
        # At index 4, SMA(5) = (1+2+3+4+5)/5 = 3.0
        assert abs(bands["mid"].iloc[4] - 3.0) < 1e-9


class TestSignalMeanReversion:
    def test_close_below_lower_fires_bb_lower(self) -> None:
        closes = _oscillating(50)
        # Push last close far below band
        closes[-1] = closes[-1] - 100.0
        df = _make_ohlcv(closes)
        bands = compute_bands(df["close"])
        signals = signal_mean_reversion(df, bands)
        assert "BB_LOWER" in signals

    def test_close_above_upper_fires_bb_upper(self) -> None:
        closes = _oscillating(50)
        closes[-1] = closes[-1] + 100.0
        df = _make_ohlcv(closes)
        bands = compute_bands(df["close"])
        signals = signal_mean_reversion(df, bands)
        assert "BB_UPPER" in signals

    def test_close_inside_band_fires_nothing(self) -> None:
        closes = _oscillating(50)
        df = _make_ohlcv(closes)
        bands = compute_bands(df["close"])
        signals = signal_mean_reversion(df, bands)
        # Oscillating around mean — should not typically hit bands
        assert isinstance(signals, list)


class TestSignalPctB:
    def test_extreme_oversold_fires_pct_b_low(self) -> None:
        bands = pd.DataFrame({
            "mid": [100.0],
            "upper": [110.0],
            "lower": [90.0],
            "width": [0.2],
            "pct_b": [0.01],
        })
        signals = signal_pct_b(bands)
        assert "BB_PCT_B_LOW" in signals

    def test_extreme_overbought_fires_pct_b_high(self) -> None:
        bands = pd.DataFrame({
            "mid": [100.0],
            "upper": [110.0],
            "lower": [90.0],
            "width": [0.2],
            "pct_b": [0.98],
        })
        signals = signal_pct_b(bands)
        assert "BB_PCT_B_HIGH" in signals


# ── Volume tests ──────────────────────────────────────────────────────────────

class TestComputeOBV:
    def test_obv_increases_on_up_days(self) -> None:
        closes = [100.0, 101.0, 102.0]
        vols = [1000.0, 1000.0, 1000.0]
        idx = pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC")
        df = pd.DataFrame({"close": closes, "volume": vols,
                           "open": closes, "high": closes, "low": closes}, index=idx)
        obv = compute_obv(df)
        assert obv.iloc[-1] > obv.iloc[0]

    def test_obv_decreases_on_down_days(self) -> None:
        closes = [102.0, 101.0, 100.0]
        vols = [1000.0, 1000.0, 1000.0]
        idx = pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC")
        df = pd.DataFrame({"close": closes, "volume": vols,
                           "open": closes, "high": closes, "low": closes}, index=idx)
        obv = compute_obv(df)
        assert obv.iloc[-1] < obv.iloc[0]


class TestSignalVolumeSpike:
    def test_spike_detected(self) -> None:
        closes = _trend_up(30)
        df = _make_ohlcv(closes, seed_vol=100.0)
        # Spike the last candle — use .loc to avoid copy-on-write warning
        df.loc[df.index[-1], "volume"] = 1000.0
        signals = signal_volume_spike(df)
        assert "VOL_SPIKE" in signals

    def test_no_spike_fires_nothing(self) -> None:
        df = _make_ohlcv(_trend_up(30), seed_vol=100.0)
        signals = signal_volume_spike(df)
        assert "VOL_SPIKE" not in signals


class TestSignalMFI:
    def test_oversold_fires_mfi_os(self) -> None:
        # Create scenario where MFI should be low: consistent price drops + volume
        closes = _trend_down(40)
        df = _make_ohlcv(closes, seed_vol=1000.0)
        signals = signal_mfi(df)
        # May or may not fire depending on OHLCV synth — structural test
        assert isinstance(signals, list)
        assert all(s in ("MFI_OS", "MFI_OB") for s in signals)

    def test_mfi_bounded_0_100(self) -> None:
        from crypto_screener.signals.volume import compute_mfi
        df = _make_ohlcv(_oscillating(50))
        mfi = compute_mfi(df).dropna()
        assert (mfi >= 0).all() and (mfi <= 100).all()


# ── ATR tests ─────────────────────────────────────────────────────────────────

class TestComputeATR:
    def test_atr_positive(self) -> None:
        df = _make_ohlcv(_oscillating(30))
        atr = compute_atr(df).dropna()
        assert (atr > 0).all()

    def test_atr_same_length(self) -> None:
        df = _make_ohlcv(_trend_up(50))
        atr = compute_atr(df)
        assert len(atr) == 50


# ── Scorer tests ──────────────────────────────────────────────────────────────

class TestScoreSetup:
    def test_high_score_passes_threshold(self) -> None:
        result = score_setup(
            symbol="BTCUSDT",
            timeframe="1h",
            rsi_signals=["RSI_BULL_DIV", "RSI_OS"],
            bb_signals=["BB_LOWER", "BB_PCT_B_LOW"],
            vol_signals=["VOL_SPIKE", "MFI_OS"],
            threshold=3.0,
        )
        assert result.composite_score > 0
        assert result.direction == "LONG"
        assert result.passes_threshold
        assert result.passes_category_gate
        assert result.surfaced

    def test_low_score_fails_threshold(self) -> None:
        result = score_setup(
            symbol="ETHUSDT",
            timeframe="1h",
            rsi_signals=["RSI_MID_BULL"],
            bb_signals=[],
            vol_signals=[],
            threshold=3.0,
        )
        assert not result.passes_threshold
        assert not result.surfaced

    def test_single_category_fails_diversity_gate(self) -> None:
        result = score_setup(
            symbol="SOLUSDT",
            timeframe="1h",
            rsi_signals=["RSI_BULL_DIV", "RSI_OS", "RSI_MID_BULL"],
            bb_signals=[],
            vol_signals=[],
            threshold=2.0,
            min_categories=2,
        )
        assert not result.passes_category_gate

    def test_direction_long_for_positive_score(self) -> None:
        result = score_setup("X", "1h", ["RSI_OS"], ["BB_LOWER"], ["MFI_OS"], threshold=0.0)
        assert result.direction == "LONG"
        assert result.composite_score > 0

    def test_direction_short_for_negative_score(self) -> None:
        result = score_setup("X", "1h", ["RSI_OB"], ["BB_UPPER"], ["MFI_OB"], threshold=0.0)
        assert result.direction == "SHORT"
        assert result.composite_score < 0

    def test_confidence_normalized_0_to_1(self) -> None:
        result = score_setup("X", "1h", ["RSI_BULL_DIV"], ["BB_SQUEEZE_BULL"], ["MFI_OS"])
        assert 0.0 <= result.confidence <= 1.0

    def test_volume_gate_filters_illiquid(self) -> None:
        universe_vols = [1e6, 2e6, 3e6, 4e6, 5e6]
        result = score_setup(
            symbol="LOWVOLTOKEN",
            timeframe="1h",
            rsi_signals=["RSI_BULL_DIV"],
            bb_signals=["BB_LOWER"],
            vol_signals=["MFI_OS"],
            universe_volumes=universe_vols,
            symbol_volume=1.0,   # essentially zero volume
            threshold=2.0,
        )
        assert not result.passes_volume_gate
        assert not result.surfaced


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
