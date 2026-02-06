"""Tests for the golden pocket strategy and signal generation."""

import numpy as np
import pandas as pd
import pytest

from src.strategy.golden_pocket import analyze_timeframe, score_confluence
from src.strategy.signals import Signal


def _make_strategy_candles(n=200):
    """
    Create synthetic OHLCV data with a swing high, pullback into golden pocket,
    and RSI divergence pattern. This simulates the ideal signal scenario.
    """
    np.random.seed(42)
    prices = []

    # Phase 1: Rise (0-60) — creating swing low → swing high
    for i in range(60):
        prices.append(95000 + i * 100 + np.random.randn() * 20)

    # Phase 2: Pullback into golden pocket (60-120)
    # Swing high ~101000, 0.618 retracement from 95000→101000 ≈ 97.3k
    for i in range(60):
        prices.append(101000 - i * 60 + np.random.randn() * 20)

    # Phase 3: Stabilize in golden pocket area (120-160)
    for i in range(40):
        prices.append(97500 + np.sin(i * 0.3) * 200 + np.random.randn() * 20)

    # Phase 4: Second dip — creates potential divergence (160-200)
    for i in range(40):
        if i < 20:
            prices.append(97300 - i * 30 + np.random.randn() * 20)
        else:
            prices.append(96700 + (i - 20) * 40 + np.random.randn() * 20)

    n = len(prices)
    df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="15min", tz="UTC"),
        "open": [p - 30 for p in prices],
        "high": [p + abs(np.random.randn() * 50) + 20 for p in prices],
        "low": [p - abs(np.random.randn() * 50) - 20 for p in prices],
        "close": prices,
        "volume": [1000 + abs(np.random.randn() * 200) for _ in range(n)],
    })
    return df


class TestAnalyzeTimeframe:
    """Test the full analysis pipeline."""

    def test_returns_list(self):
        candles = _make_strategy_candles()
        signals = analyze_timeframe(candles, "BTC/USDT:USDT", "15m")
        assert isinstance(signals, list)

    def test_signals_are_signal_objects(self):
        candles = _make_strategy_candles()
        signals = analyze_timeframe(candles, "BTC/USDT:USDT", "15m")
        for s in signals:
            assert isinstance(s, Signal)

    def test_signal_direction_valid(self):
        candles = _make_strategy_candles()
        signals = analyze_timeframe(candles, "BTC/USDT:USDT", "15m")
        for s in signals:
            assert s.direction in ("LONG", "SHORT")

    def test_signal_has_stop_loss(self):
        candles = _make_strategy_candles()
        signals = analyze_timeframe(candles, "BTC/USDT:USDT", "15m")
        for s in signals:
            assert s.stop_loss > 0
            # Stop should not be more than 5% from entry
            assert s.stop_distance_pct <= 0.051  # Small tolerance

    def test_signal_risk_pct_valid(self):
        candles = _make_strategy_candles()
        signals = analyze_timeframe(candles, "BTC/USDT:USDT", "15m")
        for s in signals:
            assert 0.01 <= s.risk_pct <= 0.02

    def test_signal_has_take_profits(self):
        candles = _make_strategy_candles()
        signals = analyze_timeframe(candles, "BTC/USDT:USDT", "15m")
        for s in signals:
            assert len(s.take_profits) > 0
            for tp in s.take_profits:
                assert "level" in tp
                assert "price" in tp
                assert "pct" in tp

    def test_insufficient_data_returns_empty(self):
        """Too few candles should return no signals."""
        small = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=10, freq="15min", tz="UTC"),
            "open": [100.0] * 10,
            "high": [101.0] * 10,
            "low": [99.0] * 10,
            "close": [100.0] * 10,
            "volume": [1000.0] * 10,
        })
        signals = analyze_timeframe(small, "BTC/USDT:USDT", "15m")
        assert signals == []


class TestScoreConfluence:
    """Test confluence scoring."""

    def test_base_score(self):
        """Base signal should return 0.7 confidence, 0.01 risk."""
        candles = _make_strategy_candles()
        div = {"index": 50, "confidence": 0.5, "confirmed": False}
        fib_levels = {0.618: 97500, 0.786: 96500, 0.886: 95500}

        conf, risk, factors = score_confluence(
            candles=candles,
            div=div,
            fib_levels=fib_levels,
            price=98000,  # Not in golden pocket → no depth bonus
            timeframe="15m",
        )
        assert conf >= 0.6
        assert risk >= 0.01

    def test_max_confluence_capped(self):
        """Risk should never exceed 0.02."""
        candles = _make_strategy_candles()
        div = {"index": 50, "confidence": 0.95, "confirmed": True}
        fib_levels = {0.618: 97500, 0.786: 96500, 0.886: 95500}

        conf, risk, factors = score_confluence(
            candles=candles,
            div=div,
            fib_levels=fib_levels,
            price=95800,  # Deep in GP
            timeframe="15m",
        )
        assert risk <= 0.02
        assert conf <= 1.0

    def test_factors_list_populated(self):
        candles = _make_strategy_candles()
        div = {"index": 50, "confidence": 0.9, "confirmed": True}
        fib_levels = {0.618: 97500, 0.786: 96500, 0.886: 95500}

        _, _, factors = score_confluence(
            candles=candles,
            div=div,
            fib_levels=fib_levels,
            price=95800,
            timeframe="15m",
        )
        assert isinstance(factors, list)


class TestSignalDataclass:
    """Test Signal dataclass properties."""

    def _make_signal(self):
        return Signal(
            direction="LONG",
            confidence=0.8,
            risk_pct=0.01,
            entry_price=98000.0,
            stop_loss=96500.0,
            take_profits=[
                {"level": 0.5, "price": 99000.0, "pct": 0.3},
                {"level": 0.382, "price": 100000.0, "pct": 0.3},
            ],
            timeframe="15m",
            symbol="BTC/USDT:USDT",
            fib_levels={0.618: 97500.0, 0.886: 95500.0},
            swing_high=101000.0,
            swing_low=95000.0,
            rsi_values={"current": 35.0, "peak1": 22.0, "peak2": 28.0},
            divergence_type="bullish",
            confluence_factors=["volume_confirmation"],
            timestamp=pd.Timestamp("2025-01-01", tz="UTC").to_pydatetime(),
            reason="Bullish RSI div at 98000",
        )

    def test_str_representation(self):
        s = self._make_signal()
        text = str(s)
        assert "LONG" in text
        assert "98000" in text

    def test_stop_distance_pct(self):
        s = self._make_signal()
        expected = abs(98000 - 96500) / 98000
        assert s.stop_distance_pct == pytest.approx(expected, rel=0.01)

    def test_risk_reward_ratio(self):
        s = self._make_signal()
        reward = 99000 - 98000  # TP1 - entry
        risk = 98000 - 96500    # entry - stop
        assert s.risk_reward_ratio == pytest.approx(reward / risk, rel=0.01)

    def test_risk_reward_no_tps(self):
        s = self._make_signal()
        s.take_profits = []
        assert s.risk_reward_ratio is None
