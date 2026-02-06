"""Tests for risk manager — validates all risk enforcement rules."""

from datetime import datetime, timezone

import pytest

from src.risk.portfolio import Portfolio, Position, TradeResult
from src.risk.risk_manager import RiskManager
from src.strategy.signals import Signal


def _make_signal(**overrides) -> Signal:
    """Create a valid test signal with optional overrides."""
    defaults = dict(
        direction="LONG",
        confidence=0.8,
        risk_pct=0.01,
        entry_price=98000.0,
        stop_loss=96500.0,
        take_profits=[{"level": 0.5, "price": 99000.0, "pct": 0.3}],
        timeframe="15m",
        symbol="BTC/USDT:USDT",
        fib_levels={0.618: 97500.0},
        swing_high=101000.0,
        swing_low=95000.0,
        rsi_values={"current": 35.0},
        divergence_type="bullish",
        confluence_factors=[],
        timestamp=datetime.now(timezone.utc),
        reason="Test signal",
    )
    defaults.update(overrides)
    return Signal(**defaults)


def _make_position(**overrides) -> Position:
    """Create a test position."""
    defaults = dict(
        symbol="BTC/USDT:USDT",
        direction="LONG",
        entry_price=98000.0,
        size_usd=5000.0,
        size_contracts=0.051,
        leverage=3,
        stop_loss=96500.0,
        take_profits=[],
        margin_used=1666.67,
        opened_at=datetime.now(timezone.utc),
    )
    defaults.update(overrides)
    return Position(**defaults)


class TestRiskManagerApproval:
    """Test that valid signals are approved."""

    def test_valid_signal_approved(self):
        rm = RiskManager(paper_mode=True)
        portfolio = Portfolio(10000.0)
        signal = _make_signal()

        allowed, reason = rm.can_open_position(signal, portfolio)
        assert allowed is True
        assert reason == "approved"


class TestDailyLossLimit:
    """Test 3% daily loss halt."""

    def test_daily_loss_halts_trading(self):
        rm = RiskManager()
        portfolio = Portfolio(10000.0)

        # Simulate a 3%+ daily loss by closing losing trades
        pos = _make_position(margin_used=1000, size_usd=10000)
        portfolio.open_position(pos)
        # Close at a loss that makes daily P&L exceed -3%
        portfolio.close_position(pos, exit_price=95000, reason="stop_loss", fees=5)

        signal = _make_signal()
        allowed, reason = rm.can_open_position(signal, portfolio)
        assert allowed is False
        assert "daily loss limit" in reason.lower() or "Daily loss limit" in reason


class TestWeeklyLossLimit:
    """Test 7% weekly loss halt."""

    def test_weekly_loss_requires_manual_review(self):
        rm = RiskManager()
        portfolio = Portfolio(10000.0)

        # Simulate 7%+ weekly loss
        for _ in range(3):
            pos = _make_position(margin_used=1000, size_usd=10000)
            portfolio.open_position(pos)
            portfolio.close_position(pos, exit_price=95500, reason="stop_loss", fees=5)

        signal = _make_signal()
        allowed, reason = rm.can_open_position(signal, portfolio)
        # May hit daily limit first, but should be rejected
        assert allowed is False


class TestMaxConcurrentPositions:
    """Test max 3 concurrent positions."""

    def test_rejects_4th_position(self):
        rm = RiskManager()
        portfolio = Portfolio(10000.0)

        # Open 3 positions with small margin so we don't trigger daily loss limit
        for i in range(3):
            pos = _make_position(
                symbol=f"TEST{i}/USDT:USDT", margin_used=50, size_usd=150,
            )
            pos.current_price = pos.entry_price  # No unrealized PnL
            portfolio.open_position(pos)

        signal = _make_signal()
        allowed, reason = rm.can_open_position(signal, portfolio)
        assert allowed is False
        assert "concurrent" in reason.lower()

    def test_allows_after_closing(self):
        rm = RiskManager()
        portfolio = Portfolio(10000.0)

        positions = []
        for i in range(3):
            pos = _make_position(
                symbol=f"TEST{i}/USDT:USDT", margin_used=100, size_usd=300,
            )
            portfolio.open_position(pos)
            positions.append(pos)

        # Close one
        portfolio.close_position(positions[0], exit_price=98500, reason="take_profit")

        signal = _make_signal()
        allowed, reason = rm.can_open_position(signal, portfolio)
        assert allowed is True


class TestStopLossRequired:
    """Test that every signal must have a stop loss."""

    def test_no_stop_rejected(self):
        rm = RiskManager()
        portfolio = Portfolio(10000.0)
        signal = _make_signal(stop_loss=0)

        allowed, reason = rm.can_open_position(signal, portfolio)
        assert allowed is False
        assert "stop" in reason.lower()


class TestStopDistance:
    """Test max 5% stop distance."""

    def test_stop_too_far_rejected(self):
        rm = RiskManager()
        portfolio = Portfolio(10000.0)
        # Entry 98000, stop 90000 = 8.16% distance
        signal = _make_signal(entry_price=98000, stop_loss=90000)

        allowed, reason = rm.can_open_position(signal, portfolio)
        assert allowed is False
        assert "stop distance" in reason.lower() or "Stop distance" in reason


class TestRiskPctValid:
    """Test risk_pct must be 0.01 or 0.02."""

    def test_risk_too_high_rejected(self):
        rm = RiskManager()
        portfolio = Portfolio(10000.0)
        signal = _make_signal(risk_pct=0.05)

        allowed, reason = rm.can_open_position(signal, portfolio)
        assert allowed is False
        assert "risk pct" in reason.lower() or "Risk pct" in reason

    def test_risk_1pct_accepted(self):
        rm = RiskManager()
        portfolio = Portfolio(10000.0)
        signal = _make_signal(risk_pct=0.01)

        allowed, _ = rm.can_open_position(signal, portfolio)
        assert allowed is True

    def test_risk_2pct_accepted(self):
        rm = RiskManager()
        portfolio = Portfolio(10000.0)
        signal = _make_signal(risk_pct=0.02)

        allowed, _ = rm.can_open_position(signal, portfolio)
        assert allowed is True


class TestTakeProfitsRequired:
    """Test that signals must have take profit levels."""

    def test_no_tp_rejected(self):
        rm = RiskManager()
        portfolio = Portfolio(10000.0)
        signal = _make_signal(take_profits=[])

        allowed, reason = rm.can_open_position(signal, portfolio)
        assert allowed is False
        assert "take profit" in reason.lower()


class TestConsecutiveLossOverride:
    """Test that 3 consecutive losses reduce risk to 0.5%."""

    def test_consecutive_loss_reduces_risk(self):
        rm = RiskManager()
        portfolio = Portfolio(10000.0)

        # Record 3 consecutive losses
        for _ in range(3):
            pos = _make_position(margin_used=100, size_usd=300)
            portfolio.open_position(pos)
            portfolio.close_position(pos, exit_price=97000, reason="stop_loss")

        assert portfolio.consecutive_losses >= 3

        signal = _make_signal(risk_pct=0.01)
        allowed, reason = rm.can_open_position(signal, portfolio)
        assert allowed is True
        # Signal's risk should have been reduced
        assert signal.risk_pct == 0.005


class TestManualReset:
    """Test halt reset functionality."""

    def test_reset_clears_halt(self):
        rm = RiskManager()
        rm._manual_review_required = True

        portfolio = Portfolio(10000.0)
        signal = _make_signal()

        allowed, _ = rm.can_open_position(signal, portfolio)
        assert allowed is False

        rm.reset_halt()

        allowed, _ = rm.can_open_position(signal, portfolio)
        assert allowed is True
