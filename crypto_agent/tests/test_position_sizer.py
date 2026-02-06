"""Tests for position sizing module."""

import pytest

from src.risk.position_sizer import calculate_position_size


class TestCalculatePositionSize:
    """Test position size calculation from risk and stop parameters."""

    def test_basic_long_position(self):
        """Standard long: 1% risk, BTC at 98000, stop at 96500."""
        result = calculate_position_size(
            equity=10000, risk_pct=0.01, entry_price=98000,
            stop_loss_price=96500, leverage=3,
        )
        assert result["risk_amount"] == pytest.approx(100.0)  # 1% of 10000
        assert result["stop_distance_pct"] == pytest.approx(
            abs(98000 - 96500) / 98000, rel=0.01
        )
        assert result["size_usd"] > 0
        assert result["margin_required"] == pytest.approx(result["size_usd"] / 3, rel=0.01)
        assert result["leverage"] == 3

    def test_risk_amount_correct(self):
        """Risk amount should be exactly equity * risk_pct."""
        result = calculate_position_size(
            equity=50000, risk_pct=0.02, entry_price=100000,
            stop_loss_price=97000, leverage=5,
        )
        assert result["risk_amount"] == pytest.approx(1000.0)  # 2% of 50000

    def test_size_formula(self):
        """size_usd = risk_amount / stop_distance_pct."""
        equity = 10000
        risk_pct = 0.01
        entry = 100000
        stop = 97000
        stop_dist = abs(entry - stop) / entry  # 3%

        result = calculate_position_size(
            equity=equity, risk_pct=risk_pct, entry_price=entry,
            stop_loss_price=stop, leverage=1,
        )
        expected_size = (equity * risk_pct) / stop_dist
        assert result["size_usd"] == pytest.approx(expected_size, rel=0.01)

    def test_5pct_stop_cap(self):
        """Stop distance exceeding 5% should be capped."""
        result = calculate_position_size(
            equity=10000, risk_pct=0.01, entry_price=98000,
            stop_loss_price=90000, leverage=3,  # ~8.2% away
        )
        assert result["stop_distance_pct"] == pytest.approx(0.05, rel=0.01)

    def test_5pct_stop_cap_short(self):
        """Short position stop cap."""
        result = calculate_position_size(
            equity=10000, risk_pct=0.01, entry_price=98000,
            stop_loss_price=108000, leverage=3,  # ~10.2% away
        )
        assert result["stop_distance_pct"] == pytest.approx(0.05, rel=0.01)

    def test_contracts_calculation(self):
        """size_contracts = size_usd / entry_price."""
        result = calculate_position_size(
            equity=10000, risk_pct=0.01, entry_price=100000,
            stop_loss_price=97000, leverage=1,
        )
        assert result["size_contracts"] == pytest.approx(
            result["size_usd"] / 100000, rel=0.01
        )

    def test_margin_with_leverage(self):
        """margin_required = size_usd / leverage."""
        for lev in [1, 3, 5, 10]:
            result = calculate_position_size(
                equity=10000, risk_pct=0.01, entry_price=100000,
                stop_loss_price=97000, leverage=lev,
            )
            assert result["margin_required"] == pytest.approx(
                result["size_usd"] / lev, rel=0.01
            )

    def test_liquidation_price_long(self):
        """Long liquidation = entry * (1 - 1/leverage)."""
        result = calculate_position_size(
            equity=10000, risk_pct=0.01, entry_price=100000,
            stop_loss_price=97000, leverage=5,
        )
        expected_liq = 100000 * (1 - 1/5)  # 80000
        assert result["liquidation_price"] == pytest.approx(expected_liq, rel=0.01)

    def test_liquidation_price_short(self):
        """Short liquidation = entry * (1 + 1/leverage)."""
        result = calculate_position_size(
            equity=10000, risk_pct=0.01, entry_price=100000,
            stop_loss_price=103000, leverage=5,
        )
        expected_liq = 100000 * (1 + 1/5)  # 120000
        assert result["liquidation_price"] == pytest.approx(expected_liq, rel=0.01)

    def test_invalid_equity_raises(self):
        with pytest.raises(ValueError, match="Equity must be positive"):
            calculate_position_size(
                equity=0, risk_pct=0.01, entry_price=100000,
                stop_loss_price=97000,
            )

    def test_invalid_risk_pct_raises(self):
        with pytest.raises(ValueError, match="Risk pct"):
            calculate_position_size(
                equity=10000, risk_pct=0.05, entry_price=100000,
                stop_loss_price=97000,
            )

    def test_invalid_leverage_raises(self):
        with pytest.raises(ValueError, match="Leverage"):
            calculate_position_size(
                equity=10000, risk_pct=0.01, entry_price=100000,
                stop_loss_price=97000, leverage=20,
            )

    def test_realistic_golden_pocket_scenario(self):
        """
        Real scenario: BTC swing 95000→105000.
        Golden pocket 0.618 = 98820, 0.886 = 96140.
        Entry at 97500, stop below 0.886 at 96000.
        """
        result = calculate_position_size(
            equity=10000, risk_pct=0.01, entry_price=97500,
            stop_loss_price=96000, leverage=3,
        )
        # Stop distance ≈ 1.54%
        assert result["stop_distance_pct"] < 0.05
        # Risk = $100
        assert result["risk_amount"] == pytest.approx(100.0)
        # Position ≈ $100 / 0.0154 ≈ $6500
        assert 5000 < result["size_usd"] < 8000
