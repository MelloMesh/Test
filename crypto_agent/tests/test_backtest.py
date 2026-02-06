"""Tests for the backtesting engine."""

import numpy as np
import pandas as pd
import pytest

from src.backtest.costs import (
    calculate_entry_cost,
    calculate_exit_cost,
    calculate_funding_cost,
    calculate_round_trip_cost,
)
from src.backtest.engine import run_backtest, _calculate_pnl
from src.backtest.report import generate_report, _calculate_max_drawdown
from src.backtest.stop_manager import check_stop_hit, check_tp_hit
from src.backtest.tp_optimizer import config_to_tp_list, evaluate_config, get_tp_configs


class TestCosts:
    """Test trading cost calculations."""

    def test_entry_cost_taker(self):
        cost = calculate_entry_cost(10000, is_taker=True)
        # 0.04% fee + 0.03% slippage = 0.07% of 10000 = 7
        assert cost == pytest.approx(7.0, rel=0.1)

    def test_entry_cost_maker(self):
        cost = calculate_entry_cost(10000, is_taker=False)
        # 0.02% fee + 0.03% slippage = 0.05% of 10000 = 5
        assert cost == pytest.approx(5.0, rel=0.1)

    def test_round_trip_cost(self):
        cost = calculate_round_trip_cost(10000)
        # Both sides taker: 2 * 7 = 14
        assert cost == pytest.approx(14.0, rel=0.1)

    def test_funding_cost_long_positive_rate(self):
        """Long pays when funding rate is positive."""
        cost = calculate_funding_cost(10000, 0.0001, "LONG", periods=1)
        assert cost == pytest.approx(1.0)  # 10000 * 0.0001

    def test_funding_cost_short_positive_rate(self):
        """Short receives when funding rate is positive."""
        cost = calculate_funding_cost(10000, 0.0001, "SHORT", periods=1)
        assert cost == pytest.approx(-1.0)  # Negative = receives

    def test_funding_cost_multiple_periods(self):
        cost = calculate_funding_cost(10000, 0.0001, "LONG", periods=3)
        assert cost == pytest.approx(3.0)


class TestStopManager:
    """Test stop and TP hit detection."""

    def test_long_stop_hit(self):
        candle = pd.Series({"high": 100, "low": 94, "close": 95, "open": 98})
        assert check_stop_hit(candle, "LONG", 95.0) == True

    def test_long_stop_not_hit(self):
        candle = pd.Series({"high": 100, "low": 96, "close": 98, "open": 97})
        assert check_stop_hit(candle, "LONG", 95.0) == False

    def test_short_stop_hit(self):
        candle = pd.Series({"high": 106, "low": 102, "close": 105, "open": 103})
        assert check_stop_hit(candle, "SHORT", 105.0) == True

    def test_short_stop_not_hit(self):
        candle = pd.Series({"high": 104, "low": 102, "close": 103, "open": 103})
        assert check_stop_hit(candle, "SHORT", 105.0) == False

    def test_long_tp_hit(self):
        candle = pd.Series({"high": 102, "low": 98, "close": 101, "open": 99})
        assert check_tp_hit(candle, "LONG", 101.5) == True

    def test_long_tp_not_hit(self):
        candle = pd.Series({"high": 100, "low": 98, "close": 99, "open": 99})
        assert check_tp_hit(candle, "LONG", 101.5) == False

    def test_short_tp_hit(self):
        candle = pd.Series({"high": 100, "low": 95, "close": 96, "open": 99})
        assert check_tp_hit(candle, "SHORT", 96.0) == True


class TestPnlCalculation:
    """Test P&L calculation."""

    def test_long_profit(self):
        pnl = _calculate_pnl("LONG", 100.0, 105.0, 1000.0)
        assert pnl == pytest.approx(50.0)  # 5% of 1000

    def test_long_loss(self):
        pnl = _calculate_pnl("LONG", 100.0, 97.0, 1000.0)
        assert pnl == pytest.approx(-30.0)  # -3% of 1000

    def test_short_profit(self):
        pnl = _calculate_pnl("SHORT", 100.0, 95.0, 1000.0)
        assert pnl == pytest.approx(50.0)  # 5% of 1000

    def test_short_loss(self):
        pnl = _calculate_pnl("SHORT", 100.0, 103.0, 1000.0)
        assert pnl == pytest.approx(-30.0)  # -3% of 1000


class TestReport:
    """Test report generation."""

    def test_empty_report(self):
        report = generate_report([], [], 10000.0)
        assert report["total_trades"] == 0
        assert report["final_equity"] == 10000.0

    def test_report_with_trades(self):
        trades = [
            {"pnl": 100.0, "direction": "LONG", "fees": 5, "risk_amount": 100,
             "timeframe": "15m", "entry_time": None, "exit_time": None},
            {"pnl": -50.0, "direction": "LONG", "fees": 5, "risk_amount": 100,
             "timeframe": "15m", "entry_time": None, "exit_time": None},
            {"pnl": 75.0, "direction": "SHORT", "fees": 5, "risk_amount": 100,
             "timeframe": "30m", "entry_time": None, "exit_time": None},
        ]
        eq_curve = [
            {"equity": 10000, "timestamp": "2025-01-01"},
            {"equity": 10100, "timestamp": "2025-01-02"},
            {"equity": 10050, "timestamp": "2025-01-03"},
            {"equity": 10125, "timestamp": "2025-01-04"},
        ]
        report = generate_report(trades, eq_curve, 10000.0)

        assert report["total_trades"] == 3
        assert report["win_count"] == 2
        assert report["loss_count"] == 1
        assert report["win_rate"] == pytest.approx(2/3, rel=0.01)
        assert report["net_pnl"] == pytest.approx(125.0)
        assert report["profit_factor"] > 1.0
        assert report["long_trades"] == 2
        assert report["short_trades"] == 1

    def test_max_drawdown_calculation(self):
        eq_curve = [
            {"equity": 10000},
            {"equity": 10500},
            {"equity": 9800},  # DD from 10500 = 700
            {"equity": 10200},
            {"equity": 9500},  # DD from 10500 = 1000 (max)
            {"equity": 10100},
        ]
        dd, dd_pct = _calculate_max_drawdown(eq_curve, 10000)
        assert dd == pytest.approx(1000.0)
        assert dd_pct == pytest.approx(0.10)


class TestTPOptimizer:
    """Test TP optimization utilities."""

    def test_get_configs_returns_list(self):
        configs = get_tp_configs()
        assert isinstance(configs, list)
        assert len(configs) > 0

    def test_configs_sum_to_1(self):
        """All TP allocations should sum to 100%."""
        for config in get_tp_configs():
            total = config["tp1_pct"] + config["tp2_pct"] + config["tp3_pct"] + config["tp4_pct"]
            assert total == pytest.approx(1.0, abs=0.01)

    def test_config_to_tp_list(self):
        config = {"tp1_pct": 0.40, "tp2_pct": 0.30, "tp3_pct": 0.20, "tp4_pct": 0.10}
        tp_list = config_to_tp_list(config)
        assert len(tp_list) == 4
        assert tp_list[0]["level"] == 0.5
        assert tp_list[0]["pct"] == 0.40

    def test_evaluate_config_empty(self):
        config = {"tp1_pct": 0.4, "tp2_pct": 0.3, "tp3_pct": 0.2, "tp4_pct": 0.1}
        result = evaluate_config(config, [], [], 10000)
        assert result["win_rate"] == 0
        assert result["profit_factor"] == 0


class TestBacktestEngine:
    """Test the full backtest engine."""

    def _make_backtest_candles(self, n=300):
        """Create synthetic data for backtesting."""
        np.random.seed(123)
        prices = [95000.0]
        for i in range(1, n):
            change = np.random.randn() * 50 + np.sin(i * 0.05) * 100
            prices.append(prices[-1] + change)

        return pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=n, freq="15min", tz="UTC"),
            "open": [p - 20 for p in prices],
            "high": [p + abs(np.random.randn() * 30) + 10 for p in prices],
            "low": [p - abs(np.random.randn() * 30) - 10 for p in prices],
            "close": prices,
            "volume": [1000 + abs(np.random.randn() * 200) for _ in range(n)],
        })

    def test_backtest_returns_tuple(self):
        candles = self._make_backtest_candles()
        trades, eq_curve, stats = run_backtest(candles)
        assert isinstance(trades, list)
        assert isinstance(eq_curve, list)
        assert isinstance(stats, dict)

    def test_backtest_equity_curve_not_empty(self):
        candles = self._make_backtest_candles()
        _, eq_curve, _ = run_backtest(candles)
        assert len(eq_curve) > 0

    def test_backtest_stats_keys(self):
        candles = self._make_backtest_candles()
        _, _, stats = run_backtest(candles)
        assert "signals_generated" in stats
        assert "signals_taken" in stats
        assert "be_stop_triggers" in stats

    def test_backtest_insufficient_data(self):
        """Too few candles should return empty results."""
        candles = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=20, freq="15min", tz="UTC"),
            "open": [100.0] * 20,
            "high": [101.0] * 20,
            "low": [99.0] * 20,
            "close": [100.0] * 20,
            "volume": [1000.0] * 20,
        })
        trades, eq_curve, stats = run_backtest(candles)
        assert len(trades) == 0
        assert len(eq_curve) == 0
