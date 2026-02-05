"""
Parameter Optimization Framework - Grid search for optimal strategy parameters.
"""

import asyncio
import json
import itertools
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, asdict
import logging

from .engine import BacktestEngine
from ..config import SystemConfig


@dataclass
class ParameterSet:
    """A set of parameters to test."""
    name: str
    min_confidence: float
    max_stop_loss_pct: float
    reward_risk_ratio: float
    kelly_fraction: float
    max_portfolio_risk_pct: float
    max_concurrent_positions: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OptimizationResult:
    """Results from testing a parameter set."""
    parameters: ParameterSet
    total_trades: int
    win_rate: float
    profit_factor: float
    total_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    calmar_ratio: float
    avg_rr: float
    max_consecutive_losses: int

    # Composite score for ranking
    optimization_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'parameters': self.parameters.to_dict()
        }


class ParameterOptimizer:
    """
    Optimizes trading strategy parameters using grid search.
    """

    def __init__(
        self,
        base_config: SystemConfig,
        output_dir: str = "optimization_results"
    ):
        """
        Initialize parameter optimizer.

        Args:
            base_config: Base system configuration
            output_dir: Directory for optimization results
        """
        self.base_config = base_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.logger = logging.getLogger(self.__class__.__name__)

    def generate_parameter_grid(
        self,
        min_confidence_values: Optional[List[float]] = None,
        max_stop_loss_values: Optional[List[float]] = None,
        reward_risk_values: Optional[List[float]] = None,
        kelly_fraction_values: Optional[List[float]] = None,
        max_portfolio_risk_values: Optional[List[float]] = None,
        max_positions_values: Optional[List[int]] = None
    ) -> List[ParameterSet]:
        """
        Generate parameter combinations for grid search.

        Args:
            min_confidence_values: Confidence threshold values to test
            max_stop_loss_values: Max stop loss % values to test
            reward_risk_values: Reward:Risk ratio values to test
            kelly_fraction_values: Kelly fraction values to test
            max_portfolio_risk_values: Max portfolio risk % values to test
            max_positions_values: Max concurrent positions to test

        Returns:
            List of parameter sets to test
        """
        # Default parameter ranges
        if min_confidence_values is None:
            min_confidence_values = [0.35, 0.40, 0.45, 0.50]

        if max_stop_loss_values is None:
            max_stop_loss_values = [2.0, 3.0, 4.0, 5.0]

        if reward_risk_values is None:
            reward_risk_values = [1.5, 2.0, 2.5, 3.0]

        if kelly_fraction_values is None:
            kelly_fraction_values = [0.1, 0.25, 0.5]

        if max_portfolio_risk_values is None:
            max_portfolio_risk_values = [5.0, 10.0, 15.0]

        if max_positions_values is None:
            max_positions_values = [3, 5, 7]

        # Generate all combinations
        parameter_sets = []
        idx = 0

        for conf, stop, rr, kelly, port_risk, max_pos in itertools.product(
            min_confidence_values,
            max_stop_loss_values,
            reward_risk_values,
            kelly_fraction_values,
            max_portfolio_risk_values,
            max_positions_values
        ):
            param_set = ParameterSet(
                name=f"config_{idx:03d}",
                min_confidence=conf,
                max_stop_loss_pct=stop,
                reward_risk_ratio=rr,
                kelly_fraction=kelly,
                max_portfolio_risk_pct=port_risk,
                max_concurrent_positions=max_pos
            )
            parameter_sets.append(param_set)
            idx += 1

        self.logger.info(f"Generated {len(parameter_sets)} parameter combinations")
        return parameter_sets

    async def run_optimization(
        self,
        parameter_sets: List[ParameterSet],
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1h",
        initial_capital: float = 10000.0,
        historical_data: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        max_parallel: int = 3
    ) -> List[OptimizationResult]:
        """
        Run optimization across all parameter sets.

        Args:
            parameter_sets: Parameter combinations to test
            symbols: Trading symbols
            start_date: Backtest start
            end_date: Backtest end
            timeframe: Candle timeframe
            initial_capital: Starting capital
            historical_data: Pre-fetched historical data (optional)
            max_parallel: Maximum parallel backtests (to avoid OOM)

        Returns:
            List of optimization results sorted by score
        """
        self.logger.info(
            f"Starting optimization with {len(parameter_sets)} configurations"
        )

        results = []

        # Run backtests in batches to avoid memory issues
        for i in range(0, len(parameter_sets), max_parallel):
            batch = parameter_sets[i:i + max_parallel]

            self.logger.info(
                f"Running batch {i // max_parallel + 1}/{(len(parameter_sets) + max_parallel - 1) // max_parallel} "
                f"({len(batch)} configs)"
            )

            # Run batch in parallel
            batch_tasks = [
                self._run_single_backtest(
                    param_set,
                    symbols,
                    start_date,
                    end_date,
                    timeframe,
                    initial_capital,
                    historical_data
                )
                for param_set in batch
            ]

            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            # Filter out exceptions and collect results
            for param_set, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    self.logger.error(f"Error testing {param_set.name}: {result}")
                elif result is not None:
                    results.append(result)

        # Sort by optimization score (descending)
        results.sort(key=lambda r: r.optimization_score, reverse=True)

        # Save results
        self._save_optimization_results(results)

        self.logger.info(
            f"Optimization complete. Best score: {results[0].optimization_score:.2f} "
            f"(config: {results[0].parameters.name})"
        )

        return results

    async def _run_single_backtest(
        self,
        param_set: ParameterSet,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        timeframe: str,
        initial_capital: float,
        historical_data: Optional[Dict[str, List[Dict[str, Any]]]]
    ) -> Optional[OptimizationResult]:
        """
        Run a single backtest with given parameters.

        Args:
            param_set: Parameter set to test
            symbols: Trading symbols
            start_date: Start date
            end_date: End date
            timeframe: Timeframe
            initial_capital: Initial capital
            historical_data: Historical data

        Returns:
            Optimization result or None on failure
        """
        try:
            # Create modified config with these parameters
            config = self._create_config(param_set)

            # Run backtest
            engine = BacktestEngine(
                config,
                data_dir=str(self.output_dir / "backtest_cache")
            )

            if historical_data:
                engine.set_historical_data(historical_data)

            backtest_results = await engine.run_backtest(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                timeframe=timeframe,
                initial_capital=initial_capital
            )

            # Extract metrics
            metrics = backtest_results.get('metrics')
            if not metrics:
                self.logger.warning(f"No metrics for {param_set.name}")
                return None

            # Calculate optimization score
            score = self._calculate_optimization_score(metrics)

            result = OptimizationResult(
                parameters=param_set,
                total_trades=metrics.get('total_trades', 0),
                win_rate=metrics.get('win_rate', 0),
                profit_factor=metrics.get('profit_factor', 0),
                total_return_pct=metrics.get('total_pnl_percent', 0),
                sharpe_ratio=metrics.get('sharpe_ratio', 0),
                sortino_ratio=metrics.get('sortino_ratio', 0),
                max_drawdown_pct=metrics.get('max_drawdown_percent', 0),
                calmar_ratio=metrics.get('calmar_ratio', 0),
                avg_rr=metrics.get('avg_rr', 0),
                max_consecutive_losses=metrics.get('max_consecutive_losses', 0),
                optimization_score=score
            )

            self.logger.info(
                f"{param_set.name}: Score={score:.2f}, WR={result.win_rate:.1f}%, "
                f"PF={result.profit_factor:.2f}, Return={result.total_return_pct:+.2f}%"
            )

            return result

        except Exception as e:
            self.logger.error(f"Backtest failed for {param_set.name}: {e}")
            return None

    def _create_config(self, param_set: ParameterSet) -> SystemConfig:
        """
        Create a config with modified parameters.

        Args:
            param_set: Parameter set

        Returns:
            Modified config
        """
        import copy
        config = copy.deepcopy(self.base_config)

        # Apply parameters
        config.signal_synthesis.min_confidence = param_set.min_confidence
        config.signal_synthesis.max_stop_loss_pct = param_set.max_stop_loss_pct
        config.signal_synthesis.reward_risk_ratio = param_set.reward_risk_ratio
        config.risk_management.kelly_fraction = param_set.kelly_fraction
        config.risk_management.max_portfolio_risk_pct = param_set.max_portfolio_risk_pct
        config.risk_management.max_concurrent_positions = param_set.max_concurrent_positions

        return config

    def _calculate_optimization_score(self, metrics: Dict[str, Any]) -> float:
        """
        Calculate composite optimization score.

        Weighted formula balancing multiple objectives:
        - Profit Factor (30%)
        - Sharpe Ratio (25%)
        - Win Rate (15%)
        - Total Return (15%)
        - Max Drawdown penalty (15%)

        Args:
            metrics: Strategy metrics

        Returns:
            Optimization score (higher is better)
        """
        # Extract metrics
        pf = metrics.get('profit_factor', 0)
        sharpe = metrics.get('sharpe_ratio', 0)
        win_rate = metrics.get('win_rate', 0)
        total_return = metrics.get('total_pnl_percent', 0)
        max_dd = metrics.get('max_drawdown_percent', 0)
        total_trades = metrics.get('total_trades', 0)

        # Minimum trades filter (need at least 10 trades for valid sample)
        if total_trades < 10:
            return 0.0

        # Normalize metrics to 0-100 scale
        # Profit Factor: 0-5 → 0-100 (2.0 = good)
        pf_score = min((pf / 2.0) * 100, 100)

        # Sharpe: -2 to 3 → 0-100 (1.0 = good)
        sharpe_score = min(max(((sharpe + 2) / 5) * 100, 0), 100)

        # Win Rate: 0-100 → 0-100 (50% = good)
        wr_score = win_rate

        # Return: -50% to +50% → 0-100 (10% = good)
        return_score = min(max(((total_return + 50) / 100) * 100, 0), 100)

        # Max DD penalty: 0-50% → 100-0 (lower DD is better)
        dd_score = max(100 - (max_dd * 2), 0)

        # Weighted composite score
        score = (
            pf_score * 0.30 +
            sharpe_score * 0.25 +
            wr_score * 0.15 +
            return_score * 0.15 +
            dd_score * 0.15
        )

        return score

    def _save_optimization_results(self, results: List[OptimizationResult]):
        """
        Save optimization results to file.

        Args:
            results: Optimization results (sorted by score)
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"optimization_{timestamp}.json"

        data = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'total_configurations': len(results),
            'results': [r.to_dict() for r in results]
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        # Also save summary report
        self._save_summary_report(results, timestamp)

        self.logger.info(f"Results saved to {filename}")

    def _save_summary_report(self, results: List[OptimizationResult], timestamp: str):
        """
        Save human-readable summary report.

        Args:
            results: Optimization results
            timestamp: Timestamp string
        """
        report_file = self.output_dir / f"optimization_summary_{timestamp}.txt"

        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("PARAMETER OPTIMIZATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Total Configurations Tested: {len(results)}\n")
            f.write(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n")

            # Top 10 configurations
            f.write("=" * 80 + "\n")
            f.write("TOP 10 CONFIGURATIONS\n")
            f.write("=" * 80 + "\n\n")

            for i, result in enumerate(results[:10], 1):
                f.write(f"#{i} - {result.parameters.name} (Score: {result.optimization_score:.2f})\n")
                f.write("-" * 80 + "\n")

                # Parameters
                f.write("Parameters:\n")
                f.write(f"  Min Confidence:      {result.parameters.min_confidence:.2f}\n")
                f.write(f"  Max Stop Loss:       {result.parameters.max_stop_loss_pct:.1f}%\n")
                f.write(f"  R:R Ratio:           {result.parameters.reward_risk_ratio:.1f}:1\n")
                f.write(f"  Kelly Fraction:      {result.parameters.kelly_fraction:.2f}\n")
                f.write(f"  Max Portfolio Risk:  {result.parameters.max_portfolio_risk_pct:.1f}%\n")
                f.write(f"  Max Positions:       {result.parameters.max_concurrent_positions}\n\n")

                # Performance
                f.write("Performance:\n")
                f.write(f"  Total Trades:        {result.total_trades}\n")
                f.write(f"  Win Rate:            {result.win_rate:.1f}%\n")
                f.write(f"  Profit Factor:       {result.profit_factor:.2f}\n")
                f.write(f"  Total Return:        {result.total_return_pct:+.2f}%\n")
                f.write(f"  Sharpe Ratio:        {result.sharpe_ratio:.2f}\n")
                f.write(f"  Sortino Ratio:       {result.sortino_ratio:.2f}\n")
                f.write(f"  Max Drawdown:        {result.max_drawdown_pct:.2f}%\n")
                f.write(f"  Calmar Ratio:        {result.calmar_ratio:.2f}\n")
                f.write(f"  Avg R:R:             {result.avg_rr:.2f}:1\n")
                f.write(f"  Max Cons. Losses:    {result.max_consecutive_losses}\n\n")

            # Best in each category
            f.write("=" * 80 + "\n")
            f.write("CATEGORY LEADERS\n")
            f.write("=" * 80 + "\n\n")

            best_pf = max(results, key=lambda r: r.profit_factor)
            f.write(f"Best Profit Factor:  {best_pf.profit_factor:.2f} ({best_pf.parameters.name})\n")

            best_sharpe = max(results, key=lambda r: r.sharpe_ratio)
            f.write(f"Best Sharpe Ratio:   {best_sharpe.sharpe_ratio:.2f} ({best_sharpe.parameters.name})\n")

            best_wr = max(results, key=lambda r: r.win_rate)
            f.write(f"Best Win Rate:       {best_wr.win_rate:.1f}% ({best_wr.parameters.name})\n")

            best_return = max(results, key=lambda r: r.total_return_pct)
            f.write(f"Best Total Return:   {best_return.total_return_pct:+.2f}% ({best_return.parameters.name})\n")

            min_dd = min(results, key=lambda r: r.max_drawdown_pct)
            f.write(f"Lowest Drawdown:     {min_dd.max_drawdown_pct:.2f}% ({min_dd.parameters.name})\n\n")

        self.logger.info(f"Summary report saved to {report_file}")
