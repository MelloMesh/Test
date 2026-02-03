"""
Edge Analysis and Signal Filtering
Identifies which signals have real edge based on backtest results
Filters by timeframe-specific thresholds
"""

import json
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class EdgeAnalyzer:
    """
    Analyzes backtest results to identify signals with genuine edge
    Applies timeframe-specific filtering thresholds
    """

    def __init__(self, config):
        self.config = config

    def load_backtest_results(self, results_path: str = "results/backtests/backtest_results_multiframe.json") -> List[Dict]:
        """Load backtest results from JSON"""
        with open(results_path, 'r') as f:
            results = json.load(f)

        logger.info(f"Loaded {len(results)} backtest results")
        return results

    def filter_by_timeframe_thresholds(self, results: List[Dict]) -> List[Dict]:
        """Filter signals that meet timeframe-specific performance thresholds"""
        filtered = []

        for result in results:
            timeframe = result['timeframe']
            thresholds = self.config.THRESHOLDS[timeframe]

            # Check minimum requirements
            meets_win_rate = result['win_rate'] >= thresholds['min_win_rate']
            meets_profit_factor = result['profit_factor'] >= thresholds['min_profit_factor']
            meets_sharpe = result['sharpe_ratio'] >= thresholds['min_sharpe']
            meets_min_trades = result['total_trades'] >= thresholds['min_trades']

            if meets_win_rate and meets_profit_factor and meets_sharpe and meets_min_trades:
                filtered.append(result)
                logger.info(f"✓ {result['signal_name']} ({timeframe}): PASSED - WR={result['win_rate']:.1%}, PF={result['profit_factor']:.2f}x")
            else:
                reasons = []
                if not meets_win_rate:
                    reasons.append(f"WR={result['win_rate']:.1%}<{thresholds['min_win_rate']:.1%}")
                if not meets_profit_factor:
                    reasons.append(f"PF={result['profit_factor']:.2f}<{thresholds['min_profit_factor']:.2f}")
                if not meets_sharpe:
                    reasons.append(f"Sharpe={result['sharpe_ratio']:.2f}<{thresholds['min_sharpe']:.2f}")
                if not meets_min_trades:
                    reasons.append(f"Trades={result['total_trades']}<{thresholds['min_trades']}")

                logger.debug(f"✗ {result['signal_name']} ({timeframe}): FAILED - {', '.join(reasons)}")

        logger.info(f"Filtered to {len(filtered)}/{len(results)} signals with edge")
        return filtered

    def rank_signals(self, results: List[Dict]) -> pd.DataFrame:
        """Rank signals by overall performance"""
        if not results:
            logger.warning("No results to rank")
            return pd.DataFrame()

        df = pd.DataFrame(results)

        # Calculate composite score
        df['composite_score'] = (
            df['win_rate'] * 0.3 +
            (df['profit_factor'] / 5.0) * 0.3 +  # Normalize PF
            (df['sharpe_ratio'] / 3.0) * 0.2 +    # Normalize Sharpe
            (1 - df['max_drawdown_pct']) * 0.2
        )

        # Rank within each timeframe
        df['rank_overall'] = df['composite_score'].rank(ascending=False)
        df['rank_within_timeframe'] = df.groupby('timeframe')['composite_score'].rank(ascending=False)

        # Sort by composite score
        df.sort_values('composite_score', ascending=False, inplace=True)

        return df

    def analyze_timeframe_consistency(self, results: List[Dict]) -> Dict:
        """
        Analyze which signals work across multiple timeframes
        Identifies signals with consistent performance
        """
        # Group by signal type/concept (not exact name)
        signal_families = {}

        for result in results:
            # Extract signal family (e.g., "RSI_Oversold", "MACD_Cross")
            name_parts = result['signal_name'].split('_')
            if len(name_parts) >= 2:
                family = '_'.join(name_parts[:2])  # First two parts
            else:
                family = result['signal_name']

            if family not in signal_families:
                signal_families[family] = []

            signal_families[family].append(result)

        # Analyze consistency
        consistency_report = {}

        for family, family_results in signal_families.items():
            timeframes = [r['timeframe'] for r in family_results]
            avg_win_rate = np.mean([r['win_rate'] for r in family_results])
            avg_profit_factor = np.mean([r['profit_factor'] for r in family_results])

            consistency_report[family] = {
                'timeframes': timeframes,
                'num_timeframes': len(set(timeframes)),
                'avg_win_rate': avg_win_rate,
                'avg_profit_factor': avg_profit_factor,
                'consistent': len(set(timeframes)) >= 2  # Works on 2+ timeframes
            }

        return consistency_report

    def analyze_regime_performance(self, results: List[Dict]) -> Dict:
        """Analyze signal performance by market regime"""
        regime_analysis = {}

        for result in results:
            signal_name = result['signal_name']
            win_rate_by_regime = result.get('win_rate_by_regime', {})
            trades_by_regime = result.get('trades_by_regime', {})

            regime_analysis[signal_name] = {
                'best_regime': max(win_rate_by_regime, key=win_rate_by_regime.get) if win_rate_by_regime else None,
                'worst_regime': min(win_rate_by_regime, key=win_rate_by_regime.get) if win_rate_by_regime else None,
                'regime_performance': win_rate_by_regime,
                'regime_trades': trades_by_regime
            }

        return regime_analysis

    def select_top_signals_per_timeframe(self, ranked_df: pd.DataFrame, top_n: int = 5) -> Dict[str, List[str]]:
        """Select top N signals for each timeframe"""
        top_signals = {}

        for timeframe in self.config.TIMEFRAMES:
            tf_signals = ranked_df[ranked_df['timeframe'] == timeframe]
            top_n_actual = min(top_n, len(tf_signals))

            top_signals[timeframe] = tf_signals.head(top_n_actual)['signal_name'].tolist()

            logger.info(f"{timeframe}: Top {top_n_actual} signals selected")
            for i, signal in enumerate(top_signals[timeframe], 1):
                signal_data = tf_signals[tf_signals['signal_name'] == signal].iloc[0]
                logger.info(f"  {i}. {signal}: WR={signal_data['win_rate']:.1%}, PF={signal_data['profit_factor']:.2f}x")

        return top_signals

    def generate_edge_report(self, results: List[Dict], output_path: str = "results/analysis/edge_analysis_multiframe.json"):
        """Generate comprehensive edge analysis report"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Filter by thresholds
        signals_with_edge = self.filter_by_timeframe_thresholds(results)

        # Rank signals
        ranked_df = self.rank_signals(signals_with_edge)

        # Timeframe consistency
        consistency = self.analyze_timeframe_consistency(signals_with_edge)

        # Regime performance
        regime_perf = self.analyze_regime_performance(signals_with_edge)

        # Top signals per timeframe
        top_signals = self.select_top_signals_per_timeframe(ranked_df, self.config.EDGE_SIGNALS_TO_DEPLOY)

        # Build report
        report = {
            'summary': {
                'total_signals_tested': len(results),
                'signals_with_edge': len(signals_with_edge),
                'edge_percentage': len(signals_with_edge) / len(results) if results else 0,
                'signals_by_timeframe': ranked_df['timeframe'].value_counts().to_dict(),
            },
            'top_signals_per_timeframe': top_signals,
            'consistency_analysis': consistency,
            'regime_performance': regime_perf,
            'ranked_signals': ranked_df.to_dict('records')
        }

        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Edge analysis report saved to {output_path}")

        # Save ranked signals CSV
        csv_path = output_path.replace('.json', '.csv')
        ranked_df.to_csv(csv_path, index=False)
        logger.info(f"Ranked signals CSV saved to {csv_path}")

        return report

    def print_summary(self, report: Dict):
        """Print edge analysis summary"""
        print("\n" + "=" * 80)
        print("EDGE ANALYSIS SUMMARY - MULTI-TIMEFRAME")
        print("=" * 80)

        summary = report['summary']
        print(f"\nTotal Signals Tested: {summary['total_signals_tested']}")
        print(f"Signals with Edge: {summary['signals_with_edge']} ({summary['edge_percentage']:.1%})")

        print("\nSignals by Timeframe:")
        for tf, count in sorted(summary['signals_by_timeframe'].items()):
            print(f"  {tf}: {count} signals")

        print("\nTop Signals per Timeframe:")
        for tf, signals in report['top_signals_per_timeframe'].items():
            print(f"\n{tf} (Top {len(signals)}):")
            for i, signal in enumerate(signals, 1):
                print(f"  {i}. {signal}")

        print("\nCross-Timeframe Consistency:")
        consistent_families = [
            family for family, data in report['consistency_analysis'].items()
            if data['consistent']
        ]
        print(f"Signal families working across multiple timeframes: {len(consistent_families)}")
        for family in consistent_families[:5]:  # Top 5
            data = report['consistency_analysis'][family]
            print(f"  {family}: {data['num_timeframes']} timeframes, WR={data['avg_win_rate']:.1%}")

        print("\n" + "=" * 80)


def main():
    """Example usage"""
    import sys
    sys.path.append("../..")
    import config

    analyzer = EdgeAnalyzer(config)

    # Load backtest results
    results = analyzer.load_backtest_results()

    # Generate edge report
    report = analyzer.generate_edge_report(results)

    # Print summary
    analyzer.print_summary(report)


if __name__ == "__main__":
    main()
