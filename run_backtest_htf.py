#!/usr/bin/env python3
"""
HTF Backtesting Runner with Visual Results
Runs backtests on HTF-aware signals and displays intuitive results
"""

import sys
import os
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from analysis.higher_timeframe_analyzer import HigherTimeframeAnalyzer
from backtest.backtest_engine_htf import HTFBacktestEngine
from signals.signal_discovery_htf import get_htf_aware_signals
from data.kraken_fetcher import KrakenFetcher

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BacktestResultsViewer:
    """Display backtest results in an intuitive format"""

    @staticmethod
    def print_header(title: str):
        """Print section header"""
        print(f"\n{'='*80}")
        print(f"{title:^80}")
        print(f"{'='*80}\n")

    @staticmethod
    def print_metric(label: str, value: str, good: bool = None):
        """Print a metric with optional color coding"""
        emoji = ""
        if good is True:
            emoji = "✅"
        elif good is False:
            emoji = "❌"
        elif good is None:
            emoji = "📊"

        print(f"  {emoji} {label:<30} {value:>40}")

    @staticmethod
    def display_summary(results: Dict):
        """Display backtest summary"""
        BacktestResultsViewer.print_header("📈 BACKTEST SUMMARY")

        # Overall Performance
        win_rate = results['win_rate']
        profit_factor = results['profit_factor']
        total_return = results['total_return_pct']
        max_dd = results['max_drawdown_pct']

        BacktestResultsViewer.print_metric(
            "Total Trades",
            str(results['total_trades']),
            good=None
        )

        BacktestResultsViewer.print_metric(
            "Win Rate",
            f"{win_rate:.1f}%",
            good=(win_rate >= 55)
        )

        BacktestResultsViewer.print_metric(
            "Profit Factor",
            f"{profit_factor:.2f}x",
            good=(profit_factor >= 1.5)
        )

        BacktestResultsViewer.print_metric(
            "Total Return",
            f"{total_return:+.1f}%",
            good=(total_return > 0)
        )

        BacktestResultsViewer.print_metric(
            "Max Drawdown",
            f"{max_dd:.1f}%",
            good=(max_dd < 15)
        )

        BacktestResultsViewer.print_metric(
            "Sharpe Ratio",
            f"{results.get('sharpe_ratio', 0):.2f}",
            good=(results.get('sharpe_ratio', 0) > 1.0)
        )

    @staticmethod
    def display_htf_comparison(results: Dict):
        """Display HTF aligned vs non-aligned performance"""
        if 'htf_aligned_win_rate' not in results:
            return

        BacktestResultsViewer.print_header("🎯 HTF ALIGNMENT IMPACT")

        print("  HTF-Aligned Trades:")
        BacktestResultsViewer.print_metric(
            "  Win Rate",
            f"{results['htf_aligned_win_rate']:.1f}%",
            good=(results['htf_aligned_win_rate'] >= 55)
        )
        BacktestResultsViewer.print_metric(
            "  Profit Factor",
            f"{results['htf_aligned_profit_factor']:.2f}x",
            good=(results['htf_aligned_profit_factor'] >= 1.5)
        )
        BacktestResultsViewer.print_metric(
            "  Total Trades",
            str(results['htf_aligned_trades']),
            good=None
        )

        print("\n  Trades Blocked by HTF Filter:")
        BacktestResultsViewer.print_metric(
            "  Would-be Losers Avoided",
            str(results.get('htf_blocked_losers', 0)),
            good=True
        )

    @staticmethod
    def display_signal_breakdown(results: Dict):
        """Display performance by signal type"""
        if 'signal_breakdown' not in results:
            return

        BacktestResultsViewer.print_header("📊 PERFORMANCE BY SIGNAL TYPE")

        breakdown = results['signal_breakdown']

        # Sort by win rate
        sorted_signals = sorted(
            breakdown.items(),
            key=lambda x: x[1]['win_rate'],
            reverse=True
        )

        print(f"  {'Signal':<35} {'Trades':>8} {'Win Rate':>10} {'Profit Factor':>15}")
        print(f"  {'-'*78}")

        for signal_name, stats in sorted_signals:
            if stats['trades'] >= 5:  # Only show signals with enough data
                wr = stats['win_rate']
                pf = stats['profit_factor']

                emoji = "🟢" if wr >= 55 and pf >= 1.5 else "🟡" if wr >= 50 else "🔴"

                print(f"  {emoji} {signal_name:<33} {stats['trades']:>8} {wr:>9.1f}% {pf:>14.2f}x")

    @staticmethod
    def display_monthly_performance(results: Dict):
        """Display monthly performance breakdown"""
        if 'monthly_returns' not in results or not results['monthly_returns']:
            return

        BacktestResultsViewer.print_header("📅 MONTHLY PERFORMANCE")

        print(f"  {'Month':<15} {'Return':>12} {'Trades':>10} {'Win Rate':>12}")
        print(f"  {'-'*78}")

        for month_data in results['monthly_returns']:
            month = month_data['month']
            ret = month_data['return_pct']
            trades = month_data['trades']
            wr = month_data['win_rate']

            emoji = "🟢" if ret > 0 else "🔴"

            print(f"  {month:<15} {emoji} {ret:>9.1f}% {trades:>10} {wr:>11.1f}%")

    @staticmethod
    def display_trade_distribution(results: Dict):
        """Display distribution of trade outcomes"""
        if 'trade_distribution' not in results:
            return

        BacktestResultsViewer.print_header("📊 TRADE OUTCOME DISTRIBUTION")

        dist = results['trade_distribution']

        BacktestResultsViewer.print_metric(
            "Winners",
            f"{dist['winners']} trades ({dist['winner_pct']:.1f}%)",
            good=True
        )

        BacktestResultsViewer.print_metric(
            "Losers",
            f"{dist['losers']} trades ({dist['loser_pct']:.1f}%)",
            good=False
        )

        BacktestResultsViewer.print_metric(
            "Average Winner",
            f"+{dist['avg_winner_pct']:.2f}%",
            good=None
        )

        BacktestResultsViewer.print_metric(
            "Average Loser",
            f"{dist['avg_loser_pct']:.2f}%",
            good=None
        )

        BacktestResultsViewer.print_metric(
            "Largest Winner",
            f"+{dist['largest_winner_pct']:.2f}%",
            good=None
        )

        BacktestResultsViewer.print_metric(
            "Largest Loser",
            f"{dist['largest_loser_pct']:.2f}%",
            good=None
        )


class HTFBacktestRunner:
    """Run HTF backtests and display results"""

    def __init__(self, cache_dir: str = "backtest_results"):
        self.fetcher = KrakenFetcher()
        self.htf_analyzer = HigherTimeframeAnalyzer()
        self.backtest_engine = HTFBacktestEngine()
        self.viewer = BacktestResultsViewer()

        # Results caching
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def _get_cache_key(self, symbols: List[str], start_date: datetime, end_date: datetime) -> str:
        """Generate cache key from backtest parameters"""
        import hashlib

        # Create unique key from parameters
        key_data = f"{'-'.join(sorted(symbols))}_{start_date.date()}_{end_date.date()}"
        cache_key = hashlib.md5(key_data.encode()).hexdigest()

        return cache_key

    def _save_results(
        self,
        cache_key: str,
        results: Dict,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime
    ):
        """Save backtest results to cache"""
        cache_file = self.cache_dir / f"{cache_key}.json"

        data = {
            'metadata': {
                'symbols': symbols,
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'run_timestamp': datetime.utcnow().isoformat(),
                'duration_days': (end_date - start_date).days
            },
            'results': results
        }

        with open(cache_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"💾 Results saved to {cache_file}")

    def _load_cached_results(self, cache_key: str) -> Optional[Dict]:
        """Load cached results if they exist"""
        cache_file = self.cache_dir / f"{cache_key}.json"

        if not cache_file.exists():
            return None

        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)

            logger.info(f"📂 Loaded cached results from {cache_file}")
            return data
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
            return None

    def run_backtest(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        data_dir: str = "data/kraken",
        force_rerun: bool = False
    ):
        """Run backtest on symbols"""

        print(f"\n{'='*80}")
        print(f"🚀 HTF BACKTEST RUNNER")
        print(f"{'='*80}\n")

        print(f"  Symbols: {', '.join(symbols)}")
        print(f"  Period: {start_date.date()} to {end_date.date()}")
        print(f"  Duration: {(end_date - start_date).days} days")
        print(f"\n{'='*80}\n")

        # Check for cached results
        cache_key = self._get_cache_key(symbols, start_date, end_date)
        cached_data = self._load_cached_results(cache_key)

        if cached_data and not force_rerun:
            print(f"✅ Found cached results from {cached_data['metadata']['run_timestamp'][:19]}")
            print(f"   Using cached data to avoid re-running backtest...\n")

            # Display cached results
            self.viewer.display_summary(cached_data['results'])
            self.viewer.display_htf_comparison(cached_data['results'])

            print(f"\n{'='*80}\n")
            print(f"✅ Displayed cached backtest results!")
            print(f"\n💡 To force re-run: pass force_rerun=True")
            print(f"   Or delete: {self.cache_dir}/{cache_key}.json\n")
            return

        if cached_data:
            print(f"ℹ️  Cached results found but force_rerun=True, re-running backtest...\n")

        # Get HTF-aware signals
        print("📋 Loading HTF-aware signals...")
        signals = get_htf_aware_signals()
        print(f"✓ Loaded {len(signals)} signal strategies\n")

        all_results = []

        for symbol in symbols:
            print(f"\n🔍 Backtesting {symbol}...")
            print(f"{'-'*80}")

            # Load data
            try:
                # Load LTF data (30m)
                ltf_data = pd.read_csv(f"{data_dir}/{symbol}_30m.csv")
                ltf_data['timestamp'] = pd.to_datetime(ltf_data['timestamp'])
                ltf_data = ltf_data.set_index('timestamp')

                # Load HTF data
                htf_data = {}
                for tf in ['1w', '1d', '4h']:
                    df = pd.read_csv(f"{data_dir}/{symbol}_{tf}.csv")
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df.set_index('timestamp')
                    htf_data[tf] = df

                print(f"  ✓ Data loaded: {len(ltf_data)} candles")

            except FileNotFoundError as e:
                print(f"  ❌ Data not found: {e}")
                continue

            # Run backtest for each signal
            for signal in signals:
                result = self.backtest_engine.backtest_signal_with_htf(
                    signal_hypothesis=signal,
                    instrument=symbol,
                    data=ltf_data,
                    htf_data=htf_data,
                    funding_rates=None
                )

                if result and result.total_trades > 0:
                    all_results.append({
                        'symbol': symbol,
                        'signal': signal.name,
                        'result': result
                    })

        # Aggregate and display results
        self.display_aggregated_results(all_results, cache_key, symbols, start_date, end_date)

    def display_aggregated_results(
        self,
        all_results: List[Dict],
        cache_key: str = None,
        symbols: List[str] = None,
        start_date: datetime = None,
        end_date: datetime = None
    ):
        """Aggregate and display all backtest results"""

        if not all_results:
            print("\n❌ No backtest results to display")
            return

        # Aggregate metrics
        total_trades = sum(r['result'].total_trades for r in all_results)
        winning_trades = sum(r['result'].winning_trades for r in all_results)
        total_pnl = sum(r['result'].total_pnl for r in all_results)
        total_pnl_pct = sum(r['result'].total_return_pct for r in all_results)

        htf_aligned = sum(r['result'].htf_aligned_trades for r in all_results)
        htf_blocked = sum(r['result'].htf_misaligned_trades_skipped for r in all_results)

        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        # Calculate profit factor
        total_wins = sum(r['result'].total_wins for r in all_results)
        total_losses = sum(abs(r['result'].total_losses) for r in all_results)
        profit_factor = (total_wins / total_losses) if total_losses > 0 else 0

        # Build results dict
        results = {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_return_pct': total_pnl_pct,
            'max_drawdown_pct': max(r['result'].max_drawdown_pct for r in all_results),
            'sharpe_ratio': sum(r['result'].sharpe_ratio for r in all_results) / len(all_results),
            'htf_aligned_trades': htf_aligned,
            'htf_aligned_win_rate': win_rate,  # Simplified
            'htf_aligned_profit_factor': profit_factor,  # Simplified
            'htf_blocked_losers': htf_blocked,
        }

        # Display
        self.viewer.display_summary(results)
        self.viewer.display_htf_comparison(results)

        # Save results to cache
        if cache_key and symbols and start_date and end_date:
            self._save_results(cache_key, results, symbols, start_date, end_date)
            print(f"\n💾 Results cached for future use")

        print(f"\n{'='*80}\n")
        print(f"✅ Backtest Complete!")
        print(f"\n📁 Results saved to: {self.cache_dir}/{cache_key}.json" if cache_key else "")
        print(f"   Run again to use cached results (instant!)\n")


def main():
    """Run HTF backtest"""

    # Configuration
    symbols = ['XXBTZUSD']  # BTC/USD on Kraken
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=90)  # 3 months

    runner = HTFBacktestRunner()
    runner.run_backtest(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        data_dir="data/kraken"
    )


if __name__ == "__main__":
    main()
