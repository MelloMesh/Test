#!/usr/bin/env python3
"""
HTF Paper Trading Performance Analyzer
Analyzes paper trading results to identify winning patterns and optimization opportunities
"""

import json
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional
from collections import defaultdict
from dataclasses import dataclass
import statistics

@dataclass
class TradeAnalysis:
    """Container for trade analysis results"""
    total_trades: int
    winners: int
    losers: int
    win_rate: float
    avg_win: float
    avg_loss: float
    avg_pnl: float
    profit_factor: float
    total_pnl: float
    avg_hold_time_hours: float


class PaperTradingAnalyzer:
    """Analyzes paper trading performance and identifies patterns"""

    def __init__(self, results_file: str = "paper_trading_results.json"):
        self.results_file = Path(results_file)
        self.trades = []
        self.load_trades()

    def load_trades(self):
        """Load trades from results file"""
        if not self.results_file.exists():
            print(f"❌ No results file found at {self.results_file}")
            print("   Run paper trading first to generate data")
            sys.exit(1)

        with open(self.results_file, 'r') as f:
            data = json.load(f)
            self.trades = data.get('closed_trades', [])

        if not self.trades:
            print("⚠️  No closed trades yet")
            print("   Let paper trading run to accumulate data")
            sys.exit(0)

    def analyze_overall_performance(self) -> TradeAnalysis:
        """Calculate overall performance metrics"""
        total = len(self.trades)
        winners = [t for t in self.trades if t['status'] == 'closed_win']
        losers = [t for t in self.trades if t['status'] == 'closed_loss']

        win_rate = (len(winners) / total * 100) if total > 0 else 0

        avg_win = statistics.mean([t['pnl'] for t in winners]) if winners else 0
        avg_loss = statistics.mean([t['pnl'] for t in losers]) if losers else 0
        avg_pnl = statistics.mean([t['pnl'] for t in self.trades]) if self.trades else 0

        total_wins = sum(t['pnl'] for t in winners)
        total_losses = abs(sum(t['pnl'] for t in losers))
        profit_factor = (total_wins / total_losses) if total_losses > 0 else 0

        total_pnl = sum(t['pnl'] for t in self.trades)

        # Calculate average hold time
        hold_times = []
        for t in self.trades:
            if t.get('entry_time') and t.get('exit_time'):
                entry = datetime.fromisoformat(t['entry_time'].replace('Z', '+00:00'))
                exit = datetime.fromisoformat(t['exit_time'].replace('Z', '+00:00'))
                hold_times.append((exit - entry).total_seconds() / 3600)

        avg_hold_time = statistics.mean(hold_times) if hold_times else 0

        return TradeAnalysis(
            total_trades=total,
            winners=len(winners),
            losers=len(losers),
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_pnl=avg_pnl,
            profit_factor=profit_factor,
            total_pnl=total_pnl,
            avg_hold_time_hours=avg_hold_time
        )

    def analyze_by_signal_type(self) -> Dict[str, TradeAnalysis]:
        """Analyze performance by signal type"""
        by_signal = defaultdict(list)
        for trade in self.trades:
            by_signal[trade['signal_name']].append(trade)

        results = {}
        for signal_name, trades in by_signal.items():
            winners = [t for t in trades if t['status'] == 'closed_win']
            losers = [t for t in trades if t['status'] == 'closed_loss']

            results[signal_name] = TradeAnalysis(
                total_trades=len(trades),
                winners=len(winners),
                losers=len(losers),
                win_rate=(len(winners) / len(trades) * 100) if trades else 0,
                avg_win=statistics.mean([t['pnl'] for t in winners]) if winners else 0,
                avg_loss=statistics.mean([t['pnl'] for t in losers]) if losers else 0,
                avg_pnl=statistics.mean([t['pnl'] for t in trades]) if trades else 0,
                profit_factor=0,  # Calculate if needed
                total_pnl=sum(t['pnl'] for t in trades),
                avg_hold_time_hours=0  # Calculate if needed
            )

        return results

    def analyze_by_symbol(self) -> Dict[str, TradeAnalysis]:
        """Analyze performance by trading pair"""
        by_symbol = defaultdict(list)
        for trade in self.trades:
            by_symbol[trade['symbol']].append(trade)

        results = {}
        for symbol, trades in by_symbol.items():
            winners = [t for t in trades if t['status'] == 'closed_win']
            losers = [t for t in trades if t['status'] == 'closed_loss']

            results[symbol] = TradeAnalysis(
                total_trades=len(trades),
                winners=len(winners),
                losers=len(losers),
                win_rate=(len(winners) / len(trades) * 100) if trades else 0,
                avg_win=statistics.mean([t['pnl'] for t in winners]) if winners else 0,
                avg_loss=statistics.mean([t['pnl'] for t in losers]) if losers else 0,
                avg_pnl=statistics.mean([t['pnl'] for t in trades]) if trades else 0,
                profit_factor=0,
                total_pnl=sum(t['pnl'] for t in trades),
                avg_hold_time_hours=0
            )

        return results

    def analyze_by_direction(self) -> Dict[str, TradeAnalysis]:
        """Analyze performance by trade direction (long/short)"""
        by_direction = defaultdict(list)
        for trade in self.trades:
            by_direction[trade['direction']].append(trade)

        results = {}
        for direction, trades in by_direction.items():
            winners = [t for t in trades if t['status'] == 'closed_win']
            losers = [t for t in trades if t['status'] == 'closed_loss']

            results[direction] = TradeAnalysis(
                total_trades=len(trades),
                winners=len(winners),
                losers=len(losers),
                win_rate=(len(winners) / len(trades) * 100) if trades else 0,
                avg_win=statistics.mean([t['pnl'] for t in winners]) if winners else 0,
                avg_loss=statistics.mean([t['pnl'] for t in losers]) if losers else 0,
                avg_pnl=statistics.mean([t['pnl'] for t in trades]) if trades else 0,
                profit_factor=0,
                total_pnl=sum(t['pnl'] for t in trades),
                avg_hold_time_hours=0
            )

        return results

    def analyze_by_htf_bias(self) -> Dict[str, TradeAnalysis]:
        """Analyze performance by HTF bias"""
        by_bias = defaultdict(list)
        for trade in self.trades:
            by_bias[trade['htf_bias']].append(trade)

        results = {}
        for bias, trades in by_bias.items():
            winners = [t for t in trades if t['status'] == 'closed_win']
            losers = [t for t in trades if t['status'] == 'closed_loss']

            results[bias] = TradeAnalysis(
                total_trades=len(trades),
                winners=len(winners),
                losers=len(losers),
                win_rate=(len(winners) / len(trades) * 100) if trades else 0,
                avg_win=statistics.mean([t['pnl'] for t in winners]) if winners else 0,
                avg_loss=statistics.mean([t['pnl'] for t in losers]) if losers else 0,
                avg_pnl=statistics.mean([t['pnl'] for t in trades]) if trades else 0,
                profit_factor=0,
                total_pnl=sum(t['pnl'] for t in trades),
                avg_hold_time_hours=0
            )

        return results

    def analyze_by_htf_alignment(self) -> Dict[str, TradeAnalysis]:
        """Analyze performance by HTF alignment strength"""
        # Bin alignment scores
        bins = {
            'Strong (75-100%)': [],
            'Moderate (50-75%)': [],
            'Weak (0-50%)': []
        }

        for trade in self.trades:
            alignment = trade.get('htf_alignment', 0)
            if alignment >= 75:
                bins['Strong (75-100%)'].append(trade)
            elif alignment >= 50:
                bins['Moderate (50-75%)'].append(trade)
            else:
                bins['Weak (0-50%)'].append(trade)

        results = {}
        for bin_name, trades in bins.items():
            if not trades:
                continue

            winners = [t for t in trades if t['status'] == 'closed_win']
            losers = [t for t in trades if t['status'] == 'closed_loss']

            results[bin_name] = TradeAnalysis(
                total_trades=len(trades),
                winners=len(winners),
                losers=len(losers),
                win_rate=(len(winners) / len(trades) * 100) if trades else 0,
                avg_win=statistics.mean([t['pnl'] for t in winners]) if winners else 0,
                avg_loss=statistics.mean([t['pnl'] for t in losers]) if losers else 0,
                avg_pnl=statistics.mean([t['pnl'] for t in trades]) if trades else 0,
                profit_factor=0,
                total_pnl=sum(t['pnl'] for t in trades),
                avg_hold_time_hours=0
            )

        return results

    def analyze_by_hour_of_day(self) -> Dict[int, TradeAnalysis]:
        """Analyze performance by hour of day (UTC)"""
        by_hour = defaultdict(list)

        for trade in self.trades:
            if trade.get('entry_time'):
                entry_time = datetime.fromisoformat(trade['entry_time'].replace('Z', '+00:00'))
                hour = entry_time.hour
                by_hour[hour].append(trade)

        results = {}
        for hour, trades in by_hour.items():
            winners = [t for t in trades if t['status'] == 'closed_win']
            losers = [t for t in trades if t['status'] == 'closed_loss']

            results[hour] = TradeAnalysis(
                total_trades=len(trades),
                winners=len(winners),
                losers=len(losers),
                win_rate=(len(winners) / len(trades) * 100) if trades else 0,
                avg_win=statistics.mean([t['pnl'] for t in winners]) if winners else 0,
                avg_loss=statistics.mean([t['pnl'] for t in losers]) if losers else 0,
                avg_pnl=statistics.mean([t['pnl'] for t in trades]) if trades else 0,
                profit_factor=0,
                total_pnl=sum(t['pnl'] for t in trades),
                avg_hold_time_hours=0
            )

        return results

    def get_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []

        # Overall performance check
        overall = self.analyze_overall_performance()
        if overall.win_rate < 45:
            recommendations.append("⚠️  Win rate below 45% - consider tightening entry criteria")
        if overall.profit_factor < 1.5:
            recommendations.append("⚠️  Profit factor below 1.5 - improve R:R or filter losing setups")

        # Signal type analysis
        signal_perf = self.analyze_by_signal_type()
        if signal_perf:
            best_signal = max(signal_perf.items(), key=lambda x: x[1].win_rate)
            worst_signal = min(signal_perf.items(), key=lambda x: x[1].win_rate)

            if best_signal[1].win_rate > 60:
                recommendations.append(f"✅ '{best_signal[0]}' performing well ({best_signal[1].win_rate:.1f}% WR) - consider increasing position size")

            if worst_signal[1].win_rate < 40 and worst_signal[1].total_trades >= 10:
                recommendations.append(f"❌ '{worst_signal[0]}' underperforming ({worst_signal[1].win_rate:.1f}% WR) - consider disabling")

        # Direction bias
        direction_perf = self.analyze_by_direction()
        if len(direction_perf) == 2:
            long_wr = direction_perf.get('long', TradeAnalysis(0,0,0,0,0,0,0,0,0,0)).win_rate
            short_wr = direction_perf.get('short', TradeAnalysis(0,0,0,0,0,0,0,0,0,0)).win_rate

            if abs(long_wr - short_wr) > 20:
                better = 'long' if long_wr > short_wr else 'short'
                recommendations.append(f"📊 {better.upper()} trades performing significantly better - consider bias")

        # HTF alignment
        alignment_perf = self.analyze_by_htf_alignment()
        if 'Strong (75-100%)' in alignment_perf and 'Weak (0-50%)' in alignment_perf:
            strong_wr = alignment_perf['Strong (75-100%)'].win_rate
            weak_wr = alignment_perf['Weak (0-50%)'].win_rate

            if strong_wr > weak_wr + 15:
                recommendations.append(f"🎯 Strong HTF alignment trades win {strong_wr - weak_wr:.1f}% more often - increase minimum alignment threshold")

        # Sample size
        if overall.total_trades < 50:
            recommendations.append(f"📈 Only {overall.total_trades} trades - collect 50-100+ for statistical significance")

        if not recommendations:
            recommendations.append("✅ System performing well - continue monitoring")

        return recommendations

    def display_report(self):
        """Display comprehensive performance report"""
        print("\n" + "="*80)
        print("                  📊 PAPER TRADING PERFORMANCE ANALYSIS")
        print("="*80)

        # Overall Performance
        overall = self.analyze_overall_performance()
        print(f"\n{'📈 OVERALL PERFORMANCE':^80}")
        print("-"*80)
        print(f"  Total Trades:       {overall.total_trades:>6}")
        print(f"  Winners/Losers:     {overall.winners:>6} / {overall.losers}")

        wr_color = "🟢" if overall.win_rate >= 50 else "🟡" if overall.win_rate >= 40 else "🔴"
        print(f"  Win Rate:           {wr_color} {overall.win_rate:>5.1f}%")

        pf_color = "🟢" if overall.profit_factor >= 2.0 else "🟡" if overall.profit_factor >= 1.5 else "🔴"
        print(f"  Profit Factor:      {pf_color} {overall.profit_factor:>5.2f}")

        pnl_color = "🟢" if overall.total_pnl > 0 else "🔴"
        print(f"  Total P&L:          {pnl_color} ${overall.total_pnl:>,.2f}")
        print(f"  Avg Win:            🟢 ${overall.avg_win:>,.2f}")
        print(f"  Avg Loss:           🔴 ${overall.avg_loss:>,.2f}")
        print(f"  Avg P&L/Trade:      {'🟢' if overall.avg_pnl > 0 else '🔴'} ${overall.avg_pnl:>,.2f}")
        print(f"  Avg Hold Time:      ⏱️  {overall.avg_hold_time_hours:.1f} hours")

        # Signal Type Performance
        print(f"\n{'🎯 PERFORMANCE BY SIGNAL TYPE':^80}")
        print("-"*80)
        signal_perf = self.analyze_by_signal_type()

        # Sort by total P&L
        sorted_signals = sorted(signal_perf.items(), key=lambda x: x[1].total_pnl, reverse=True)

        for signal_name, perf in sorted_signals:
            wr_emoji = "🟢" if perf.win_rate >= 50 else "🟡" if perf.win_rate >= 40 else "🔴"
            pnl_emoji = "🟢" if perf.total_pnl > 0 else "🔴"

            print(f"\n  {signal_name}")
            print(f"    Trades: {perf.total_trades} | Win Rate: {wr_emoji} {perf.win_rate:.1f}% | "
                  f"P&L: {pnl_emoji} ${perf.total_pnl:,.2f} | Avg: ${perf.avg_pnl:,.2f}")

        # Symbol Performance
        print(f"\n{'💰 PERFORMANCE BY SYMBOL':^80}")
        print("-"*80)
        symbol_perf = self.analyze_by_symbol()

        # Sort by total P&L
        sorted_symbols = sorted(symbol_perf.items(), key=lambda x: x[1].total_pnl, reverse=True)[:10]

        for symbol, perf in sorted_symbols:
            wr_emoji = "🟢" if perf.win_rate >= 50 else "🟡" if perf.win_rate >= 40 else "🔴"
            pnl_emoji = "🟢" if perf.total_pnl > 0 else "🔴"

            print(f"  {symbol:<15} | Trades: {perf.total_trades:>3} | WR: {wr_emoji} {perf.win_rate:>5.1f}% | "
                  f"P&L: {pnl_emoji} ${perf.total_pnl:>8,.2f}")

        # Direction Performance
        print(f"\n{'↕️  PERFORMANCE BY DIRECTION':^80}")
        print("-"*80)
        direction_perf = self.analyze_by_direction()

        for direction, perf in direction_perf.items():
            emoji = "🟢" if direction == "long" else "🔴"
            wr_emoji = "🟢" if perf.win_rate >= 50 else "🟡" if perf.win_rate >= 40 else "🔴"

            print(f"  {emoji} {direction.upper():<8} | Trades: {perf.total_trades:>3} | "
                  f"WR: {wr_emoji} {perf.win_rate:>5.1f}% | P&L: ${perf.total_pnl:>8,.2f}")

        # HTF Bias Performance
        print(f"\n{'🔄 PERFORMANCE BY HTF BIAS':^80}")
        print("-"*80)
        bias_perf = self.analyze_by_htf_bias()

        for bias, perf in sorted(bias_perf.items(), key=lambda x: x[1].total_pnl, reverse=True):
            wr_emoji = "🟢" if perf.win_rate >= 50 else "🟡" if perf.win_rate >= 40 else "🔴"

            print(f"  {bias:<12} | Trades: {perf.total_trades:>3} | "
                  f"WR: {wr_emoji} {perf.win_rate:>5.1f}% | P&L: ${perf.total_pnl:>8,.2f}")

        # HTF Alignment Performance
        print(f"\n{'🎯 PERFORMANCE BY HTF ALIGNMENT STRENGTH':^80}")
        print("-"*80)
        alignment_perf = self.analyze_by_htf_alignment()

        alignment_order = ['Strong (75-100%)', 'Moderate (50-75%)', 'Weak (0-50%)']
        for alignment in alignment_order:
            if alignment in alignment_perf:
                perf = alignment_perf[alignment]
                wr_emoji = "🟢" if perf.win_rate >= 50 else "🟡" if perf.win_rate >= 40 else "🔴"

                print(f"  {alignment:<20} | Trades: {perf.total_trades:>3} | "
                      f"WR: {wr_emoji} {perf.win_rate:>5.1f}% | P&L: ${perf.total_pnl:>8,.2f}")

        # Time of Day Performance (top 5 hours)
        print(f"\n{'🕐 PERFORMANCE BY HOUR OF DAY (UTC) - Top 5':^80}")
        print("-"*80)
        hour_perf = self.analyze_by_hour_of_day()

        # Sort by total trades to show most active hours
        sorted_hours = sorted(hour_perf.items(), key=lambda x: x[1].total_trades, reverse=True)[:5]

        for hour, perf in sorted_hours:
            wr_emoji = "🟢" if perf.win_rate >= 50 else "🟡" if perf.win_rate >= 40 else "🔴"

            print(f"  {hour:02d}:00 UTC    | Trades: {perf.total_trades:>3} | "
                  f"WR: {wr_emoji} {perf.win_rate:>5.1f}% | P&L: ${perf.total_pnl:>8,.2f}")

        # Recommendations
        print(f"\n{'💡 RECOMMENDATIONS':^80}")
        print("-"*80)
        recommendations = self.get_recommendations()
        for rec in recommendations:
            print(f"  {rec}")

        print("\n" + "="*80)
        print(f"  Analysis generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print("="*80 + "\n")


def main():
    """Main entry point"""
    print("\n🔍 Loading paper trading results...")

    analyzer = PaperTradingAnalyzer()
    analyzer.display_report()

    print("💾 To export detailed analysis, check paper_trading_results.json")
    print("🔄 Run this script anytime to see updated analysis\n")


if __name__ == "__main__":
    main()
