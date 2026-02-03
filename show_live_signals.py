#!/usr/bin/env python3
"""
Show Live Signals - Display signals that meet lenient R:R based thresholds
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

import config


def load_backtest_results():
    """Load backtest results"""
    results_file = Path("results/backtests/backtest_results_multiframe.json")

    if not results_file.exists():
        print("❌ No backtest results found. Run backtesting first:")
        print("   ./run.sh --mode backtest --no-fetch")
        return []

    with open(results_file, 'r') as f:
        return json.load(f)


def filter_by_live_thresholds(results):
    """Filter signals using lenient live trading thresholds"""
    qualified = []

    for result in results:
        tf = result['timeframe']
        thresholds = config.LIVE_THRESHOLDS[tf]

        # Check if meets lenient criteria
        meets_wr = result['win_rate'] >= thresholds['min_win_rate']
        meets_pf = result['profit_factor'] >= thresholds['min_profit_factor']
        meets_trades = result['total_trades'] >= thresholds['min_trades']

        if meets_wr and meets_pf and meets_trades:
            qualified.append(result)

    return qualified


def calculate_breakeven_metrics(win_rate, rr_ratio):
    """Calculate if signal is profitable at given R:R"""
    # Break-even WR = 1 / (1 + R:R)
    breakeven_wr = 1 / (1 + rr_ratio)

    # Expected value per trade (assuming $100 risk)
    risk = 100
    reward = risk * rr_ratio

    wins = win_rate * reward
    losses = (1 - win_rate) * risk

    expected_value = wins - losses

    return breakeven_wr, expected_value


def print_signal_detail(signal, rank):
    """Print detailed signal information"""
    tf = signal['timeframe']
    rr = config.LIVE_THRESHOLDS[tf]['required_rr']
    breakeven_wr, exp_value = calculate_breakeven_metrics(signal['win_rate'], rr)

    print(f"\n{'='*80}")
    print(f"#{rank}. {signal['signal_name']} ({signal['timeframe']})")
    print(f"{'='*80}")

    print(f"\n📊 PERFORMANCE:")
    print(f"  Win Rate: {signal['win_rate']:.1%} (Break-even: {breakeven_wr:.1%}) {'✓' if signal['win_rate'] > breakeven_wr else '✗'}")
    print(f"  Profit Factor: {signal['profit_factor']:.2f}x")
    print(f"  Sharpe Ratio: {signal['sharpe_ratio']:.2f}")
    print(f"  Total Trades: {signal['total_trades']}")

    print(f"\n💰 PROFITABILITY (at {rr}:1 R:R):")
    print(f"  Expected Value: ${exp_value:.2f} per $100 risked")
    print(f"  Per Trade P&L: {exp_value:.1f}% per position")
    print(f"  Monthly Est. (50 trades): {exp_value * 50:.1f}% return")

    print(f"\n📈 STATISTICS:")
    print(f"  Avg Win: {signal['avg_win_pips']:.1f} pips")
    print(f"  Avg Loss: {signal['avg_loss_pips']:.1f} pips")
    print(f"  Largest Win: {signal['largest_win_pips']:.1f} pips")
    print(f"  Largest Loss: {signal['largest_loss_pips']:.1f} pips")
    print(f"  Max Drawdown: {signal['max_drawdown_pct']:.1%}")

    print(f"\n⏱️  TIMING:")
    print(f"  Avg Hold: {signal['avg_hold_duration_minutes']:.0f} minutes")
    print(f"  Expected Range: {config.TIMEFRAME_CHARACTERISTICS[tf]['typical_hold_minutes']}")

    print(f"\n💸 COSTS:")
    print(f"  Slippage: ${signal['total_slippage_cost']:.2f}")
    print(f"  Funding: ${signal['total_funding_cost']:.2f}")
    print(f"  Total Cost: ${signal['total_transaction_cost']:.2f}")

    if 'win_rate_by_regime' in signal:
        print(f"\n🌍 REGIME PERFORMANCE:")
        for regime, wr in signal['win_rate_by_regime'].items():
            print(f"  {regime.capitalize()}: {wr:.1%}")


def main():
    """Main function"""
    print("\n" + "="*80)
    print("LIVE TRADING SIGNALS - R:R BASED FILTERING")
    print("="*80)

    # Load results
    results = load_backtest_results()

    if not results:
        return

    print(f"\n✓ Loaded {len(results)} backtest results")

    # Filter by lenient thresholds
    qualified_signals = filter_by_live_thresholds(results)

    print(f"\n📊 THRESHOLD COMPARISON:")
    print(f"\nLenient Thresholds (R:R Based):")
    for tf, thresh in config.LIVE_THRESHOLDS.items():
        rr = thresh['required_rr']
        breakeven = 1 / (1 + rr)
        print(f"  {tf}: WR >{thresh['min_win_rate']:.0%}, PF >{thresh['min_profit_factor']:.1f}x (R:R={rr}:1, breakeven={breakeven:.1%})")

    print(f"\n✓ {len(qualified_signals)}/{len(results)} signals qualified for live trading")

    if not qualified_signals:
        print("\n❌ No signals met the lenient thresholds.")
        print("\nTop 5 closest signals:")

        # Sort by win rate
        sorted_results = sorted(results, key=lambda x: x['win_rate'], reverse=True)

        for i, sig in enumerate(sorted_results[:5], 1):
            tf = sig['timeframe']
            required_wr = config.LIVE_THRESHOLDS[tf]['min_win_rate']
            print(f"  {i}. {sig['signal_name']} ({sig['timeframe']}): "
                  f"WR={sig['win_rate']:.1%} (need {required_wr:.1%}), "
                  f"PF={sig['profit_factor']:.2f}x")

        print("\n💡 TIP: With random mock data, no signals will qualify.")
        print("   Use real market data for actual profitable signals.")
        return

    # Sort by expected value
    for sig in qualified_signals:
        tf = sig['timeframe']
        rr = config.LIVE_THRESHOLDS[tf]['required_rr']
        _, exp_val = calculate_breakeven_metrics(sig['win_rate'], rr)
        sig['expected_value'] = exp_val

    qualified_signals.sort(key=lambda x: x['expected_value'], reverse=True)

    # Print summary table
    print(f"\n{'='*80}")
    print("QUALIFIED SIGNALS SUMMARY")
    print(f"{'='*80}")
    print(f"{'#':<3} {'Signal Name':<40} {'TF':<4} {'WR':>6} {'PF':>6} {'EV':>8}")
    print("-"*80)

    for i, sig in enumerate(qualified_signals, 1):
        print(f"{i:<3} {sig['signal_name']:<40} {sig['timeframe']:<4} "
              f"{sig['win_rate']:>6.1%} {sig['profit_factor']:>6.2f}x "
              f"${sig['expected_value']:>7.2f}")

    # Print detailed info for top 5
    print(f"\n{'='*80}")
    print("TOP 5 SIGNAL DETAILS")
    print(f"{'='*80}")

    for i, signal in enumerate(qualified_signals[:5], 1):
        print_signal_detail(signal, i)

    # Save to file
    output_file = Path("results/trades/live_qualified_signals.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(qualified_signals, f, indent=2, default=str)

    print(f"\n{'='*80}")
    print(f"✓ Saved {len(qualified_signals)} qualified signals to: {output_file}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
