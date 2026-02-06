"""
EV Validation Runner.

Runs the full backtest pipeline on multiple synthetic datasets
and computes aggregate EV statistics. This validates the theoretical
EV estimates from the improvement loop.

Usage:
    python3 -m src.backtest.run_validation
"""

import numpy as np

from src.backtest.engine import run_backtest
from src.backtest.report import generate_report
from src.backtest.synthetic_data import generate_btc_like_data
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_validation(
    seeds: list[int] | None = None,
    n_candles: int = 5000,
    initial_equity: float = 10000.0,
) -> dict:
    """
    Run backtests across multiple random seeds and aggregate results.

    Args:
        seeds: List of random seeds. More seeds = more robust estimate.
        n_candles: Candles per dataset (~52 days of 15m data at 5000).
        initial_equity: Starting equity per backtest.

    Returns:
        Aggregated results dict with EV estimate.
    """
    if seeds is None:
        seeds = [42, 123, 456, 789, 2024, 7, 99, 314, 555, 1001]

    all_trades = []
    all_reports = []
    per_seed_stats = []

    print("=" * 70)
    print("EV VALIDATION: Running backtests across multiple datasets")
    print(f"Seeds: {len(seeds)} | Candles per run: {n_candles}")
    print("=" * 70)

    for seed in seeds:
        candles = generate_btc_like_data(n_candles=n_candles, seed=seed)

        trades, eq_curve, stats = run_backtest(
            candles=candles,
            symbol="BTC/USDT:USDT",
            timeframe="15m",
            initial_equity=initial_equity,
        )

        report = generate_report(
            trades, eq_curve, initial_equity,
            stats["signals_generated"], stats["signals_taken"], stats["be_stop_triggers"],
        )

        all_trades.extend(trades)
        all_reports.append(report)

        r_multiples = [t.get("r_multiple", 0) for t in trades]
        avg_r = np.mean(r_multiples) if r_multiples else 0
        win_r = [r for r in r_multiples if r > 0]
        loss_r = [r for r in r_multiples if r <= 0]

        seed_stat = {
            "seed": seed,
            "trades": len(trades),
            "signals": stats["signals_generated"],
            "taken": stats["signals_taken"],
            "win_rate": report["win_rate"],
            "avg_r": avg_r,
            "avg_win_r": np.mean(win_r) if win_r else 0,
            "avg_loss_r": np.mean(loss_r) if loss_r else 0,
            "profit_factor": report["profit_factor"],
            "max_dd_pct": report["max_drawdown_pct"],
            "return_pct": report["return_pct"],
            "be_stops": stats["be_stop_triggers"],
        }
        per_seed_stats.append(seed_stat)

        status = "+" if report["net_pnl"] > 0 else "-"
        print(
            f"  Seed {seed:>5}: {len(trades):>3} trades | "
            f"WR={report['win_rate']:.0%} | "
            f"AvgR={avg_r:+.2f} | "
            f"PF={report['profit_factor']:.2f} | "
            f"DD={report['max_drawdown_pct']:.1%} | "
            f"{status}${abs(report['net_pnl']):.0f}"
        )

    # Aggregate stats
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)

    total_trades = len(all_trades)
    if total_trades == 0:
        print("NO TRADES GENERATED across all datasets.")
        print("This means the strategy filters are too restrictive for synthetic data.")
        print("=" * 70)
        return {
            "total_trades": 0,
            "ev": 0,
            "win_rate": 0,
            "avg_win_r": 0,
            "avg_loss_r": 0,
            "profit_factor": 0,
            "per_seed": per_seed_stats,
        }

    r_multiples = [t.get("r_multiple", 0) for t in all_trades]
    win_trades = [r for r in r_multiples if r > 0]
    loss_trades = [r for r in r_multiples if r <= 0]

    win_rate = len(win_trades) / total_trades
    avg_win_r = np.mean(win_trades) if win_trades else 0
    avg_loss_r = abs(np.mean(loss_trades)) if loss_trades else 0

    # EV = (WR × AvgWinR) - ((1-WR) × AvgLossR)
    ev = (win_rate * avg_win_r) - ((1 - win_rate) * avg_loss_r)

    # Aggregated metrics
    avg_pf = np.mean([r["profit_factor"] for r in all_reports if r["profit_factor"] < float("inf")])
    avg_dd = np.mean([r["max_drawdown_pct"] for r in all_reports])
    avg_return = np.mean([r["return_pct"] for r in all_reports])
    median_r = np.median(r_multiples)

    # Exit reason breakdown
    reasons = {}
    for t in all_trades:
        reason = t.get("reason", "unknown")
        reasons[reason] = reasons.get(reason, 0) + 1

    print(f"Total Trades:        {total_trades} across {len(seeds)} seeds")
    print(f"Avg Trades/Seed:     {total_trades / len(seeds):.1f}")
    print()
    print(f"Win Rate:            {win_rate:.1%}")
    print(f"Avg Win R:           {avg_win_r:+.2f}R")
    print(f"Avg Loss R:          {avg_loss_r:.2f}R")
    print(f"Median R:            {median_r:+.2f}R")
    print()
    print(f"*** EXPECTED VALUE:  {ev:+.3f}R ***")
    print()
    print(f"Profit Factor:       {avg_pf:.3f}")
    print(f"Avg Max Drawdown:    {avg_dd:.2%}")
    print(f"Avg Return:          {avg_return:.2%}")
    print()

    # Exit reason breakdown
    print("Exit Reasons:")
    for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
        pct = count / total_trades
        avg_r_for_reason = np.mean([
            t.get("r_multiple", 0) for t in all_trades if t.get("reason") == reason
        ])
        print(f"  {reason:<25} {count:>4} ({pct:>5.1%}) | AvgR={avg_r_for_reason:+.2f}")

    # R distribution
    print()
    print("R-Multiple Distribution:")
    r_arr = np.array(r_multiples)
    for threshold in [-2, -1, 0, 1, 2, 3, 5, 10]:
        if threshold < 0:
            count = np.sum(r_arr <= threshold)
            print(f"  R <= {threshold:>3}: {count:>4} ({count/total_trades:>5.1%})")
        else:
            count = np.sum(r_arr >= threshold)
            print(f"  R >= {threshold:>3}: {count:>4} ({count/total_trades:>5.1%})")

    print("=" * 70)

    # EV target check
    if ev >= 2.0:
        print(f"TARGET MET: EV = {ev:.3f}R (target: 2-5R)")
    elif ev >= 1.0:
        print(f"PROGRESS: EV = {ev:.3f}R — getting closer to 2R target")
    elif ev > 0:
        print(f"POSITIVE EV: {ev:.3f}R — profitable but below 2R target")
    else:
        print(f"NEGATIVE EV: {ev:.3f}R — needs improvement")

    print("=" * 70)

    return {
        "total_trades": total_trades,
        "ev": round(ev, 4),
        "win_rate": round(win_rate, 4),
        "avg_win_r": round(avg_win_r, 4),
        "avg_loss_r": round(avg_loss_r, 4),
        "median_r": round(median_r, 4),
        "profit_factor": round(avg_pf, 4),
        "avg_max_dd": round(avg_dd, 4),
        "avg_return": round(avg_return, 4),
        "exit_reasons": reasons,
        "per_seed": per_seed_stats,
    }


if __name__ == "__main__":
    results = run_validation()
