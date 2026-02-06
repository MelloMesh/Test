"""
Backtesting performance report generation.
Calculates all key metrics and outputs summary tables.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


def generate_report(
    trade_results: list[dict],
    equity_curve: list[dict],
    initial_equity: float,
    signals_generated: int = 0,
    signals_taken: int = 0,
    be_stop_triggers: int = 0,
) -> dict:
    """
    Generate a comprehensive backtest performance report.

    Args:
        trade_results: List of trade dicts with keys:
            - pnl, direction, entry_price, exit_price, reason, timeframe,
              entry_time, exit_time, fees
        equity_curve: List of dicts with keys: timestamp, equity
        initial_equity: Starting equity.
        signals_generated: Total signals detected.
        signals_taken: Signals that passed risk manager.
        be_stop_triggers: How many times stop was moved to breakeven.

    Returns:
        Report dict with all metrics.
    """
    if not trade_results:
        return _empty_report(initial_equity)

    pnls = [t["pnl"] for t in trade_results]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    total_trades = len(trade_results)
    win_count = len(wins)
    loss_count = len(losses)
    win_rate = win_count / total_trades if total_trades > 0 else 0

    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    net_pnl = sum(pnls)
    total_cost_r = sum(t.get("cost_r", 0) for t in trade_results)

    # Max drawdown from equity curve
    max_dd, max_dd_pct = _calculate_max_drawdown(equity_curve, initial_equity)

    # Sharpe ratio (annualized, assuming 15m candles ≈ 35040 per year)
    if len(pnls) > 1:
        returns = np.array(pnls) / initial_equity
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
    else:
        sharpe = 0

    # R-multiples (directly from trade results or computed from PnL/risk)
    r_multiples = []
    for t in trade_results:
        if "r_multiple" in t:
            r_multiples.append(t["r_multiple"])
        else:
            risk = t.get("risk_amount", 0)
            if risk > 0:
                r_multiples.append(t["pnl"] / risk)
    avg_r = np.mean(r_multiples) if r_multiples else 0
    total_r = sum(r_multiples) if r_multiples else 0

    # Holding times
    holding_times = []
    for t in trade_results:
        if "entry_time" in t and "exit_time" in t and t["entry_time"] and t["exit_time"]:
            delta = t["exit_time"] - t["entry_time"]
            holding_times.append(delta.total_seconds() / 3600)  # hours
    avg_holding_hours = np.mean(holding_times) if holding_times else 0

    # Per-direction breakdown
    long_trades = [t for t in trade_results if t.get("direction") == "LONG"]
    short_trades = [t for t in trade_results if t.get("direction") == "SHORT"]

    # Per-timeframe breakdown
    tf_breakdown = {}
    for t in trade_results:
        tf = t.get("timeframe", "unknown")
        if tf not in tf_breakdown:
            tf_breakdown[tf] = {"trades": 0, "wins": 0, "pnl": 0}
        tf_breakdown[tf]["trades"] += 1
        tf_breakdown[tf]["pnl"] += t["pnl"]
        if t["pnl"] > 0:
            tf_breakdown[tf]["wins"] += 1

    report = {
        "total_trades": total_trades,
        "win_count": win_count,
        "loss_count": loss_count,
        "win_rate": round(win_rate, 4),
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "net_pnl": round(net_pnl, 2),
        "profit_factor": round(profit_factor, 3),
        "max_drawdown": round(max_dd, 2),
        "max_drawdown_pct": round(max_dd_pct, 4),
        "sharpe_ratio": round(sharpe, 3),
        "avg_r_multiple": round(avg_r, 3),
        "total_r": round(total_r, 3),
        "total_cost_r": round(total_cost_r, 3),
        "largest_win": round(max(pnls), 2) if pnls else 0,
        "largest_loss": round(min(pnls), 2) if pnls else 0,
        "avg_holding_hours": round(avg_holding_hours, 2),
        "signals_generated": signals_generated,
        "signals_taken": signals_taken,
        "be_stop_triggers": be_stop_triggers,
        "initial_equity": initial_equity,
        "final_equity": round(initial_equity + net_pnl, 2),
        "return_pct": round(net_pnl / initial_equity, 4),
        "long_trades": len(long_trades),
        "short_trades": len(short_trades),
        "long_pnl": round(sum(t["pnl"] for t in long_trades), 2),
        "short_pnl": round(sum(t["pnl"] for t in short_trades), 2),
        "timeframe_breakdown": tf_breakdown,
    }

    return report


def print_report(report: dict) -> None:
    """Print a formatted report to the logger."""
    lines = [
        "",
        "=" * 60,
        "BACKTEST REPORT",
        "=" * 60,
        f"Total Trades:      {report['total_trades']}",
        f"Win Rate:          {report['win_rate']:.1%} ({report['win_count']}W / {report['loss_count']}L)",
        f"Profit Factor:     {report['profit_factor']:.3f}",
        "",
        "--- R-Multiple Summary ---",
        f"Avg R-Multiple:    {report['avg_r_multiple']:+.3f}R",
        f"Total R:           {report['total_r']:+.3f}R",
        f"Total Costs:       {report['total_cost_r']:.3f}R",
        "",
        "--- Dollar P&L ---",
        f"Net P&L:           ${report['net_pnl']:.2f} ({report['return_pct']:.2%})",
        f"Max Drawdown:      ${report['max_drawdown']:.2f} ({report['max_drawdown_pct']:.2%})",
        f"Sharpe Ratio:      {report['sharpe_ratio']:.3f}",
        f"Largest Win:       ${report['largest_win']:.2f}",
        f"Largest Loss:      ${report['largest_loss']:.2f}",
        f"Avg Holding Time:  {report['avg_holding_hours']:.1f} hours",
        "",
        f"Signals Generated: {report['signals_generated']}",
        f"Signals Taken:     {report['signals_taken']}",
        f"BE Stop Triggers:  {report['be_stop_triggers']}",
        "",
        f"Long Trades:  {report['long_trades']} (P&L: ${report['long_pnl']:.2f})",
        f"Short Trades: {report['short_trades']} (P&L: ${report['short_pnl']:.2f})",
        "",
        f"Initial Equity: ${report['initial_equity']:.2f}",
        f"Final Equity:   ${report['final_equity']:.2f}",
        "=" * 60,
    ]

    # Timeframe breakdown
    if report.get("timeframe_breakdown"):
        lines.append("\nTimeframe Breakdown:")
        for tf, data in report["timeframe_breakdown"].items():
            wr = data["wins"] / data["trades"] if data["trades"] > 0 else 0
            lines.append(f"  {tf}: {data['trades']} trades | WR={wr:.0%} | PnL=${data['pnl']:.2f}")

    for line in lines:
        logger.info(line)

    # Also print to stdout
    print("\n".join(lines))


def save_equity_curve(equity_curve: list[dict], filename: str = "equity_curve.csv") -> Path:
    """Save the equity curve to CSV."""
    path = DATA_DIR / filename
    df = pd.DataFrame(equity_curve)
    df.to_csv(path, index=False)
    logger.info(f"Equity curve saved to {path}")
    return path


def print_tp_optimization_table(configs: list[dict]) -> None:
    """Print a formatted TP optimization comparison table."""
    header = f"{'Config':<8} {'TP1(0.5)':<10} {'TP2(0.382)':<11} {'TP3(0.236)':<11} {'TP4(ext)':<10} {'Win%':<6} {'PF':<6} {'AvgR':<6} {'MaxDD':<7}"
    print("\n" + "=" * 80)
    print("TP OPTIMIZATION RESULTS")
    print("=" * 80)
    print(header)
    print("-" * 80)

    for i, cfg in enumerate(configs):
        line = (
            f"{chr(65+i):<8} "
            f"{cfg.get('tp1_pct', 0):.0%}{'':<6} "
            f"{cfg.get('tp2_pct', 0):.0%}{'':<7} "
            f"{cfg.get('tp3_pct', 0):.0%}{'':<7} "
            f"{cfg.get('tp4_pct', 0):.0%}{'':<6} "
            f"{cfg.get('win_rate', 0):.0%}{'':<2} "
            f"{cfg.get('profit_factor', 0):.2f}{'':<2} "
            f"{cfg.get('avg_r', 0):.2f}{'':<2} "
            f"{cfg.get('max_dd_pct', 0):.1%}"
        )
        print(line)

    # Recommend best config
    valid = [c for c in configs if c.get("max_dd_pct", 1) < 0.10]
    if valid:
        best = max(valid, key=lambda c: c.get("profit_factor", 0))
        idx = configs.index(best)
        print(f"\nRecommended: Config {chr(65+idx)} (highest PF with MaxDD < 10%)")
    print("=" * 80)


def _calculate_max_drawdown(
    equity_curve: list[dict], initial_equity: float
) -> tuple[float, float]:
    """Calculate maximum drawdown from equity curve."""
    if not equity_curve:
        return 0.0, 0.0

    equities = [e["equity"] for e in equity_curve]
    peak = initial_equity
    max_dd = 0.0

    for eq in equities:
        if eq > peak:
            peak = eq
        dd = peak - eq
        if dd > max_dd:
            max_dd = dd

    max_dd_pct = max_dd / initial_equity if initial_equity > 0 else 0
    return max_dd, max_dd_pct


def _empty_report(initial_equity: float) -> dict:
    """Return an empty report when no trades were made."""
    return {
        "total_trades": 0, "win_count": 0, "loss_count": 0,
        "win_rate": 0, "gross_profit": 0, "gross_loss": 0,
        "net_pnl": 0, "profit_factor": 0, "max_drawdown": 0,
        "max_drawdown_pct": 0, "sharpe_ratio": 0, "avg_r_multiple": 0,
        "total_r": 0, "total_cost_r": 0,
        "largest_win": 0, "largest_loss": 0, "avg_holding_hours": 0,
        "signals_generated": 0, "signals_taken": 0,
        "be_stop_triggers": 0, "initial_equity": initial_equity,
        "final_equity": initial_equity, "return_pct": 0,
        "long_trades": 0, "short_trades": 0, "long_pnl": 0, "short_pnl": 0,
        "timeframe_breakdown": {},
    }
