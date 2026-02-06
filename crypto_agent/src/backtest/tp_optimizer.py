"""
Take profit allocation optimizer.
Tests multiple TP allocation configurations against backtest data
and recommends the best risk-adjusted configuration.
"""

from itertools import product

from src.utils.logger import get_logger

logger = get_logger(__name__)

# TP configurations to test
# Each config is (TP1%, TP2%, TP3%, TP4%) at fib levels (0.5, 0.382, 0.236, ext)
DEFAULT_TP_CONFIGS = [
    {"tp1_pct": 0.40, "tp2_pct": 0.30, "tp3_pct": 0.20, "tp4_pct": 0.10},
    {"tp1_pct": 0.50, "tp2_pct": 0.25, "tp3_pct": 0.15, "tp4_pct": 0.10},
    {"tp1_pct": 0.30, "tp2_pct": 0.30, "tp3_pct": 0.30, "tp4_pct": 0.10},
    {"tp1_pct": 0.30, "tp2_pct": 0.30, "tp3_pct": 0.20, "tp4_pct": 0.20},
    {"tp1_pct": 0.25, "tp2_pct": 0.25, "tp3_pct": 0.25, "tp4_pct": 0.25},
    {"tp1_pct": 0.60, "tp2_pct": 0.20, "tp3_pct": 0.10, "tp4_pct": 0.10},
    {"tp1_pct": 0.20, "tp2_pct": 0.30, "tp3_pct": 0.30, "tp4_pct": 0.20},
    {"tp1_pct": 0.35, "tp2_pct": 0.35, "tp3_pct": 0.20, "tp4_pct": 0.10},
]


def get_tp_configs() -> list[dict]:
    """Return the list of TP configurations to test."""
    return DEFAULT_TP_CONFIGS


def config_to_tp_list(config: dict) -> list[dict]:
    """
    Convert a TP config dict to the list format used by the strategy.

    Returns:
        List of {'level': float, 'pct': float} dicts.
    """
    return [
        {"level": 0.5, "pct": config["tp1_pct"]},
        {"level": 0.382, "pct": config["tp2_pct"]},
        {"level": 0.236, "pct": config["tp3_pct"]},
        {"level": 1.272, "pct": config["tp4_pct"]},
    ]


def evaluate_config(
    config: dict,
    trade_results: list[dict],
    equity_curve: list[dict],
    initial_equity: float,
) -> dict:
    """
    Evaluate a TP config's performance from backtest results.

    Args:
        config: TP allocation config.
        trade_results: List of trade result dicts.
        equity_curve: Equity curve data.
        initial_equity: Starting equity.

    Returns:
        Config dict enriched with performance metrics.
    """
    if not trade_results:
        return {**config, "win_rate": 0, "profit_factor": 0, "avg_r": 0, "max_dd_pct": 0}

    pnls = [t["pnl"] for t in trade_results]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    win_rate = len(wins) / len(pnls) if pnls else 0
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 1
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

    # R-multiples
    r_multiples = []
    for t in trade_results:
        risk = t.get("risk_amount", 0)
        if risk > 0:
            r_multiples.append(t["pnl"] / risk)
    avg_r = sum(r_multiples) / len(r_multiples) if r_multiples else 0

    # Max drawdown
    peak = initial_equity
    max_dd = 0
    for e in equity_curve:
        eq = e["equity"]
        if eq > peak:
            peak = eq
        dd = peak - eq
        if dd > max_dd:
            max_dd = dd
    max_dd_pct = max_dd / initial_equity if initial_equity > 0 else 0

    return {
        **config,
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 3),
        "avg_r": round(avg_r, 3),
        "max_dd_pct": round(max_dd_pct, 4),
        "total_trades": len(trade_results),
        "net_pnl": round(sum(pnls), 2),
    }


def recommend_best_config(results: list[dict]) -> dict | None:
    """
    Recommend the best TP config based on:
    1. Max drawdown < 10%
    2. Highest profit factor among valid configs.

    Returns:
        The recommended config dict, or None if no valid configs.
    """
    valid = [r for r in results if r.get("max_dd_pct", 1.0) < 0.10]
    if not valid:
        # Fallback: just pick lowest drawdown
        valid = sorted(results, key=lambda r: r.get("max_dd_pct", 1.0))
        if valid:
            logger.warning("No config with MaxDD < 10%. Using lowest drawdown config.")
            return valid[0]
        return None

    best = max(valid, key=lambda r: r.get("profit_factor", 0))
    logger.info(
        f"Recommended TP config: TP1={best['tp1_pct']:.0%}, TP2={best['tp2_pct']:.0%}, "
        f"TP3={best['tp3_pct']:.0%}, TP4={best['tp4_pct']:.0%} "
        f"(PF={best['profit_factor']:.2f}, WR={best['win_rate']:.0%}, MaxDD={best['max_dd_pct']:.1%})"
    )
    return best
