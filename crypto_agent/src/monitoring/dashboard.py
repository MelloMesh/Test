"""
Terminal dashboard for monitoring the trading agent.
Displays current status, positions, and recent activity.
"""

from datetime import datetime, timezone

from src.execution.position_tracker import PositionTracker
from src.risk.portfolio import Portfolio
from src.utils.logger import get_logger

logger = get_logger(__name__)


def print_dashboard(
    portfolio: Portfolio,
    tracker: PositionTracker,
    current_prices: dict[str, float],
    fib_info: dict[str, dict] | None = None,
    last_signal_time: datetime | None = None,
) -> None:
    """
    Print a terminal status dashboard.

    Args:
        portfolio: Current portfolio state.
        tracker: Position tracker.
        current_prices: Dict of symbol → price.
        fib_info: Optional dict of symbol → {golden_pocket_start, golden_pocket_end}.
        last_signal_time: When the last signal was generated.
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    summary = portfolio.summary()

    lines = [
        "",
        f"{'='*60}",
        f" CRYPTO TRADING AGENT — {now}",
        f"{'='*60}",
        "",
        f" Equity:    ${summary['equity']:>12,.2f}  (cash: ${summary['cash']:>10,.2f})",
        f" Daily P&L: ${summary['daily_pnl']:>12,.2f}  ({summary['daily_pnl_pct']:>+.2%})",
        f" Weekly P&L:${summary['weekly_pnl']:>12,.2f}  ({summary['weekly_pnl_pct']:>+.2%})",
        f" Exposure:  {summary['total_exposure_pct']:>12.2%}",
        f" Positions: {summary['open_positions']}/3",
        f" Trades:    {summary['total_trades']}  (losing streak: {summary['consecutive_losses']})",
        "",
    ]

    # Current prices
    lines.append(" Prices:")
    for symbol, price in current_prices.items():
        line = f"   {symbol}: ${price:,.2f}"
        if fib_info and symbol in fib_info:
            fi = fib_info[symbol]
            gp_start = fi.get("golden_pocket_start", 0)
            gp_end = fi.get("golden_pocket_end", 0)
            in_gp = gp_end <= price <= gp_start if gp_start > gp_end else gp_start <= price <= gp_end
            gp_status = " [IN GOLDEN POCKET]" if in_gp else ""
            line += f"  GP: {gp_end:,.2f}–{gp_start:,.2f}{gp_status}"
        lines.append(line)

    # Open positions
    if tracker.open_count > 0:
        lines.append("")
        lines.append(" Open Positions:")
        for pos_info in tracker.summary():
            pnl_str = f"${pos_info['unrealized_pnl']:+,.2f}"
            be_str = " [BE]" if pos_info['stop_at_be'] else ""
            lines.append(
                f"   {pos_info['direction']:5s} {pos_info['symbol']}: "
                f"entry={pos_info['entry']:,.2f}  now={pos_info['current']:,.2f}  "
                f"PnL={pnl_str}{be_str}"
            )
    else:
        lines.append("")
        lines.append(" No open positions — waiting for signals...")

    if last_signal_time:
        lines.append(f"\n Last signal: {last_signal_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")

    lines.append(f"{'='*60}")

    output = "\n".join(lines)
    print(output)
    logger.debug("Dashboard refreshed")
