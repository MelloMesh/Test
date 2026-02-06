"""
Portfolio tracker for equity, open positions, and P&L.
Tracks daily/weekly performance for circuit breaker enforcement.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Position:
    """An open trading position."""

    symbol: str
    direction: str               # "LONG" or "SHORT"
    entry_price: float
    size_usd: float
    size_contracts: float
    leverage: int
    stop_loss: float
    take_profits: list[dict]
    margin_used: float
    opened_at: datetime
    signal_reason: str = ""

    # Mutable state
    current_price: float = 0.0
    stop_moved_to_be: bool = False
    partial_closes: list[dict] = field(default_factory=list)

    @property
    def unrealized_pnl(self) -> float:
        """Calculate unrealized P&L based on current price."""
        if self.current_price <= 0:
            return 0.0
        if self.direction == "LONG":
            return (self.current_price - self.entry_price) / self.entry_price * self.size_usd
        else:
            return (self.entry_price - self.current_price) / self.entry_price * self.size_usd

    @property
    def remaining_size_pct(self) -> float:
        """Percentage of original position still open."""
        closed_pct = sum(pc.get("pct", 0) for pc in self.partial_closes)
        return max(0.0, 1.0 - closed_pct)


@dataclass
class TradeResult:
    """Result of a closed trade."""

    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    size_usd: float
    realized_pnl: float
    fees: float
    opened_at: datetime
    closed_at: datetime
    reason: str  # "stop_loss", "take_profit", "manual"


class Portfolio:
    """
    Track portfolio equity, positions, and trade history.
    Provides data for risk manager circuit breakers.
    """

    def __init__(self, initial_equity: float = 10000.0):
        self.initial_equity = initial_equity
        self.cash = initial_equity
        self.positions: list[Position] = []
        self.trade_history: list[TradeResult] = []
        self._start_of_day_equity: float = initial_equity
        self._start_of_week_equity: float = initial_equity
        self._day_start: datetime = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        self._week_start: datetime = self._day_start - timedelta(
            days=self._day_start.weekday()
        )

    @property
    def equity(self) -> float:
        """Total equity = cash + unrealized P&L of open positions."""
        unrealized = sum(p.unrealized_pnl for p in self.positions)
        return self.cash + unrealized

    @property
    def total_margin_used(self) -> float:
        """Total margin locked in open positions."""
        return sum(p.margin_used for p in self.positions)

    @property
    def total_exposure_pct(self) -> float:
        """Total position exposure as percentage of equity."""
        if self.equity <= 0:
            return 0.0
        total_size = sum(p.size_usd * p.remaining_size_pct for p in self.positions)
        return total_size / self.equity

    @property
    def open_position_count(self) -> int:
        return len(self.positions)

    @property
    def consecutive_losses(self) -> int:
        """Count consecutive losing trades from most recent."""
        count = 0
        for trade in reversed(self.trade_history):
            if trade.realized_pnl < 0:
                count += 1
            else:
                break
        return count

    @property
    def daily_pnl(self) -> float:
        """P&L since start of current day."""
        self._maybe_reset_daily()
        return self.equity - self._start_of_day_equity

    @property
    def daily_pnl_pct(self) -> float:
        """Daily P&L as percentage of start-of-day equity."""
        if self._start_of_day_equity <= 0:
            return 0.0
        return self.daily_pnl / self._start_of_day_equity

    @property
    def weekly_pnl(self) -> float:
        """P&L since start of current week."""
        self._maybe_reset_weekly()
        return self.equity - self._start_of_week_equity

    @property
    def weekly_pnl_pct(self) -> float:
        """Weekly P&L as percentage of start-of-week equity."""
        if self._start_of_week_equity <= 0:
            return 0.0
        return self.weekly_pnl / self._start_of_week_equity

    def open_position(self, position: Position) -> None:
        """Add a new position to the portfolio."""
        self.cash -= position.margin_used
        self.positions.append(position)
        logger.info(
            f"Opened {position.direction} {position.symbol}: "
            f"${position.size_usd:.2f} @ {position.entry_price:.2f} "
            f"(margin: ${position.margin_used:.2f})"
        )

    def close_position(
        self,
        position: Position,
        exit_price: float,
        reason: str = "manual",
        fees: float = 0.0,
    ) -> TradeResult:
        """Close a position and record the trade result."""
        if position.direction == "LONG":
            pnl = (exit_price - position.entry_price) / position.entry_price * position.size_usd
        else:
            pnl = (position.entry_price - exit_price) / position.entry_price * position.size_usd

        pnl -= fees
        self.cash += position.margin_used + pnl

        result = TradeResult(
            symbol=position.symbol,
            direction=position.direction,
            entry_price=position.entry_price,
            exit_price=exit_price,
            size_usd=position.size_usd,
            realized_pnl=pnl,
            fees=fees,
            opened_at=position.opened_at,
            closed_at=datetime.now(timezone.utc),
            reason=reason,
        )
        self.trade_history.append(result)

        if position in self.positions:
            self.positions.remove(position)

        logger.info(
            f"Closed {position.direction} {position.symbol} @ {exit_price:.2f}: "
            f"PnL=${pnl:.2f} ({reason})"
        )
        return result

    def _maybe_reset_daily(self) -> None:
        """Reset daily tracking if a new day has started."""
        now = datetime.now(timezone.utc)
        today = now.replace(hour=0, minute=0, second=0, microsecond=0)
        if today > self._day_start:
            self._start_of_day_equity = self.equity
            self._day_start = today

    def _maybe_reset_weekly(self) -> None:
        """Reset weekly tracking if a new week has started."""
        now = datetime.now(timezone.utc)
        this_monday = now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(
            days=now.weekday()
        )
        if this_monday > self._week_start:
            self._start_of_week_equity = self.equity
            self._week_start = this_monday

    def summary(self) -> dict:
        """Return a portfolio summary dict."""
        return {
            "equity": round(self.equity, 2),
            "cash": round(self.cash, 2),
            "open_positions": self.open_position_count,
            "total_margin_used": round(self.total_margin_used, 2),
            "total_exposure_pct": round(self.total_exposure_pct, 4),
            "daily_pnl": round(self.daily_pnl, 2),
            "daily_pnl_pct": round(self.daily_pnl_pct, 4),
            "weekly_pnl": round(self.weekly_pnl, 2),
            "weekly_pnl_pct": round(self.weekly_pnl_pct, 4),
            "total_trades": len(self.trade_history),
            "consecutive_losses": self.consecutive_losses,
        }
