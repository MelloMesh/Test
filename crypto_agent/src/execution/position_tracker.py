"""
Position tracker — syncs local position state with exchange.
For paper mode, tracks positions locally.
For live mode, queries exchange for current positions.
"""

from datetime import datetime, timezone

from src.config import LIVE_TRADING_ENABLED
from src.risk.portfolio import Portfolio, Position
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PositionTracker:
    """
    Track open positions and sync with exchange state.
    """

    def __init__(self, portfolio: Portfolio, exchange=None):
        self.portfolio = portfolio
        self.exchange = exchange
        self.is_live = LIVE_TRADING_ENABLED and exchange is not None
        # Map of symbol → {position, stop_order_id, tp_order_ids, stop_moved_to_be}
        self.tracked: dict[str, dict] = {}

    def add_position(
        self,
        position: Position,
        stop_order_id: str = "",
        tp_order_ids: list[str] | None = None,
    ) -> None:
        """Register a new position for tracking."""
        key = f"{position.symbol}_{position.direction}"
        self.tracked[key] = {
            "position": position,
            "stop_order_id": stop_order_id,
            "tp_order_ids": tp_order_ids or [],
            "stop_moved_to_be": False,
        }
        logger.info(f"Tracking position: {key}")

    def remove_position(self, symbol: str, direction: str) -> None:
        """Remove a position from tracking."""
        key = f"{symbol}_{direction}"
        if key in self.tracked:
            del self.tracked[key]
            logger.info(f"Removed tracked position: {key}")

    def get_tracked(self, symbol: str, direction: str) -> dict | None:
        """Get tracked position data."""
        return self.tracked.get(f"{symbol}_{direction}")

    def update_prices(self, prices: dict[str, float]) -> None:
        """
        Update current prices for all tracked positions.

        Args:
            prices: Dict mapping symbol to current price.
        """
        for key, data in self.tracked.items():
            pos = data["position"]
            if pos.symbol in prices:
                pos.current_price = prices[pos.symbol]

    def sync_with_exchange(self) -> None:
        """
        Sync local state with exchange (live mode only).
        Checks if positions/orders still exist on the exchange.
        """
        if not self.is_live or not self.exchange:
            return

        try:
            positions = self.exchange.fetch_positions()
            exchange_positions = {
                p["symbol"]: p for p in positions
                if p.get("contracts", 0) > 0
            }

            for key in list(self.tracked.keys()):
                pos = self.tracked[key]["position"]
                if pos.symbol not in exchange_positions:
                    logger.warning(
                        f"Position {key} no longer exists on exchange — removing"
                    )
                    del self.tracked[key]

        except Exception as e:
            logger.error(f"Failed to sync with exchange: {e}")

    @property
    def open_count(self) -> int:
        return len(self.tracked)

    def summary(self) -> list[dict]:
        """Return summary of all tracked positions."""
        result = []
        for key, data in self.tracked.items():
            pos = data["position"]
            result.append({
                "symbol": pos.symbol,
                "direction": pos.direction,
                "entry": pos.entry_price,
                "current": pos.current_price,
                "size_usd": pos.size_usd,
                "unrealized_pnl": pos.unrealized_pnl,
                "stop_at_be": data["stop_moved_to_be"],
            })
        return result
