"""
Order management — place, cancel, and modify orders via ccxt or paper mode.
Routes all orders through paper mode unless LIVE_TRADING_ENABLED=true.
"""

import time

from src.config import LIVE_TRADING_ENABLED, MARGIN_MODE
from src.execution import paper_mode
from src.utils.logger import get_logger

logger = get_logger(__name__)


class OrderManager:
    """
    Manages order execution.
    Routes to paper mode (mock) or live exchange based on config.
    """

    def __init__(self, exchange=None):
        self.exchange = exchange
        self.is_live = LIVE_TRADING_ENABLED and exchange is not None
        self._order_delay_ms = 100  # Rate limit courtesy between order calls

        if self.is_live:
            logger.warning("OrderManager initialized in LIVE mode")
        else:
            logger.info("OrderManager initialized in PAPER mode")

    def place_entry(
        self,
        symbol: str,
        direction: str,
        size_contracts: float,
        current_price: float,
        leverage: int = 3,
    ) -> dict:
        """
        Place an entry market order.

        Args:
            symbol: Trading pair.
            direction: "LONG" or "SHORT".
            size_contracts: Position size in base currency.
            current_price: Current market price.
            leverage: Leverage to set.

        Returns:
            Order response dict.
        """
        side = "buy" if direction == "LONG" else "sell"

        if not self.is_live:
            return paper_mode.place_market_order(symbol, side, size_contracts, current_price)

        # Live mode
        try:
            # Set leverage and margin mode
            self.exchange.set_leverage(leverage, symbol)
            self.exchange.set_margin_mode(MARGIN_MODE, symbol)
            time.sleep(self._order_delay_ms / 1000)

            order = self.exchange.create_order(
                symbol=symbol,
                type="market",
                side=side,
                amount=size_contracts,
            )
            logger.info(f"LIVE ENTRY: {side} {size_contracts:.6f} {symbol} → {order['id']}")
            return order
        except Exception as e:
            logger.error(f"Failed to place entry order: {e}")
            raise

    def place_stop_loss(
        self,
        symbol: str,
        direction: str,
        size_contracts: float,
        stop_price: float,
    ) -> dict:
        """Place a stop loss order."""
        # Stop side is opposite of position direction
        side = "sell" if direction == "LONG" else "buy"

        if not self.is_live:
            return paper_mode.place_stop_loss_order(symbol, side, size_contracts, stop_price)

        try:
            time.sleep(self._order_delay_ms / 1000)
            order = self.exchange.create_order(
                symbol=symbol,
                type="stop_market",
                side=side,
                amount=size_contracts,
                params={"stopPrice": stop_price, "closePosition": False},
            )
            logger.info(f"LIVE STOP: {side} {size_contracts:.6f} {symbol} @ {stop_price:.2f} → {order['id']}")
            return order
        except Exception as e:
            logger.error(f"Failed to place stop loss: {e}")
            raise

    def place_take_profit(
        self,
        symbol: str,
        direction: str,
        size_contracts: float,
        tp_price: float,
    ) -> dict:
        """Place a take profit order."""
        side = "sell" if direction == "LONG" else "buy"

        if not self.is_live:
            return paper_mode.place_take_profit_order(symbol, side, size_contracts, tp_price)

        try:
            time.sleep(self._order_delay_ms / 1000)
            order = self.exchange.create_order(
                symbol=symbol,
                type="take_profit_market",
                side=side,
                amount=size_contracts,
                params={"stopPrice": tp_price, "closePosition": False},
            )
            logger.info(f"LIVE TP: {side} {size_contracts:.6f} {symbol} @ {tp_price:.2f} → {order['id']}")
            return order
        except Exception as e:
            logger.error(f"Failed to place take profit: {e}")
            raise

    def cancel_order(self, symbol: str, order_id: str) -> dict:
        """Cancel an open order."""
        if not self.is_live:
            return paper_mode.cancel_order(symbol, order_id)

        try:
            result = self.exchange.cancel_order(order_id, symbol)
            logger.info(f"LIVE CANCEL: order {order_id} for {symbol}")
            return result
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            raise

    def move_stop_to_breakeven(
        self,
        symbol: str,
        direction: str,
        old_order_id: str,
        size_contracts: float,
        entry_price: float,
    ) -> dict:
        """
        Move stop loss to breakeven (entry price).
        Cancels old stop and places new one.
        """
        logger.info(f"Moving stop to breakeven for {symbol} @ {entry_price:.2f}")

        if not self.is_live:
            return paper_mode.modify_stop_loss(symbol, old_order_id, entry_price, size_contracts)

        try:
            self.cancel_order(symbol, old_order_id)
            time.sleep(self._order_delay_ms / 1000)
            return self.place_stop_loss(symbol, direction, size_contracts, entry_price)
        except Exception as e:
            logger.error(f"Failed to move stop to BE: {e}")
            raise
