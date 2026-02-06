"""
Main trading agent loop.
Monitors markets, generates signals, and executes trades.

Usage:
    python3 -m src.agent.trading_agent --mode paper
    python3 -m src.agent.trading_agent --mode paper --symbols BTC/USDT:USDT,ETH/USDT:USDT
"""

import argparse
import asyncio
import signal
import sys
import time
from datetime import datetime, timezone

import pandas as pd

from src.config import (
    DEFAULT_LEVERAGE,
    DEFAULT_SYMBOLS,
    LIVE_TRADING_ENABLED,
    STRATEGY_TIMEFRAMES,
)
from src.data.fetcher import fetch_candles
from src.execution.order_manager import OrderManager
from src.execution.position_tracker import PositionTracker
from src.execution.stop_handler import check_and_move_stops
from src.exchange import get_exchange, get_public_exchange
from src.indicators.fibonacci import (
    calculate_fib_levels,
    detect_swings,
    get_latest_swing_pair,
    is_in_golden_pocket,
)
from src.monitoring.dashboard import print_dashboard
from src.risk.portfolio import Portfolio, Position
from src.risk.position_sizer import calculate_position_size
from src.risk.risk_manager import RiskManager
from src.strategy.golden_pocket import analyze_timeframe
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TradingAgent:
    """
    Main trading agent that orchestrates all components.

    Loop:
    1. Fetch latest candles (15m and 30m)
    2. Detect swings → calculate fib levels
    3. Check if price is in golden pocket
    4. If yes: compute RSI, check for divergence
    5. If signal: score confluence → risk check → size → place orders
    6. Monitor open positions for stop-to-breakeven
    7. Display dashboard
    8. Sleep until next candle close
    """

    def __init__(
        self,
        symbols: list[str] | None = None,
        timeframes: list[str] | None = None,
        initial_equity: float = 10000.0,
        leverage: int = DEFAULT_LEVERAGE,
        paper_mode: bool = True,
    ):
        self.symbols = symbols or DEFAULT_SYMBOLS
        self.timeframes = timeframes or STRATEGY_TIMEFRAMES
        self.leverage = leverage
        self.paper_mode = paper_mode

        # Components
        self.public_exchange = get_public_exchange()
        self.private_exchange = None
        if not paper_mode and LIVE_TRADING_ENABLED:
            self.private_exchange = get_exchange(private=True)

        self.portfolio = Portfolio(initial_equity)
        self.risk_manager = RiskManager(paper_mode=paper_mode)
        self.order_manager = OrderManager(self.private_exchange)
        self.tracker = PositionTracker(self.portfolio, self.private_exchange)

        # State
        self.candles_cache: dict[str, dict[str, pd.DataFrame]] = {}  # symbol → {tf → df}
        self.fib_info: dict[str, dict] = {}
        self.last_signal_time: datetime | None = None
        self.running = True
        self.cycle_count = 0

        # Graceful shutdown
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _handle_shutdown(self, signum, frame):
        logger.info("Shutdown signal received. Stopping gracefully...")
        self.running = False

    def run(self) -> None:
        """Main trading loop."""
        mode_str = "PAPER" if self.paper_mode else "LIVE"
        logger.info(f"Trading agent starting in {mode_str} mode")
        logger.info(f"Symbols: {', '.join(self.symbols)}")
        logger.info(f"Timeframes: {', '.join(self.timeframes)}")
        logger.info(f"Leverage: {self.leverage}x")

        # Initial data fetch
        self._fetch_all_candles()

        # Show initial state
        prices = self._get_current_prices()
        if prices:
            logger.info("Connected. Current prices:")
            for sym, price in prices.items():
                logger.info(f"  {sym}: ${price:,.2f}")

        while self.running:
            try:
                self.cycle_count += 1
                self._run_cycle()
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error in trading cycle: {e}", exc_info=True)
                time.sleep(10)  # Back off on errors

        logger.info("Trading agent stopped.")
        logger.info(f"Final portfolio: {self.portfolio.summary()}")

    def _run_cycle(self) -> None:
        """Execute one cycle of the trading loop."""
        # 1. Fetch latest candles
        self._fetch_all_candles()

        # 2. Get current prices
        prices = self._get_current_prices()
        self.tracker.update_prices(prices)

        # 3. Update fib levels for each symbol
        self._update_fib_levels()

        # 4. Check for signals on each symbol/timeframe
        for symbol in self.symbols:
            for tf in self.timeframes:
                candles = self._get_candles(symbol, tf)
                if candles is None or len(candles) < 50:
                    continue

                signals = analyze_timeframe(candles, symbol, tf, anticipatory=True)

                for sig in signals:
                    self._process_signal(sig)

        # 5. Monitor stops for breakeven
        candles_for_stops = {}
        for symbol in self.symbols:
            # Use 15m candles for stop monitoring
            candles_for_stops[symbol] = self._get_candles(symbol, "15m")
        be_moves = check_and_move_stops(self.tracker, self.order_manager, candles_for_stops)
        if be_moves > 0:
            logger.info(f"Moved {be_moves} stop(s) to breakeven")

        # 6. Display dashboard
        print_dashboard(
            self.portfolio,
            self.tracker,
            prices,
            self.fib_info,
            self.last_signal_time,
        )

        # 7. Sleep until next candle period
        # For 15m candles, sleep ~60 seconds between checks
        sleep_time = 60
        logger.debug(f"Cycle {self.cycle_count} complete. Sleeping {sleep_time}s...")
        time.sleep(sleep_time)

    def _process_signal(self, sig) -> None:
        """Process a trade signal through risk management and execution."""
        # Skip old signals (more than 2 candle periods old)
        now = datetime.now(timezone.utc)
        if hasattr(sig.timestamp, 'tzinfo') and sig.timestamp.tzinfo is not None:
            age_seconds = (now - sig.timestamp).total_seconds()
        else:
            age_seconds = 0

        if age_seconds > 3600:  # Older than 1 hour
            return

        # Risk check
        allowed, reason = self.risk_manager.can_open_position(sig, self.portfolio)
        if not allowed:
            logger.debug(f"Signal rejected by risk manager: {reason}")
            return

        # Position sizing
        try:
            sizing = calculate_position_size(
                equity=self.portfolio.equity,
                risk_pct=sig.risk_pct,
                entry_price=sig.entry_price,
                stop_loss_price=sig.stop_loss,
                leverage=self.leverage,
            )
        except ValueError as e:
            logger.warning(f"Position sizing failed: {e}")
            return

        # Check margin
        if sizing["margin_required"] > self.portfolio.cash:
            logger.warning("Insufficient margin for position")
            return

        # Execute entry
        entry_order = self.order_manager.place_entry(
            symbol=sig.symbol,
            direction=sig.direction,
            size_contracts=sizing["size_contracts"],
            current_price=sig.entry_price,
            leverage=self.leverage,
        )

        # Place stop loss
        stop_order = self.order_manager.place_stop_loss(
            symbol=sig.symbol,
            direction=sig.direction,
            size_contracts=sizing["size_contracts"],
            stop_price=sig.stop_loss,
        )

        # Place take profits
        tp_order_ids = []
        for tp in sig.take_profits:
            tp_size = sizing["size_contracts"] * tp["pct"]
            tp_order = self.order_manager.place_take_profit(
                symbol=sig.symbol,
                direction=sig.direction,
                size_contracts=tp_size,
                tp_price=tp["price"],
            )
            tp_order_ids.append(tp_order.get("id", ""))

        # Create position and add to portfolio
        position = Position(
            symbol=sig.symbol,
            direction=sig.direction,
            entry_price=sig.entry_price,
            size_usd=sizing["size_usd"],
            size_contracts=sizing["size_contracts"],
            leverage=self.leverage,
            stop_loss=sig.stop_loss,
            take_profits=sig.take_profits,
            margin_used=sizing["margin_required"],
            opened_at=datetime.now(timezone.utc),
            signal_reason=sig.reason,
            current_price=sig.entry_price,
        )

        self.portfolio.open_position(position)
        self.tracker.add_position(
            position,
            stop_order_id=stop_order.get("id", ""),
            tp_order_ids=tp_order_ids,
        )

        self.last_signal_time = datetime.now(timezone.utc)
        logger.info(f"TRADE ENTERED: {sig}")

    def _fetch_all_candles(self) -> None:
        """Fetch latest candles for all symbols and timeframes."""
        for symbol in self.symbols:
            if symbol not in self.candles_cache:
                self.candles_cache[symbol] = {}

            for tf in self.timeframes:
                try:
                    df = fetch_candles(
                        symbol=symbol,
                        timeframe=tf,
                        limit=200,
                        exchange=self.public_exchange,
                    )
                    self.candles_cache[symbol][tf] = df
                except Exception as e:
                    logger.error(f"Failed to fetch {symbol} {tf}: {e}")

    def _get_candles(self, symbol: str, timeframe: str) -> pd.DataFrame | None:
        """Get cached candles for a symbol/timeframe."""
        return self.candles_cache.get(symbol, {}).get(timeframe)

    def _get_current_prices(self) -> dict[str, float]:
        """Fetch current prices for all symbols."""
        prices = {}
        for symbol in self.symbols:
            try:
                ticker = self.public_exchange.fetch_ticker(symbol)
                prices[symbol] = ticker["last"]
            except Exception as e:
                logger.error(f"Failed to fetch price for {symbol}: {e}")
        return prices

    def _update_fib_levels(self) -> None:
        """Update Fibonacci levels for all symbols."""
        for symbol in self.symbols:
            candles = self._get_candles(symbol, "30m")  # Use 30m for structure
            if candles is None or len(candles) < 50:
                continue

            swings = detect_swings(candles)
            pair = get_latest_swing_pair(swings)
            if pair is None:
                continue

            swing_low, swing_high = pair
            fib_levels = calculate_fib_levels(swing_high["price"], swing_low["price"])

            if fib_levels:
                self.fib_info[symbol] = {
                    "golden_pocket_start": fib_levels.get(0.618, 0),
                    "golden_pocket_end": fib_levels.get(0.886, 0),
                    "swing_high": swing_high["price"],
                    "swing_low": swing_low["price"],
                    "fib_levels": fib_levels,
                }


def main():
    parser = argparse.ArgumentParser(description="Crypto Trading Agent")
    parser.add_argument(
        "--mode", choices=["paper", "live"], default="paper",
        help="Trading mode (default: paper)"
    )
    parser.add_argument(
        "--symbols", default=None,
        help="Comma-separated symbols (default: BTC/USDT:USDT,ETH/USDT:USDT)"
    )
    parser.add_argument(
        "--equity", type=float, default=10000.0,
        help="Initial equity (default: 10000)"
    )
    parser.add_argument(
        "--leverage", type=int, default=DEFAULT_LEVERAGE,
        help=f"Leverage (default: {DEFAULT_LEVERAGE})"
    )

    args = parser.parse_args()

    symbols = args.symbols.split(",") if args.symbols else None
    paper = args.mode == "paper"

    if not paper and not LIVE_TRADING_ENABLED:
        print("ERROR: Live mode requires LIVE_TRADING_ENABLED=true in .env")
        print("This is a safety check. Set it only when you're ready for real trading.")
        sys.exit(1)

    agent = TradingAgent(
        symbols=symbols,
        initial_equity=args.equity,
        leverage=args.leverage,
        paper_mode=paper,
    )
    agent.run()


if __name__ == "__main__":
    main()
