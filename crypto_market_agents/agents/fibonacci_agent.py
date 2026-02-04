"""
Fibonacci Agent - Detects Fibonacci retracements, extensions, and golden pocket zones.

This agent analyzes price swings to identify:
- Fibonacci retracement levels (23.6%, 38.2%, 50%, 61.8%, 75%, 78.6%)
- Golden Pocket zones (75%-78.6% deep retracement - prime entry zone)
- Fibonacci extension targets (127.2%, 161.8%, 261.8%)
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone
import logging

from .base_agent import BaseAgent
from ..exchange.base import BaseExchange
from ..schemas import FibonacciLevels


class FibonacciAgent(BaseAgent):
    """
    Agent for detecting Fibonacci levels and golden pocket zones.

    Analyzes recent price swings to identify key Fibonacci retracement
    and extension levels that institutions often use for entries and targets.
    """

    # Standard Fibonacci ratios
    RETRACEMENT_LEVELS = [0.236, 0.382, 0.500, 0.618, 0.75, 0.786]
    EXTENSION_LEVELS = [1.272, 1.618, 2.618]
    GOLDEN_POCKET_RANGE = (0.75, 0.786)  # 75% - 78.6% (deep retracement - prime entry zone)

    def __init__(
        self,
        exchange: BaseExchange,
        symbols: List[str],
        lookback: int = 100,
        min_swing_size: float = 0.02,  # 2% minimum swing
        update_interval: int = 300
    ):
        """
        Initialize Fibonacci Agent.

        Args:
            exchange: Exchange adapter instance
            symbols: List of trading symbols to analyze
            lookback: Number of candles to look back for swing detection
            min_swing_size: Minimum swing size as percentage (2% default)
            update_interval: Seconds between updates
        """
        super().__init__(
            name="Fibonacci Agent",
            exchange=exchange,
            update_interval=update_interval
        )

        self.symbols = symbols
        self.lookback = lookback
        self.min_swing_size = min_swing_size

        # Storage for detected Fibonacci levels
        self.fib_levels: Dict[str, FibonacciLevels] = {}  # symbol -> levels
        self.current_prices: Dict[str, float] = {}

        self.logger = logging.getLogger(self.__class__.__name__)

    async def execute(self):
        """Main execution loop for Fibonacci detection."""
        try:
            if not self.symbols:
                self.symbols = await self.exchange.get_trading_symbols()
                self.logger.info(f"Loaded {len(self.symbols)} symbols from exchange")

            golden_pockets_found = 0
            total_retracements = 0

            for symbol in self.symbols:
                # Get current price
                ticker = await self.exchange.get_ticker(symbol)
                if not ticker or 'last_price' not in ticker:
                    continue

                current_price = ticker['last_price']
                self.current_prices[symbol] = current_price

                # Detect Fibonacci levels
                fib_levels = await self._detect_fibonacci_levels(symbol, current_price)

                if fib_levels:
                    self.fib_levels[symbol] = fib_levels
                    total_retracements += 1

                    if fib_levels.in_golden_pocket:
                        golden_pockets_found += 1

                # Small delay to avoid overwhelming the API
                await asyncio.sleep(0.1)

            # Summary logging
            self.logger.info(
                f"Fibonacci detection complete: {total_retracements} active retracements, "
                f"{golden_pockets_found} in golden pocket zones"
            )

        except Exception as e:
            self.logger.error(f"Error in Fibonacci detection: {e}", exc_info=True)

    async def _detect_fibonacci_levels(
        self,
        symbol: str,
        current_price: float
    ) -> Optional[FibonacciLevels]:
        """
        Detect Fibonacci levels for a symbol.

        Args:
            symbol: Trading symbol
            current_price: Current price of the asset

        Returns:
            FibonacciLevels object or None
        """
        try:
            # Get klines data (daily for swing detection)
            klines = await self.exchange.get_klines(
                symbol=symbol,
                interval='1440',  # Daily
                limit=self.lookback
            )

            if len(klines) < 20:
                return None

            # Find most recent significant swing
            swing_high, swing_low, swing_direction = self._find_recent_swing(klines)

            if not swing_high or not swing_low:
                return None

            # Calculate swing size
            swing_size_pct = abs(swing_high['price'] - swing_low['price']) / swing_low['price']

            if swing_size_pct < self.min_swing_size:
                return None  # Swing too small to be significant

            # Calculate Fibonacci retracement levels
            retracements = self._calculate_retracements(
                swing_high['price'],
                swing_low['price'],
                swing_direction
            )

            # Calculate Fibonacci extension targets
            extensions = self._calculate_extensions(
                swing_high['price'],
                swing_low['price'],
                swing_direction
            )

            # Check if current price is in golden pocket (75%-78.6%)
            golden_pocket_low = retracements.get(0.786, 0)
            golden_pocket_high = retracements.get(0.75, 0)

            in_golden_pocket = False
            if swing_direction == 'bullish':
                # For bullish swing, golden pocket is below current swing high
                in_golden_pocket = golden_pocket_low <= current_price <= golden_pocket_high
            else:
                # For bearish swing, golden pocket is above current swing low
                in_golden_pocket = golden_pocket_high <= current_price <= golden_pocket_low

            # Calculate distance from golden pocket
            gp_mid = (golden_pocket_low + golden_pocket_high) / 2
            # Prevent division by zero
            distance_from_gp = abs(current_price - gp_mid) / current_price if current_price > 0 else 1.0

            return FibonacciLevels(
                symbol=symbol,
                swing_high=swing_high['price'],
                swing_low=swing_low['price'],
                swing_direction=swing_direction,
                swing_size_pct=swing_size_pct,
                retracement_levels=retracements,
                extension_levels=extensions,
                golden_pocket_low=golden_pocket_low,
                golden_pocket_high=golden_pocket_high,
                in_golden_pocket=in_golden_pocket,
                distance_from_golden_pocket=distance_from_gp,
                current_price=current_price,
                timestamp=datetime.now(timezone.utc)
            )

        except Exception as e:
            self.logger.error(f"Error detecting Fibonacci levels for {symbol}: {e}")
            return None

    def _find_recent_swing(
        self,
        klines: List[Dict[str, Any]]
    ) -> Tuple[Optional[Dict], Optional[Dict], Optional[str]]:
        """
        Find the most recent significant swing high and low.

        Args:
            klines: List of kline data

        Returns:
            Tuple of (swing_high, swing_low, direction)
        """
        if len(klines) < 10:
            return None, None, None

        highs = [k['high'] for k in klines]
        lows = [k['low'] for k in klines]
        timestamps = [k['timestamp'] for k in klines]

        # Find highest high and lowest low in recent data
        max_high = max(highs[-30:]) if len(highs) >= 30 else max(highs)
        min_low = min(lows[-30:]) if len(lows) >= 30 else min(lows)

        # Find their indices
        max_high_idx = None
        min_low_idx = None

        for i in range(len(highs) - 1, max(0, len(highs) - 30), -1):
            if highs[i] == max_high and max_high_idx is None:
                max_high_idx = i
            if lows[i] == min_low and min_low_idx is None:
                min_low_idx = i

        if max_high_idx is None or min_low_idx is None:
            return None, None, None

        swing_high = {
            'price': max_high,
            'timestamp': timestamps[max_high_idx],
            'index': max_high_idx
        }

        swing_low = {
            'price': min_low,
            'timestamp': timestamps[min_low_idx],
            'index': min_low_idx
        }

        # Determine swing direction (which came first)
        if min_low_idx < max_high_idx:
            direction = 'bullish'  # Low then high = bullish swing
        else:
            direction = 'bearish'  # High then low = bearish swing

        return swing_high, swing_low, direction

    def _calculate_retracements(
        self,
        swing_high: float,
        swing_low: float,
        direction: str
    ) -> Dict[float, float]:
        """
        Calculate Fibonacci retracement levels.

        Args:
            swing_high: Swing high price
            swing_low: Swing low price
            direction: 'bullish' or 'bearish'

        Returns:
            Dict mapping Fibonacci ratio to price level
        """
        price_range = swing_high - swing_low
        retracements = {}

        for ratio in self.RETRACEMENT_LEVELS:
            if direction == 'bullish':
                # For bullish swing, retracements go down from high
                level = swing_high - (price_range * ratio)
            else:
                # For bearish swing, retracements go up from low
                level = swing_low + (price_range * ratio)

            retracements[ratio] = level

        return retracements

    def _calculate_extensions(
        self,
        swing_high: float,
        swing_low: float,
        direction: str
    ) -> Dict[float, float]:
        """
        Calculate Fibonacci extension levels (targets).

        Args:
            swing_high: Swing high price
            swing_low: Swing low price
            direction: 'bullish' or 'bearish'

        Returns:
            Dict mapping Fibonacci extension ratio to price level
        """
        price_range = swing_high - swing_low
        extensions = {}

        for ratio in self.EXTENSION_LEVELS:
            if direction == 'bullish':
                # For bullish swing, extensions go up from high
                level = swing_high + (price_range * (ratio - 1))
            else:
                # For bearish swing, extensions go down from low
                level = swing_low - (price_range * (ratio - 1))

            extensions[ratio] = level

        return extensions

    def get_fibonacci_levels(self, symbol: str) -> Optional[FibonacciLevels]:
        """Get Fibonacci levels for a symbol."""
        return self.fib_levels.get(symbol)

    def is_in_golden_pocket(self, symbol: str) -> bool:
        """Check if price is currently in golden pocket zone."""
        fib = self.fib_levels.get(symbol)
        return fib.in_golden_pocket if fib else False

    def get_status(self) -> Dict[str, Any]:
        """Get agent status."""
        golden_pockets = sum(1 for fib in self.fib_levels.values() if fib.in_golden_pocket)

        return {
            "name": self.name,
            "running": self.running,
            "symbols_analyzed": len(self.symbols),
            "total_fibonacci_levels": len(self.fib_levels),
            "symbols_in_golden_pocket": golden_pockets,
            "last_update": self.last_update.isoformat() if self.last_update else None
        }
