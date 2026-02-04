"""
HTF Support/Resistance Detection Agent.

Detects support and resistance levels across multiple higher timeframes (HTF)
and identifies confluence zones where multiple timeframe levels cluster.
"""

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import logging

from .base_agent import BaseAgent
from ..exchange.base import BaseExchange
from ..schemas import SRLevel, SRConfluence


class SRDetectionAgent(BaseAgent):
    """
    Agent for detecting support/resistance levels across higher timeframes.

    Analyzes price action across multiple timeframes to identify key levels
    where price has historically reversed or consolidated. Weights levels
    by timeframe importance and identifies confluence zones.
    """

    def __init__(
        self,
        exchange: BaseExchange,
        symbols: List[str],
        timeframes: Optional[List[str]] = None,
        lookback: int = 100,
        min_touches: int = 2,
        confluence_tolerance: float = 0.015,
        update_interval: int = 300
    ):
        """
        Initialize S/R Detection Agent.

        Args:
            exchange: Exchange adapter instance
            symbols: List of trading symbols to analyze
            timeframes: List of timeframes to analyze (default: 1M, 1w, 3d, 1d, 4h, 1h)
            lookback: Number of candles to look back
            min_touches: Minimum touches for a valid level
            confluence_tolerance: Price tolerance for clustering (1.5% default)
            update_interval: Seconds between updates
        """
        super().__init__(
            name="S/R Detection",
            exchange=exchange,
            update_interval=update_interval
        )

        self.symbols = symbols
        self.lookback = lookback
        self.min_touches = min_touches
        self.confluence_tolerance = confluence_tolerance

        # Default HTF timeframes with weights (higher = more important)
        self.timeframes = timeframes or ['1M', '1w', '3d', '1d', '4h', '1h']
        self.timeframe_weights = {
            '1M': 10,  # Monthly - Institutional levels
            '1w': 8,   # Weekly - Major swing levels
            '3d': 6,   # 3-day - Strong intermediate
            '1d': 5,   # Daily - Key day trading levels
            '4h': 3,   # 4-hour - Intraday levels
            '1h': 2    # 1-hour - Short-term levels
        }

        # Map our interval format to minutes
        self.interval_map = {
            '1M': '43200',  # 30 days * 24 * 60
            '1w': '10080',  # 7 days * 24 * 60
            '3d': '4320',   # 3 days * 24 * 60
            '1d': '1440',   # 24 * 60
            '4h': '240',
            '1h': '60'
        }

        # Storage for detected levels
        self.sr_levels: Dict[str, List[SRLevel]] = {}  # symbol -> levels
        self.confluence_zones: Dict[str, List[SRConfluence]] = {}  # symbol -> zones
        self.current_prices: Dict[str, float] = {}

        self.logger = logging.getLogger(self.__class__.__name__)

    async def execute(self):
        """Main execution loop for S/R detection."""
        try:
            # Get symbols from exchange if not provided
            if not self.symbols:
                self.symbols = await self.exchange.get_trading_symbols()
                self.logger.info(f"Loaded {len(self.symbols)} symbols from exchange")

            for symbol in self.symbols:
                # Get current price
                ticker = await self.exchange.get_ticker(symbol)
                if not ticker or 'last_price' not in ticker:
                    continue

                current_price = ticker['last_price']
                self.current_prices[symbol] = current_price

                # Detect S/R levels across all timeframes
                all_levels = []

                for timeframe in self.timeframes:
                    levels = await self._detect_levels_for_timeframe(
                        symbol, timeframe, current_price
                    )
                    all_levels.extend(levels)

                # Store levels
                self.sr_levels[symbol] = all_levels

                # Find confluence zones
                confluence_zones = self._find_confluence_zones(all_levels, current_price)
                self.confluence_zones[symbol] = confluence_zones

                self.logger.debug(
                    f"{symbol}: Found {len(all_levels)} S/R levels, "
                    f"{len(confluence_zones)} confluence zones"
                )

                # Small delay to avoid overwhelming the API
                await asyncio.sleep(0.1)

        except Exception as e:
            self.logger.error(f"Error in S/R detection: {e}")

    async def _detect_levels_for_timeframe(
        self,
        symbol: str,
        timeframe: str,
        current_price: float
    ) -> List[SRLevel]:
        """
        Detect S/R levels for a specific timeframe.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe to analyze
            current_price: Current price of the asset

        Returns:
            List of detected S/R levels
        """
        try:
            # Get klines data
            interval = self.interval_map.get(timeframe, '60')
            klines = await self.exchange.get_klines(
                symbol=symbol,
                interval=interval,
                limit=self.lookback
            )

            if len(klines) < 10:
                return []

            # Extract highs and lows
            highs = [k['high'] for k in klines]
            lows = [k['low'] for k in klines]
            timestamps = [k['timestamp'] for k in klines]

            # Find swing highs (resistance)
            resistance_levels = self._find_swing_points(
                highs, timestamps, 'resistance', timeframe
            )

            # Find swing lows (support)
            support_levels = self._find_swing_points(
                lows, timestamps, 'support', timeframe
            )

            # Combine and filter
            all_levels = resistance_levels + support_levels

            # Filter levels that are too far from current price (>20%)
            filtered_levels = [
                level for level in all_levels
                if abs(level.price - current_price) / current_price < 0.20
            ]

            return filtered_levels

        except Exception as e:
            self.logger.error(f"Error detecting levels for {symbol} {timeframe}: {e}")
            return []

    def _find_swing_points(
        self,
        prices: List[float],
        timestamps: List[datetime],
        level_type: str,
        timeframe: str
    ) -> List[SRLevel]:
        """
        Find swing high/low points in price data.

        Args:
            prices: List of prices (highs or lows)
            timestamps: Corresponding timestamps
            level_type: 'support' or 'resistance'
            timeframe: Timeframe being analyzed

        Returns:
            List of S/R levels
        """
        if len(prices) < 5:
            return []

        swing_points = []
        window = 3  # Look 3 candles on each side

        # Find swing points (local extrema)
        for i in range(window, len(prices) - window):
            is_swing = False

            if level_type == 'resistance':
                # Swing high: higher than surrounding candles
                is_swing = all(
                    prices[i] >= prices[j]
                    for j in range(i - window, i + window + 1)
                    if j != i
                )
            else:
                # Swing low: lower than surrounding candles
                is_swing = all(
                    prices[i] <= prices[j]
                    for j in range(i - window, i + window + 1)
                    if j != i
                )

            if is_swing:
                swing_points.append({
                    'price': prices[i],
                    'timestamp': timestamps[i],
                    'index': i
                })

        # Cluster nearby swing points (within 0.5%)
        clustered = self._cluster_swing_points(swing_points)

        # Create SRLevel objects
        levels = []
        timeframe_weight = self.timeframe_weights.get(timeframe, 1)

        for cluster in clustered:
            if cluster['touches'] >= self.min_touches:
                level = SRLevel(
                    price=cluster['price'],
                    strength=cluster['touches'] * timeframe_weight,
                    touches=cluster['touches'],
                    timeframe=timeframe,
                    level_type=level_type,
                    timeframe_weight=timeframe_weight,
                    first_seen=cluster['first_seen'],
                    last_touch=cluster['last_touch']
                )
                levels.append(level)

        return levels

    def _cluster_swing_points(
        self,
        swing_points: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Cluster nearby swing points into levels.

        Args:
            swing_points: List of swing point dicts

        Returns:
            List of clustered levels
        """
        if not swing_points:
            return []

        # Sort by price
        sorted_points = sorted(swing_points, key=lambda x: x['price'])

        clusters = []
        current_cluster = [sorted_points[0]]

        for point in sorted_points[1:]:
            # Check if point is within 0.5% of cluster average
            cluster_avg = sum(p['price'] for p in current_cluster) / len(current_cluster)

            if abs(point['price'] - cluster_avg) / cluster_avg < 0.005:
                current_cluster.append(point)
            else:
                # Finalize current cluster
                if len(current_cluster) > 0:
                    clusters.append(self._finalize_cluster(current_cluster))
                current_cluster = [point]

        # Don't forget last cluster
        if len(current_cluster) > 0:
            clusters.append(self._finalize_cluster(current_cluster))

        return clusters

    def _finalize_cluster(self, points: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create final cluster statistics."""
        return {
            'price': sum(p['price'] for p in points) / len(points),
            'touches': len(points),
            'first_seen': min(p['timestamp'] for p in points),
            'last_touch': max(p['timestamp'] for p in points)
        }

    def _find_confluence_zones(
        self,
        levels: List[SRLevel],
        current_price: float
    ) -> List[SRConfluence]:
        """
        Find zones where multiple timeframe levels cluster.

        Args:
            levels: All detected S/R levels
            current_price: Current price

        Returns:
            List of confluence zones
        """
        if not levels:
            return []

        # Sort levels by price
        sorted_levels = sorted(levels, key=lambda x: x.price)

        zones = []
        i = 0

        while i < len(sorted_levels):
            zone_levels = [sorted_levels[i]]
            base_price = sorted_levels[i].price

            # Find all levels within confluence_tolerance
            j = i + 1
            while j < len(sorted_levels):
                if abs(sorted_levels[j].price - base_price) / base_price < self.confluence_tolerance:
                    zone_levels.append(sorted_levels[j])
                    j += 1
                else:
                    break

            # Create confluence zone if we have multiple levels
            if len(zone_levels) >= 2:
                # Calculate zone center and score
                avg_price = sum(level.price for level in zone_levels) / len(zone_levels)
                confluence_score = sum(level.strength for level in zone_levels)

                # Determine zone type
                support_count = sum(1 for level in zone_levels if level.level_type == 'support')
                resistance_count = sum(1 for level in zone_levels if level.level_type == 'resistance')

                if support_count > resistance_count:
                    zone_type = 'support'
                elif resistance_count > support_count:
                    zone_type = 'resistance'
                else:
                    zone_type = 'both'

                # Calculate distance from current price
                distance_percent = abs(avg_price - current_price) / current_price

                zone = SRConfluence(
                    price=avg_price,
                    confluence_score=confluence_score,
                    levels=zone_levels,
                    distance_percent=distance_percent,
                    zone_type=zone_type,
                    strength=len(zone_levels)
                )
                zones.append(zone)

            i = j if j > i else i + 1

        # Sort zones by confluence score (strongest first)
        zones.sort(key=lambda x: x.confluence_score, reverse=True)

        return zones

    def get_htf_confluence(self, symbol: str, current_price: float) -> Optional[SRConfluence]:
        """
        Get the nearest significant HTF confluence zone.

        Args:
            symbol: Trading symbol
            current_price: Current price

        Returns:
            Nearest confluence zone or None
        """
        if symbol not in self.confluence_zones:
            return None

        zones = self.confluence_zones[symbol]

        # Find zones within 2% of current price
        nearby_zones = [
            zone for zone in zones
            if zone.distance_percent < 0.02
        ]

        if not nearby_zones:
            return None

        # Return strongest nearby zone
        return nearby_zones[0]

    def get_levels_for_symbol(self, symbol: str) -> List[SRLevel]:
        """Get all detected S/R levels for a symbol."""
        return self.sr_levels.get(symbol, [])

    def get_confluence_zones(self, symbol: str) -> List[SRConfluence]:
        """Get all confluence zones for a symbol."""
        return self.confluence_zones.get(symbol, [])

    def get_status(self) -> Dict[str, Any]:
        """Get agent status."""
        total_levels = sum(len(levels) for levels in self.sr_levels.values())
        total_zones = sum(len(zones) for zones in self.confluence_zones.values())

        return {
            "name": self.name,
            "running": self.running,
            "symbols_analyzed": len(self.symbols),
            "total_levels": total_levels,
            "total_confluence_zones": total_zones,
            "timeframes": self.timeframes,
            "last_update": self.last_update.isoformat() if self.last_update else None
        }
