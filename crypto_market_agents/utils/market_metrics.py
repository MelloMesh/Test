"""
Market Metrics Utilities - Liquidity and volatility calculations.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass
class AssetMetrics:
    """Comprehensive metrics for an asset."""
    symbol: str
    liquidity_usd_24h: float
    liquidity_tier: str  # 'ultra', 'high', 'medium', 'low'
    base_stop_loss_pct: float
    volatility_multiplier: float
    recommended_stop_pct: float
    market_cap_tier: Optional[str] = None  # 'large', 'mid', 'small', 'micro'


class MarketMetricsCalculator:
    """
    Calculates market metrics for dynamic risk management.

    Adjusts stop losses based on:
    - Liquidity (24h volume in USD)
    - Volatility (recent price swings)
    - Market cap tier
    """

    # Liquidity tiers (24h volume USD)
    LIQUIDITY_TIERS = {
        'ultra': 100_000_000,   # >$100M - BTC, ETH, major alts
        'high': 10_000_000,      # $10M-$100M - Top 50 alts
        'medium': 1_000_000,     # $1M-$10M - Mid-cap alts
        'low': 0                 # <$1M - Small/meme coins
    }

    # Base stop loss percentages by liquidity tier
    BASE_STOPS = {
        'ultra': 0.75,    # 0.75% - BTC/ETH level assets
        'high': 1.5,      # 1.5% - Top tier alts
        'medium': 2.5,    # 2.5% - Mid-cap alts
        'low': 4.0        # 4.0% - Low liquidity/meme coins
    }

    # Volatility multiplier ranges
    VOLATILITY_RANGES = {
        'very_low': (0.8, 1.0),      # Low volatility -> tighter stops
        'low': (1.0, 1.2),
        'normal': (1.2, 1.5),
        'high': (1.5, 2.0),
        'very_high': (2.0, 3.0)      # High volatility -> wider stops
    }

    @staticmethod
    def get_liquidity_tier(liquidity_usd_24h: float) -> str:
        """
        Determine liquidity tier based on 24h volume.

        Args:
            liquidity_usd_24h: 24-hour trading volume in USD

        Returns:
            Liquidity tier string
        """
        if liquidity_usd_24h >= MarketMetricsCalculator.LIQUIDITY_TIERS['ultra']:
            return 'ultra'
        elif liquidity_usd_24h >= MarketMetricsCalculator.LIQUIDITY_TIERS['high']:
            return 'high'
        elif liquidity_usd_24h >= MarketMetricsCalculator.LIQUIDITY_TIERS['medium']:
            return 'medium'
        else:
            return 'low'

    @staticmethod
    def calculate_volatility_multiplier(
        intraday_range_pct: float,
        lookback_volatility: Optional[float] = None
    ) -> float:
        """
        Calculate volatility multiplier for stop loss adjustment.

        Args:
            intraday_range_pct: Today's high-low range percentage
            lookback_volatility: Optional longer-term volatility metric

        Returns:
            Volatility multiplier (1.0 = normal, >1.0 = increase stops)
        """
        # Base multiplier on intraday range
        if intraday_range_pct < 2:
            base_multiplier = 0.9  # Very low volatility
        elif intraday_range_pct < 5:
            base_multiplier = 1.0  # Low volatility
        elif intraday_range_pct < 10:
            base_multiplier = 1.3  # Normal volatility
        elif intraday_range_pct < 20:
            base_multiplier = 1.7  # High volatility
        else:
            base_multiplier = 2.5  # Very high volatility (meme coins)

        # Adjust with lookback volatility if available
        if lookback_volatility is not None:
            # Blend current and historical volatility
            final_multiplier = (base_multiplier * 0.6) + (lookback_volatility * 0.4)
            return min(final_multiplier, 3.0)  # Cap at 3x

        return min(base_multiplier, 3.0)

    @staticmethod
    def calculate_asset_metrics(
        symbol: str,
        liquidity_usd_24h: float,
        intraday_range_pct: float,
        volatility_ratio: Optional[float] = None
    ) -> AssetMetrics:
        """
        Calculate comprehensive asset metrics for risk management.

        Args:
            symbol: Trading symbol
            liquidity_usd_24h: 24-hour volume in USD
            intraday_range_pct: Intraday high-low range percentage
            volatility_ratio: Optional volatility ratio from price action

        Returns:
            AssetMetrics object with all calculated values
        """
        # Determine liquidity tier
        tier = MarketMetricsCalculator.get_liquidity_tier(liquidity_usd_24h)

        # Get base stop loss for this tier
        base_stop = MarketMetricsCalculator.BASE_STOPS[tier]

        # Calculate volatility multiplier
        vol_multiplier = MarketMetricsCalculator.calculate_volatility_multiplier(
            intraday_range_pct,
            volatility_ratio
        )

        # Calculate recommended stop loss percentage
        recommended_stop = base_stop * vol_multiplier

        # Cap at reasonable limits
        recommended_stop = max(0.5, min(recommended_stop, 5.0))

        return AssetMetrics(
            symbol=symbol,
            liquidity_usd_24h=liquidity_usd_24h,
            liquidity_tier=tier,
            base_stop_loss_pct=base_stop,
            volatility_multiplier=vol_multiplier,
            recommended_stop_pct=recommended_stop
        )

    @staticmethod
    def get_stop_loss_for_asset(
        symbol: str,
        liquidity_usd_24h: float,
        intraday_range_pct: float,
        volatility_ratio: Optional[float] = None
    ) -> float:
        """
        Quick method to get recommended stop loss percentage.

        Args:
            symbol: Trading symbol
            liquidity_usd_24h: 24-hour volume in USD
            intraday_range_pct: Intraday range percentage
            volatility_ratio: Optional volatility ratio

        Returns:
            Recommended stop loss percentage
        """
        metrics = MarketMetricsCalculator.calculate_asset_metrics(
            symbol,
            liquidity_usd_24h,
            intraday_range_pct,
            volatility_ratio
        )
        return metrics.recommended_stop_pct

    @staticmethod
    def is_high_liquidity_asset(liquidity_usd_24h: float) -> bool:
        """Check if asset is high liquidity (>$10M daily volume)."""
        return liquidity_usd_24h >= MarketMetricsCalculator.LIQUIDITY_TIERS['high']

    @staticmethod
    def is_volatile_asset(intraday_range_pct: float) -> bool:
        """Check if asset is highly volatile (>10% intraday range)."""
        return intraday_range_pct > 10.0
