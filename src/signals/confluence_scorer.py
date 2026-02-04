"""
Confluence Scoring System
Evaluates signal quality based on multiple factors to determine order type
"""

from dataclasses import dataclass
from typing import Optional, Dict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class ConfluenceScore:
    """Result of confluence scoring"""
    total_score: int
    order_type: str  # 'MARKET', 'LIMIT', 'SKIP'
    trade_type: str  # 'TREND_CONTINUATION_LONG', 'MEAN_REVERSION_SHORT', etc.
    position_size_multiplier: float
    limit_price: Optional[float]
    reasoning: str


class ConfluenceScorer:
    """
    Scores trading signals based on multiple confluences

    Scoring Factors (max 12 points):
    - HTF Alignment (0-2 pts): How aligned are weekly/daily/4h?
    - Price at S/R (0-3 pts): Are we at a key level?
    - LTF Technical (0-2 pts): Quality of LTF signal (divergence, etc.)
    - Volume (0-2 pts): Volume confirmation
    - Order Blocks (0-2 pts): At institutional accumulation/distribution zone
    - Fair Value Gaps (0-2 pts): Price filling imbalance
    - Multiple Timeframe Confirmation (0-1 pt): Signal on multiple TFs

    Decision:
    - 7+ points = MARKET order (ultra high conviction)
    - 4-6 points = MARKET order (high conviction)
    - 2-3 points = LIMIT order (wait for better price)
    - 0-1 points = SKIP (not enough confluence)
    """

    def __init__(self):
        # Scoring thresholds
        self.market_threshold_ultra = 7  # Ultra high conviction
        self.market_threshold = 4         # High conviction
        self.limit_threshold = 2          # Medium conviction
        # Below limit_threshold = SKIP

        # S/R proximity threshold (percentage)
        self.sr_proximity_perfect = 0.005  # Within 0.5% = perfect
        self.sr_proximity_good = 0.015     # Within 1.5% = good
        self.sr_proximity_ok = 0.03        # Within 3% = ok

    def evaluate_signal(
        self,
        symbol: str,
        signal_name: str,
        direction: str,
        current_price: float,
        htf_context,  # HTFContext object
        has_divergence: bool = False,
        has_volume_confirmation: bool = False,
        multi_tf_confirmation: bool = False,
        rsi_value: Optional[float] = None
    ) -> ConfluenceScore:
        """
        Evaluate signal and determine order type

        Args:
            symbol: Trading symbol
            signal_name: Signal type
            direction: 'long' or 'short'
            current_price: Current market price
            htf_context: HTFContext object with market context
            has_divergence: Does signal have RSI/MACD divergence?
            has_volume_confirmation: Volume spike supporting direction?
            multi_tf_confirmation: Signal confirmed on multiple LTFs?
            rsi_value: Current RSI value (if applicable)

        Returns:
            ConfluenceScore with order decision
        """
        score = 0
        reasons = []

        # === 1. HTF ALIGNMENT (0-2 points) ===
        if htf_context.alignment_score >= 70:
            score += 2
            reasons.append(f"Strong HTF alignment ({htf_context.alignment_score:.0f}%)")
        elif htf_context.alignment_score >= 50:
            score += 1
            reasons.append(f"Moderate HTF alignment ({htf_context.alignment_score:.0f}%)")

        # === 2. PRICE AT S/R LEVEL (0-3 points) - MOST IMPORTANT ===
        sr_score, sr_reason, at_level = self._score_sr_location(
            direction, current_price, htf_context
        )
        score += sr_score
        if sr_reason:
            reasons.append(sr_reason)

        # === 3. LTF TECHNICAL SIGNALS (0-2 points) ===
        if has_divergence:
            score += 1
            reasons.append("Divergence detected")

        # RSI oversold/overbought
        if rsi_value is not None:
            if direction == "long" and rsi_value < 30:
                score += 1
                reasons.append(f"RSI oversold ({rsi_value:.0f})")
            elif direction == "short" and rsi_value > 70:
                score += 1
                reasons.append(f"RSI overbought ({rsi_value:.0f})")

        # === 4. VOLUME CONFIRMATION (0-2 points) ===
        if has_volume_confirmation:
            score += 2
            reasons.append("Volume confirmation")

        # === 5. INSTITUTIONAL CONCEPTS (0-4 points total) ===
        # Order Blocks (0-2 points) - Institutional zones
        if "Order_Block" in signal_name:
            score += 2
            reasons.append("At institutional order block zone")

        # Fair Value Gaps (0-2 points) - Imbalance fills
        if "FVG" in signal_name:
            score += 2
            reasons.append("Filling fair value gap (imbalance)")

        # === 6. MULTI-TIMEFRAME CONFIRMATION (0-1 point) ===
        if multi_tf_confirmation:
            score += 1
            reasons.append("Multi-TF confirmation")

        # === DETERMINE ORDER TYPE AND TRADE TYPE ===
        order_type, trade_type, position_multiplier, limit_price = self._determine_order_type(
            score, direction, current_price, htf_context, at_level
        )

        # Build reasoning string
        reasoning = f"{symbol} {direction.upper()} {signal_name}: Score {score}/10 - " + ", ".join(reasons)

        return ConfluenceScore(
            total_score=score,
            order_type=order_type,
            trade_type=trade_type,
            position_size_multiplier=position_multiplier,
            limit_price=limit_price,
            reasoning=reasoning
        )

    def _score_sr_location(
        self,
        direction: str,
        current_price: float,
        htf_context
    ) -> tuple:
        """
        Score based on price proximity to S/R levels

        Returns: (score, reason, at_level_type)
        """
        score = 0
        reason = None
        at_level = None

        # Check weekly levels (most important)
        if direction == "long":
            # For longs, we want to be near support
            if htf_context.weekly_support:
                dist = abs(current_price - htf_context.weekly_support) / current_price
                if dist < self.sr_proximity_perfect:
                    score = 3
                    reason = f"Perfect entry at weekly support (${htf_context.weekly_support:,.2f})"
                    at_level = 'weekly_support'
                elif dist < self.sr_proximity_good:
                    score = 2
                    reason = f"Near weekly support (${htf_context.weekly_support:,.2f})"
                    at_level = 'weekly_support'
                elif dist < self.sr_proximity_ok:
                    score = 1
                    reason = f"Approaching weekly support"
                    at_level = 'weekly_support'

            # Also check daily support if not at weekly
            if score == 0 and htf_context.daily_support:
                dist = abs(current_price - htf_context.daily_support) / current_price
                if dist < self.sr_proximity_good:
                    score = 2
                    reason = f"At daily support (${htf_context.daily_support:,.2f})"
                    at_level = 'daily_support'

        elif direction == "short":
            # For shorts, we want to be near resistance
            if htf_context.weekly_resistance:
                dist = abs(current_price - htf_context.weekly_resistance) / current_price
                if dist < self.sr_proximity_perfect:
                    score = 3
                    reason = f"Perfect entry at weekly resistance (${htf_context.weekly_resistance:,.2f})"
                    at_level = 'weekly_resistance'
                elif dist < self.sr_proximity_good:
                    score = 2
                    reason = f"Near weekly resistance (${htf_context.weekly_resistance:,.2f})"
                    at_level = 'weekly_resistance'
                elif dist < self.sr_proximity_ok:
                    score = 1
                    reason = f"Approaching weekly resistance"
                    at_level = 'weekly_resistance'

            # Check daily resistance if not at weekly
            if score == 0 and htf_context.daily_resistance:
                dist = abs(current_price - htf_context.daily_resistance) / current_price
                if dist < self.sr_proximity_good:
                    score = 2
                    reason = f"At daily resistance (${htf_context.daily_resistance:,.2f})"
                    at_level = 'daily_resistance'

        return score, reason, at_level

    def _determine_order_type(
        self,
        score: int,
        direction: str,
        current_price: float,
        htf_context,
        at_level: Optional[str]
    ) -> tuple:
        """
        Determine order type, trade type, position size, and limit price

        Returns: (order_type, trade_type, position_multiplier, limit_price)
        """
        # Default: SKIP
        if score < self.limit_threshold:
            return "SKIP", "INSUFFICIENT_CONFLUENCE", 0.0, None

        # Determine if this is trend-following or mean reversion
        is_with_trend = (
            (direction == "long" and htf_context.primary_bias == "bullish") or
            (direction == "short" and htf_context.primary_bias == "bearish")
        )

        is_counter_trend = (
            (direction == "long" and htf_context.primary_bias == "bearish") or
            (direction == "short" and htf_context.primary_bias == "bullish")
        )

        # === MARKET ORDERS ===
        if score >= self.market_threshold:

            # Trend continuation (highest conviction)
            if is_with_trend and at_level in ['weekly_support', 'daily_support', 'weekly_resistance', 'daily_resistance']:
                if score >= self.market_threshold_ultra:
                    trade_type = f"TREND_CONTINUATION_{direction.upper()}_ULTRA"
                    position_multiplier = 1.5  # Larger size
                else:
                    trade_type = f"TREND_CONTINUATION_{direction.upper()}"
                    position_multiplier = 1.2

            # Mean reversion (medium conviction)
            elif is_counter_trend and at_level in ['weekly_resistance', 'weekly_support', 'daily_resistance', 'daily_support']:
                trade_type = f"MEAN_REVERSION_{direction.upper()}"
                position_multiplier = 0.7  # Smaller size (riskier)

            # Trend-following mid-range
            elif is_with_trend:
                trade_type = f"TREND_FOLLOWING_{direction.upper()}"
                position_multiplier = 1.0  # Normal size

            # Other (neutral HTF or ranging)
            else:
                trade_type = f"OPPORTUNITY_{direction.upper()}"
                position_multiplier = 0.8

            return "MARKET", trade_type, position_multiplier, None

        # === LIMIT ORDERS ===
        else:  # score >= limit_threshold but < market_threshold
            trade_type = f"LIMIT_{direction.upper()}"
            position_multiplier = 1.0

            # Calculate limit price based on direction and S/R levels
            if direction == "long":
                # Place limit at support
                if htf_context.weekly_support:
                    limit_price = htf_context.weekly_support
                elif htf_context.daily_support:
                    limit_price = htf_context.daily_support
                else:
                    # No clear support, place limit 2% below current price
                    limit_price = current_price * 0.98

            else:  # short
                # Place limit at resistance
                if htf_context.weekly_resistance:
                    limit_price = htf_context.weekly_resistance
                elif htf_context.daily_resistance:
                    limit_price = htf_context.daily_resistance
                else:
                    # No clear resistance, place limit 2% above current price
                    limit_price = current_price * 1.02

            return "LIMIT", trade_type, position_multiplier, limit_price


# Singleton instance
_scorer_instance = None

def get_confluence_scorer() -> ConfluenceScorer:
    """Get global confluence scorer instance"""
    global _scorer_instance
    if _scorer_instance is None:
        _scorer_instance = ConfluenceScorer()
    return _scorer_instance
