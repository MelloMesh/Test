"""
Signal Synthesis Agent - Integrates all agent outputs into actionable trading signals.
"""

from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from .base_agent import BaseAgent
from ..exchange.base import BaseExchange
from ..schemas import TradingSignal, PriceActionSignal, MomentumSignal, VolumeSignal
from ..config import SignalSynthesisConfig
from ..utils.market_metrics import MarketMetricsCalculator


# Signal synthesis constants
MIN_STOP_DISTANCE_PCT = 0.1  # Minimum 0.1% distance between entry and stop
SR_CONFLUENCE_DISTANCE_PCT = 0.01  # Within 1% for S/R confluence
SR_CONFLUENCE_EXTENDED_PCT = 0.02  # Within 2% for extended S/R check
FIBONACCI_GP_DISTANCE_PCT = 0.02  # Within 2% of golden pocket
PSYCHOLOGICAL_LEVEL_TOLERANCE_LARGE = 0.01  # 1% tolerance for large prices
PSYCHOLOGICAL_LEVEL_TOLERANCE_SMALL = 0.02  # 2% tolerance for small prices
CONFLUENCE_SCORE_LIMIT_THRESHOLD = 4  # Minimum confluence score for LIMIT orders


class SignalSynthesisAgent(BaseAgent):
    """
    Agent that synthesizes signals from all other agents.

    Generates actionable trading signals with:
    - Entry/exit levels
    - Stop-loss and take-profit targets
    - Confidence scores
    - Rationale
    """

    def __init__(
        self,
        exchange: BaseExchange,
        config: SignalSynthesisConfig,
        price_action_agent,
        momentum_agent,
        volume_spike_agent,
        sr_agent=None,
        fibonacci_agent=None,
        learning_agent=None
    ):
        """
        Initialize Signal Synthesis Agent.

        Args:
            exchange: Exchange adapter
            config: Signal synthesis configuration
            price_action_agent: Price action agent instance
            momentum_agent: Momentum agent instance
            volume_spike_agent: Volume spike agent instance
            sr_agent: S/R detection agent instance (optional)
            fibonacci_agent: Fibonacci agent instance (optional)
            learning_agent: Learning agent instance (optional)
        """
        super().__init__(
            name="SignalSynthesis",
            exchange=exchange,
            update_interval=config.update_interval
        )
        self.config = config
        self.price_action_agent = price_action_agent
        self.momentum_agent = momentum_agent
        self.volume_spike_agent = volume_spike_agent
        self.sr_agent = sr_agent
        self.fibonacci_agent = fibonacci_agent
        self.learning_agent = learning_agent

    async def execute(self):
        """Execute signal synthesis."""
        # Get latest signals from all agents
        price_signals = self.price_action_agent.get_latest_signals()
        momentum_signals = self.momentum_agent.get_latest_signals()
        volume_signals = self.volume_spike_agent.get_latest_signals()

        # Create lookup dictionaries
        price_by_symbol = {s.symbol: s for s in price_signals}
        momentum_by_symbol = {}
        for s in momentum_signals:
            if s.symbol not in momentum_by_symbol:
                momentum_by_symbol[s.symbol] = []
            momentum_by_symbol[s.symbol].append(s)

        volume_by_symbol = {s.symbol: s for s in volume_signals}

        # Find symbols with signals from multiple agents
        all_symbols = set(price_by_symbol.keys()) | set(momentum_by_symbol.keys()) | set(volume_by_symbol.keys())

        trading_signals = []

        for symbol in all_symbols:
            try:
                signal = self._synthesize_signal(
                    symbol,
                    price_by_symbol.get(symbol),
                    momentum_by_symbol.get(symbol, []),
                    volume_by_symbol.get(symbol)
                )

                if signal and signal.confidence >= self.config.min_confidence:
                    trading_signals.append(signal)

            except Exception as e:
                self.logger.error(f"Error synthesizing signal for {symbol}: {e}")

        # Sort by confidence
        trading_signals.sort(key=lambda x: x.confidence, reverse=True)

        # Update signals
        self.latest_signals = trading_signals
        self.signals_generated += len(trading_signals)

        # Log signal generation stats
        self.logger.info(
            f"Signal Synthesis: Analyzed {len(all_symbols)} symbols, "
            f"generated {len(trading_signals)} signals above confidence {self.config.min_confidence}"
        )

        if trading_signals:
            self.logger.info(
                f"Generated {len(trading_signals)} trading signals. "
                f"Top confidence: {trading_signals[0].confidence:.2f}"
            )

    def _synthesize_signal(
        self,
        symbol: str,
        price_signal: Optional[PriceActionSignal],
        momentum_signals: List[MomentumSignal],
        volume_signal: Optional[VolumeSignal]
    ) -> Optional[TradingSignal]:
        """
        Synthesize a trading signal from multiple agent signals.

        Args:
            symbol: Trading symbol
            price_signal: Price action signal
            momentum_signals: List of momentum signals for different timeframes
            volume_signal: Volume spike signal

        Returns:
            Trading signal or None
        """
        if not price_signal:
            return None

        # Get S/R data if available
        sr_confluence = None
        sr_levels = None
        if self.sr_agent:
            current_price = price_signal.price
            sr_confluence = self.sr_agent.get_htf_confluence(symbol, current_price)
            sr_levels = self.sr_agent.get_levels_for_symbol(symbol)

        # Get Fibonacci data if available
        fib_levels = None
        if self.fibonacci_agent:
            fib_levels = self.fibonacci_agent.get_fibonacci_levels(symbol)

        # Get learning insights if available
        learning_insights = None
        if self.learning_agent:
            learning_insights = self.learning_agent.get_insights(symbol)

        # Determine direction and confidence
        direction, confidence, rationale_parts = self._determine_direction(
            price_signal,
            momentum_signals,
            volume_signal,
            sr_confluence,
            fib_levels,
            learning_insights
        )

        if not direction or confidence < self.config.min_confidence:
            return None

        # Calculate entry, stop, and target levels
        entry = price_signal.price
        stop, target = self._calculate_levels(
            entry,
            direction,
            price_signal,
            volume_signal,
            fib_levels,
            self.config.reward_risk_ratio,
            self.config.max_stop_loss_pct,
            sr_levels
        )

        # Build rationale
        rationale = " | ".join(rationale_parts)

        # Determine order type and confluence score
        order_type, confluence_score = self._determine_order_type(
            price_signal,
            volume_signal,
            sr_confluence,
            fib_levels,
            entry
        )

        # Create trading signal
        signal = TradingSignal(
            asset=symbol,
            direction=direction,
            entry=entry,
            stop=stop,
            target=target,
            confidence=confidence,
            rationale=rationale,
            timestamp=datetime.now(timezone.utc),
            order_type=order_type,
            confluence_score=confluence_score,
            price_signal=price_signal.to_dict() if price_signal else None,
            momentum_signal=momentum_signals[0].to_dict() if momentum_signals else None,
            volume_signal=volume_signal.to_dict() if volume_signal else None,
            sr_data=sr_confluence.to_dict() if sr_confluence else None,
            fib_data=fib_levels.to_dict() if fib_levels else None,
            learning_insights=learning_insights.to_dict() if learning_insights else None
        )

        # Open paper trade if learning agent is enabled
        if self.learning_agent and self.learning_agent.paper_trading:
            import asyncio
            asyncio.create_task(
                self.learning_agent.open_paper_trade(
                    signal,
                    sr_levels=[level.to_dict() for level in sr_levels[:5]] if sr_levels else None,
                    volume_data=volume_signal.to_dict() if volume_signal else None,
                    momentum_data=momentum_signals[0].to_dict() if momentum_signals else None
                )
            )

        return signal

    def _determine_direction(
        self,
        price_signal: PriceActionSignal,
        momentum_signals: List[MomentumSignal],
        volume_signal: Optional[VolumeSignal],
        sr_confluence=None,
        fib_levels=None,
        learning_insights=None
    ) -> tuple:
        """
        Determine trade direction and confidence.

        Returns:
            Tuple of (direction, confidence, rationale_parts)
        """
        direction = None
        confidence = 0.0
        rationale_parts = []

        # Price action analysis (15% weight)
        price_score = 0
        if price_signal.breakout_detected:
            if price_signal.price_change_pct > 0:
                price_score += 0.15
                rationale_parts.append(f"Bullish breakout (+{price_signal.price_change_pct:.1f}%)")
            else:
                price_score -= 0.15
                rationale_parts.append(f"Bearish breakout ({price_signal.price_change_pct:.1f}%)")

        if price_signal.volatility_ratio > 2.0:
            rationale_parts.append(f"High volatility ({price_signal.volatility_ratio:.1f}x)")

        # Momentum analysis (15% weight)
        momentum_score = 0
        if momentum_signals:
            # Use the strongest momentum signal
            strongest_momentum = max(momentum_signals, key=lambda x: x.strength_score)

            if strongest_momentum.status == "oversold":
                momentum_score += 0.15
                rationale_parts.append(
                    f"Oversold RSI {strongest_momentum.rsi:.1f} on {strongest_momentum.timeframe}"
                )
            elif strongest_momentum.status == "overbought":
                momentum_score -= 0.15
                rationale_parts.append(
                    f"Overbought RSI {strongest_momentum.rsi:.1f} on {strongest_momentum.timeframe}"
                )

            # OBV confirmation
            if strongest_momentum.obv_change_pct > 10:
                momentum_score += 0.03
                rationale_parts.append(f"OBV rising (+{strongest_momentum.obv_change_pct:.1f}%)")
            elif strongest_momentum.obv_change_pct < -10:
                momentum_score -= 0.03
                rationale_parts.append(f"OBV falling ({strongest_momentum.obv_change_pct:.1f}%)")

        # Volume analysis (10% weight)
        volume_score = 0
        if volume_signal and volume_signal.spike_detected:
            volume_score += 0.10
            rationale_parts.append(
                f"Volume spike (z-score: {volume_signal.volume_zscore:.1f}, "
                f"+{volume_signal.volume_change_pct:.1f}%)"
            )

        # HTF S/R Analysis (20% weight)
        sr_score = 0
        if sr_confluence:
            # Boost score if near strong S/R level
            if sr_confluence.distance_percent < SR_CONFLUENCE_DISTANCE_PCT:
                sr_weight = min(sr_confluence.confluence_score / 100, 0.20)

                if sr_confluence.zone_type == 'support':
                    sr_score += sr_weight
                    rationale_parts.append(
                        f"HTF {sr_confluence.zone_type} @ ${sr_confluence.price:,.2f} "
                        f"({sr_confluence.strength} levels)"
                    )
                elif sr_confluence.zone_type == 'resistance':
                    sr_score -= sr_weight
                    rationale_parts.append(
                        f"HTF {sr_confluence.zone_type} @ ${sr_confluence.price:,.2f} "
                        f"({sr_confluence.strength} levels)"
                    )

        # Fibonacci Analysis (20% weight - Golden Pocket!)
        fib_score = 0
        if fib_levels:
            # Golden Pocket is a major institutional entry zone
            if fib_levels.in_golden_pocket:
                if fib_levels.swing_direction == 'bullish':
                    # Price at golden pocket of bullish swing = BUY opportunity
                    fib_score += 0.20
                    rationale_parts.append(
                        f"🎯 Golden Pocket (${fib_levels.golden_pocket_low:.2f}-${fib_levels.golden_pocket_high:.2f}) "
                        f"in {fib_levels.swing_direction} swing"
                    )
                else:
                    # Price at golden pocket of bearish swing = SELL opportunity
                    fib_score -= 0.20
                    rationale_parts.append(
                        f"🎯 Golden Pocket (${fib_levels.golden_pocket_low:.2f}-${fib_levels.golden_pocket_high:.2f}) "
                        f"in {fib_levels.swing_direction} swing"
                    )
            # Near golden pocket
            elif fib_levels.distance_from_golden_pocket < FIBONACCI_GP_DISTANCE_PCT:
                gp_boost = 0.10 * (1 - fib_levels.distance_from_golden_pocket / FIBONACCI_GP_DISTANCE_PCT)
                if fib_levels.swing_direction == 'bullish':
                    fib_score += gp_boost
                else:
                    fib_score -= gp_boost
                rationale_parts.append(
                    f"Near Golden Pocket ({fib_levels.distance_from_golden_pocket * 100:.1f}% away)"
                )

        # Learning Agent Insights (20% weight)
        learning_score = 0
        if learning_insights:
            learning_score = learning_insights.confidence_adjustment
            if abs(learning_score) > 0:
                rationale_parts.append(
                    f"Learning: {learning_insights.recommended_action} "
                    f"({learning_insights.win_rate_overall:.0f}% win rate)"
                )

        # Calculate total score
        total_score = price_score + momentum_score + volume_score + sr_score + fib_score + learning_score

        # Apply learning context multiplier
        if learning_insights:
            total_score *= learning_insights.context_multiplier

        # Determine direction
        if total_score > 0.3:
            direction = "LONG"
            confidence = min(1.0, abs(total_score))
        elif total_score < -0.3:
            direction = "SHORT"
            confidence = min(1.0, abs(total_score))

        # Debug logging for rejected signals (sample every 100th symbol)
        if not direction and hash(price_signal.symbol) % 100 == 0:
            self.logger.debug(
                f"{price_signal.symbol}: No signal - score {total_score:.3f} "
                f"(P:{price_score:.2f} M:{momentum_score:.2f} V:{volume_score:.2f} "
                f"SR:{sr_score:.2f} FIB:{fib_score:.2f} L:{learning_score:.2f})"
            )

        return direction, confidence, rationale_parts

    def _calculate_levels(
        self,
        entry: float,
        direction: str,
        price_signal: PriceActionSignal,
        volume_signal: Optional[VolumeSignal],
        fib_levels=None,
        reward_risk_ratio: float = 2.0,
        max_stop_loss_pct: float = 5.0,
        sr_levels=None
    ) -> tuple:
        """
        Calculate stop-loss and take-profit levels with dynamic sizing.

        Uses:
        - Liquidity-based stop loss (tight for BTC/ETH, wider for altcoins)
        - Fibonacci extension targets when available
        - S/R levels for optimal placement

        Args:
            entry: Entry price
            direction: Trade direction
            price_signal: Price action signal
            volume_signal: Volume signal (for liquidity data)
            fib_levels: Fibonacci levels (for extension targets)
            reward_risk_ratio: Reward-to-risk ratio
            max_stop_loss_pct: Maximum stop loss percentage
            sr_levels: S/R levels for better placement

        Returns:
            Tuple of (stop, target)
        """
        # Input validation
        if entry <= 0:
            self.logger.error(f"Invalid entry price: {entry}")
            # Fallback to simple percentage
            entry = price_signal.price if price_signal.price > 0 else 1.0

        if direction not in ["LONG", "SHORT"]:
            self.logger.error(f"Invalid direction: {direction}, defaulting to LONG")
            direction = "LONG"

        if reward_risk_ratio <= 0:
            self.logger.warning(f"Invalid R:R ratio: {reward_risk_ratio}, using 2.0")
            reward_risk_ratio = 2.0

        # Calculate dynamic stop loss based on liquidity and volatility
        liquidity_usd = 0
        if volume_signal:
            liquidity_usd = price_signal.price * volume_signal.volume_24h

        # Get recommended stop loss percentage using market metrics
        stop_pct = MarketMetricsCalculator.get_stop_loss_for_asset(
            symbol=price_signal.symbol,
            liquidity_usd_24h=liquidity_usd,
            intraday_range_pct=price_signal.intraday_range_pct,
            volatility_ratio=price_signal.volatility_ratio
        )

        # Cap at max allowed
        stop_pct = min(stop_pct, max_stop_loss_pct)

        # Calculate stop loss and target based on direction
        if direction == "LONG":
            # === STOP LOSS CALCULATION ===
            # Start with dynamic stop based on liquidity/volatility
            dynamic_stop = entry * (1 - stop_pct / 100)

            # Adjust with S/R levels if available
            if sr_levels:
                supports = [
                    level for level in sr_levels
                    if level.level_type == 'support' and level.price < entry
                ]
                if supports:
                    nearest_support = max(supports, key=lambda x: x.price)
                    sr_stop = nearest_support.price * 0.998  # Slightly below support
                    # Use the tighter of the two (more conservative)
                    stop = max(sr_stop, dynamic_stop)
                else:
                    stop = dynamic_stop
            else:
                stop = dynamic_stop

            # === TARGET CALCULATION ===
            risk = entry - stop

            # Priority 1: Fibonacci 1.618 extension (golden ratio target)
            if fib_levels and fib_levels.swing_direction == 'bullish':
                fib_target = fib_levels.extension_levels.get(1.618)
                if fib_target and fib_target > entry:
                    # Use Fibonacci extension as primary target
                    target = fib_target
                else:
                    # Fall back to risk-reward ratio
                    target = entry + (risk * reward_risk_ratio)
            # Priority 2: S/R resistance level (conservative)
            elif sr_levels:
                resistances = [
                    level for level in sr_levels
                    if level.level_type == 'resistance' and level.price > entry
                ]
                if resistances:
                    nearest_resistance = min(resistances, key=lambda x: x.price)
                    sr_target = nearest_resistance.price * 0.998
                    rr_target = entry + (risk * reward_risk_ratio)
                    # Use the more conservative (closer) target
                    target = min(sr_target, rr_target)
                else:
                    target = entry + (risk * reward_risk_ratio)
            else:
                # Default: risk-reward ratio
                target = entry + (risk * reward_risk_ratio)

        else:  # SHORT
            # === STOP LOSS CALCULATION ===
            dynamic_stop = entry * (1 + stop_pct / 100)

            # Adjust with S/R levels if available
            if sr_levels:
                resistances = [
                    level for level in sr_levels
                    if level.level_type == 'resistance' and level.price > entry
                ]
                if resistances:
                    nearest_resistance = min(resistances, key=lambda x: x.price)
                    sr_stop = nearest_resistance.price * 1.002  # Slightly above resistance
                    # Use the tighter of the two (more conservative)
                    stop = min(sr_stop, dynamic_stop)
                else:
                    stop = dynamic_stop
            else:
                stop = dynamic_stop

            # === TARGET CALCULATION ===
            risk = stop - entry

            # Priority 1: Fibonacci 1.618 extension (golden ratio target)
            if fib_levels and fib_levels.swing_direction == 'bearish':
                fib_target = fib_levels.extension_levels.get(1.618)
                if fib_target and fib_target < entry:
                    # Use Fibonacci extension as primary target
                    target = fib_target
                else:
                    # Fall back to risk-reward ratio
                    target = entry - (risk * reward_risk_ratio)
            # Priority 2: S/R support level (conservative)
            elif sr_levels:
                supports = [
                    level for level in sr_levels
                    if level.level_type == 'support' and level.price < entry
                ]
                if supports:
                    nearest_support = max(supports, key=lambda x: x.price)
                    sr_target = nearest_support.price * 1.002
                    rr_target = entry - (risk * reward_risk_ratio)
                    # Use the more conservative (closer) target
                    target = max(sr_target, rr_target)
                else:
                    target = entry - (risk * reward_risk_ratio)
            else:
                # Default: risk-reward ratio
                target = entry - (risk * reward_risk_ratio)

        # Final validation: ensure stop and target are valid
        stop_distance_pct = abs(stop - entry) / entry * 100
        if stop_distance_pct < MIN_STOP_DISTANCE_PCT:
            self.logger.warning(
                f"{price_signal.symbol}: Stop too close to entry ({stop_distance_pct:.2f}%), "
                f"adjusting to minimum {MIN_STOP_DISTANCE_PCT}%"
            )
            if direction == "LONG":
                stop = entry * (1 - MIN_STOP_DISTANCE_PCT / 100)
            else:
                stop = entry * (1 + MIN_STOP_DISTANCE_PCT / 100)

        # Ensure target is in correct direction
        if direction == "LONG" and target <= entry:
            self.logger.error(f"{price_signal.symbol}: Invalid LONG target {target} <= entry {entry}")
            target = entry * 1.01  # Fallback to 1% profit
        elif direction == "SHORT" and target >= entry:
            self.logger.error(f"{price_signal.symbol}: Invalid SHORT target {target} >= entry {entry}")
            target = entry * 0.99  # Fallback to 1% profit

        return round(stop, 8), round(target, 8)

    def _determine_order_type(
        self,
        price_signal: PriceActionSignal,
        volume_signal: Optional[VolumeSignal],
        sr_confluence=None,
        fib_levels=None,
        entry_price: float = 0
    ) -> tuple:
        """
        Determine order type (LIMIT vs MARKET) based on confluence.

        LIMIT orders are used when:
        - High confluence (HTF S/R + Fibonacci + Psychological levels)
        - Better fills expected at specific levels
        - No time urgency

        MARKET orders are used when:
        - Breakout scenarios (time-sensitive)
        - Volume spikes (momentum plays)
        - Low confluence (no clear level to wait for)

        Args:
            price_signal: Price action signal
            volume_signal: Volume signal
            sr_confluence: S/R confluence data
            fib_levels: Fibonacci levels
            entry_price: Entry price

        Returns:
            Tuple of (order_type, confluence_score)
        """
        confluence_score = 0

        # Check for HTF S/R confluence (+2 to +4 points)
        if sr_confluence and sr_confluence.distance_percent < SR_CONFLUENCE_EXTENDED_PCT:
            confluence_score += min(sr_confluence.strength, 4)

        # Check for Golden Pocket (+4 points - institutional zone!)
        if fib_levels and fib_levels.in_golden_pocket:
            confluence_score += 4

        # Check for psychological levels (+2 points)
        if entry_price > 0 and self._is_psychological_level(entry_price):
            confluence_score += 2

        # FORCE MARKET if breakout detected (time-sensitive)
        if price_signal.breakout_detected:
            return "MARKET", confluence_score

        # FORCE MARKET if volume spike (momentum play)
        if volume_signal and volume_signal.spike_detected:
            return "MARKET", confluence_score

        # Use LIMIT for high confluence
        if confluence_score >= CONFLUENCE_SCORE_LIMIT_THRESHOLD:
            return "LIMIT", confluence_score

        # Default to MARKET for lower confluence
        return "MARKET", confluence_score

    def _is_psychological_level(self, price: float) -> bool:
        """
        Check if price is near a psychological level.

        Psychological levels: Round numbers (e.g., $100, $1000, $50, etc.)

        Args:
            price: Price to check

        Returns:
            True if near psychological level
        """
        # Round to nearest psychological level
        if price >= 1000:
            # For large prices, check for $1000 intervals
            round_level = round(price / 1000) * 1000
            return abs(price - round_level) / price < 0.01  # Within 1%
        elif price >= 100:
            # For mid prices, check for $100 intervals
            round_level = round(price / 100) * 100
            return abs(price - round_level) / price < 0.01
        elif price >= 10:
            # For small prices, check for $10 intervals
            round_level = round(price / 10) * 10
            return abs(price - round_level) / price < 0.02
        elif price >= 1:
            # For very small prices, check for $1 intervals
            round_level = round(price)
            return abs(price - round_level) / price < 0.02
        else:
            # For sub-$1 prices, check for $0.10 or $0.01 intervals
            if price >= 0.1:
                round_level = round(price * 10) / 10
                return abs(price - round_level) / price < 0.03
            else:
                round_level = round(price * 100) / 100
                return abs(price - round_level) / price < 0.05

        return False
