"""
Signal Synthesis Agent - Integrates all agent outputs into actionable trading signals.
"""

from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from .base_agent import BaseAgent
from ..exchange.base import BaseExchange
from ..schemas import TradingSignal, PriceActionSignal, MomentumSignal, VolumeSignal
from ..config import SignalSynthesisConfig


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
            self.config.reward_risk_ratio,
            self.config.max_stop_loss_pct,
            sr_levels
        )

        # Build rationale
        rationale = " | ".join(rationale_parts)

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
            price_signal=price_signal.to_dict() if price_signal else None,
            momentum_signal=momentum_signals[0].to_dict() if momentum_signals else None,
            volume_signal=volume_signal.to_dict() if volume_signal else None,
            sr_data=sr_confluence.to_dict() if sr_confluence else None,
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

        # Price action analysis (20% weight)
        price_score = 0
        if price_signal.breakout_detected:
            if price_signal.price_change_pct > 0:
                price_score += 0.2
                rationale_parts.append(f"Bullish breakout (+{price_signal.price_change_pct:.1f}%)")
            else:
                price_score -= 0.2
                rationale_parts.append(f"Bearish breakout ({price_signal.price_change_pct:.1f}%)")

        if price_signal.volatility_ratio > 2.0:
            rationale_parts.append(f"High volatility ({price_signal.volatility_ratio:.1f}x)")

        # Momentum analysis (20% weight)
        momentum_score = 0
        if momentum_signals:
            # Use the strongest momentum signal
            strongest_momentum = max(momentum_signals, key=lambda x: x.strength_score)

            if strongest_momentum.status == "oversold":
                momentum_score += 0.2
                rationale_parts.append(
                    f"Oversold RSI {strongest_momentum.rsi:.1f} on {strongest_momentum.timeframe}"
                )
            elif strongest_momentum.status == "overbought":
                momentum_score -= 0.2
                rationale_parts.append(
                    f"Overbought RSI {strongest_momentum.rsi:.1f} on {strongest_momentum.timeframe}"
                )

            # OBV confirmation
            if strongest_momentum.obv_change_pct > 10:
                momentum_score += 0.05
                rationale_parts.append(f"OBV rising (+{strongest_momentum.obv_change_pct:.1f}%)")
            elif strongest_momentum.obv_change_pct < -10:
                momentum_score -= 0.05
                rationale_parts.append(f"OBV falling ({strongest_momentum.obv_change_pct:.1f}%)")

        # Volume analysis (15% weight)
        volume_score = 0
        if volume_signal and volume_signal.spike_detected:
            volume_score += 0.15
            rationale_parts.append(
                f"Volume spike (z-score: {volume_signal.volume_zscore:.1f}, "
                f"+{volume_signal.volume_change_pct:.1f}%)"
            )

        # HTF S/R Analysis (25% weight - highest!)
        sr_score = 0
        if sr_confluence:
            # Boost score if near strong S/R level
            if sr_confluence.distance_percent < 0.01:  # Within 1%
                sr_weight = min(sr_confluence.confluence_score / 100, 0.25)

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
        total_score = price_score + momentum_score + volume_score + sr_score + learning_score

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
                f"SR:{sr_score:.2f} L:{learning_score:.2f})"
            )

        return direction, confidence, rationale_parts

    def _calculate_levels(
        self,
        entry: float,
        direction: str,
        price_signal: PriceActionSignal,
        reward_risk_ratio: float,
        max_stop_loss_pct: float,
        sr_levels=None
    ) -> tuple:
        """
        Calculate stop-loss and take-profit levels.

        Args:
            entry: Entry price
            direction: Trade direction
            price_signal: Price action signal
            reward_risk_ratio: Reward-to-risk ratio
            max_stop_loss_pct: Maximum stop loss percentage
            sr_levels: S/R levels for better placement

        Returns:
            Tuple of (stop, target)
        """
        # Use volatility-based stop loss as baseline
        stop_pct = min(
            price_signal.intraday_range_pct / 2,
            max_stop_loss_pct
        )

        # Ensure minimum stop loss
        stop_pct = max(stop_pct, 0.5)

        # Adjust stop/target based on S/R levels if available
        if sr_levels:
            if direction == "LONG":
                # Find nearest support below entry for stop
                supports = [
                    level for level in sr_levels
                    if level.level_type == 'support' and level.price < entry
                ]
                if supports:
                    nearest_support = max(supports, key=lambda x: x.price)
                    sr_stop = nearest_support.price * 0.998  # Slightly below support
                    volatility_stop = entry * (1 - stop_pct / 100)
                    # Use the tighter stop
                    stop = max(sr_stop, volatility_stop)
                else:
                    stop = entry * (1 - stop_pct / 100)

                # Find nearest resistance above entry for target
                resistances = [
                    level for level in sr_levels
                    if level.level_type == 'resistance' and level.price > entry
                ]
                if resistances:
                    nearest_resistance = min(resistances, key=lambda x: x.price)
                    sr_target = nearest_resistance.price * 0.998  # Slightly below resistance
                    risk = entry - stop
                    volatility_target = entry + (risk * reward_risk_ratio)
                    # Use the more conservative target
                    target = min(sr_target, volatility_target)
                else:
                    risk = entry - stop
                    target = entry + (risk * reward_risk_ratio)

            else:  # SHORT
                # Find nearest resistance above entry for stop
                resistances = [
                    level for level in sr_levels
                    if level.level_type == 'resistance' and level.price > entry
                ]
                if resistances:
                    nearest_resistance = min(resistances, key=lambda x: x.price)
                    sr_stop = nearest_resistance.price * 1.002  # Slightly above resistance
                    volatility_stop = entry * (1 + stop_pct / 100)
                    # Use the tighter stop
                    stop = min(sr_stop, volatility_stop)
                else:
                    stop = entry * (1 + stop_pct / 100)

                # Find nearest support below entry for target
                supports = [
                    level for level in sr_levels
                    if level.level_type == 'support' and level.price < entry
                ]
                if supports:
                    nearest_support = max(supports, key=lambda x: x.price)
                    sr_target = nearest_support.price * 1.002  # Slightly above support
                    risk = stop - entry
                    volatility_target = entry - (risk * reward_risk_ratio)
                    # Use the more conservative target
                    target = max(sr_target, volatility_target)
                else:
                    risk = stop - entry
                    target = entry - (risk * reward_risk_ratio)

        else:
            # No S/R levels, use volatility-based calculation
            if direction == "LONG":
                stop = entry * (1 - stop_pct / 100)
                target = entry * (1 + (stop_pct * reward_risk_ratio) / 100)
            else:  # SHORT
                stop = entry * (1 + stop_pct / 100)
                target = entry * (1 - (stop_pct * reward_risk_ratio) / 100)

        return round(stop, 8), round(target, 8)
