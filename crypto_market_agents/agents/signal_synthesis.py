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
        volume_spike_agent
    ):
        """
        Initialize Signal Synthesis Agent.

        Args:
            exchange: Exchange adapter
            config: Signal synthesis configuration
            price_action_agent: Price action agent instance
            momentum_agent: Momentum agent instance
            volume_spike_agent: Volume spike agent instance
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

        # Determine direction and confidence
        direction, confidence, rationale_parts = self._determine_direction(
            price_signal,
            momentum_signals,
            volume_signal
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
            self.config.max_stop_loss_pct
        )

        # Build rationale
        rationale = " | ".join(rationale_parts)

        return TradingSignal(
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
            volume_signal=volume_signal.to_dict() if volume_signal else None
        )

    def _determine_direction(
        self,
        price_signal: PriceActionSignal,
        momentum_signals: List[MomentumSignal],
        volume_signal: Optional[VolumeSignal]
    ) -> tuple:
        """
        Determine trade direction and confidence.

        Returns:
            Tuple of (direction, confidence, rationale_parts)
        """
        direction = None
        confidence = 0.0
        rationale_parts = []

        # Price action analysis
        price_score = 0
        if price_signal.breakout_detected:
            if price_signal.price_change_pct > 0:
                price_score += 0.3
                rationale_parts.append(f"Bullish breakout (+{price_signal.price_change_pct:.1f}%)")
            else:
                price_score -= 0.3
                rationale_parts.append(f"Bearish breakout ({price_signal.price_change_pct:.1f}%)")

        if price_signal.volatility_ratio > 2.0:
            rationale_parts.append(f"High volatility ({price_signal.volatility_ratio:.1f}x)")

        # Momentum analysis
        momentum_score = 0
        if momentum_signals:
            # Use the strongest momentum signal
            strongest_momentum = max(momentum_signals, key=lambda x: x.strength_score)

            if strongest_momentum.status == "oversold":
                momentum_score += 0.3
                rationale_parts.append(
                    f"Oversold RSI {strongest_momentum.rsi:.1f} on {strongest_momentum.timeframe}"
                )
            elif strongest_momentum.status == "overbought":
                momentum_score -= 0.3
                rationale_parts.append(
                    f"Overbought RSI {strongest_momentum.rsi:.1f} on {strongest_momentum.timeframe}"
                )

            # OBV confirmation
            if strongest_momentum.obv_change_pct > 10:
                momentum_score += 0.1
                rationale_parts.append(f"OBV rising (+{strongest_momentum.obv_change_pct:.1f}%)")
            elif strongest_momentum.obv_change_pct < -10:
                momentum_score -= 0.1
                rationale_parts.append(f"OBV falling ({strongest_momentum.obv_change_pct:.1f}%)")

        # Volume analysis
        volume_score = 0
        if volume_signal and volume_signal.spike_detected:
            volume_score += 0.2
            rationale_parts.append(
                f"Volume spike (z-score: {volume_signal.volume_zscore:.1f}, "
                f"+{volume_signal.volume_change_pct:.1f}%)"
            )

        # Calculate total score
        total_score = price_score + momentum_score + volume_score

        # Determine direction
        if total_score > 0.3:
            direction = "LONG"
            confidence = min(1.0, abs(total_score))
        elif total_score < -0.3:
            direction = "SHORT"
            confidence = min(1.0, abs(total_score))

        return direction, confidence, rationale_parts

    def _calculate_levels(
        self,
        entry: float,
        direction: str,
        price_signal: PriceActionSignal,
        reward_risk_ratio: float,
        max_stop_loss_pct: float
    ) -> tuple:
        """
        Calculate stop-loss and take-profit levels.

        Args:
            entry: Entry price
            direction: Trade direction
            price_signal: Price action signal
            reward_risk_ratio: Reward-to-risk ratio
            max_stop_loss_pct: Maximum stop loss percentage

        Returns:
            Tuple of (stop, target)
        """
        # Use volatility-based stop loss
        stop_pct = min(
            price_signal.intraday_range_pct / 2,
            max_stop_loss_pct
        )

        # Ensure minimum stop loss
        stop_pct = max(stop_pct, 0.5)

        if direction == "LONG":
            stop = entry * (1 - stop_pct / 100)
            target = entry * (1 + (stop_pct * reward_risk_ratio) / 100)
        else:  # SHORT
            stop = entry * (1 + stop_pct / 100)
            target = entry * (1 - (stop_pct * reward_risk_ratio) / 100)

        return round(stop, 8), round(target, 8)
