from dataclasses import asdict
from typing import Dict, List

from crypto_agents.agents.base import BaseAgent
from crypto_agents.schemas import MarketSnapshot, Signal
from crypto_agents.utils import now_ms


class SignalSynthesisAgent(BaseAgent):
    name = "signal_synthesis"

    def __init__(self, config, output_dir: str, bus):
        super().__init__(config, output_dir, bus)

    async def run(self, snapshot: MarketSnapshot):
        signals: List[Dict] = []
        price_action = snapshot.tickers.get("price_action", {})
        momentum = snapshot.tickers.get("momentum", {})
        volume_spike = snapshot.tickers.get("volume_spike", {})
        for symbol in snapshot.symbols:
            pa = price_action.get(symbol)
            mo = momentum.get(symbol)
            vs = volume_spike.get(symbol)
            if not pa or not mo or not vs:
                continue
            direction = None
            rationale_parts = []
            confidence = 0.5
            if mo["rsi"] >= self.config.strategy.rsi_overbought and pa["deltas"][1] > 0:
                direction = "short"
                rationale_parts.append("overbought RSI with upside momentum")
                confidence += 0.15
            if mo["rsi"] <= self.config.strategy.rsi_oversold and pa["deltas"][1] < 0:
                direction = "long"
                rationale_parts.append("oversold RSI with downside momentum")
                confidence += 0.15
            if vs["interpretation"] != "neutral":
                rationale_parts.append(vs["interpretation"])
                confidence += 0.1
            if pa["volatility_ratio"] >= self.config.strategy.volatility_ratio_threshold:
                rationale_parts.append("elevated volatility")
                confidence += 0.05
            if not direction:
                continue
            price = pa["price"]
            stop_offset = min(price * self.config.strategy.max_stop_pct, price * 0.01)
            if direction == "long":
                stop = price - stop_offset
                target = price + stop_offset * self.config.strategy.reward_to_risk
            else:
                stop = price + stop_offset
                target = price - stop_offset * self.config.strategy.reward_to_risk
            signals.append(
                asdict(
                    Signal(
                        asset=symbol,
                        direction=direction,
                        entry=price,
                        stop=stop,
                        target=target,
                        confidence=min(confidence, 0.95),
                        rationale="; ".join(rationale_parts),
                        timestamp=now_ms(),
                    )
                )
            )
        return await self.emit(signals)
