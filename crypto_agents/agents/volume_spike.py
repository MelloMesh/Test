from dataclasses import asdict
from typing import Dict, List

from crypto_agents.agents.base import BaseAgent
from crypto_agents.schemas import MarketSnapshot, VolumeSpikeMetrics
from crypto_agents.utils import now_ms, z_score


class VolumeSpikeAgent(BaseAgent):
    name = "volume_spike"

    def __init__(self, config, output_dir: str, bus):
        super().__init__(config, output_dir, bus)

    async def run(self, snapshot: MarketSnapshot):
        metrics: List[Dict] = []
        for symbol in snapshot.symbols:
            klines = snapshot.klines.get(symbol, [])
            if len(klines) < self.config.windows.volume_window:
                continue
            volumes = [float(kline[5]) for kline in klines]
            current = volumes[-1]
            baseline = sum(volumes[-self.config.windows.volume_window : -1]) / (
                self.config.windows.volume_window - 1
            )
            score = z_score(volumes[-self.config.windows.volume_window : -1], current)
            interpretation = "neutral"
            if score >= self.config.strategy.volume_spike_z:
                interpretation = "potential accumulation"
            elif score <= -self.config.strategy.volume_spike_z:
                interpretation = "potential distribution"
            metrics.append(
                asdict(
                    VolumeSpikeMetrics(
                        symbol=symbol,
                        current_volume=current,
                        baseline_volume=baseline,
                        z_score=score,
                        interpretation=interpretation,
                        timestamp=now_ms(),
                    )
                )
            )
        return await self.emit(metrics)
