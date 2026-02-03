from dataclasses import asdict
from typing import Dict, List

from crypto_agents.agents.base import BaseAgent
from crypto_agents.config import AppConfig
from crypto_agents.schemas import MarketSnapshot, PriceActionMetrics
from crypto_agents.utils import now_ms, percent_change


class PriceActionAgent(BaseAgent):
    name = "price_action"

    def __init__(self, config: AppConfig, output_dir: str, bus):
        super().__init__(config, output_dir, bus)

    async def run(self, snapshot: MarketSnapshot):
        metrics: List[Dict] = []
        lookbacks = self.config.windows.lookbacks
        for symbol in snapshot.symbols:
            klines = snapshot.klines.get(symbol, [])
            if len(klines) < max(lookbacks) + 1:
                continue
            closes = [float(kline[4]) for kline in klines]
            highs = [float(kline[2]) for kline in klines]
            lows = [float(kline[3]) for kline in klines]
            current = closes[-1]
            deltas = {}
            for window in lookbacks:
                deltas[window] = percent_change(current, closes[-1 - window])
            intraday_range = (max(highs) - min(lows)) / current if current else 0.0
            recent_vol = max(highs[-self.config.windows.volatility_window:]) - min(
                lows[-self.config.windows.volatility_window:]
            )
            full_vol = max(highs) - min(lows)
            volatility_ratio = (recent_vol / full_vol) if full_vol else 0.0
            metrics.append(
                asdict(
                    PriceActionMetrics(
                        symbol=symbol,
                        price=current,
                        deltas=deltas,
                        intraday_range=intraday_range,
                        volatility_ratio=volatility_ratio,
                        timestamp=now_ms(),
                    )
                )
            )
        return await self.emit(metrics)
