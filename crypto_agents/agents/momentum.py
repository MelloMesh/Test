from dataclasses import asdict
from typing import Dict, List

from crypto_agents.agents.base import BaseAgent
from crypto_agents.config import AppConfig
from crypto_agents.schemas import MarketSnapshot, MomentumMetrics
from crypto_agents.utils import compute_obv, compute_rsi, now_ms


class MomentumAgent(BaseAgent):
    name = "momentum"

    def __init__(self, config: AppConfig, output_dir: str, bus):
        super().__init__(config, output_dir, bus)

    async def run(self, snapshot: MarketSnapshot):
        metrics: List[Dict] = []
        for symbol in snapshot.symbols:
            klines = snapshot.klines.get(symbol, [])
            if len(klines) < self.config.windows.rsi_period + 1:
                continue
            closes = [float(kline[4]) for kline in klines]
            volumes = [float(kline[5]) for kline in klines]
            rsi = compute_rsi(closes, self.config.windows.rsi_period)
            obv = compute_obv(closes[-self.config.windows.obv_period:], volumes[-self.config.windows.obv_period:])
            metrics.append(
                asdict(
                    MomentumMetrics(
                        symbol=symbol,
                        rsi=rsi,
                        obv=obv,
                        timeframe=self.config.binance.klines_interval,
                        timestamp=now_ms(),
                    )
                )
            )
        return await self.emit(metrics)
