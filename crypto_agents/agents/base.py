import asyncio
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

from crypto_agents.config import AppConfig
from crypto_agents.schemas import AgentOutput, MarketSnapshot
from crypto_agents.utils import now_ms, write_json_log


class BaseAgent:
    name = "base"

    def __init__(self, config: AppConfig, output_dir: str, bus: asyncio.Queue):
        self.config = config
        self.output_dir = Path(output_dir)
        self.bus = bus

    async def run(self, snapshot: MarketSnapshot) -> AgentOutput:
        raise NotImplementedError

    async def emit(self, payload: List[Dict[str, Any]]) -> AgentOutput:
        output = AgentOutput(agent=self.name, timestamp=now_ms(), payload=payload)
        await write_json_log(self.output_dir / f"{self.name}.jsonl", asdict(output))
        await self.bus.put(asdict(output))
        return output
