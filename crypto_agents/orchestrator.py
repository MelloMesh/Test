import asyncio
import contextlib
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

from crypto_agents.agents.momentum import MomentumAgent
from crypto_agents.agents.price_action import PriceActionAgent
from crypto_agents.agents.signal_synthesis import SignalSynthesisAgent
from crypto_agents.agents.volume_spike import VolumeSpikeAgent
from crypto_agents.client import BinanceFuturesClient
from crypto_agents.config import AppConfig
from crypto_agents.schemas import MarketSnapshot, TradeWatchlist
from crypto_agents.utils import now_ms, setup_logger, write_json_log


class Orchestrator:
    def __init__(self, config: AppConfig):
        self.config = config
        self.logger = setup_logger("orchestrator", config.output.log_dir)
        self.bus: asyncio.Queue = asyncio.Queue()
        self.output_dir = config.output.log_dir
        self.agents = {
            "price_action": PriceActionAgent(config, self.output_dir, self.bus),
            "momentum": MomentumAgent(config, self.output_dir, self.bus),
            "volume_spike": VolumeSpikeAgent(config, self.output_dir, self.bus),
            "signal_synthesis": SignalSynthesisAgent(config, self.output_dir, self.bus),
        }
        self._agent_cache: Dict[str, Dict[str, Any]] = {
            "price_action": {},
            "momentum": {},
            "volume_spike": {},
            "signal_synthesis": {},
        }

    async def run(self):
        async with BinanceFuturesClient(self.config) as client:
            symbols = await self._load_symbols(client)
            await client.start_price_stream(symbols)
            await client.wait_for_ws()
            report_task = asyncio.create_task(self._report_loop())
            try:
                while True:
                    snapshot = await self._fetch_snapshot(client, symbols)
                    await self._run_agents(snapshot)
                    await asyncio.sleep(self.config.poll_interval_sec)
            finally:
                report_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await report_task

    async def _load_symbols(self, client: BinanceFuturesClient) -> List[str]:
        info = await client.exchange_info()
        raw_symbols = [
            symbol
            for symbol in info.get("symbols", [])
            if symbol.get("quoteAsset") == self.config.quote_asset
            and symbol.get("contractType") == "PERPETUAL"
            and symbol.get("status") == "TRADING"
        ]
        tickers = await client.ticker_24h()
        ticker_map = {ticker["symbol"]: ticker for ticker in tickers}
        filtered = []
        for symbol in raw_symbols:
            ticker = ticker_map.get(symbol["symbol"])
            if not ticker:
                continue
            notional_volume = float(ticker.get("quoteVolume", 0))
            if notional_volume < self.config.liquidity.min_notional_volume:
                continue
            filtered.append(symbol["symbol"])
            if len(filtered) >= self.config.max_symbols:
                break
        open_interest = await self._fetch_open_interest(client, filtered)
        eligible = [
            symbol
            for symbol in filtered
            if open_interest.get(symbol, 0.0) >= self.config.liquidity.min_open_interest
        ]
        self.logger.info("Loaded %s symbols", len(eligible))
        return eligible

    async def _fetch_open_interest(
        self, client: BinanceFuturesClient, symbols: List[str]
    ) -> Dict[str, float]:
        results: Dict[str, float] = {}
        semaphore = asyncio.Semaphore(10)

        async def fetch(symbol: str):
            async with semaphore:
                data = await client.open_interest(symbol)
                results[symbol] = float(data.get("openInterest", 0.0))

        await asyncio.gather(*(fetch(symbol) for symbol in symbols))
        return results

    async def _fetch_snapshot(
        self, client: BinanceFuturesClient, symbols: List[str]
    ) -> MarketSnapshot:
        tickers = await client.ticker_24h()
        ticker_map = {ticker["symbol"]: ticker for ticker in tickers}
        klines_data = await self._fetch_klines(client, symbols)
        mark_prices = await client.mark_prices()
        volumes = {
            symbol: float(ticker_map[symbol]["quoteVolume"])
            for symbol in symbols
            if symbol in ticker_map
        }
        prices = {
            symbol: float(ticker_map[symbol]["lastPrice"])
            for symbol in symbols
            if symbol in ticker_map
        }
        snapshot = MarketSnapshot(
            symbols=symbols,
            prices=prices,
            klines=klines_data,
            tickers=ticker_map,
            open_interest={},
            mark_prices=mark_prices,
            volumes=volumes,
            timestamp=now_ms(),
        )
        return snapshot

    async def _fetch_klines(
        self, client: BinanceFuturesClient, symbols: List[str]
    ) -> Dict[str, List[List[Any]]]:
        semaphore = asyncio.Semaphore(8)
        results: Dict[str, List[List[Any]]] = {}

        async def fetch(symbol: str):
            async with semaphore:
                results[symbol] = await client.klines(symbol)

        await asyncio.gather(*(fetch(symbol) for symbol in symbols))
        return results

    async def _run_agents(self, snapshot: MarketSnapshot):
        tasks = []
        for agent_name, agent in self.agents.items():
            if agent_name == "signal_synthesis":
                continue
            timeout = self.config.agent_timeouts.get(agent_name, 10.0)
            tasks.append(asyncio.wait_for(agent.run(snapshot), timeout=timeout))
        outputs = await asyncio.gather(*tasks, return_exceptions=True)
        for output in outputs:
            if isinstance(output, Exception):
                self.logger.error("Agent failure: %s", output)
                continue
            for item in output.payload:
                symbol = item.get("symbol")
                if symbol:
                    self._agent_cache[output.agent][symbol] = item
        signal_snapshot = MarketSnapshot(
            symbols=snapshot.symbols,
            prices=snapshot.prices,
            klines=snapshot.klines,
            tickers={
                "price_action": self._agent_cache["price_action"],
                "momentum": self._agent_cache["momentum"],
                "volume_spike": self._agent_cache["volume_spike"],
            },
            open_interest=snapshot.open_interest,
            mark_prices=snapshot.mark_prices,
            volumes=snapshot.volumes,
            timestamp=now_ms(),
        )
        synthesis = self.agents["signal_synthesis"]
        synthesis_output = await synthesis.run(signal_snapshot)
        self._agent_cache["signal_synthesis"] = {
            signal["asset"]: signal for signal in synthesis_output.payload
        }

    async def _report_loop(self):
        while True:
            await asyncio.sleep(self.config.output.report_interval_sec)
            signals = list(self._agent_cache.get("signal_synthesis", {}).values())
            report = TradeWatchlist(timestamp=now_ms(), signals=signals)
            await write_json_log(
                Path(self.output_dir) / "trade_watchlist.jsonl", asdict(report)
            )


async def main():
    config = AppConfig()
    orchestrator = Orchestrator(config)
    await orchestrator.run()


if __name__ == "__main__":
    asyncio.run(main())
