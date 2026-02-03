import asyncio
import contextlib
import json
from typing import Any, Dict, List, Optional

import aiohttp
import websockets

from crypto_agents.config import AppConfig
from crypto_agents.utils import jittered_backoff, now_ms


class BinanceFuturesClient:
    def __init__(self, config: AppConfig):
        self.config = config
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws_tasks: List[asyncio.Task] = []
        self._ws_connected = asyncio.Event()
        self._ws_lock = asyncio.Lock()
        self._ws_active = 0
        self._prices: Dict[str, float] = {}
        self._mark_prices: Dict[str, float] = {}
        self._last_ws_message = 0

    async def __aenter__(self):
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()

    async def close(self):
        for task in self._ws_tasks:
            task.cancel()
        for task in self._ws_tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        if self._session:
            await self._session.close()

    async def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        url = f"{self.config.binance.rest_base}{path}"
        attempt = 0
        while True:
            try:
                async with self._session.get(url, params=params, timeout=10) as response:
                    if response.status in {418, 429} or response.status >= 500:
                        raise aiohttp.ClientResponseError(
                            response.request_info,
                            response.history,
                            status=response.status,
                            message=await response.text(),
                            headers=response.headers,
                        )
                    response.raise_for_status()
                    return await response.json()
            except (aiohttp.ClientResponseError, asyncio.TimeoutError):
                attempt += 1
                backoff = jittered_backoff(
                    attempt, self.config.backoff_base, self.config.backoff_cap, self.config.jitter
                )
                await asyncio.sleep(backoff)

    async def exchange_info(self) -> Dict[str, Any]:
        return await self._get("/fapi/v1/exchangeInfo")

    async def ticker_24h(self) -> List[Dict[str, Any]]:
        return await self._get("/fapi/v1/ticker/24hr")

    async def open_interest(self, symbol: str) -> Dict[str, Any]:
        return await self._get("/fapi/v1/openInterest", params={"symbol": symbol})

    async def klines(self, symbol: str) -> List[List[Any]]:
        params = {
            "symbol": symbol,
            "interval": self.config.binance.klines_interval,
            "limit": self.config.kline_limit,
        }
        return await self._get("/fapi/v1/klines", params=params)

    async def start_price_stream(self, symbols: List[str]):
        if self._ws_tasks:
            return
        streams = [f"{symbol.lower()}@markPrice" for symbol in symbols]
        max_streams = self.config.max_ws_streams_per_connection
        allowed = self.config.rate_limits.ws_max_connections * max_streams
        if len(streams) > allowed:
            streams = streams[:allowed]
        for idx in range(0, len(streams), max_streams):
            chunk = streams[idx : idx + max_streams]
            self._ws_tasks.append(asyncio.create_task(self._run_ws(chunk)))

    async def _run_ws(self, streams: List[str]):
        attempt = 0
        while True:
            try:
                query = "/".join(streams)
                url = f"{self.config.binance.ws_base}?streams={query}"
                async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                    async with self._ws_lock:
                        self._ws_active += 1
                        self._ws_connected.set()
                    attempt = 0
                    async for message in ws:
                        payload = json.loads(message)
                        data = payload.get("data", {})
                        symbol = data.get("s")
                        mark_price = data.get("p")
                        if symbol and mark_price:
                            self._mark_prices[symbol] = float(mark_price)
                            self._last_ws_message = now_ms()
            except asyncio.CancelledError:
                async with self._ws_lock:
                    self._ws_active = max(0, self._ws_active - 1)
                    if self._ws_active == 0:
                        self._ws_connected.clear()
                raise
            except Exception:
                async with self._ws_lock:
                    self._ws_active = max(0, self._ws_active - 1)
                    if self._ws_active == 0:
                        self._ws_connected.clear()
                attempt += 1
                backoff = jittered_backoff(
                    attempt, self.config.backoff_base, self.config.backoff_cap, self.config.jitter
                )
                await asyncio.sleep(backoff)

    async def mark_prices(self) -> Dict[str, float]:
        return dict(self._mark_prices)

    async def wait_for_ws(self, timeout: float = 5.0) -> bool:
        try:
            await asyncio.wait_for(self._ws_connected.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False
