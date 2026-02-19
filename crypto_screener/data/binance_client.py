"""
data/binance_client.py — Binance public REST API client.

Responsibilities:
- Fetch top-N symbols by 24h USDT volume
- Fetch OHLCV klines for a given symbol + timeframe
- Return clean pandas DataFrames with UTC timestamps
- Aggressive in-memory caching; exponential backoff on errors

Rate-limit/backoff pattern ported from the JS apiFetch() in index.html.
"""

from __future__ import annotations

import time
import logging
from functools import lru_cache
from typing import Optional

import pandas as pd
import requests

from crypto_screener import config

logger = logging.getLogger(__name__)

# ── In-memory cache ───────────────────────────────────────────────────────────
_cache: dict[str, tuple[float, object]] = {}


def _cache_get(key: str) -> Optional[object]:
    """Return cached value if still fresh, else None."""
    if key in _cache:
        ts, val = _cache[key]
        if time.time() - ts < config.CACHE_TTL_SECONDS:
            return val
    return None


def _cache_set(key: str, val: object) -> None:
    _cache[key] = (time.time(), val)


# ── HTTP helper ───────────────────────────────────────────────────────────────

def _get(path: str, params: dict | None = None) -> dict | list:
    """
    GET request with exponential backoff retry.
    Mirrors the JS apiFetch() retry pattern: wait 2s → 4s → 8s → 16s.
    """
    url = config.BINANCE_BASE_URL + path
    delay = 2.0
    last_exc: Exception = RuntimeError("unreachable")

    for attempt in range(config.MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=config.REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < config.MAX_RETRIES:
                logger.warning("Request failed (attempt %d/%d): %s — retrying in %.0fs",
                               attempt + 1, config.MAX_RETRIES, exc, delay)
                time.sleep(delay)
                delay *= 2
            else:
                logger.error("All retries exhausted for %s", url)

    raise last_exc


# ── Public API ────────────────────────────────────────────────────────────────

def get_top_symbols(n: int = config.TOP_N_SYMBOLS) -> list[str]:
    """
    Return top-N USDT symbols ranked by 24h quote volume.
    Uses /api/v3/ticker/24hr — no auth required.
    """
    cache_key = f"top_symbols_{n}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached  # type: ignore[return-value]

    data: list[dict] = _get("/api/v3/ticker/24hr")  # type: ignore[assignment]

    quote = config.QUOTE_ASSET
    excluded = config.EXCLUDED_BASE_ASSETS
    usdt_pairs = [
        d for d in data
        if d["symbol"].endswith(quote)
        and not d["symbol"].endswith(f"DOWN{quote}")
        and not d["symbol"].endswith(f"UP{quote}")
        and d.get("status", "TRADING") == "TRADING"
        and d["symbol"][: -len(quote)] not in excluded
    ]

    ranked = sorted(usdt_pairs, key=lambda d: float(d["quoteVolume"]), reverse=True)
    symbols = [d["symbol"] for d in ranked[:n]]

    _cache_set(cache_key, symbols)
    logger.info("Top %d symbols fetched: %s … %s", n, symbols[0], symbols[-1])
    return symbols


_BINANCE_MAX_LIMIT = 1000  # Binance hard cap per klines request


def _parse_raw_klines(raw: list[list]) -> pd.DataFrame:
    """Convert raw Binance kline list to a clean OHLCV DataFrame."""
    df = pd.DataFrame(raw, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ])
    df = df[["open_time", "open", "high", "low", "close", "volume"]].copy()
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.rename(columns={"open_time": "timestamp"})
    df = df.set_index("timestamp")
    df = df.astype(float)
    return df


def get_ohlcv(
    symbol: str,
    interval: str = config.DEFAULT_TIMEFRAME,
    limit: int = config.KLINE_LIMIT,
) -> pd.DataFrame:
    """
    Fetch OHLCV klines from Binance.

    Returns a DataFrame indexed by UTC datetime with columns:
        open, high, low, close, volume  (all float64)

    Automatically paginates for limit > 1000 (Binance hard cap per request).
    Uses /api/v3/klines — no auth required.
    """
    cache_key = f"ohlcv_{symbol}_{interval}_{limit}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached  # type: ignore[return-value]

    if limit <= _BINANCE_MAX_LIMIT:
        # Single request — fast path
        raw: list[list] = _get(  # type: ignore[assignment]
            "/api/v3/klines",
            params={"symbol": symbol, "interval": interval, "limit": limit},
        )
        if not raw:
            raise ValueError(f"Empty kline response for {symbol}/{interval}")
        df = _parse_raw_klines(raw)
    else:
        # Paginate backwards: fetch 1000 at a time using endTime
        frames: list[pd.DataFrame] = []
        remaining = limit
        end_time_ms: int | None = None  # None = most recent candle

        while remaining > 0:
            batch_size = min(remaining, _BINANCE_MAX_LIMIT)
            params: dict = {"symbol": symbol, "interval": interval, "limit": batch_size}
            if end_time_ms is not None:
                params["endTime"] = end_time_ms

            raw = _get("/api/v3/klines", params=params)  # type: ignore[assignment]
            if not raw:
                break

            batch_df = _parse_raw_klines(raw)
            frames.append(batch_df)
            remaining -= len(batch_df)

            # Walk backwards: next request ends just before the earliest candle
            earliest_ms = int(batch_df.index.min().timestamp() * 1000)
            if end_time_ms is not None and earliest_ms >= end_time_ms:
                break  # no more history available
            end_time_ms = earliest_ms - 1

            if len(batch_df) < batch_size:
                break  # reached the beginning of available history

            time.sleep(0.1)  # gentle rate limit between pagination requests

        if not frames:
            raise ValueError(f"Empty kline response for {symbol}/{interval}")

        df = pd.concat(frames)
        df = df[~df.index.duplicated(keep="first")]

    df = df.sort_index()
    _cache_set(cache_key, df)
    logger.debug("Fetched %d candles for %s/%s", len(df), symbol, interval)
    return df


def get_24h_volumes(symbols: list[str]) -> dict[str, float]:
    """
    Return a {symbol: quote_volume} mapping for the given symbols.
    Used by the scorer to gate illiquid setups.
    """
    cache_key = f"volumes_{'_'.join(sorted(symbols))}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached  # type: ignore[return-value]

    data: list[dict] = _get("/api/v3/ticker/24hr")  # type: ignore[assignment]
    vol_map = {d["symbol"]: float(d["quoteVolume"]) for d in data}
    result = {s: vol_map.get(s, 0.0) for s in symbols}

    _cache_set(cache_key, result)
    return result
