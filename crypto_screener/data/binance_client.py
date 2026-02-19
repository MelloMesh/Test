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
from typing import Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from crypto_screener import config

logger = logging.getLogger(__name__)

# ── Persistent HTTP session with connection pooling ───────────────────────────
# Reuses TCP connections across all requests — critical for 100-symbol scans.
# urllib3 handles the connection pool; we do our own retry logic on top.
_session = requests.Session()
_adapter = HTTPAdapter(
    pool_connections=30,
    pool_maxsize=30,
    max_retries=Retry(total=0),  # no urllib3 retries — we handle them ourselves
)
_session.mount("https://", _adapter)
_session.mount("http://", _adapter)

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


def clear_cache() -> None:
    """Evict all cached entries. Call at the start of each watch-mode scan cycle."""
    _cache.clear()


# ── HTTP helper ───────────────────────────────────────────────────────────────

def _get(path: str, params: dict | None = None) -> dict | list:
    """
    GET request using the shared session (connection pooling) with exponential
    backoff retry. Retry logs are DEBUG-level to keep watch-mode output clean.
    """
    url = config.BINANCE_BASE_URL + path
    delay = 1.0
    last_exc: Exception = RuntimeError("unreachable")

    for attempt in range(config.MAX_RETRIES + 1):
        try:
            resp = _session.get(url, params=params, timeout=config.REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < config.MAX_RETRIES:
                logger.debug("Request failed (attempt %d/%d): %s — retrying in %.0fs",
                             attempt + 1, config.MAX_RETRIES, exc, delay)
                time.sleep(delay)
                delay *= 2
            else:
                logger.debug("All retries exhausted for %s", url)

    raise last_exc


# ── Shared ticker data (avoids double-fetch of /api/v3/ticker/24hr) ───────────

def _get_all_tickers() -> list[dict]:
    """
    Fetch /api/v3/ticker/24hr once and cache under a stable short key.

    Both get_top_symbols() and get_24h_volumes() share this result.
    This avoids two separate weight-40 requests (each call to ticker/24hr
    costs Binance weight 40) — a significant saving against the 1200/min limit.
    """
    cache_key = "tickers_all"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached  # type: ignore[return-value]

    data: list[dict] = _get("/api/v3/ticker/24hr")  # type: ignore[assignment]
    _cache_set(cache_key, data)
    return data


# ── Public API ────────────────────────────────────────────────────────────────

def get_top_symbols(n: int = config.TOP_N_SYMBOLS) -> list[str]:
    """
    Return top-N USDT symbols ranked by 24h quote volume.

    Applies two liquidity filters:
    1. Absolute floor: MIN_VOLUME_USD ($20M by default) — hard minimum regardless
       of universe rank. Eliminates micro-caps with manipulable volume.
    2. Relative rank: top-N after the absolute floor.

    Uses /api/v3/ticker/24hr — no auth required (shared via _get_all_tickers).
    """
    cache_key = f"top_symbols_{n}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached  # type: ignore[return-value]

    data = _get_all_tickers()

    quote = config.QUOTE_ASSET
    excluded = config.EXCLUDED_BASE_ASSETS
    min_vol = config.MIN_VOLUME_USD

    usdt_pairs = [
        d for d in data
        if d["symbol"].endswith(quote)
        and not d["symbol"].endswith(f"DOWN{quote}")
        and not d["symbol"].endswith(f"UP{quote}")
        and d.get("status", "TRADING") == "TRADING"
        and d["symbol"][: -len(quote)] not in excluded
        and float(d["quoteVolume"]) >= min_vol  # absolute liquidity floor
    ]

    ranked = sorted(usdt_pairs, key=lambda d: float(d["quoteVolume"]), reverse=True)
    symbols = [d["symbol"] for d in ranked[:n]]

    _cache_set(cache_key, symbols)
    logger.info(
        "Top %d symbols fetched (≥$%.0fM 24h vol): %s … %s",
        len(symbols), min_vol / 1e6, symbols[0] if symbols else "?", symbols[-1] if symbols else "?"
    )
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

    Raises ValueError if fewer than config.MIN_CANDLES candles are returned —
    this guards all indicator computations against sparse data on new listings.
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

    # Guard: reject symbols with insufficient history for EMA-200 warmup.
    if len(df) < config.MIN_CANDLES:
        raise ValueError(
            f"Insufficient data for {symbol}/{interval}: "
            f"{len(df)} candles returned, need ≥{config.MIN_CANDLES}"
        )

    _cache_set(cache_key, df)
    logger.debug("Fetched %d candles for %s/%s", len(df), symbol, interval)
    return df


def get_24h_volumes(symbols: list[str]) -> dict[str, float]:
    """
    Return a {symbol: quote_volume} mapping for the given symbols.
    Used by the scorer to gate illiquid setups.

    Reuses the cached ticker data from _get_all_tickers() — no extra API call.
    """
    data = _get_all_tickers()
    vol_map = {d["symbol"]: float(d["quoteVolume"]) for d in data}
    return {s: vol_map.get(s, 0.0) for s in symbols}
