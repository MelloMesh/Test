import asyncio
import json
import logging
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List


def setup_logger(name: str, log_dir: str) -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        file_handler = logging.FileHandler(Path(log_dir) / f"{name}.log")
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s"
        )
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
    return logger


def now_ms() -> int:
    return int(time.time() * 1000)


def percent_change(current: float, previous: float) -> float:
    if previous == 0:
        return 0.0
    return (current - previous) / previous * 100.0


def compute_rsi(closes: List[float], period: int) -> float:
    if len(closes) < period + 1:
        return 50.0
    gains = []
    losses = []
    for idx in range(-period, 0):
        delta = closes[idx] - closes[idx - 1]
        if delta >= 0:
            gains.append(delta)
            losses.append(0.0)
        else:
            gains.append(0.0)
            losses.append(abs(delta))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def compute_obv(closes: List[float], volumes: List[float]) -> float:
    if len(closes) < 2:
        return 0.0
    obv = 0.0
    for idx in range(1, len(closes)):
        if closes[idx] > closes[idx - 1]:
            obv += volumes[idx]
        elif closes[idx] < closes[idx - 1]:
            obv -= volumes[idx]
    return obv


def z_score(series: Iterable[float], current: float) -> float:
    values = list(series)
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    std = math.sqrt(variance)
    if std == 0:
        return 0.0
    return (current - mean) / std


def jittered_backoff(attempt: int, base: float, cap: float, jitter: float) -> float:
    raw = min(cap, base * (2 ** attempt))
    return raw * (1 + random.uniform(-jitter, jitter))


_log_lock = asyncio.Lock()


async def write_json_log(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, separators=(",", ":"))
    async with _log_lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(data + "\n")
