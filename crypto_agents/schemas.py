from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class PriceActionMetrics:
    symbol: str
    price: float
    deltas: Dict[int, float]
    intraday_range: float
    volatility_ratio: float
    timestamp: int


@dataclass
class MomentumMetrics:
    symbol: str
    rsi: float
    obv: float
    timeframe: str
    timestamp: int


@dataclass
class VolumeSpikeMetrics:
    symbol: str
    current_volume: float
    baseline_volume: float
    z_score: float
    interpretation: str
    timestamp: int


@dataclass
class Signal:
    asset: str
    direction: str
    entry: float
    stop: float
    target: float
    confidence: float
    rationale: str
    timestamp: int


@dataclass
class AgentOutput:
    agent: str
    timestamp: int
    payload: List[Dict]


@dataclass
class MarketSnapshot:
    symbols: List[str]
    prices: Dict[str, float]
    klines: Dict[str, List[Dict]]
    tickers: Dict[str, Dict]
    open_interest: Dict[str, float]
    mark_prices: Dict[str, float]
    volumes: Dict[str, float]
    timestamp: int


@dataclass
class TradeWatchlist:
    timestamp: int
    signals: List[Signal]
    notes: Optional[str] = None
