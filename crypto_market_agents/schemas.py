"""
Shared data schemas for inter-agent communication.
"""

from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any
from datetime import datetime
import json


@dataclass
class MarketData:
    """Basic market data structure."""
    symbol: str
    price: float
    volume_24h: float
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class PriceActionSignal:
    """Signal from Price Action Agent."""
    symbol: str
    price: float
    price_change_pct: float
    intraday_range_pct: float
    volatility_ratio: float
    breakout_detected: bool
    timeframe: str
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class MomentumSignal:
    """Signal from Momentum Agent."""
    symbol: str
    rsi: float
    obv: float
    obv_change_pct: float
    status: str  # "overbought", "oversold", "neutral"
    strength_score: float  # 0-100
    timeframe: str
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class VolumeSignal:
    """Signal from Volume Spike Agent."""
    symbol: str
    volume_24h: float
    volume_change_pct: float
    volume_zscore: float
    spike_detected: bool
    baseline_volume: float
    liquidity_usd: float
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class TradingSignal:
    """Consolidated trading signal from Signal Synthesis Agent."""
    asset: str
    direction: str  # "LONG" or "SHORT"
    entry: float
    stop: float
    target: float
    confidence: float  # 0-1
    rationale: str
    timestamp: datetime

    # Supporting data
    price_signal: Optional[Dict[str, Any]] = None
    momentum_signal: Optional[Dict[str, Any]] = None
    volume_signal: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            "timestamp": self.timestamp.isoformat()
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class AgentStatus:
    """Status information for an agent."""
    agent_name: str
    status: str  # "running", "stopped", "error"
    last_update: datetime
    signals_generated: int
    errors: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            "last_update": self.last_update.isoformat()
        }


@dataclass
class SystemReport:
    """Consolidated system report."""
    timestamp: datetime
    active_agents: int
    trading_signals: List[TradingSignal]
    agent_statuses: List[AgentStatus]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "active_agents": self.active_agents,
            "trading_signals": [s.to_dict() for s in self.trading_signals],
            "agent_statuses": [s.to_dict() for s in self.agent_statuses]
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
