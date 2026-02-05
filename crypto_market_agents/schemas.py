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
class SRLevel:
    """Support or resistance level from a specific timeframe."""
    price: float
    strength: int
    touches: int
    timeframe: str
    level_type: str  # 'support' or 'resistance'
    timeframe_weight: int
    first_seen: datetime
    last_touch: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            "first_seen": self.first_seen.isoformat(),
            "last_touch": self.last_touch.isoformat()
        }


@dataclass
class SRConfluence:
    """Confluence zone where multiple S/R levels cluster."""
    price: float
    confluence_score: int
    levels: List[SRLevel]
    distance_percent: float
    zone_type: str  # 'support', 'resistance', or 'both'
    strength: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "price": self.price,
            "confluence_score": self.confluence_score,
            "levels": [level.to_dict() for level in self.levels],
            "distance_percent": self.distance_percent,
            "zone_type": self.zone_type,
            "strength": self.strength
        }


@dataclass
class FibonacciLevels:
    """Fibonacci retracement and extension levels for a symbol."""
    symbol: str
    swing_high: float
    swing_low: float
    swing_direction: str  # 'bullish' or 'bearish'
    swing_size_pct: float
    retracement_levels: Dict[float, float]  # ratio -> price
    extension_levels: Dict[float, float]  # ratio -> price
    golden_pocket_low: float
    golden_pocket_high: float
    in_golden_pocket: bool
    distance_from_golden_pocket: float
    current_price: float
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class PaperTrade:
    """Paper trade record for learning and strategy validation."""
    trade_id: str
    symbol: str
    direction: str  # 'LONG' or 'SHORT'
    entry_price: float
    stop_loss: float
    take_profit: float
    entry_time: datetime
    exit_time: Optional[datetime]
    exit_price: Optional[float]
    pnl: Optional[float]
    pnl_percent: Optional[float]
    outcome: str  # 'WIN', 'LOSS', 'BREAKEVEN', 'OPEN'

    # Signal data that triggered the trade
    signal_data: Dict[str, Any]

    # Market context at entry
    sr_levels: Optional[List[Dict[str, Any]]] = None
    volume_data: Optional[Dict[str, Any]] = None
    momentum_data: Optional[Dict[str, Any]] = None

    # Learning metadata
    notes: str = ""
    confidence_at_entry: float = 0.0

    # R:R tracking for optimal target analysis
    target_hit: bool = False  # Whether initial target (2:1 R:R) was hit
    target_hit_time: Optional[datetime] = None  # When target was hit
    max_favorable_excursion: float = 0.0  # Best price achieved (in favor)
    max_rr_achieved: float = 0.0  # Maximum R:R ratio achieved

    def to_dict(self) -> Dict[str, Any]:
        data = {
            **asdict(self),
            "entry_time": self.entry_time.isoformat()
        }
        if self.exit_time:
            data["exit_time"] = self.exit_time.isoformat()
        if self.target_hit_time:
            data["target_hit_time"] = self.target_hit_time.isoformat()
        return data


@dataclass
class StrategyMetrics:
    """Performance metrics for trading strategy."""
    total_trades: int
    wins: int
    losses: int
    breakeven: int
    open_trades: int
    win_rate: float
    avg_rr: float
    profit_factor: float
    total_pnl_percent: float
    avg_win_percent: float
    avg_loss_percent: float
    largest_win_percent: float
    largest_loss_percent: float
    timestamp: datetime

    # Streak tracking (added in optimization)
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    current_streak: int = 0  # Positive for wins, negative for losses

    # Risk metrics (added in optimization)
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown_percent: float = 0.0
    calmar_ratio: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class LearningInsights:
    """Insights from the learning agent for signal optimization."""
    symbol: str
    confidence_adjustment: float  # Adjustment to add to confidence score
    context_multiplier: float  # Multiplier for overall score
    win_rate_at_sr: float  # Win rate when trading at S/R levels
    win_rate_overall: float
    recommended_action: str  # 'increase_position', 'decrease_position', 'avoid', 'normal'
    reasoning: str
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
    order_type: str = "MARKET"  # "LIMIT" or "MARKET"
    confluence_score: int = 0  # Confluence score for limit order priority

    # Supporting data
    price_signal: Optional[Dict[str, Any]] = None
    momentum_signal: Optional[Dict[str, Any]] = None
    volume_signal: Optional[Dict[str, Any]] = None
    sr_data: Optional[Dict[str, Any]] = None
    fib_data: Optional[Dict[str, Any]] = None
    learning_insights: Optional[Dict[str, Any]] = None

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
            "agent_statuses": [s.to_dict() if hasattr(s, 'to_dict') else s for s in self.agent_statuses]
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
