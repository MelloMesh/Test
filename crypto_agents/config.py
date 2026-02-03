from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class RateLimitConfig:
    max_weight_per_minute: int = 1200
    max_requests_per_minute: int = 2400
    ws_max_connections: int = 5


@dataclass
class LiquidityFilter:
    min_notional_volume: float = 5_000_000
    min_open_interest: float = 2_000_000


@dataclass
class WindowConfig:
    lookbacks: List[int] = field(default_factory=lambda: [1, 5, 15, 60])
    volatility_window: int = 60
    rsi_period: int = 14
    obv_period: int = 20
    volume_window: int = 20


@dataclass
class StrategyConfig:
    breakout_threshold: float = 0.02
    rapid_move_threshold: float = 0.03
    volatility_ratio_threshold: float = 1.5
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    volume_spike_z: float = 2.0
    reward_to_risk: float = 2.0
    max_stop_pct: float = 0.02


@dataclass
class OutputConfig:
    log_dir: str = "./logs"
    report_interval_sec: int = 300


@dataclass
class BinanceConfig:
    rest_base: str = "https://fapi.binance.com"
    ws_base: str = "wss://fstream.binance.com/stream"
    klines_interval: str = "1m"


@dataclass
class AppConfig:
    rate_limits: RateLimitConfig = field(default_factory=RateLimitConfig)
    liquidity: LiquidityFilter = field(default_factory=LiquidityFilter)
    windows: WindowConfig = field(default_factory=WindowConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    binance: BinanceConfig = field(default_factory=BinanceConfig)
    quote_asset: str = "USDT"
    max_symbols: int = 200
    poll_interval_sec: int = 30
    kline_limit: int = 200
    websocket_depth: int = 20
    max_ws_streams_per_connection: int = 200
    backoff_base: float = 0.5
    backoff_cap: float = 10.0
    jitter: float = 0.25
    agent_timeouts: Dict[str, float] = field(default_factory=lambda: {
        "price_action": 10.0,
        "momentum": 10.0,
        "volume_spike": 10.0,
        "signal_synthesis": 10.0,
    })
