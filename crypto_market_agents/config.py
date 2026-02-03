"""
Configuration management for the crypto market agents system.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import os


@dataclass
class ExchangeConfig:
    """Configuration for exchange connection."""
    name: str = "bybit"
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    testnet: bool = False
    rate_limit_per_second: int = 10
    max_retries: int = 3
    retry_delay: float = 2.0
    timeout: int = 30


@dataclass
class PriceActionConfig:
    """Configuration for Price Action Agent."""
    enabled: bool = True
    lookback_windows: List[int] = field(default_factory=lambda: [5, 15, 60])  # minutes
    breakout_threshold: float = 0.03  # 3% price change
    volatility_threshold: float = 2.0  # 2x average volatility
    update_interval: int = 60  # seconds


@dataclass
class MomentumConfig:
    """Configuration for Momentum Agent."""
    enabled: bool = True
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    obv_lookback: int = 20
    timeframes: List[str] = field(default_factory=lambda: ["5", "15", "60"])  # minutes
    update_interval: int = 60  # seconds


@dataclass
class VolumeConfig:
    """Configuration for Volume Spike Agent."""
    enabled: bool = True
    min_liquidity_usd: float = 1000000.0  # $1M minimum 24h volume
    spike_threshold_zscore: float = 2.0
    spike_threshold_percentile: float = 95.0
    lookback_periods: int = 100
    update_interval: int = 60  # seconds


@dataclass
class SignalSynthesisConfig:
    """Configuration for Signal Synthesis Agent."""
    enabled: bool = True
    reward_risk_ratio: float = 2.0
    max_stop_loss_pct: float = 2.0
    min_confidence: float = 0.6
    update_interval: int = 300  # seconds (5 minutes)


@dataclass
class SystemConfig:
    """Main system configuration."""
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    price_action: PriceActionConfig = field(default_factory=PriceActionConfig)
    momentum: MomentumConfig = field(default_factory=MomentumConfig)
    volume: VolumeConfig = field(default_factory=VolumeConfig)
    signal_synthesis: SignalSynthesisConfig = field(default_factory=SignalSynthesisConfig)

    # Global settings
    log_level: str = "INFO"
    log_file: str = "crypto_agents.log"
    output_dir: str = "output"
    data_cache_ttl: int = 60  # seconds

    @classmethod
    def from_env(cls) -> "SystemConfig":
        """Load configuration from environment variables."""
        config = cls()

        # Exchange configuration
        config.exchange.api_key = os.getenv("EXCHANGE_API_KEY")
        config.exchange.api_secret = os.getenv("EXCHANGE_API_SECRET")
        config.exchange.testnet = os.getenv("EXCHANGE_TESTNET", "false").lower() == "true"
        config.exchange.name = os.getenv("EXCHANGE_NAME", "bybit")

        # System configuration
        config.log_level = os.getenv("LOG_LEVEL", "INFO")

        return config
