"""
Configuration management for the crypto market agents system.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import os


@dataclass
class ExchangeConfig:
    """Configuration for exchange connection."""
    name: str = "coinbase"  # Default to US-compliant exchange
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
    min_confidence: float = 0.35  # Temporarily reduced from 0.6 for testing
    update_interval: int = 300  # seconds (5 minutes)


@dataclass
class SRDetectionConfig:
    """Configuration for S/R Detection Agent."""
    enabled: bool = True
    timeframes: List[str] = field(default_factory=lambda: ['1M', '1w', '3d', '1d', '4h', '1h'])
    lookback: int = 100
    min_touches: int = 2
    confluence_tolerance: float = 0.015  # 1.5% price tolerance for clustering
    update_interval: int = 300  # seconds (5 minutes)


@dataclass
class FibonacciConfig:
    """Configuration for Fibonacci Agent."""
    enabled: bool = True
    lookback: int = 100
    min_swing_size: float = 0.02  # 2% minimum swing
    update_interval: int = 300  # seconds (5 minutes)


@dataclass
class LearningConfig:
    """Configuration for Learning Agent."""
    enabled: bool = True
    paper_trading: bool = True
    data_dir: str = "data"
    update_interval: int = 300  # seconds (5 minutes)
    min_trades_before_learning: int = 20
    auto_optimize: bool = True


@dataclass
class TelegramConfig:
    """Configuration for Telegram Bot."""
    enabled: bool = True
    bot_token: str = ""  # Loaded from environment
    chat_id: str = ""  # Loaded from environment
    send_signals: bool = True  # Send trading signals
    send_alerts: bool = True  # Send system alerts
    send_trade_updates: bool = True  # Send trade execution updates
    max_signals_per_batch: int = 5  # Maximum signals to send at once


@dataclass
class SystemConfig:
    """Main system configuration."""
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    price_action: PriceActionConfig = field(default_factory=PriceActionConfig)
    momentum: MomentumConfig = field(default_factory=MomentumConfig)
    volume: VolumeConfig = field(default_factory=VolumeConfig)
    signal_synthesis: SignalSynthesisConfig = field(default_factory=SignalSynthesisConfig)
    sr_detection: SRDetectionConfig = field(default_factory=SRDetectionConfig)
    fibonacci: FibonacciConfig = field(default_factory=FibonacciConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)

    # Global settings
    log_level: str = "INFO"
    log_file: str = "crypto_agents.log"
    output_dir: str = "output"
    data_cache_ttl: int = 60  # seconds
    symbols: List[str] = field(default_factory=list)  # Can be set for specific symbols

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

        # Telegram configuration
        config.telegram.bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        config.telegram.chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        config.telegram.enabled = bool(config.telegram.bot_token and config.telegram.chat_id)

        return config
