# Crypto Market Agents - Multi-Agent Trading Analysis System

A modular, production-ready system of autonomous agents for cryptocurrency market analysis. Built with async Python, this system analyzes live market data across multiple dimensions to generate consolidated trading insights and risk-managed signals.

## Overview

The system consists of four specialized agents that operate in parallel, each focusing on a specific analytical domain:

1. **Price Action Agent** - Monitors breakouts, rapid price changes, and volatility
2. **Momentum Agent** - Computes RSI and OBV indicators across multiple timeframes
3. **Volume Spike Agent** - Detects whale activity and unusual volume patterns
4. **Signal Synthesis Agent** - Integrates all data to generate actionable trading signals

All agents are independently callable and share data via structured JSON schemas, enabling modular development and easy extension.

## Key Features

- **Exchange Abstraction Layer** - Easily swap between exchanges without changing agent logic
- **US Compliance Focus** - Designed with US-accessible exchanges in mind
- **Async Parallel Execution** - All agents run concurrently for maximum efficiency
- **Rate Limiting & Backoff** - Built-in protection against API limits
- **Comprehensive Logging** - Full audit trail of all operations
- **JSON Output** - Structured data output for integration with other systems
- **Periodic Reporting** - Consolidated reports generated every 5 minutes

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Agent Orchestrator                        │
│  (Manages lifecycle, generates consolidated reports)        │
└───────────────┬─────────────────────────────────────────────┘
                │
        ┌───────┼───────┬────────────┬──────────────┐
        │       │       │            │              │
        ▼       ▼       ▼            ▼              ▼
    ┌─────┐ ┌─────┐ ┌──────┐  ┌──────────┐  ┌─────────────┐
    │Price│ │Momentum│Volume│  │  Signal  │  │  Exchange   │
    │Action│ │       │Spike │  │ Synthesis│  │  Adapter    │
    └─────┘ └─────┘ └──────┘  └──────────┘  └──────┬──────┘
                                                     │
                                              ┌──────▼──────┐
                                              │  Exchange   │
                                              │   API       │
                                              │ (Bybit/etc) │
                                              └─────────────┘
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd crypto_market_agents
```

2. Install dependencies:
```bash
pip install aiohttp
```

3. (Optional) Set environment variables:
```bash
export EXCHANGE_NAME=bybit
export EXCHANGE_API_KEY=your_api_key_here      # Optional for public endpoints
export EXCHANGE_API_SECRET=your_api_secret_here # Optional for public endpoints
export LOG_LEVEL=INFO
```

## Usage

### Basic Usage

Run the system with default configuration:

```bash
python -m crypto_market_agents.main
```

### Programmatic Usage

```python
import asyncio
from crypto_market_agents.config import SystemConfig
from crypto_market_agents.orchestrator import AgentOrchestrator

async def run_agents():
    # Load configuration
    config = SystemConfig.from_env()

    # Create and initialize orchestrator
    orchestrator = AgentOrchestrator(config)

    if await orchestrator.initialize():
        # Run forever (or until interrupted)
        await orchestrator.run_forever()

if __name__ == "__main__":
    asyncio.run(run_agents())
```

### Custom Configuration

```python
from crypto_market_agents.config import (
    SystemConfig,
    ExchangeConfig,
    PriceActionConfig,
    MomentumConfig,
    VolumeConfig,
    SignalSynthesisConfig
)

# Create custom configuration
config = SystemConfig(
    exchange=ExchangeConfig(
        name="bybit",
        rate_limit_per_second=5,
        testnet=True
    ),
    price_action=PriceActionConfig(
        breakout_threshold=0.05,  # 5% breakout threshold
        update_interval=30        # Update every 30 seconds
    ),
    momentum=MomentumConfig(
        rsi_overbought=75.0,
        rsi_oversold=25.0,
        timeframes=["5", "15", "60"]
    ),
    volume=VolumeConfig(
        min_liquidity_usd=500000,
        spike_threshold_zscore=2.5
    ),
    signal_synthesis=SignalSynthesisConfig(
        reward_risk_ratio=3.0,  # 3:1 reward-to-risk
        min_confidence=0.7      # Minimum 70% confidence
    )
)

# Use custom configuration
orchestrator = AgentOrchestrator(config)
```

### Accessing Individual Agents

```python
# Run individual agents independently
from crypto_market_agents.exchange.adapter import ExchangeFactory
from crypto_market_agents.agents.price_action import PriceActionAgent
from crypto_market_agents.config import ExchangeConfig, PriceActionConfig

async def run_price_action_only():
    # Create exchange
    exchange = ExchangeFactory.create(ExchangeConfig())
    await exchange.connect()

    # Create and start agent
    agent = PriceActionAgent(exchange, PriceActionConfig())
    await agent.start()

    # Let it run for a while
    await asyncio.sleep(300)

    # Get signals
    signals = agent.get_latest_signals()
    for signal in signals:
        if signal.breakout_detected:
            print(f"Breakout detected: {signal.symbol} - {signal.price_change_pct:.2f}%")

    await agent.stop()
    await exchange.disconnect()
```

## Output Format

### Trading Signal Example

```json
{
  "asset": "BTCUSDT",
  "direction": "LONG",
  "entry": 45000.0,
  "stop": 44100.0,
  "target": 46800.0,
  "confidence": 0.75,
  "rationale": "Bullish breakout (+3.2%) | Oversold RSI 28.5 on 15m | Volume spike (z-score: 2.3, +45.2%)",
  "timestamp": "2025-01-15T10:30:00.000000Z",
  "price_signal": { ... },
  "momentum_signal": { ... },
  "volume_signal": { ... }
}
```

### System Report Example

Reports are saved to the `output/` directory every 5 minutes:

```json
{
  "timestamp": "2025-01-15T10:30:00.000000Z",
  "active_agents": 4,
  "trading_signals": [
    {
      "asset": "BTCUSDT",
      "direction": "LONG",
      ...
    }
  ],
  "agent_statuses": [
    {
      "agent_name": "PriceAction",
      "status": "running",
      "signals_generated": 150,
      "errors": 0,
      "last_update": "2025-01-15T10:30:00.000000Z"
    }
  ]
}
```

## Exchange Abstraction

The system uses an exchange abstraction layer, making it easy to swap exchanges without modifying agent code.

### Supported Exchanges

Currently implemented:
- **Bybit** (Note: Restricted for US users - public data may be accessible)

### US-Compliant Alternatives

The system is designed to easily integrate US-accessible exchanges:

- **Coinbase Advanced Trade** - Fully US-compliant
- **Kraken** - Available in most US states
- **Gemini** - New York-based, fully regulated
- **Binance.US** - US-specific version

### Adding a New Exchange

1. Create a new adapter implementing `BaseExchange`:

```python
from crypto_market_agents.exchange.base import BaseExchange

class CoinbaseExchange(BaseExchange):
    async def connect(self) -> bool:
        # Implementation
        pass

    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        # Implementation
        pass

    # ... implement all abstract methods
```

2. Register it in `exchange/adapter.py`:

```python
if exchange_name == "coinbase":
    return CoinbaseExchange(...)
```

3. Update configuration:

```python
config.exchange.name = "coinbase"
```

## Agent Details

### Price Action Agent

**Purpose**: Detect significant price movements and breakouts

**Metrics**:
- Price change percentage over multiple timeframes
- Intraday range (high-low spread)
- Volatility ratio (current vs historical)

**Signals Generated**:
- Breakout detection (threshold-based)
- Abnormal volatility alerts

### Momentum Agent

**Purpose**: Identify overbought/oversold conditions and momentum trends

**Indicators**:
- RSI (Relative Strength Index)
- OBV (On-Balance Volume)

**Timeframes**: Configurable (default: 5m, 15m, 60m)

**Signals Generated**:
- Overbought conditions (RSI > 70)
- Oversold conditions (RSI < 30)
- Volume momentum trends

### Volume Spike Agent

**Purpose**: Detect unusual volume activity indicating whale accumulation/distribution

**Analysis**:
- Z-score based spike detection
- Percentile-based thresholds
- Liquidity filtering (minimum 24h volume)

**Signals Generated**:
- Volume spike alerts with statistical confidence
- Baseline vs current volume comparison

### Signal Synthesis Agent

**Purpose**: Integrate all agent outputs into actionable trading signals

**Process**:
1. Correlate price, momentum, and volume data
2. Score each signal component
3. Determine trade direction (LONG/SHORT)
4. Calculate entry, stop-loss, and take-profit levels
5. Compute confidence score (0-1)
6. Generate rationale explaining the setup

**Risk Management**:
- Configurable reward-to-risk ratio (default: 2:1)
- Maximum stop-loss percentage (default: 2%)
- Minimum confidence threshold (default: 60%)

## Performance & Compliance

### Rate Limiting

The system includes sophisticated rate limiting:

- Token bucket algorithm
- Per-exchange rate limits
- Automatic backoff on 429 responses
- Exponential retry with jitter

### Error Handling

- Graceful degradation on API failures
- Automatic reconnection
- Per-agent error tracking
- Comprehensive logging

### US Accessibility Considerations

**Important**: Bybit restricts US users from trading. While some public market data endpoints may be accessible, users should:

1. Verify compliance with local regulations
2. Check Bybit's current terms of service
3. Consider using US-compliant exchanges instead

The system logs warnings and provides information about US-accessible alternatives.

## Configuration Reference

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `EXCHANGE_NAME` | Exchange to use | `bybit` |
| `EXCHANGE_API_KEY` | API key (optional for public data) | `None` |
| `EXCHANGE_API_SECRET` | API secret (optional for public data) | `None` |
| `EXCHANGE_TESTNET` | Use testnet | `false` |
| `LOG_LEVEL` | Logging level | `INFO` |

### Configuration Classes

- `ExchangeConfig` - Exchange connection settings
- `PriceActionConfig` - Price action agent settings
- `MomentumConfig` - Momentum agent settings
- `VolumeConfig` - Volume spike agent settings
- `SignalSynthesisConfig` - Signal synthesis settings
- `SystemConfig` - Overall system configuration

## Development

### Project Structure

```
crypto_market_agents/
├── __init__.py
├── config.py              # Configuration management
├── schemas.py             # Shared data schemas
├── orchestrator.py        # Main orchestration logic
├── main.py               # Entry point
├── exchange/
│   ├── base.py           # Abstract exchange interface
│   ├── bybit.py          # Bybit implementation
│   └── adapter.py        # Exchange factory
├── agents/
│   ├── base_agent.py     # Base agent class
│   ├── price_action.py   # Price Action Agent
│   ├── momentum.py       # Momentum Agent
│   ├── volume_spike.py   # Volume Spike Agent
│   └── signal_synthesis.py # Signal Synthesis Agent
└── utils/
    ├── logging.py        # Logging utilities
    └── rate_limiter.py   # Rate limiting utilities
```

### Extending the System

**Adding a New Agent**:

1. Create a new agent class inheriting from `BaseAgent`
2. Implement the `execute()` method
3. Define signal schema in `schemas.py`
4. Register in `orchestrator.py`

**Adding New Indicators**:

Add calculation methods to existing agents or create new specialized agents.

**Custom Signal Logic**:

Modify `SignalSynthesisAgent._determine_direction()` to customize signal generation logic.

## Logging

Logs are written to both console and file (`crypto_agents.log` by default).

**Log Levels**:
- `DEBUG` - Detailed diagnostic information
- `INFO` - General informational messages
- `WARNING` - Warning messages (e.g., US accessibility warnings)
- `ERROR` - Error messages

## Troubleshooting

### Connection Issues

**Problem**: Unable to connect to exchange

**Solutions**:
- Check internet connection
- Verify exchange API is accessible from your location
- Check if using correct API endpoints (testnet vs mainnet)
- Review firewall/proxy settings

### Rate Limiting

**Problem**: Frequent 429 errors

**Solutions**:
- Reduce `rate_limit_per_second` in configuration
- Increase `update_interval` for agents
- Check exchange rate limit documentation

### No Signals Generated

**Problem**: Agents running but no trading signals

**Solutions**:
- Lower `min_confidence` threshold
- Adjust breakout/spike thresholds
- Verify market has sufficient volatility
- Check that all required agents are enabled

## Legal Disclaimer

**This software is for educational and research purposes only.**

- Not financial advice
- No warranty of any kind
- Use at your own risk
- Verify compliance with local regulations
- Check exchange terms of service
- Consult with legal and financial advisors before trading

## License

MIT License - See LICENSE file for details

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Implement your changes with tests
4. Submit a pull request

## Support

For issues, questions, or contributions, please open an issue on GitHub.

---

**Built with Python 3.8+ | Async/Await | Production-Ready**
