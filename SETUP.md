# Setup Guide

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment (Optional)

Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env` with your settings (API keys are optional for public data).

### 3. Run Tests

Verify the system is working:

```bash
python test_system.py
```

### 4. Run the System

```bash
# Run with default configuration
python -m crypto_market_agents.main

# Or run the example
python example.py
```

## Configuration Options

### Method 1: Environment Variables

Set environment variables before running:

```bash
export EXCHANGE_NAME=bybit
export LOG_LEVEL=DEBUG
python -m crypto_market_agents.main
```

### Method 2: .env File

Create a `.env` file (see `.env.example` for template):

```
EXCHANGE_NAME=bybit
EXCHANGE_TESTNET=false
LOG_LEVEL=INFO
```

### Method 3: Programmatic Configuration

```python
from crypto_market_agents.config import SystemConfig, ExchangeConfig

config = SystemConfig(
    exchange=ExchangeConfig(
        name="bybit",
        testnet=True,
        rate_limit_per_second=5
    ),
    log_level="DEBUG"
)

orchestrator = AgentOrchestrator(config)
```

## US Compliance

**IMPORTANT**: Bybit restricts US users. For production use in the US:

### Option 1: Use Coinbase (Recommended for US)

1. Create adapter for Coinbase (see `exchange/bybit.py` as template)
2. Implement `BaseExchange` interface
3. Register in `exchange/adapter.py`
4. Set `EXCHANGE_NAME=coinbase`

### Option 2: Use Other US-Compliant Exchanges

- **Kraken**: Available in most US states
- **Gemini**: New York-based, fully regulated
- **Binance.US**: US-specific version

## Directory Structure

After running, you'll see:

```
Test/
├── crypto_market_agents/     # Main package
├── output/                   # Generated reports (created automatically)
│   ├── latest_report.json
│   └── report_*.json
├── crypto_agents.log         # Log file
└── ...
```

## Troubleshooting

### No module named 'aiohttp'

```bash
pip install aiohttp
```

### Connection errors

- Check internet connection
- Verify exchange is accessible from your location
- Try testnet mode: `EXCHANGE_TESTNET=true`

### No signals generated

- Lower confidence threshold in configuration
- Increase lookback periods
- Ensure market has sufficient activity

## Next Steps

1. Review generated reports in `output/`
2. Customize agent configurations
3. Implement US-compliant exchange adapter
4. Integrate with your trading strategy

## Advanced Usage

### Custom Agent

Create a new agent by inheriting from `BaseAgent`:

```python
from crypto_market_agents.agents.base_agent import BaseAgent

class MyCustomAgent(BaseAgent):
    async def execute(self):
        # Your logic here
        pass
```

### Custom Exchange

Implement `BaseExchange` interface:

```python
from crypto_market_agents.exchange.base import BaseExchange

class MyExchange(BaseExchange):
    async def connect(self) -> bool:
        # Implementation
        pass

    # ... implement all required methods
```

### Integration

Use the orchestrator in your application:

```python
import asyncio
from crypto_market_agents.orchestrator import AgentOrchestrator
from crypto_market_agents.config import SystemConfig

async def my_app():
    config = SystemConfig.from_env()
    orchestrator = AgentOrchestrator(config)

    await orchestrator.initialize()
    await orchestrator.start()

    while True:
        # Your application logic
        signals = orchestrator.get_latest_signals()

        # Process signals
        for signal in signals:
            if signal.confidence > 0.8:
                print(f"High confidence signal: {signal.asset}")

        await asyncio.sleep(60)

asyncio.run(my_app())
```

## Support

For issues or questions:

1. Check the logs in `crypto_agents.log`
2. Review the README.md
3. Check configuration in `config.py`
4. Open an issue on GitHub
