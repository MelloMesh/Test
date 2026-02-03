# Using Coinbase Advanced Trade (US-Compliant)

## Why Coinbase?

**Coinbase is fully US-compliant and accessible from all 50 states.**

- ✅ Regulated by FinCEN
- ✅ Public market data freely accessible
- ✅ No VPN or geo-restrictions for US users
- ✅ Reliable API with good documentation

## Quick Start

### 1. Set Environment Variable

```bash
export EXCHANGE_NAME=coinbase
```

Or update your `.env` file:

```bash
# .env
EXCHANGE_NAME=coinbase
LOG_LEVEL=INFO
```

### 2. Run the System

```bash
# Run with Coinbase
python -m crypto_market_agents.main
```

That's it! The system will now use Coinbase instead of Bybit.

## Programmatic Usage

```python
from crypto_market_agents.config import SystemConfig, ExchangeConfig
from crypto_market_agents.orchestrator import AgentOrchestrator
import asyncio

async def main():
    # Configure for Coinbase
    config = SystemConfig()
    config.exchange = ExchangeConfig(
        name="coinbase",
        rate_limit_per_second=5  # Conservative rate limit
    )

    # Create and run orchestrator
    orchestrator = AgentOrchestrator(config)

    if await orchestrator.initialize():
        print("System initialized with Coinbase!")
        await orchestrator.run_forever()

if __name__ == "__main__":
    asyncio.run(main())
```

## Testing Coinbase Connection

Test that Coinbase is working:

```bash
python test_coinbase.py
```

Expected output:
```
✓ Created exchange adapter: CoinbaseExchange
✓ US Accessible: True
✓ Connected to Coinbase Advanced Trade API
✓ Found 200+ trading symbols
✓ Ticker data retrieved
```

## Available Symbols

Coinbase uses different quote currencies than Bybit:

- **USD pairs**: BTC-USD, ETH-USD, etc.
- **USDC pairs**: BTC-USDC, ETH-USDC, etc.
- **USDT pairs**: Limited availability

The adapter automatically handles symbol format conversion:
- Input: `BTCUSD` → Coinbase API: `BTC-USD`
- Input: `ETHUSDC` → Coinbase API: `ETH-USDC`

## Features Supported

All exchange abstraction features work with Coinbase:

| Feature | Supported | Notes |
|---------|-----------|-------|
| Get Trading Symbols | ✅ | All online products |
| Get Ticker | ✅ | Real-time prices |
| Get Multiple Tickers | ✅ | Batch retrieval |
| Get Candlesticks/OHLCV | ✅ | Multiple timeframes |
| Get Order Book | ✅ | Market depth data |
| Get Recent Trades | ✅ | Latest trades |
| Rate Limiting | ✅ | Token bucket + backoff |
| Error Handling | ✅ | Automatic retries |

## Timeframes

Coinbase supports these candlestick intervals:

| Interval | Coinbase Format |
|----------|----------------|
| 1 minute | ONE_MINUTE |
| 5 minutes | FIVE_MINUTE |
| 15 minutes | FIFTEEN_MINUTE |
| 30 minutes | THIRTY_MINUTE |
| 1 hour | ONE_HOUR |
| 6 hours | SIX_HOUR |
| 1 day | ONE_DAY |

## Rate Limits

Coinbase public API rate limits:

- **Public endpoints**: ~10 requests/second recommended
- **Authenticated**: Higher limits with API keys

The adapter includes automatic rate limiting and exponential backoff.

## API Keys (Optional)

For public market data, **no API keys are required**.

If you want to add trading functionality later:

1. Create API keys at: https://www.coinbase.com/settings/api
2. Set environment variables:
   ```bash
   export EXCHANGE_API_KEY=your_key_here
   export EXCHANGE_API_SECRET=your_secret_here
   ```

## Comparison: Bybit vs Coinbase

| Feature | Bybit | Coinbase |
|---------|-------|----------|
| US Access | ❌ Restricted | ✅ Full access |
| Regulation | Limited | ✅ FinCEN regulated |
| Quote Currencies | USDT | USD, USDC, USDT |
| Market Data | Excellent | Excellent |
| API Documentation | Good | Excellent |
| Rate Limits | Generous | Moderate |

## Complete Example

Full working example with Coinbase:

```python
"""
Complete example using Coinbase Advanced Trade.
"""

import asyncio
from crypto_market_agents.config import SystemConfig, ExchangeConfig
from crypto_market_agents.orchestrator import AgentOrchestrator


async def main():
    print("Starting Crypto Market Agents with Coinbase...\n")

    # Configure for Coinbase
    config = SystemConfig()
    config.exchange = ExchangeConfig(
        name="coinbase",
        rate_limit_per_second=5,
        max_retries=3,
        timeout=30
    )

    # Customize agent settings if needed
    config.price_action.breakout_threshold = 0.04  # 4% breakout
    config.momentum.rsi_overbought = 75.0
    config.momentum.rsi_oversold = 25.0
    config.volume.min_liquidity_usd = 1000000  # $1M minimum
    config.signal_synthesis.min_confidence = 0.65  # 65% minimum

    # Create orchestrator
    orchestrator = AgentOrchestrator(config)

    # Initialize
    if not await orchestrator.initialize():
        print("Failed to initialize. Check logs.")
        return

    print("✓ System initialized successfully with Coinbase")
    print("✓ All agents started")
    print("\nMonitoring market... Press Ctrl+C to stop\n")

    # Run forever
    await orchestrator.run_forever()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nShutdown requested. Goodbye!")
```

Save as `run_with_coinbase.py` and run:

```bash
python run_with_coinbase.py
```

## Troubleshooting

### Cannot connect to Coinbase

**Check**:
1. Internet connection
2. Firewall/proxy settings
3. Coinbase API status: https://status.coinbase.com/

### No symbols found

Coinbase may return different symbol formats. The adapter handles common cases, but you can customize symbol parsing in `crypto_market_agents/exchange/coinbase.py`.

### Rate limit errors

Reduce `rate_limit_per_second` in configuration:

```python
config.exchange.rate_limit_per_second = 3  # More conservative
```

## Next Steps

1. ✅ You're now using a US-compliant exchange!
2. Review output reports in `output/latest_report.json`
3. Customize agent thresholds in `config.py`
4. Monitor logs in `crypto_agents.log`

## Resources

- **Coinbase API Docs**: https://docs.cloud.coinbase.com/advanced-trade-api/docs
- **API Status**: https://status.coinbase.com/
- **Support**: https://help.coinbase.com/

---

**You're all set! The system now works with a fully US-compliant exchange.** 🎉
