# Quick Start Guide - Crypto Market Agents

## TL;DR - Get Started in 3 Commands

```bash
pip install aiohttp
python test_coinbase.py  # Verify connection
python run_with_coinbase.py  # Run the system
```

## What You Get

**4 AI Agents analyzing crypto markets in real-time:**

1. 📈 **Price Action** - Breakouts & volatility
2. 📊 **Momentum** - RSI & OBV indicators
3. 📦 **Volume Spikes** - Whale activity detection
4. 🎯 **Signal Synthesis** - Trading signals with entry/stop/target

**Output every 5 minutes:**
```json
{
  "asset": "BTCUSD",
  "direction": "LONG",
  "entry": 45000.00,
  "stop": 44100.00,
  "target": 46800.00,
  "confidence": 0.75,
  "rationale": "Bullish breakout (+3.2%) | Oversold RSI 28.5..."
}
```

## For US Users (Recommended)

### Using Coinbase (Default)

```bash
# Already configured! Just run:
python run_with_coinbase.py
```

No API keys needed for market data.

### Environment Setup (Optional)

```bash
# Create .env file
echo "EXCHANGE_NAME=coinbase" > .env
echo "LOG_LEVEL=INFO" >> .env
```

## For Non-US Users

### Using Bybit

```bash
export EXCHANGE_NAME=bybit
python -m crypto_market_agents.main
```

**Note**: Bybit restricts US users.

## File Output

Reports saved to:
- `output/latest_report.json` - Most recent signals
- `output/report_*.json` - Historical reports
- `crypto_agents.log` - System logs

## Customize Settings

Edit `crypto_market_agents/config.py` or use environment variables:

```bash
# More conservative breakout detection
export BREAKOUT_THRESHOLD=0.05  # 5% instead of 3%

# Higher confidence requirement
export MIN_CONFIDENCE=0.75  # 75% instead of 60%
```

Or programmatically:

```python
from crypto_market_agents.config import SystemConfig

config = SystemConfig()
config.price_action.breakout_threshold = 0.05
config.signal_synthesis.min_confidence = 0.75
```

## Test Without Real Data

Run the demo with simulated market data:

```bash
python demo.py
```

## What Each Script Does

| Script | Purpose |
|--------|---------|
| `run_with_coinbase.py` | ✅ Main entry point for US users |
| `test_coinbase.py` | Test Coinbase connection |
| `demo.py` | Demo with simulated data |
| `test_system.py` | Full system verification |
| `example.py` | Programming examples |

## Supported Exchanges

| Exchange | US Access | Status |
|----------|-----------|--------|
| **Coinbase** | ✅ Yes | **Default** |
| Kraken | ✅ Yes | Not implemented |
| Gemini | ✅ Yes | Not implemented |
| Binance.US | ✅ Yes | Not implemented |
| Bybit | ❌ No | Implemented |

## Adding More Exchanges

To add Kraken, Gemini, etc.:

1. Copy `crypto_market_agents/exchange/coinbase.py`
2. Rename and modify for new exchange
3. Add to `exchange/adapter.py`
4. Done! No agent code changes needed.

## Common Issues

### "Cannot connect to host"

**Solution**: Check internet connection. Verify exchange API is accessible.

### "No trading signals generated"

**Solutions**:
- Lower `min_confidence` threshold
- Adjust breakout percentages
- Ensure market has activity (not weekend/holiday)

### Rate limit errors

**Solution**: Reduce requests per second:
```python
config.exchange.rate_limit_per_second = 3
```

## Next Steps

1. ✅ Run `python run_with_coinbase.py`
2. 📊 Check `output/latest_report.json` after 5 minutes
3. 📈 Monitor logs in `crypto_agents.log`
4. ⚙️ Customize thresholds in config files
5. 🔧 Add your own custom agents (see `agents/` folder)

## Documentation

- **README.md** - Complete documentation
- **SETUP.md** - Detailed setup guide
- **COINBASE_SETUP.md** - Coinbase-specific guide
- **PROJECT_SUMMARY.md** - Technical architecture

## Support

1. Check logs: `crypto_agents.log`
2. Run tests: `python test_system.py`
3. Read docs in this repository
4. Review code comments

---

**You're ready to go!** 🚀

The system is:
- ✅ US-compliant (Coinbase)
- ✅ Production-ready
- ✅ Fully documented
- ✅ Easy to extend

Run `python run_with_coinbase.py` to start analyzing markets!
