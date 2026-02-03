"""
Run the Crypto Market Agents system with Coinbase (US-compliant).
"""

import asyncio
from crypto_market_agents.config import SystemConfig, ExchangeConfig
from crypto_market_agents.orchestrator import AgentOrchestrator


async def main():
    """Run the system with Coinbase."""

    print("="*80)
    print("CRYPTO MARKET AGENTS - COINBASE EDITION")
    print("="*80)
    print("\nUsing Coinbase Advanced Trade (US-Compliant)")
    print("No API keys required for public market data\n")

    # Configure for Coinbase
    config = SystemConfig()
    config.exchange = ExchangeConfig(
        name="coinbase",
        rate_limit_per_second=5,  # Conservative rate limit
        max_retries=3,
        timeout=30
    )

    # Optional: Customize agent settings
    # config.price_action.breakout_threshold = 0.04  # 4% breakout
    # config.momentum.rsi_overbought = 75.0
    # config.volume.min_liquidity_usd = 1000000

    # Create orchestrator
    orchestrator = AgentOrchestrator(config)

    # Initialize
    print("Initializing system...")
    if not await orchestrator.initialize():
        print("\n❌ Failed to initialize. Check your internet connection and logs.")
        print("\nTroubleshooting:")
        print("  1. Ensure you have internet access")
        print("  2. Check firewall/proxy settings")
        print("  3. Verify Coinbase API is accessible from your location")
        print("  4. Review logs in crypto_agents.log")
        return 1

    print("\n✅ System initialized successfully!")
    print("✅ Connected to Coinbase Advanced Trade API")
    print("✅ All 4 agents started and running\n")

    print("-"*80)
    print("MONITORING ACTIVE")
    print("-"*80)
    print("\nThe system is now:")
    print("  • Scanning price action for breakouts")
    print("  • Computing RSI and OBV indicators")
    print("  • Detecting volume spikes")
    print("  • Generating trading signals\n")

    print("Reports are saved to: output/latest_report.json")
    print("Updated every 5 minutes\n")

    print("Press Ctrl+C to stop\n")
    print("="*80 + "\n")

    # Run forever
    try:
        await orchestrator.run_forever()
    except KeyboardInterrupt:
        print("\n\nShutdown requested...")
    finally:
        await orchestrator.stop()
        print("✅ All agents stopped")
        print("✅ System shutdown complete")

    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        exit(exit_code)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
