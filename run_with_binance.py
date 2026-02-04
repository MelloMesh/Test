"""
Run the Crypto Market Agents system with Binance Futures (globally accessible).
"""

import asyncio
import logging
import sys
from dotenv import load_dotenv
from crypto_market_agents.config import SystemConfig, ExchangeConfig
from crypto_market_agents.orchestrator import AgentOrchestrator


def setup_logging():
    """Configure root logger to catch all agent logs."""
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Remove existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler('crypto_agents.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Reduce noise from HTTP libraries
    logging.getLogger('aiohttp').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)


async def main():
    """Run the system with Binance Futures."""

    # Setup logging FIRST
    setup_logging()

    # Load environment variables from .env file
    load_dotenv()

    print("=" * 80)
    print("CRYPTO MARKET AGENTS - BINANCE FUTURES EDITION")
    print("=" * 80)
    print("\nUsing Binance Futures API (Globally Accessible)")
    print("No geo-restrictions - works from anywhere!\n")

    # Configure for Binance Futures
    config = SystemConfig()
    config.exchange = ExchangeConfig(
        name="binance",
        testnet=True,  # In our setup, testnet=True means use Futures
        rate_limit_per_second=20,  # Binance allows higher rate limits
        max_retries=3,
        timeout=30
    )

    # Create orchestrator
    orchestrator = AgentOrchestrator(config)

    # Initialize
    print("Initializing system...")
    if not await orchestrator.initialize():
        print("\n❌ Failed to initialize. Check logs for details.")
        return 1

    print("\n✅ System initialized successfully!")
    print("✅ Connected to Binance Futures API")
    print("✅ All 6 agents started and running\n")

    print("-" * 80)
    print("MONITORING ACTIVE")
    print("-" * 80)
    print("\nThe system is now:")
    print("  • Scanning price action for breakouts")
    print("  • Computing RSI and OBV indicators")
    print("  • Detecting volume spikes")
    print("  • Analyzing HTF support/resistance levels (1M, 1w, 3d, 1d, 4h, 1h)")
    print("  • Paper trading and learning from outcomes")
    print("  • Generating optimized trading signals\n")

    print("Reports are saved to: output/latest_report.json")
    print("Updated every 5 minutes\n")

    print("Press Ctrl+C to stop\n")
    print("=" * 80 + "\n")

    # Run
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
