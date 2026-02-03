"""
Example usage of the Crypto Market Agents system.

This script demonstrates different ways to use the agents.
"""

import asyncio
from crypto_market_agents.config import SystemConfig, ExchangeConfig, SignalSynthesisConfig
from crypto_market_agents.orchestrator import AgentOrchestrator


async def example_basic():
    """Basic example: Run all agents with default configuration."""
    print("=" * 80)
    print("Example 1: Basic Usage")
    print("=" * 80)

    # Load configuration from environment
    config = SystemConfig.from_env()

    # Create orchestrator
    orchestrator = AgentOrchestrator(config)

    # Initialize
    if not await orchestrator.initialize():
        print("Failed to initialize")
        return

    # Start agents
    await orchestrator.start()

    # Run for 5 minutes
    print("Running agents for 5 minutes...")
    await asyncio.sleep(300)

    # Get latest signals
    signals = orchestrator.get_latest_signals()
    print(f"\nGenerated {len(signals)} trading signals:")
    for i, signal in enumerate(signals[:5], 1):
        print(f"\n{i}. {signal.asset} - {signal.direction}")
        print(f"   Entry: {signal.entry:.8f}")
        print(f"   Stop: {signal.stop:.8f}")
        print(f"   Target: {signal.target:.8f}")
        print(f"   Confidence: {signal.confidence:.2%}")
        print(f"   Rationale: {signal.rationale}")

    # Stop
    await orchestrator.stop()


async def example_custom_config():
    """Example with custom configuration."""
    print("\n" + "=" * 80)
    print("Example 2: Custom Configuration")
    print("=" * 80)

    # Create custom configuration
    config = SystemConfig()

    # Customize exchange settings
    config.exchange = ExchangeConfig(
        name="bybit",
        testnet=False,
        rate_limit_per_second=5  # Conservative rate limit
    )

    # Customize signal synthesis
    config.signal_synthesis = SignalSynthesisConfig(
        reward_risk_ratio=3.0,   # 3:1 reward-to-risk
        max_stop_loss_pct=1.5,   # Max 1.5% stop loss
        min_confidence=0.75,     # Minimum 75% confidence
        update_interval=180      # Report every 3 minutes
    )

    # Create and run
    orchestrator = AgentOrchestrator(config)

    if await orchestrator.initialize():
        await orchestrator.start()
        print("Running with custom configuration...")
        await asyncio.sleep(180)
        await orchestrator.stop()


async def example_individual_agent():
    """Example: Run a single agent independently."""
    print("\n" + "=" * 80)
    print("Example 3: Individual Agent")
    print("=" * 80)

    from crypto_market_agents.exchange.adapter import ExchangeFactory
    from crypto_market_agents.agents.price_action import PriceActionAgent
    from crypto_market_agents.config import PriceActionConfig

    # Create exchange
    exchange_config = ExchangeConfig()
    exchange = ExchangeFactory.create(exchange_config)

    if not await exchange.connect():
        print("Failed to connect to exchange")
        return

    # Create price action agent
    price_config = PriceActionConfig(
        breakout_threshold=0.05,  # 5% breakout threshold
        update_interval=30
    )

    agent = PriceActionAgent(exchange, price_config)

    # Start agent
    await agent.start()

    # Let it run
    print("Running Price Action Agent for 2 minutes...")
    await asyncio.sleep(120)

    # Get signals
    signals = agent.get_latest_signals()
    breakouts = [s for s in signals if s.breakout_detected]

    print(f"\nDetected {len(breakouts)} breakouts:")
    for signal in breakouts[:10]:
        print(f"  {signal.symbol}: {signal.price_change_pct:+.2f}% "
              f"(volatility: {signal.volatility_ratio:.2f}x)")

    # Stop
    await agent.stop()
    await exchange.disconnect()


async def example_signal_monitoring():
    """Example: Continuous signal monitoring."""
    print("\n" + "=" * 80)
    print("Example 4: Continuous Monitoring")
    print("=" * 80)

    config = SystemConfig.from_env()
    orchestrator = AgentOrchestrator(config)

    if not await orchestrator.initialize():
        print("Failed to initialize")
        return

    await orchestrator.start()

    print("Monitoring signals... Press Ctrl+C to stop\n")

    try:
        # Monitor for new high-confidence signals
        last_count = 0

        while True:
            await asyncio.sleep(10)

            signals = orchestrator.get_latest_signals()
            high_conf = [s for s in signals if s.confidence >= 0.8]

            if len(high_conf) > last_count:
                print(f"\n[{asyncio.get_event_loop().time():.0f}] "
                      f"New high-confidence signals: {len(high_conf)}")

                for signal in high_conf[last_count:]:
                    print(f"  🎯 {signal.asset} {signal.direction} @ {signal.entry:.8f}")
                    print(f"     Confidence: {signal.confidence:.2%}")
                    print(f"     {signal.rationale}")

                last_count = len(high_conf)

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        await orchestrator.stop()


async def main():
    """Run all examples."""
    print("\nCrypto Market Agents - Example Usage\n")
    print("This will demonstrate the system with different configurations.\n")

    # Run examples
    try:
        # Example 1: Basic usage
        await example_basic()

        # Example 2: Custom configuration
        # await example_custom_config()

        # Example 3: Individual agent
        # await example_individual_agent()

        # Example 4: Continuous monitoring
        # await example_signal_monitoring()

    except KeyboardInterrupt:
        print("\n\nExamples interrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("NOTE: This example uses Bybit, which restricts US users.")
    print("For production use in the US, modify the exchange configuration")
    print("to use a US-compliant exchange like Coinbase, Kraken, or Gemini.")
    print("=" * 80 + "\n")

    asyncio.run(main())
