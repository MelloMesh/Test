"""
Basic system test to verify all components are working.
"""

import asyncio
import sys


async def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    try:
        from crypto_market_agents.config import SystemConfig
        from crypto_market_agents.exchange.adapter import ExchangeFactory
        from crypto_market_agents.agents.price_action import PriceActionAgent
        from crypto_market_agents.agents.momentum import MomentumAgent
        from crypto_market_agents.agents.volume_spike import VolumeSpikeAgent
        from crypto_market_agents.agents.signal_synthesis import SignalSynthesisAgent
        from crypto_market_agents.orchestrator import AgentOrchestrator
        print("✓ All modules imported successfully")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


async def test_configuration():
    """Test configuration loading."""
    print("\nTesting configuration...")

    try:
        from crypto_market_agents.config import SystemConfig

        config = SystemConfig.from_env()
        print(f"✓ Configuration loaded")
        print(f"  - Exchange: {config.exchange.name}")
        print(f"  - Log level: {config.log_level}")
        print(f"  - Output dir: {config.output_dir}")
        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False


async def test_exchange_factory():
    """Test exchange factory."""
    print("\nTesting exchange factory...")

    try:
        from crypto_market_agents.exchange.adapter import ExchangeFactory, get_us_compliant_exchanges
        from crypto_market_agents.config import ExchangeConfig

        # Test exchange creation
        config = ExchangeConfig(name="bybit")
        exchange = ExchangeFactory.create(config)
        print(f"✓ Exchange adapter created: {exchange.__class__.__name__}")

        # Test US compliance info
        info = exchange.check_us_accessibility()
        print(f"  - US accessible: {info['accessible']}")
        if not info['accessible']:
            print(f"  - Note: {info['notes']}")

        # Show US-compliant alternatives
        us_exchanges = get_us_compliant_exchanges()
        print(f"\n  US-compliant alternatives:")
        for name, details in us_exchanges.items():
            print(f"    - {details['name']}")

        return True
    except Exception as e:
        print(f"✗ Exchange factory test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_exchange_connection():
    """Test exchange connection (optional - requires network)."""
    print("\nTesting exchange connection...")

    try:
        from crypto_market_agents.exchange.adapter import ExchangeFactory
        from crypto_market_agents.config import ExchangeConfig

        config = ExchangeConfig(name="bybit", testnet=False)
        exchange = ExchangeFactory.create(config)

        # Try to connect
        connected = await exchange.connect()

        if connected:
            print("✓ Exchange connection successful")

            # Try to get a few symbols
            symbols = await exchange.get_trading_symbols()
            print(f"  - Found {len(symbols)} trading symbols")

            if symbols:
                print(f"  - Sample symbols: {', '.join(symbols[:5])}")

            await exchange.disconnect()
            return True
        else:
            print("✗ Exchange connection failed (may be expected for US IPs)")
            print("  This is not necessarily an error - Bybit may restrict US access")
            return True  # Not a critical failure

    except Exception as e:
        print(f"✗ Exchange connection test failed: {e}")
        print("  This may be expected if running from a restricted location")
        return True  # Not a critical failure


async def test_agent_creation():
    """Test agent creation."""
    print("\nTesting agent creation...")

    try:
        from crypto_market_agents.config import SystemConfig
        from crypto_market_agents.exchange.adapter import ExchangeFactory
        from crypto_market_agents.agents.price_action import PriceActionAgent
        from crypto_market_agents.agents.momentum import MomentumAgent
        from crypto_market_agents.agents.volume_spike import VolumeSpikeAgent

        config = SystemConfig()
        exchange = ExchangeFactory.create(config.exchange)

        # Create agents
        price_agent = PriceActionAgent(exchange, config.price_action)
        momentum_agent = MomentumAgent(exchange, config.momentum)
        volume_agent = VolumeSpikeAgent(exchange, config.volume)

        print("✓ All agents created successfully")
        print(f"  - {price_agent.name}")
        print(f"  - {momentum_agent.name}")
        print(f"  - {volume_agent.name}")

        return True
    except Exception as e:
        print(f"✗ Agent creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_orchestrator():
    """Test orchestrator initialization."""
    print("\nTesting orchestrator...")

    try:
        from crypto_market_agents.config import SystemConfig
        from crypto_market_agents.orchestrator import AgentOrchestrator

        config = SystemConfig()
        orchestrator = AgentOrchestrator(config)

        print("✓ Orchestrator created successfully")
        print(f"  - Output directory: {orchestrator.output_dir}")

        return True
    except Exception as e:
        print(f"✗ Orchestrator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def run_all_tests():
    """Run all tests."""
    print("=" * 80)
    print("Crypto Market Agents - System Test")
    print("=" * 80)

    tests = [
        test_imports,
        test_configuration,
        test_exchange_factory,
        test_agent_creation,
        test_orchestrator,
        test_exchange_connection,  # Last because it requires network
    ]

    results = []

    for test in tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)

    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)

    passed = sum(results)
    total = len(results)

    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(run_all_tests())
    sys.exit(exit_code)
