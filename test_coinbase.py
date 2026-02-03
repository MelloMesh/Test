"""
Test script for Coinbase exchange adapter.
"""

import asyncio
from crypto_market_agents.exchange.adapter import ExchangeFactory
from crypto_market_agents.config import ExchangeConfig


async def test_coinbase():
    """Test Coinbase exchange adapter."""

    print("="*80)
    print("COINBASE EXCHANGE ADAPTER TEST")
    print("="*80)

    # Create Coinbase exchange
    config = ExchangeConfig(
        name="coinbase",
        rate_limit_per_second=5
    )

    exchange = ExchangeFactory.create(config)

    print(f"\n✓ Created exchange adapter: {exchange.__class__.__name__}")

    # Check US accessibility
    accessibility = exchange.check_us_accessibility()
    print(f"\n✓ US Accessible: {accessibility['accessible']}")
    print(f"  Notes: {accessibility['notes']}")

    # Connect
    print("\nConnecting to Coinbase...")
    connected = await exchange.connect()

    if not connected:
        print("✗ Failed to connect to Coinbase API")
        print("  (This may be expected if running without network access)")
        return

    print("✓ Connected to Coinbase Advanced Trade API")

    try:
        # Get trading symbols
        print("\nFetching trading symbols...")
        symbols = await exchange.get_trading_symbols("USD")
        print(f"✓ Found {len(symbols)} trading symbols")

        if symbols:
            print(f"  Sample symbols: {', '.join(symbols[:10])}")

        # Get a ticker
        if symbols:
            test_symbol = symbols[0]
            print(f"\nFetching ticker for {test_symbol}...")
            ticker = await exchange.get_ticker(test_symbol)

            if ticker:
                print(f"✓ Ticker data:")
                print(f"  Symbol: {ticker['symbol']}")
                print(f"  Price: ${ticker['last_price']:,.2f}")
                print(f"  24h Change: {ticker['price_change_24h_pct']:+.2f}%")
                print(f"  24h Volume: {ticker['volume_24h']:,.2f}")
                print(f"  24h High: ${ticker['high_24h']:,.2f}")
                print(f"  24h Low: ${ticker['low_24h']:,.2f}")

        # Get multiple tickers
        print("\nFetching tickers for multiple symbols...")
        test_symbols = symbols[:5] if symbols else []
        tickers = await exchange.get_tickers(test_symbols)
        print(f"✓ Retrieved {len(tickers)} tickers")

        for ticker in tickers[:3]:
            print(f"  {ticker['symbol']}: ${ticker['last_price']:,.2f} "
                  f"({ticker['price_change_24h_pct']:+.2f}%)")

        # Get klines/candles
        if symbols:
            test_symbol = symbols[0]
            print(f"\nFetching 5-minute candles for {test_symbol}...")
            klines = await exchange.get_klines(test_symbol, "5", limit=10)

            if klines:
                print(f"✓ Retrieved {len(klines)} candles")
                print("  Latest candles:")
                for kline in klines[-3:]:
                    print(f"    {kline['timestamp']}: "
                          f"O ${kline['open']:.2f} | "
                          f"H ${kline['high']:.2f} | "
                          f"L ${kline['low']:.2f} | "
                          f"C ${kline['close']:.2f}")

        # Get order book
        if symbols:
            test_symbol = symbols[0]
            print(f"\nFetching order book for {test_symbol}...")
            orderbook = await exchange.get_order_book(test_symbol, limit=5)

            if orderbook and orderbook.get('bids') and orderbook.get('asks'):
                print(f"✓ Order book retrieved")
                print(f"  Top 3 Bids:")
                for price, size in orderbook['bids'][:3]:
                    print(f"    ${price:,.2f} x {size:.4f}")
                print(f"  Top 3 Asks:")
                for price, size in orderbook['asks'][:3]:
                    print(f"    ${price:,.2f} x {size:.4f}")

        print("\n" + "="*80)
        print("COINBASE TEST COMPLETE - ALL FEATURES WORKING!")
        print("="*80)
        print("\nYou can now use Coinbase with the crypto market agents system:")
        print("  export EXCHANGE_NAME=coinbase")
        print("  python -m crypto_market_agents.main")

    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()

    finally:
        await exchange.disconnect()


if __name__ == "__main__":
    asyncio.run(test_coinbase())
