"""
Test script for Telegram bot integration.

Loads credentials from .env file and sends test messages.
IMPORTANT: Ensure .env file has TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID set.
"""

import asyncio
import os
from datetime import datetime, timezone
from dotenv import load_dotenv

from crypto_market_agents.schemas import TradingSignal
from crypto_market_agents.integrations.telegram_bot import TelegramBot


async def test_telegram_bot():
    """Test Telegram bot functionality."""
    # Load environment variables
    load_dotenv()

    print("=" * 80)
    print("TELEGRAM BOT INTEGRATION TEST")
    print("=" * 80)
    print()

    # Check credentials
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not bot_token:
        print("❌ ERROR: TELEGRAM_BOT_TOKEN not set in .env file")
        return

    if not chat_id:
        print("❌ ERROR: TELEGRAM_CHAT_ID not set in .env file")
        return

    print(f"✅ Bot Token: {bot_token[:20]}...")
    print(f"✅ Chat ID: {chat_id}")
    print()

    # Initialize bot
    bot = TelegramBot(bot_token=bot_token, chat_id=chat_id)

    try:
        # Test 1: Simple message
        print("Test 1: Sending simple message...")
        success = await bot.send_message("🤖 *Test Message from Crypto Market Agents*\n\nSystem initialized successfully!")
        print(f"  Result: {'✅ Success' if success else '❌ Failed'}")
        await asyncio.sleep(2)

        # Test 2: Trading signal
        print("\nTest 2: Sending trading signal...")
        signal = TradingSignal(
            asset="ETHUSDT",
            direction="LONG",
            entry=2450.00,
            stop=2400.00,
            target=2680.00,
            confidence=0.72,
            rationale="Bullish breakout (+3.2%) | Oversold RSI 28.4 on 60 | HTF support @ $2,445.00 (3 levels) | 🎯 Golden Pocket ($2,438-$2,453) in bullish swing | Learning: normal (62% win rate)",
            timestamp=datetime.now(timezone.utc),
            order_type="LIMIT",
            confluence_score=6
        )
        success = await bot.send_signal(signal)
        print(f"  Result: {'✅ Success' if success else '❌ Failed'}")
        await asyncio.sleep(2)

        # Test 3: Alert
        print("\nTest 3: Sending alert...")
        success = await bot.send_alert("Test Alert", "This is a test alert from the system", critical=False)
        print(f"  Result: {'✅ Success' if success else '❌ Failed'}")
        await asyncio.sleep(2)

        # Test 4: Trade update
        print("\nTest 4: Sending trade update...")
        success = await bot.send_trade_update("BTCUSDT", "OPENED", 50000.00, pnl_percent=None)
        print(f"  Result: {'✅ Success' if success else '❌ Failed'}")
        await asyncio.sleep(2)

        # Test 5: Error notification
        print("\nTest 5: Sending error notification...")
        success = await bot.send_error("Sample error message", context="Testing error handler")
        print(f"  Result: {'✅ Success' if success else '❌ Failed'}")

        print()
        print("=" * 80)
        print("ALL TESTS COMPLETE")
        print("=" * 80)
        print()
        print("✅ Check your Telegram to see the messages!")

    finally:
        # Cleanup
        await bot.close()


if __name__ == "__main__":
    asyncio.run(test_telegram_bot())
