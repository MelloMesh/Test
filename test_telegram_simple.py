"""
Simple Telegram bot test - sends one message
"""

import asyncio
from telegram import Bot
import json
from pathlib import Path


async def test_simple():
    """Send a single test message"""

    # Load credentials
    config_path = Path(".telegram_config.json")
    if not config_path.exists():
        print("ERROR: .telegram_config.json not found")
        return

    with open(config_path) as f:
        creds = json.load(f)

    token = creds.get("token")
    chat_id = creds.get("chat_id")

    if not token or not chat_id:
        print("ERROR: Missing token or chat_id in config")
        return

    # Create bot and send message
    bot = Bot(token=token)

    message = """
🎉 Telegram Bot Connected! 🎉

Your trading bot is now ready to send live signals.

✅ Connection successful
✅ Authentication verified
✅ Ready to receive trade notifications

Test completed at {time}
""".format(time=asyncio.get_event_loop().time())

    try:
        await bot.send_message(chat_id=chat_id, text=message)
        print("✅ SUCCESS! Check your Telegram app for the message.")
        print(f"   Bot: @Mello_trading_bot")
        print(f"   Chat ID: {chat_id}")
    except Exception as e:
        print(f"❌ FAILED: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you've started a chat with @Mello_trading_bot")
        print("2. Send /start to the bot first")
        print("3. Verify your token and chat ID are correct")


if __name__ == "__main__":
    asyncio.run(test_simple())
