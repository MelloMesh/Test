"""
Check Telegram bot credentials without sending messages
"""

import asyncio
from telegram import Bot
import json
from pathlib import Path


async def check_bot():
    """Verify bot credentials"""

    # Load credentials
    config_path = Path(".telegram_config.json")
    with open(config_path) as f:
        creds = json.load(f)

    token = creds.get("token")
    chat_id = creds.get("chat_id")

    print(f"Token: {token[:10]}...{token[-5:]}")
    print(f"Chat ID: {chat_id}")
    print("\nChecking bot credentials...")

    bot = Bot(token=token)

    try:
        # Get bot info (doesn't send messages)
        me = await bot.get_me()
        print(f"\n✅ Bot is VALID!")
        print(f"   Username: @{me.username}")
        print(f"   Name: {me.first_name}")
        print(f"   Bot ID: {me.id}")
        print(f"\n🎯 Configuration is correct!")
        print(f"\nNext steps:")
        print(f"1. Make sure you've sent /start to @{me.username} in Telegram")
        print(f"2. The bot will send you live trading signals")
        print(f"3. Run the real-time scanner to start receiving signals")
        return True
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nThis could be:")
        print("- Invalid bot token")
        print("- Network/firewall blocking Telegram API")
        print("- Temporary Telegram API issue")
        return False


if __name__ == "__main__":
    asyncio.run(check_bot())
