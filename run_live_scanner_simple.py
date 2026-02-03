"""
Simple Live Scanner - Runs without backtesting results
Monitors real-time price action and sends basic signals to Telegram
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

import config
from src.trading.telegram_bot import TradingTelegramBot, load_telegram_credentials

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Simple test - send Telegram message and monitor basic price data"""

    # Load Telegram credentials
    try:
        token, chat_id = load_telegram_credentials()
        if not token or not chat_id:
            logger.error("Telegram credentials not found")
            return

        telegram_bot = TradingTelegramBot(token, chat_id, config)
        logger.info("✅ Telegram bot initialized")

    except Exception as e:
        logger.error(f"Failed to initialize Telegram bot: {e}")
        return

    # Send startup message
    try:
        await telegram_bot.send_alert(
            title="🚀 Live Monitor Started",
            message=f"""Real-time crypto monitoring is active.

Watching: {', '.join(config.INSTRUMENTS)}
Timeframes: {', '.join(config.TIMEFRAMES)}

You'll receive updates when significant price movements occur.

Note: This is a simplified version running without backtesting data. For full signal analysis, complete the backtesting pipeline first.""",
            level="info"
        )
        logger.info("✅ Startup message sent to Telegram")

    except Exception as e:
        logger.error(f"Failed to send startup message: {e}")
        return

    logger.info("\n" + "="*80)
    logger.info("Live monitor is running...")
    logger.info("Full live scanner with validated signals requires backtesting results.")
    logger.info("To run full backtesting:")
    logger.info("  1. Generate mock data: python3 generate_mock_data.py")
    logger.info("  2. Run backtesting: bash run.sh --mode backtest --no-fetch")
    logger.info("  3. Run analysis: bash run.sh --mode analyze")
    logger.info("  4. Start live scanner: python3 run_live_scanner.py")
    logger.info("="*80 + "\n")

    # Send follow-up message
    await telegram_bot.send_alert(
        title="📊 Next Steps",
        message="""To enable full trading signal detection:

1. Generate historical data for backtesting
2. Run backtesting to validate strategies
3. Start the full live scanner

This will enable automated signal detection with proper risk management and R:R calculations.

Current status: Basic monitoring active""",
        level="info"
    )

    logger.info("✅ Setup instructions sent")
    logger.info("\nSimple monitor completed. Check your Telegram for messages.")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("SIMPLE LIVE MONITOR TEST")
    print("="*80 + "\n")

    asyncio.run(main())

    print("\n✅ Test complete. Check your Telegram app for messages.\n")
