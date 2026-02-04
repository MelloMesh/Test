"""
Telegram Bot for Trading Signal Notifications
Send real-time alerts and manage positions via Telegram
"""

import asyncio
from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, ContextTypes
from datetime import datetime
from typing import List, Dict, Optional
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class TradingTelegramBot:
    """
    Telegram bot for sending trading signals and managing positions
    """

    def __init__(self, token: str, chat_id: str, config):
        """
        Initialize Telegram bot

        Args:
            token: Telegram bot token from @BotFather
            chat_id: Your Telegram chat ID (get from @userinfobot)
            config: Trading system config
        """
        self.token = token
        self.chat_id = chat_id
        self.config = config
        self.bot = Bot(token=token)
        self.app = None

        # Track sent signals
        self.sent_signals: List[Dict] = []
        self.active_positions: List[Dict] = []

    async def send_signal_alert(self, signal):
        """
        Send trading signal notification

        Args:
            signal: LiveSignal object
        """
        # Format message
        direction_emoji = "🟢" if signal.direction == "long" else "🔴"
        confidence_stars = "⭐" * int(signal.confidence * 5)

        message = f"""
{direction_emoji} **NEW TRADING SIGNAL** {direction_emoji}

📊 **Signal:** {signal.signal_name}
💰 **Instrument:** {signal.instrument} ({signal.timeframe})
📈 **Direction:** {signal.direction.upper()}

💵 **ENTRY:**
   Price: ${signal.entry_price:,.2f}

🛡️ **RISK MANAGEMENT:**
   Stop Loss: ${signal.stop_loss:,.2f} ({((signal.stop_loss - signal.entry_price)/signal.entry_price * 100):.2f}%)
   Take Profit: ${signal.take_profit:,.2f} ({((signal.take_profit - signal.entry_price)/signal.entry_price * 100):.2f}%)
   R:R Ratio: 1:{signal.risk_reward_ratio:.1f}

💰 **POSITION:**
   Size: ${signal.position_size_usd:,.2f}
   Units: {signal.position_size_units:.4f}
   Risk: ${signal.risk_amount_usd:,.2f}
   Potential: ${signal.reward_amount_usd:,.2f}

📊 **QUALITY:**
   Confidence: {signal.confidence:.0%} {confidence_stars}
   Backtest WR: {signal.backtest_win_rate:.1%}
   Backtest PF: {signal.backtest_profit_factor:.2f}x

🌍 **CONTEXT:**
   Trend: {signal.macro_trend.capitalize()}
   Regime: {signal.regime.capitalize()}
   Higher TF: {"✅" if signal.higher_tf_aligned else "❌"}

⏰ **Time:** {signal.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")}

_{signal.entry_reason}_
"""

        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode='Markdown'
            )

            # Track sent signal
            self.sent_signals.append({
                'signal_id': signal.signal_id,
                'sent_time': datetime.utcnow(),
                'signal_data': signal
            })

            logger.info(f"✓ Sent signal alert for {signal.signal_name} to Telegram")

        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")

    async def send_position_update(self, position_id: str, status: str, pnl: float, exit_reason: str = ""):
        """
        Send position update notification

        Args:
            position_id: Position identifier
            status: Position status (filled, stopped, target_hit, etc.)
            pnl: Profit/loss in USD
            exit_reason: Why position was closed
        """
        pnl_emoji = "✅" if pnl > 0 else "❌"
        pnl_pct = (pnl / self.config.STARTING_CAPITAL_USDT) * 100

        message = f"""
{pnl_emoji} **POSITION UPDATE** {pnl_emoji}

🆔 Position: {position_id}
📊 Status: {status.upper()}

💰 **P&L:**
   USD: ${pnl:+,.2f}
   Return: {pnl_pct:+.2f}%

📝 **Reason:** {exit_reason or "Manual close"}

⏰ {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")}
"""

        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode='Markdown'
            )
        except Exception as e:
            logger.error(f"Failed to send position update: {e}")

    async def send_daily_summary(self, stats: Dict):
        """
        Send daily performance summary

        Args:
            stats: Dict with daily statistics
        """
        total_pnl = stats.get('total_pnl', 0)
        total_trades = stats.get('total_trades', 0)
        wins = stats.get('winning_trades', 0)
        losses = stats.get('losing_trades', 0)
        win_rate = wins / total_trades if total_trades > 0 else 0

        pnl_emoji = "✅" if total_pnl > 0 else "❌"

        message = f"""
📊 **DAILY SUMMARY** 📊

📅 Date: {datetime.utcnow().strftime("%Y-%m-%d")}

{pnl_emoji} **PERFORMANCE:**
   Total P&L: ${total_pnl:+,.2f}
   Return: {(total_pnl / self.config.STARTING_CAPITAL_USDT * 100):+.2f}%

📈 **TRADES:**
   Total: {total_trades}
   Wins: {wins} ✅
   Losses: {losses} ❌
   Win Rate: {win_rate:.1%}

💰 **CAPITAL:**
   Starting: ${self.config.STARTING_CAPITAL_USDT:,.2f}
   Current: ${self.config.STARTING_CAPITAL_USDT + total_pnl:,.2f}

Keep up the disciplined trading! 🚀
"""

        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode='Markdown'
            )
        except Exception as e:
            logger.error(f"Failed to send daily summary: {e}")

    async def send_alert(self, title: str, message: str, level: str = "info"):
        """
        Send custom alert

        Args:
            title: Alert title
            message: Alert message
            level: Alert level (info, warning, error)
        """
        emoji_map = {
            'info': 'ℹ️',
            'warning': '⚠️',
            'error': '🚨'
        }

        emoji = emoji_map.get(level, 'ℹ️')

        formatted_message = f"""
{emoji} **{title.upper()}** {emoji}

{message}

⏰ {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")}
"""

        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=formatted_message,
                parse_mode='Markdown'
            )
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")

    # Command handlers for interactive bot
    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        message = """
🤖 **Trading Bot Active** 🤖

Available commands:
/status - Current positions & stats
/signals - Recent signals
/close [id] - Close position
/stats - Trading statistics
/help - Show this message

Ready to send you trading signals! 📊
"""
        await update.message.reply_text(message, parse_mode='Markdown')

    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command - show active positions"""
        if not self.active_positions:
            await update.message.reply_text("No active positions.")
            return

        message = "📊 **ACTIVE POSITIONS:**\n\n"

        for pos in self.active_positions:
            message += f"• {pos['instrument']} {pos['direction'].upper()}\n"
            message += f"  Entry: ${pos['entry_price']:.2f}\n"
            message += f"  Current P&L: ${pos.get('current_pnl', 0):+.2f}\n\n"

        await update.message.reply_text(message, parse_mode='Markdown')

    async def cmd_signals(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /signals command - show recent signals"""
        if not self.sent_signals:
            await update.message.reply_text("No signals sent yet.")
            return

        recent = self.sent_signals[-5:]  # Last 5 signals

        message = "📊 **RECENT SIGNALS:**\n\n"

        for sig in recent:
            sig_data = sig['signal_data']
            message += f"• {sig_data.signal_name}\n"
            message += f"  {sig_data.instrument} {sig_data.direction.upper()}\n"
            message += f"  Time: {sig['sent_time'].strftime('%H:%M:%S')}\n\n"

        await update.message.reply_text(message, parse_mode='Markdown')

    async def cmd_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stats command - show trading statistics"""
        # Would load from performance tracker
        message = """
📊 **TRADING STATISTICS**

Total Signals: {total}
Active Positions: {active}
Win Rate: {wr}%
Total P&L: ${pnl:,.2f}

Keep trading systematically! 🚀
""".format(
            total=len(self.sent_signals),
            active=len(self.active_positions),
            wr="N/A",
            pnl=0.0
        )

        await update.message.reply_text(message, parse_mode='Markdown')

    def setup_handlers(self):
        """Setup command handlers"""
        if not self.app:
            self.app = Application.builder().token(self.token).build()

        self.app.add_handler(CommandHandler("start", self.cmd_start))
        self.app.add_handler(CommandHandler("status", self.cmd_status))
        self.app.add_handler(CommandHandler("signals", self.cmd_signals))
        self.app.add_handler(CommandHandler("stats", self.cmd_stats))

        logger.info("✓ Telegram bot handlers configured")

    async def run_bot(self):
        """Run the bot (for interactive commands)"""
        self.setup_handlers()
        await self.app.run_polling()


def load_telegram_credentials() -> tuple:
    """
    Load Telegram credentials from config file or environment

    Returns:
        (token, chat_id) tuple
    """
    import os

    token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')

    if not token or not chat_id:
        config_file = Path('.telegram_config.json')
        if config_file.exists():
            with open(config_file, 'r') as f:
                creds = json.load(f)
                token = creds.get('token')
                chat_id = creds.get('chat_id')

    if not token or not chat_id:
        print("\n⚠️  Telegram credentials not found!")
        print("\nTo use Telegram notifications:")
        print("1. Talk to @BotFather on Telegram")
        print("2. Create a new bot with /newbot")
        print("3. Get your bot token")
        print("4. Talk to @userinfobot to get your chat ID")
        print("5. Set environment variables:")
        print("   export TELEGRAM_BOT_TOKEN='your-token'")
        print("   export TELEGRAM_CHAT_ID='your-chat-id'")
        print("\nOr create .telegram_config.json:")
        print('{"token": "your-token", "chat_id": "your-chat-id"}')
        return None, None

    return token, chat_id


async def test_telegram_bot():
    """Test Telegram bot functionality"""
    import sys
    sys.path.append("../..")
    import config

    token, chat_id = load_telegram_credentials()

    if not token or not chat_id:
        return

    bot = TradingTelegramBot(token, chat_id, config)

    print("Testing Telegram bot...")

    # Test alert
    await bot.send_alert(
        title="Test Alert",
        message="This is a test message from your trading bot!",
        level="info"
    )

    print("✓ Test message sent! Check your Telegram.")

    # Create mock signal for testing
    from dataclasses import dataclass
    from datetime import datetime

    @dataclass
    class MockSignal:
        signal_id = "TEST_001"
        signal_name = "Test_Signal_5m"
        instrument = "BTCUSDT.P"
        timeframe = "5m"
        direction = "long"
        timestamp = datetime.utcnow()
        entry_price = 45000.0
        stop_loss = 44500.0
        take_profit = 46000.0
        risk_reward_ratio = 2.0
        position_size_usd = 1000.0
        position_size_units = 0.0222
        risk_amount_usd = 100.0
        reward_amount_usd = 200.0
        confidence = 0.65
        backtest_win_rate = 0.62
        backtest_profit_factor = 1.5
        higher_tf_aligned = True
        macro_trend = "bullish"
        regime = "trending"
        entry_reason = "RSI oversold with trend confirmation"
        status = "active"

    signal = MockSignal()

    # Test signal alert
    await bot.send_signal_alert(signal)

    print("✓ Signal alert sent! Check your Telegram.")


if __name__ == "__main__":
    asyncio.run(test_telegram_bot())
