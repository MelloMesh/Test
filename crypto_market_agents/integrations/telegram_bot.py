"""
Telegram Bot Integration - Send trading signals and system updates via Telegram.

Security: Bot token and chat IDs are loaded from environment variables.
"""

import os
import asyncio
import logging
from typing import List, Optional
from datetime import datetime
import aiohttp

from ..schemas import TradingSignal, SystemReport
from ..utils.signal_formatter import format_signal_telegram


# Telegram API limits
MAX_MESSAGE_LENGTH = 4096  # Telegram's maximum message length
RATE_LIMIT_RETRY_DELAY = 5  # Default retry delay for rate limits (seconds)


class TelegramBot:
    """
    Telegram bot for sending trading signals and system notifications.

    Environment Variables Required:
        TELEGRAM_BOT_TOKEN: Bot token from @BotFather
        TELEGRAM_CHAT_ID: Chat ID to send messages to (can be user ID or group ID)
    """

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
        enabled: bool = True
    ):
        """
        Initialize Telegram bot.

        Args:
            bot_token: Telegram bot token (defaults to env var)
            chat_id: Chat ID to send messages (defaults to env var)
            enabled: Whether bot is enabled
        """
        self.logger = logging.getLogger(self.__class__.__name__)

        # Load from environment variables if not provided
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")

        self.enabled = enabled and self.bot_token and self.chat_id

        if not self.enabled:
            if not self.bot_token:
                self.logger.warning("Telegram bot disabled: TELEGRAM_BOT_TOKEN not set")
            if not self.chat_id:
                self.logger.warning("Telegram bot disabled: TELEGRAM_CHAT_ID not set")
        else:
            self.logger.info(f"Telegram bot initialized for chat ID: {self.chat_id}")

        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
        self.session: Optional[aiohttp.ClientSession] = None

    async def _ensure_session(self):
        """Ensure aiohttp session is created."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()

    async def send_message(
        self,
        text: str,
        parse_mode: str = "Markdown",
        disable_notification: bool = False,
        _retry_count: int = 0
    ) -> bool:
        """
        Send a text message to Telegram.

        Args:
            text: Message text (will be truncated if > 4096 chars)
            parse_mode: Parse mode (Markdown, HTML, or None)
            disable_notification: Send silently
            _retry_count: Internal retry counter for rate limiting

        Returns:
            True if message sent successfully
        """
        if not self.enabled:
            return False

        # Truncate message if too long
        if len(text) > MAX_MESSAGE_LENGTH:
            truncate_at = MAX_MESSAGE_LENGTH - 30
            text = text[:truncate_at] + "\n\n...(message truncated)"
            self.logger.warning(f"Message truncated from {len(text)} to {MAX_MESSAGE_LENGTH} chars")

        try:
            await self._ensure_session()

            url = f"{self.base_url}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": parse_mode,
                "disable_notification": disable_notification
            }

            async with self.session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    self.logger.debug(f"Telegram message sent successfully")
                    return True
                elif response.status == 429:
                    # Rate limited - retry after delay
                    retry_after = int(response.headers.get('Retry-After', RATE_LIMIT_RETRY_DELAY))
                    if _retry_count < 3:  # Max 3 retries
                        self.logger.warning(f"Rate limited, retrying after {retry_after}s (attempt {_retry_count + 1}/3)")
                        await asyncio.sleep(retry_after)
                        return await self.send_message(text, parse_mode, disable_notification, _retry_count + 1)
                    else:
                        self.logger.error("Rate limit retry limit exceeded")
                        return False
                else:
                    error_text = await response.text()
                    self.logger.error(f"Telegram API error {response.status}: {error_text}")
                    return False

        except asyncio.TimeoutError:
            self.logger.error("Telegram message timeout")
            return False
        except Exception as e:
            self.logger.error(f"Error sending Telegram message: {e}")
            return False

    async def send_signal(self, signal: TradingSignal, silent: bool = False) -> bool:
        """
        Send a trading signal to Telegram.

        Args:
            signal: Trading signal to send
            silent: Send without notification

        Returns:
            True if sent successfully
        """
        if not self.enabled:
            return False

        try:
            # Format signal for Telegram
            message = format_signal_telegram(signal)

            # Send message
            return await self.send_message(message, parse_mode="Markdown", disable_notification=silent)

        except Exception as e:
            self.logger.error(f"Error sending signal to Telegram: {e}")
            return False

    async def send_signals_batch(self, signals: List[TradingSignal], max_signals: int = 5) -> int:
        """
        Send multiple signals to Telegram (limited to avoid spam).

        Args:
            signals: List of trading signals
            max_signals: Maximum number of signals to send

        Returns:
            Number of signals successfully sent
        """
        if not self.enabled or not signals:
            return 0

        sent_count = 0
        for signal in signals[:max_signals]:
            if await self.send_signal(signal, silent=(sent_count > 0)):
                sent_count += 1
                # Small delay to avoid rate limiting
                await asyncio.sleep(0.5)

        if len(signals) > max_signals:
            self.logger.info(f"Sent top {max_signals} signals, {len(signals) - max_signals} omitted")

        return sent_count

    async def send_system_update(self, report: SystemReport) -> bool:
        """
        Send a system status update.

        Args:
            report: System report

        Returns:
            True if sent successfully
        """
        if not self.enabled:
            return False

        try:
            # Format system report
            message = f"""
🤖 *System Status Update*

📊 *Agents:* {len(report.agent_statuses)} active
🎯 *Signals:* {report.total_signals_generated} generated
📈 *Top Signals:* {len(report.top_signals)}

⏰ *Updated:* {report.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}

_System running normally_
"""

            return await self.send_message(message, parse_mode="Markdown", disable_notification=True)

        except Exception as e:
            self.logger.error(f"Error sending system update: {e}")
            return False

    async def send_alert(self, title: str, message: str, critical: bool = False) -> bool:
        """
        Send an alert/notification.

        Args:
            title: Alert title
            message: Alert message
            critical: Whether this is a critical alert

        Returns:
            True if sent successfully
        """
        if not self.enabled:
            return False

        emoji = "🚨" if critical else "⚠️"
        formatted_message = f"{emoji} *{title}*\n\n{message}"

        return await self.send_message(
            formatted_message,
            parse_mode="Markdown",
            disable_notification=not critical
        )

    async def send_error(self, error_message: str, context: str = "") -> bool:
        """
        Send an error notification.

        Args:
            error_message: Error message
            context: Additional context

        Returns:
            True if sent successfully
        """
        if not self.enabled:
            return False

        message = f"❌ *Error Occurred*\n\n`{error_message}`"
        if context:
            message += f"\n\n*Context:* {context}"

        return await self.send_message(message, parse_mode="Markdown")

    async def send_trade_update(
        self,
        symbol: str,
        action: str,
        price: float,
        pnl_percent: Optional[float] = None
    ) -> bool:
        """
        Send a trade execution update.

        Args:
            symbol: Trading symbol
            action: Action taken (e.g., "OPENED", "CLOSED")
            price: Execution price
            pnl_percent: P&L percentage (for closed trades)

        Returns:
            True if sent successfully
        """
        if not self.enabled:
            return False

        emoji_map = {
            "OPENED": "🟢",
            "CLOSED": "⚪",
            "STOPPED": "🔴",
            "TARGET": "🎯"
        }

        emoji = emoji_map.get(action, "📊")
        message = f"{emoji} *{action}* {symbol} @ ${price:,.4f}"

        if pnl_percent is not None:
            pnl_emoji = "📈" if pnl_percent > 0 else "📉"
            message += f"\n{pnl_emoji} P&L: {pnl_percent:+.2f}%"

        return await self.send_message(message, parse_mode="Markdown", disable_notification=True)

    async def close(self):
        """Close the aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.logger.debug("Telegram bot session closed")

    def __del__(self):
        """Cleanup on deletion."""
        if self.session and not self.session.closed:
            # Create event loop if needed for cleanup
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.close())
                else:
                    loop.run_until_complete(self.close())
            except Exception:
                pass  # Best effort cleanup


# Convenience function for sending signals
async def send_signal_to_telegram(signal: TradingSignal, bot_token: str, chat_id: str) -> bool:
    """
    Convenience function to send a single signal.

    Args:
        signal: Trading signal
        bot_token: Telegram bot token
        chat_id: Chat ID

    Returns:
        True if sent successfully
    """
    bot = TelegramBot(bot_token=bot_token, chat_id=chat_id)
    try:
        return await bot.send_signal(signal)
    finally:
        await bot.close()
