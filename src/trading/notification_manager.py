"""
Advanced Notification Manager
Handles daily summaries, weekly reports, alert grouping, and risk management
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class DailyStats:
    """Daily trading statistics"""
    date: str
    trades_opened: int = 0
    trades_closed: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    best_trade: Optional[Dict] = None
    worst_trade: Optional[Dict] = None
    market_orders: int = 0
    limit_orders: int = 0
    limit_fills: int = 0
    limit_overrides: int = 0
    limit_cancellations: int = 0


@dataclass
class WeeklyStats:
    """Weekly trading statistics"""
    week_start: str
    week_end: str
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    pair_pnl: Dict[str, float] = field(default_factory=dict)
    trade_type_pnl: Dict[str, float] = field(default_factory=dict)
    market_fills: int = 0
    limit_fills: int = 0
    avg_rr_achieved: float = 0.0
    rr_list: List[float] = field(default_factory=list)


@dataclass
class BufferedSignal:
    """Signal waiting to be sent"""
    symbol: str
    message: str
    title: str
    timestamp: datetime


class NotificationManager:
    """
    Manages advanced notifications with:
    - Daily summaries
    - Weekly reports
    - Alert grouping
    - Price proximity alerts for pending limits
    - Risk management alerts
    """

    def __init__(self, telegram_bot, paper_trading_engine):
        self.telegram_bot = telegram_bot
        self.engine = paper_trading_engine

        # Daily/Weekly tracking
        self.daily_stats: Dict[str, DailyStats] = {}
        self.weekly_stats: Optional[WeeklyStats] = None
        self.last_daily_summary = None
        self.last_weekly_summary = None

        # Alert grouping
        self.signal_buffer: List[BufferedSignal] = []
        self.buffer_timeout = 300  # 5 minutes
        self.last_signal_time = None

        # Price alerts for limits
        self.limit_alert_sent: Dict[str, bool] = {}  # Track which limits we've alerted on
        self.limit_alert_threshold = 0.01  # 1% proximity

        # Risk management tracking
        self.consecutive_losses = 0
        self.daily_starting_capital = self.engine.current_capital
        self.win_rate_alert_sent = False

        # Notification times (UTC)
        self.daily_summary_hour = 20  # 8 PM UTC
        self.weekly_summary_day = 6  # Sunday

        # Initialize today's stats
        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        self.daily_stats[today] = DailyStats(date=today)

    async def start(self):
        """Start the notification manager background tasks"""
        asyncio.create_task(self._daily_summary_task())
        asyncio.create_task(self._weekly_summary_task())
        asyncio.create_task(self._signal_buffer_task())
        asyncio.create_task(self._limit_proximity_task())
        asyncio.create_task(self._risk_monitoring_task())
        logger.info("✅ Notification manager started")

    # ============================================================================
    # DAILY SUMMARY
    # ============================================================================

    async def _daily_summary_task(self):
        """Send daily summary at specified time"""
        while True:
            try:
                now = datetime.now(timezone.utc)

                # Check if it's time to send (and we haven't sent today)
                if now.hour == self.daily_summary_hour and now.minute < 5:
                    today = now.strftime('%Y-%m-%d')

                    if self.last_daily_summary != today:
                        await self._send_daily_summary()
                        self.last_daily_summary = today

                # Check every minute
                await asyncio.sleep(60)

            except Exception as e:
                logger.error(f"Error in daily summary task: {e}")
                await asyncio.sleep(60)

    async def _send_daily_summary(self):
        """Send daily trading summary"""
        try:
            today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
            stats = self.daily_stats.get(today, DailyStats(date=today))

            if stats.trades_closed == 0:
                # No trades today
                message = f"""
📅 **DAILY SUMMARY** - {today}

No trades closed today.

📊 **ACTIVITY:**
   New Signals: {stats.trades_opened}
   Market Orders: {stats.market_orders}
   Limit Orders: {stats.limit_orders}
   Limit Fills: {stats.limit_fills}
   Limit Overrides: {stats.limit_overrides}
   Limit Cancellations: {stats.limit_cancellations}

💼 **ACCOUNT:**
   Current Capital: ${self.engine.current_capital:,.2f}
   Open Trades: {len(self.engine.open_trades)}
   Pending Limits: {len(self.engine.pending_limits)}
"""
            else:
                win_rate = (stats.wins / stats.trades_closed * 100) if stats.trades_closed > 0 else 0

                best_trade_info = "None"
                if stats.best_trade:
                    best_trade_info = f"{stats.best_trade['symbol']} ({stats.best_trade['pnl_pct']:+.2f}%)"

                worst_trade_info = "None"
                if stats.worst_trade:
                    worst_trade_info = f"{stats.worst_trade['symbol']} ({stats.worst_trade['pnl_pct']:+.2f}%)"

                message = f"""
📅 **DAILY SUMMARY** - {today}

📊 **TRADES:**
   Closed: {stats.trades_closed}
   Wins: {stats.wins} | Losses: {stats.losses}
   Win Rate: {win_rate:.1f}%
   Total P&L: ${stats.total_pnl:+,.2f}

🏆 **BEST TRADE:**
   {best_trade_info}

📉 **WORST TRADE:**
   {worst_trade_info}

📈 **ORDER ACTIVITY:**
   New Signals: {stats.trades_opened}
   Market Orders: {stats.market_orders}
   Limit Orders: {stats.limit_orders}
   Limit Fills: {stats.limit_fills}
   Limit Overrides: {stats.limit_overrides}
   Limit Cancellations: {stats.limit_cancellations}

💼 **ACCOUNT:**
   Current Capital: ${self.engine.current_capital:,.2f}
   Open Trades: {len(self.engine.open_trades)}
   Pending Limits: {len(self.engine.pending_limits)}
"""

            await self.telegram_bot.send_alert(
                title=f"📅 Daily Summary - {today}",
                message=message,
                level="info"
            )

            # Reset for tomorrow
            tomorrow = (datetime.now(timezone.utc) + timedelta(days=1)).strftime('%Y-%m-%d')
            self.daily_stats[tomorrow] = DailyStats(date=tomorrow)

            # Reset daily capital tracking
            self.daily_starting_capital = self.engine.current_capital

        except Exception as e:
            logger.error(f"Failed to send daily summary: {e}")

    # ============================================================================
    # WEEKLY SUMMARY
    # ============================================================================

    async def _weekly_summary_task(self):
        """Send weekly summary on Sunday"""
        while True:
            try:
                now = datetime.now(timezone.utc)

                # Check if it's Sunday at specified hour
                if now.weekday() == self.weekly_summary_day and now.hour == self.daily_summary_hour and now.minute < 5:
                    week_str = now.strftime('%Y-W%W')

                    if self.last_weekly_summary != week_str:
                        await self._send_weekly_summary()
                        self.last_weekly_summary = week_str

                await asyncio.sleep(60)

            except Exception as e:
                logger.error(f"Error in weekly summary task: {e}")
                await asyncio.sleep(60)

    async def _send_weekly_summary(self):
        """Send weekly performance report"""
        try:
            # Gather stats from last 7 days
            now = datetime.now(timezone.utc)
            week_start = (now - timedelta(days=7)).strftime('%Y-%m-%d')
            week_end = now.strftime('%Y-%m-%d')

            # Analyze closed trades from last 7 days
            recent_trades = [
                t for t in self.engine.closed_trades
                if t.exit_time and t.exit_time >= (now - timedelta(days=7))
            ]

            if not recent_trades:
                message = f"""
📊 **WEEKLY REPORT** - {week_start} to {week_end}

No trades closed this week.

💼 **ACCOUNT:**
   Current Capital: ${self.engine.current_capital:,.2f}
"""
            else:
                total_pnl = sum(t.pnl for t in recent_trades)
                wins = sum(1 for t in recent_trades if t.pnl > 0)
                losses = len(recent_trades) - wins
                win_rate = (wins / len(recent_trades) * 100) if recent_trades else 0

                # Best performing pairs
                pair_pnl = defaultdict(float)
                for t in recent_trades:
                    pair_pnl[t.symbol] += t.pnl

                best_pairs = sorted(pair_pnl.items(), key=lambda x: x[1], reverse=True)[:3]
                best_pairs_str = "\n   ".join([f"{pair}: ${pnl:+,.2f}" for pair, pnl in best_pairs])

                # Trade type performance
                trade_type_pnl = defaultdict(float)
                trade_type_count = defaultdict(int)
                for t in recent_trades:
                    trade_type_pnl[t.trade_type] += t.pnl
                    trade_type_count[t.trade_type] += 1

                best_trade_type = max(trade_type_pnl.items(), key=lambda x: x[1]) if trade_type_pnl else ("N/A", 0)

                # Market vs Limit
                market_fills = sum(1 for t in recent_trades if t.order_type == 'MARKET')
                limit_fills = len(recent_trades) - market_fills

                # Average R:R
                rr_list = []
                for t in recent_trades:
                    risk_pct = abs((t.stop_loss - t.entry_price) / t.entry_price * 100)
                    if risk_pct > 0:
                        rr = abs(t.pnl_pct / risk_pct)
                        rr_list.append(rr)
                avg_rr = sum(rr_list) / len(rr_list) if rr_list else 0

                message = f"""
📊 **WEEKLY REPORT** - {week_start} to {week_end}

📈 **PERFORMANCE:**
   Total Trades: {len(recent_trades)}
   Wins: {wins} | Losses: {losses}
   Win Rate: {win_rate:.1f}%
   Total P&L: ${total_pnl:+,.2f}
   Avg R:R Achieved: {avg_rr:.2f}R

🏆 **BEST PERFORMING PAIRS:**
   {best_pairs_str}

💡 **MOST PROFITABLE TRADE TYPE:**
   {best_trade_type[0]}: ${best_trade_type[1]:+,.2f}
   ({trade_type_count.get(best_trade_type[0], 0)} trades)

📋 **ORDER TYPES:**
   Market Fills: {market_fills} ({market_fills/len(recent_trades)*100:.0f}%)
   Limit Fills: {limit_fills} ({limit_fills/len(recent_trades)*100:.0f}%)

💼 **ACCOUNT:**
   Current Capital: ${self.engine.current_capital:,.2f}
   Total Return: {((self.engine.current_capital - self.engine.initial_capital) / self.engine.initial_capital * 100):+.1f}%
"""

            await self.telegram_bot.send_alert(
                title=f"📊 Weekly Report - Week of {week_start}",
                message=message,
                level="info"
            )

        except Exception as e:
            logger.error(f"Failed to send weekly summary: {e}")

    # ============================================================================
    # ALERT GROUPING
    # ============================================================================

    async def _signal_buffer_task(self):
        """Periodically flush buffered signals"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute

                if not self.signal_buffer:
                    continue

                # Check if buffer timeout expired
                if self.last_signal_time:
                    elapsed = (datetime.now(timezone.utc) - self.last_signal_time).total_seconds()

                    if elapsed >= self.buffer_timeout:
                        await self._flush_signal_buffer()

            except Exception as e:
                logger.error(f"Error in signal buffer task: {e}")

    async def buffer_signal(self, symbol: str, message: str, title: str):
        """
        Add signal to buffer for grouping

        If multiple signals arrive within 5 minutes, they'll be grouped
        """
        signal = BufferedSignal(
            symbol=symbol,
            message=message,
            title=title,
            timestamp=datetime.now(timezone.utc)
        )

        self.signal_buffer.append(signal)
        self.last_signal_time = datetime.now(timezone.utc)

        # If this is the first signal, set a timer to flush
        if len(self.signal_buffer) == 1:
            asyncio.create_task(self._delayed_flush())

    async def _delayed_flush(self):
        """Flush buffer after timeout"""
        await asyncio.sleep(self.buffer_timeout)
        await self._flush_signal_buffer()

    async def _flush_signal_buffer(self):
        """Send all buffered signals (grouped if multiple)"""
        if not self.signal_buffer:
            return

        try:
            if len(self.signal_buffer) == 1:
                # Single signal, send normally
                signal = self.signal_buffer[0]
                await self.telegram_bot.send_alert(
                    title=signal.title,
                    message=signal.message,
                    level="info"
                )
            else:
                # Multiple signals, send grouped
                symbols = [s.symbol for s in self.signal_buffer]
                symbols_str = ", ".join(symbols)

                grouped_message = f"""
🔥 **{len(self.signal_buffer)} SIGNALS DETECTED** 🔥

Pairs: {symbols_str}

Multiple signals fired within 5 minutes during volatile conditions.
Check individual signals below:

"""
                for i, signal in enumerate(self.signal_buffer, 1):
                    grouped_message += f"\n{i}. **{signal.symbol}**\n{signal.title}\n"

                await self.telegram_bot.send_alert(
                    title=f"🔥 {len(self.signal_buffer)} Signals Detected",
                    message=grouped_message,
                    level="info"
                )

            # Clear buffer
            self.signal_buffer = []
            self.last_signal_time = None

        except Exception as e:
            logger.error(f"Failed to flush signal buffer: {e}")

    # ============================================================================
    # LIMIT PROXIMITY ALERTS
    # ============================================================================

    async def _limit_proximity_task(self):
        """Check if price is approaching pending limits"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute

                for symbol, trade in self.engine.pending_limits.items():
                    # Skip if already alerted
                    order_key = f"{symbol}_{trade.signal_name}_{trade.direction}"
                    if self.limit_alert_sent.get(order_key, False):
                        continue

                    # Get current price
                    try:
                        from datetime import timedelta
                        end = datetime.now(timezone.utc)
                        start = end - timedelta(minutes=5)
                        df = self.engine.exchange.get_ohlcv(symbol, '1m', start, end, limit=1)

                        if df.empty:
                            continue

                        current_price = df.iloc[-1]['close']

                        # Check proximity
                        distance = abs(current_price - trade.limit_price) / trade.limit_price

                        if distance <= self.limit_alert_threshold:
                            await self._send_limit_proximity_alert(trade, current_price, distance)
                            self.limit_alert_sent[order_key] = True

                    except Exception as e:
                        logger.error(f"Error checking proximity for {symbol}: {e}")

            except Exception as e:
                logger.error(f"Error in limit proximity task: {e}")
                await asyncio.sleep(60)

    async def _send_limit_proximity_alert(self, trade, current_price: float, distance: float):
        """Send alert when price approaching limit"""
        try:
            emoji = "🟢" if trade.direction == 'long' else "🔴"
            distance_pct = distance * 100

            message = f"""
⚠️ **LIMIT ORDER APPROACHING** ⚠️

💰 **ORDER:**
   Symbol: {trade.symbol}
   Direction: {trade.direction.upper()}
   Limit Price: ${trade.limit_price:,.4f}
   Current Price: ${current_price:,.4f}
   Distance: {distance_pct:.2f}%

⏱️ **INFO:**
   Trade Type: {trade.trade_type}
   Confluence: {trade.confluence_score}/10
   Pending: {((datetime.now(timezone.utc) - trade.entry_time).total_seconds() / 60):.0f} minutes

🎯 **HEADS UP:**
   Price is within 1% of your limit order!
   Order should fill soon if price continues.
"""

            await self.telegram_bot.send_alert(
                title=f"⚠️ {trade.symbol} Approaching Limit",
                message=message,
                level="info"
            )

        except Exception as e:
            logger.error(f"Failed to send limit proximity alert: {e}")

    # ============================================================================
    # RISK MANAGEMENT ALERTS
    # ============================================================================

    async def _risk_monitoring_task(self):
        """Monitor risk metrics and send alerts"""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes

                # Check consecutive losses
                await self._check_consecutive_losses()

                # Check daily drawdown
                await self._check_daily_drawdown()

                # Check win rate
                await self._check_win_rate()

            except Exception as e:
                logger.error(f"Error in risk monitoring task: {e}")
                await asyncio.sleep(300)

    async def _check_consecutive_losses(self):
        """Alert on 3 consecutive losses"""
        if len(self.engine.closed_trades) < 3:
            return

        last_three = self.engine.closed_trades[-3:]
        if all(t.pnl < 0 for t in last_three):
            # Only alert once
            if self.consecutive_losses != 3:
                self.consecutive_losses = 3

                message = f"""
⚠️ **RISK ALERT: 3 CONSECUTIVE LOSSES** ⚠️

📉 **RECENT TRADES:**
   1. {last_three[0].symbol}: ${last_three[0].pnl:,.2f}
   2. {last_three[1].symbol}: ${last_three[1].pnl:,.2f}
   3. {last_three[2].symbol}: ${last_three[2].pnl:,.2f}

💡 **RECOMMENDATION:**
   Consider taking a break to reassess strategy.
   Check if market conditions have changed.
   Review HTF context before next trade.

💼 **ACCOUNT:**
   Current Capital: ${self.engine.current_capital:,.2f}
   Return: {((self.engine.current_capital - self.engine.initial_capital) / self.engine.initial_capital * 100):+.1f}%
"""

                await self.telegram_bot.send_alert(
                    title="⚠️ Risk Alert: 3 Losses in a Row",
                    message=message,
                    level="warning"
                )
        else:
            # Reset counter if not all losses
            self.consecutive_losses = 0

    async def _check_daily_drawdown(self):
        """Alert on 2% daily drawdown"""
        if self.daily_starting_capital == 0:
            return

        drawdown = ((self.engine.current_capital - self.daily_starting_capital) /
                   self.daily_starting_capital) * 100

        if drawdown <= -2.0:
            message = f"""
🚨 **RISK ALERT: DAILY DRAWDOWN -2%** 🚨

📊 **DAILY PERFORMANCE:**
   Starting Capital: ${self.daily_starting_capital:,.2f}
   Current Capital: ${self.engine.current_capital:,.2f}
   Drawdown: {drawdown:.2f}%

💡 **RECOMMENDATION:**
   Daily loss limit approaching.
   Consider stopping trading for today.
   Protect your capital - tomorrow is another day.

📈 **TODAY'S STATS:**
   Open Trades: {len(self.engine.open_trades)}
   Pending Limits: {len(self.engine.pending_limits)}
"""

            await self.telegram_bot.send_alert(
                title="🚨 Daily Drawdown Alert: -2%",
                message=message,
                level="warning"
            )

    async def _check_win_rate(self):
        """Alert if win rate drops below 50%"""
        if self.engine.total_trades < 10:  # Need minimum trades
            return

        win_rate = (self.engine.winning_trades / self.engine.total_trades * 100)

        if win_rate < 50 and not self.win_rate_alert_sent:
            self.win_rate_alert_sent = True

            message = f"""
⚠️ **PERFORMANCE ALERT: WIN RATE BELOW 50%** ⚠️

📊 **STATISTICS:**
   Total Trades: {self.engine.total_trades}
   Winners: {self.engine.winning_trades}
   Losers: {self.engine.losing_trades}
   Win Rate: {win_rate:.1f}%

💡 **REVIEW:**
   Win rate below 50% threshold.
   Review recent trades for patterns.
   Check if confluence scoring needs adjustment.
   Verify HTF analysis is accurate.

💼 **ACCOUNT:**
   Total P&L: ${self.engine.total_pnl:+,.2f}
   Return: {((self.engine.current_capital - self.engine.initial_capital) / self.engine.initial_capital * 100):+.1f}%
"""

            await self.telegram_bot.send_alert(
                title="⚠️ Win Rate Below 50%",
                message=message,
                level="warning"
            )

        elif win_rate >= 55:
            # Reset alert if win rate recovers
            self.win_rate_alert_sent = False

    # ============================================================================
    # TRACKING METHODS (called by paper trading engine)
    # ============================================================================

    def track_trade_opened(self, trade, order_type: str):
        """Track when a new trade/order is opened"""
        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        stats = self.daily_stats.get(today)

        if stats:
            stats.trades_opened += 1
            if order_type == 'MARKET':
                stats.market_orders += 1
            else:
                stats.limit_orders += 1

    def track_trade_closed(self, trade):
        """Track when a trade closes"""
        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        stats = self.daily_stats.get(today)

        if stats:
            stats.trades_closed += 1
            stats.total_pnl += trade.pnl

            if trade.pnl > 0:
                stats.wins += 1
                if stats.best_trade is None or trade.pnl_pct > stats.best_trade['pnl_pct']:
                    stats.best_trade = {
                        'symbol': trade.symbol,
                        'pnl': trade.pnl,
                        'pnl_pct': trade.pnl_pct
                    }
            else:
                stats.losses += 1
                if stats.worst_trade is None or trade.pnl_pct < stats.worst_trade['pnl_pct']:
                    stats.worst_trade = {
                        'symbol': trade.symbol,
                        'pnl': trade.pnl,
                        'pnl_pct': trade.pnl_pct
                    }

    def track_limit_filled(self, symbol: str):
        """Track when a limit order fills"""
        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        stats = self.daily_stats.get(today)
        if stats:
            stats.limit_fills += 1

    def track_limit_override(self, symbol: str):
        """Track when a limit overrides to market"""
        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        stats = self.daily_stats.get(today)
        if stats:
            stats.limit_overrides += 1

    def track_limit_cancelled(self, symbol: str):
        """Track when a limit is cancelled"""
        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        stats = self.daily_stats.get(today)
        if stats:
            stats.limit_cancellations += 1
