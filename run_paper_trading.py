#!/usr/bin/env python3
"""
HTF Paper Trading / Forward Testing Mode
Simulates trades from live signals without risking real money
Tracks performance in real-time
"""

import sys
import os
import asyncio
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import config
from analysis.higher_timeframe_analyzer import HigherTimeframeAnalyzer, HTFContext
from data.binance_fetcher import BinanceFetcher
from signals.signal_discovery_htf import get_htf_aware_signals
from trading.telegram_bot import TradingTelegramBot, load_telegram_credentials
from trading.notification_manager import NotificationManager
from signals.limit_order_manager import get_limit_manager
from signals.confluence_scorer import get_confluence_scorer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PaperTrade:
    """Represents a simulated trade"""
    symbol: str
    signal_name: str
    direction: str  # 'long' or 'short'
    entry_price: float
    entry_time: datetime
    stop_loss: float
    take_profit: float
    position_size: float  # USDT
    htf_bias: str
    htf_alignment: float
    order_type: str = 'MARKET'  # 'MARKET' or 'LIMIT'
    limit_price: Optional[float] = None  # For limit orders
    trade_type: str = ''  # TREND_CONTINUATION, MEAN_REVERSION, etc.
    confluence_score: int = 0
    status: str = 'open'  # 'open', 'pending_limit', 'closed_win', 'closed_loss'
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    reason: str = ''


class PaperTradingEngine:
    """Manages paper trades and tracks performance"""

    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.open_trades: Dict[str, PaperTrade] = {}
        self.pending_limits: Dict[str, PaperTrade] = {}  # Limit orders waiting to fill
        self.closed_trades: List[PaperTrade] = []

        self.exchange = BinanceFetcher()
        self.htf_analyzer = HigherTimeframeAnalyzer(config)

        # Limit management
        self.limit_manager = get_limit_manager()
        self.confluence_scorer = get_confluence_scorer()

        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0

        # Load Telegram bot
        self.telegram_bot = None
        token, chat_id = load_telegram_credentials()
        if token and chat_id:
            self.telegram_bot = TradingTelegramBot(token, chat_id, config)
            logger.info("✅ Telegram bot initialized")
        else:
            logger.warning("⚠️ Telegram bot not configured")

        # Initialize notification manager (requires telegram_bot)
        self.notification_manager = None
        if self.telegram_bot:
            self.notification_manager = NotificationManager(self.telegram_bot, self)
            logger.info("✅ Notification manager initialized")

        # Results file
        self.results_file = "paper_trading_results.json"
        self.load_results()

    def load_results(self):
        """Load previous results if they exist"""
        if Path(self.results_file).exists():
            try:
                with open(self.results_file, 'r') as f:
                    data = json.load(f)
                    self.current_capital = data.get('current_capital', self.initial_capital)
                    self.closed_trades = [
                        PaperTrade(**trade) for trade in data.get('closed_trades', [])
                    ]
                    self.total_trades = len(self.closed_trades)
                    self.winning_trades = sum(1 for t in self.closed_trades if t.status == 'closed_win')
                    self.losing_trades = sum(1 for t in self.closed_trades if t.status == 'closed_loss')
                    self.total_pnl = sum(t.pnl for t in self.closed_trades)

                    logger.info(f"📂 Loaded {len(self.closed_trades)} previous trades")
            except Exception as e:
                logger.error(f"Error loading results: {e}")

    def save_results(self):
        """Save results to file"""
        try:
            data = {
                'initial_capital': self.initial_capital,
                'current_capital': self.current_capital,
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'losing_trades': self.losing_trades,
                'total_pnl': self.total_pnl,
                'closed_trades': [asdict(t) for t in self.closed_trades],
                'last_updated': datetime.now(timezone.utc).isoformat()
            }

            with open(self.results_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)

            logger.debug(f"💾 Results saved to {self.results_file}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")

    def open_trade(
        self,
        symbol: str,
        signal_name: str,
        direction: str,
        entry_price: float,
        htf_context: HTFContext,
        order_type: str = 'MARKET',
        limit_price: Optional[float] = None,
        trade_type: str = '',
        confluence_score: int = 0,
        position_size_multiplier: float = 1.0
    ) -> Optional[PaperTrade]:
        """
        Open a new paper trade (market or limit order)

        Args:
            order_type: 'MARKET' or 'LIMIT'
            limit_price: Price for limit orders
            trade_type: Type of trade (TREND_CONTINUATION, MEAN_REVERSION, etc.)
            confluence_score: Signal confluence score 0-10
            position_size_multiplier: Multiplier for position size based on confluence
        """

        # Check if we already have a trade open for this symbol
        if symbol in self.open_trades or symbol in self.pending_limits:
            logger.debug(f"Already have open/pending trade for {symbol}, skipping")
            return None

        # Calculate position size (base 2% * multiplier)
        base_pct = 2.0
        position_size_pct = base_pct * position_size_multiplier
        position_size = self.current_capital * (position_size_pct / 100)

        # Calculate stop loss and take profit (2% risk, 4% reward = 1:2 R:R)
        risk_pct = 0.02
        reward_pct = 0.04

        # Use limit price for SL/TP calculation if it's a limit order
        price_for_calc = limit_price if order_type == 'LIMIT' else entry_price

        if direction == 'long':
            stop_loss = price_for_calc * (1 - risk_pct)
            take_profit = price_for_calc * (1 + reward_pct)
        else:  # short
            stop_loss = price_for_calc * (1 + risk_pct)
            take_profit = price_for_calc * (1 - reward_pct)

        # Create trade
        trade = PaperTrade(
            symbol=symbol,
            signal_name=signal_name,
            direction=direction,
            entry_price=limit_price if order_type == 'LIMIT' else entry_price,
            entry_time=datetime.now(timezone.utc),
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size,
            htf_bias=htf_context.primary_bias,
            htf_alignment=htf_context.alignment_score,
            order_type=order_type,
            limit_price=limit_price,
            trade_type=trade_type,
            confluence_score=confluence_score,
            status='pending_limit' if order_type == 'LIMIT' else 'open'
        )

        # Add to appropriate tracking dict
        if order_type == 'LIMIT':
            self.pending_limits[symbol] = trade
            logger.info(f"\n{'='*80}")
            logger.info(f"📋 PLACED {direction.upper()} LIMIT ORDER: {symbol}")
            logger.info(f"{'='*80}")
            logger.info(f"  Signal: {signal_name}")
            logger.info(f"  Trade Type: {trade_type}")
            logger.info(f"  Confluence Score: {confluence_score}/10")
            logger.info(f"  Current Price: ${entry_price:,.4f}")
            logger.info(f"  Limit Price: ${limit_price:,.4f}")
            logger.info(f"  Stop Loss: ${stop_loss:,.4f} ({-risk_pct*100:.1f}%)")
            logger.info(f"  Take Profit: ${take_profit:,.4f} (+{reward_pct*100:.1f}%)")
            logger.info(f"  Position Size: ${position_size:,.2f} ({position_size_multiplier:.1f}x)")
            logger.info(f"  HTF Bias: {htf_context.primary_bias.upper()} ({htf_context.alignment_score:.0f}%)")
            logger.info(f"{'='*80}\n")
        else:
            self.open_trades[symbol] = trade
            self.total_trades += 1
            logger.info(f"\n{'='*80}")
            logger.info(f"🚀 OPENED {direction.upper()} MARKET TRADE: {symbol}")
            logger.info(f"{'='*80}")
            logger.info(f"  Signal: {signal_name}")
            logger.info(f"  Trade Type: {trade_type}")
            logger.info(f"  Confluence Score: {confluence_score}/10")
            logger.info(f"  Entry: ${entry_price:,.4f}")
            logger.info(f"  Stop Loss: ${stop_loss:,.4f} ({-risk_pct*100:.1f}%)")
            logger.info(f"  Take Profit: ${take_profit:,.4f} (+{reward_pct*100:.1f}%)")
            logger.info(f"  Position Size: ${position_size:,.2f} ({position_size_multiplier:.1f}x)")
            logger.info(f"  HTF Bias: {htf_context.primary_bias.upper()} ({htf_context.alignment_score:.0f}%)")
            logger.info(f"{'='*80}\n")

        # Track in notification manager
        if self.notification_manager:
            self.notification_manager.track_trade_opened(trade, order_type)

        # Send Telegram notification
        if self.telegram_bot:
            asyncio.create_task(self._send_trade_opened_notification(trade, htf_context))

        return trade

    def check_pending_limits(self):
        """Check pending limit orders for fills or overrides"""

        if not self.pending_limits:
            return

        symbols_to_fill = []
        symbols_to_cancel = []

        for symbol, trade in self.pending_limits.items():
            try:
                # Get current price
                end = datetime.now(timezone.utc)
                start = end - timedelta(minutes=5)
                df = self.exchange.get_ohlcv(symbol, '1m', start, end, limit=5)

                if df.empty:
                    continue

                current_price = df.iloc[-1]['close']

                # Build order key for limit manager
                order_key = f"{symbol}_{trade.signal_name}_{trade.direction}"

                # Check if limit filled (price touched limit)
                filled = False
                if trade.direction == 'long':
                    # Long limit fills if price touched limit or below
                    filled = df['low'].min() <= trade.limit_price
                else:
                    # Short limit fills if price touched limit or above
                    filled = df['high'].max() >= trade.limit_price

                if filled:
                    logger.info(f"✅ LIMIT FILLED: {symbol} @ ${trade.limit_price:,.4f}")
                    # Convert to open trade
                    trade.status = 'open'
                    trade.entry_price = trade.limit_price
                    self.open_trades[symbol] = trade
                    symbols_to_fill.append(symbol)
                    self.total_trades += 1

                    # Track in notification manager
                    if self.notification_manager:
                        self.notification_manager.track_limit_filled(symbol)

                    # Send Telegram notification
                    if self.telegram_bot:
                        asyncio.create_task(self._send_limit_filled_notification(trade))

                    # Remove from limit manager
                    self.limit_manager.remove_limit_order(order_key)
                    continue

                # Check if should override to market
                override_decision = self.limit_manager.check_limit_override(
                    order_key=order_key,
                    current_price=current_price,
                    current_confluence_score=trade.confluence_score,  # Would need re-evaluation in real system
                    rsi_value=None,  # Would fetch from data in real system
                    has_divergence=False,
                    volume_spike=1.0
                )

                if override_decision == 'CANCEL_LIMIT_GO_MARKET':
                    logger.info(f"🚀 OVERRIDING TO MARKET: {symbol} @ ${current_price:,.4f}")

                    old_limit_price = trade.limit_price

                    # Convert to market order
                    trade.status = 'open'
                    trade.entry_price = current_price
                    trade.order_type = 'MARKET'

                    # Recalculate SL/TP based on market entry
                    risk_pct = 0.02
                    reward_pct = 0.04
                    if trade.direction == 'long':
                        trade.stop_loss = current_price * (1 - risk_pct)
                        trade.take_profit = current_price * (1 + reward_pct)
                    else:
                        trade.stop_loss = current_price * (1 + risk_pct)
                        trade.take_profit = current_price * (1 - reward_pct)

                    self.open_trades[symbol] = trade
                    symbols_to_fill.append(symbol)
                    self.total_trades += 1

                    # Track in notification manager
                    if self.notification_manager:
                        self.notification_manager.track_limit_override(symbol)

                    # Send Telegram notification
                    if self.telegram_bot:
                        asyncio.create_task(self._send_limit_override_notification(trade, old_limit_price, current_price))

                    # Remove from limit manager
                    self.limit_manager.remove_limit_order(order_key)

                elif override_decision == 'CANCEL_LIMIT':
                    logger.info(f"❌ CANCELING LIMIT: {symbol} - Setup invalidated")
                    symbols_to_cancel.append(symbol)

                    # Track in notification manager
                    if self.notification_manager:
                        self.notification_manager.track_limit_cancelled(symbol)

                    # Send Telegram notification
                    if self.telegram_bot:
                        asyncio.create_task(self._send_limit_cancelled_notification(trade, current_price))

                    self.limit_manager.remove_limit_order(order_key)

            except Exception as e:
                logger.error(f"Error checking limit for {symbol}: {e}")

        # Remove filled/canceled limits
        for symbol in symbols_to_fill + symbols_to_cancel:
            if symbol in self.pending_limits:
                del self.pending_limits[symbol]

    def check_and_close_trades(self):
        """Check open trades and close if SL/TP hit"""

        if not self.open_trades:
            return

        symbols_to_close = []

        for symbol, trade in self.open_trades.items():
            # Get current price
            try:
                # Fetch latest candle
                end = datetime.now(timezone.utc)
                start = end - timedelta(minutes=5)
                df = self.exchange.get_ohlcv(symbol, '1m', start, end, limit=1)

                if df.empty:
                    continue

                current_price = df.iloc[-1]['close']

                # Check if SL or TP hit
                should_close = False
                close_reason = ''

                if trade.direction == 'long':
                    if current_price <= trade.stop_loss:
                        should_close = True
                        close_reason = 'Stop Loss'
                    elif current_price >= trade.take_profit:
                        should_close = True
                        close_reason = 'Take Profit'
                else:  # short
                    if current_price >= trade.stop_loss:
                        should_close = True
                        close_reason = 'Stop Loss'
                    elif current_price <= trade.take_profit:
                        should_close = True
                        close_reason = 'Take Profit'

                if should_close:
                    self.close_trade(symbol, current_price, close_reason)
                    symbols_to_close.append(symbol)

            except Exception as e:
                logger.error(f"Error checking {symbol}: {e}")

        # Remove closed trades from open trades dict
        for symbol in symbols_to_close:
            del self.open_trades[symbol]

    def close_trade(self, symbol: str, exit_price: float, reason: str):
        """Close a trade and calculate P&L"""

        if symbol not in self.open_trades:
            return

        trade = self.open_trades[symbol]
        trade.exit_price = exit_price
        trade.exit_time = datetime.now(timezone.utc)
        trade.reason = reason

        # Calculate P&L
        if trade.direction == 'long':
            pnl_pct = ((exit_price - trade.entry_price) / trade.entry_price) * 100
        else:  # short
            pnl_pct = ((trade.entry_price - exit_price) / trade.entry_price) * 100

        pnl = trade.position_size * (pnl_pct / 100)

        trade.pnl = pnl
        trade.pnl_pct = pnl_pct

        # Update status
        if pnl > 0:
            trade.status = 'closed_win'
            self.winning_trades += 1
        else:
            trade.status = 'closed_loss'
            self.losing_trades += 1

        # Update capital
        self.current_capital += pnl
        self.total_pnl += pnl

        # Move to closed trades
        self.closed_trades.append(trade)

        # Track in notification manager
        if self.notification_manager:
            self.notification_manager.track_trade_closed(trade)

        # Log
        emoji = "🟢" if pnl > 0 else "🔴"
        logger.info(f"\n{'='*80}")
        logger.info(f"{emoji} CLOSED {trade.direction.upper()} TRADE: {symbol}")
        logger.info(f"{'='*80}")
        logger.info(f"  Reason: {reason}")
        logger.info(f"  Entry: ${trade.entry_price:,.4f}")
        logger.info(f"  Exit: ${exit_price:,.4f}")
        logger.info(f"  P&L: ${pnl:+,.2f} ({pnl_pct:+.2f}%)")
        logger.info(f"  Duration: {(trade.exit_time - trade.entry_time).total_seconds() / 3600:.1f} hours")
        logger.info(f"  Current Capital: ${self.current_capital:,.2f}")
        logger.info(f"{'='*80}\n")

        # Send Telegram notification
        if self.telegram_bot:
            asyncio.create_task(self._send_trade_closed_notification(trade))

        # Save results
        self.save_results()

    async def _send_trade_opened_notification(self, trade: PaperTrade, htf_context: HTFContext):
        """Send Telegram notification for opened trade"""
        try:
            emoji = "🟢" if trade.direction == 'long' else "🔴"
            order_emoji = "🚀" if trade.order_type == 'MARKET' else "📋"

            # Build entry info based on order type
            if trade.order_type == 'MARKET':
                entry_info = f"Entry: ${trade.entry_price:,.4f}"
                title = f"Paper Trade Opened: {trade.symbol}"
            else:
                entry_info = f"Current: ${trade.entry_price:,.4f}\n   Limit Price: ${trade.limit_price:,.4f}"
                title = f"Limit Order Placed: {trade.symbol}"

            # Confluence bar
            score = trade.confluence_score
            filled_bars = "█" * score
            empty_bars = "░" * (10 - score)
            confluence_bar = filled_bars + empty_bars

            message = f"""
{emoji} **PAPER {trade.order_type} ORDER** {order_emoji}

💰 **TRADE:**
   Symbol: {trade.symbol}
   Direction: **{trade.direction.upper()}**
   {entry_info}
   Trade Type: {trade.trade_type}

⚡ **CONFLUENCE:**
   {confluence_bar} {score}/10

🛡️ **RISK MANAGEMENT:**
   Stop Loss: ${trade.stop_loss:,.4f}
   Take Profit: ${trade.take_profit:,.4f}
   Position Size: ${trade.position_size:,.2f}

📊 **HTF CONTEXT:**
   Bias: {htf_context.primary_bias.upper()}
   Alignment: {htf_context.alignment_score:.0f}%
   Regime: {htf_context.regime.upper()}

💼 **ACCOUNT:**
   Current Capital: ${self.current_capital:,.2f}
"""

            await self.telegram_bot.send_alert(
                title=title,
                message=message,
                level="info"
            )
        except Exception as e:
            logger.error(f"Failed to send Telegram notification: {e}")

    async def _send_limit_filled_notification(self, trade: PaperTrade):
        """Send notification when limit order fills"""
        try:
            emoji = "🟢" if trade.direction == 'long' else "🔴"

            # Calculate time pending
            time_pending = (datetime.now(timezone.utc) - trade.entry_time).total_seconds() / 60

            message = f"""
✅ **LIMIT ORDER FILLED** ✅

💰 **TRADE:**
   Symbol: {trade.symbol}
   Direction: **{trade.direction.upper()}**
   Entry: ${trade.entry_price:,.4f}
   Trade Type: {trade.trade_type}

⏱️ **TIMING:**
   Time Pending: {time_pending:.0f} minutes

🛡️ **RISK MANAGEMENT:**
   Stop Loss: ${trade.stop_loss:,.4f}
   Take Profit: ${trade.take_profit:,.4f}
   Position Size: ${trade.position_size:,.2f}

💼 **ACCOUNT:**
   Current Capital: ${self.current_capital:,.2f}
   Open Trades: {len(self.open_trades)}
"""

            await self.telegram_bot.send_alert(
                title=f"✅ Limit Filled: {trade.symbol}",
                message=message,
                level="info"
            )
        except Exception as e:
            logger.error(f"Failed to send limit filled notification: {e}")

    async def _send_limit_override_notification(self, trade: PaperTrade, old_limit_price: float, market_price: float):
        """Send notification when limit overrides to market"""
        try:
            emoji = "🟢" if trade.direction == 'long' else "🔴"

            # Calculate price difference
            if trade.direction == 'long':
                price_diff_pct = ((market_price - old_limit_price) / old_limit_price) * 100
            else:
                price_diff_pct = ((old_limit_price - market_price) / old_limit_price) * 100

            # Calculate time pending
            time_pending = (datetime.now(timezone.utc) - trade.entry_time).total_seconds() / 60

            message = f"""
🚀 **LIMIT OVERRIDDEN TO MARKET** 🚀

💰 **TRADE:**
   Symbol: {trade.symbol}
   Direction: **{trade.direction.upper()}**
   Original Limit: ${old_limit_price:,.4f}
   Market Entry: ${market_price:,.4f}
   Slippage: {price_diff_pct:+.2f}%
   Trade Type: {trade.trade_type}

⚡ **REASON:**
   Confluence improved or price drifting away
   Better to enter now than miss the move

⏱️ **TIMING:**
   Time Pending: {time_pending:.0f} minutes

🛡️ **RISK MANAGEMENT:**
   Stop Loss: ${trade.stop_loss:,.4f}
   Take Profit: ${trade.take_profit:,.4f}
   Position Size: ${trade.position_size:,.2f}

💼 **ACCOUNT:**
   Current Capital: ${self.current_capital:,.2f}
   Open Trades: {len(self.open_trades)}
"""

            await self.telegram_bot.send_alert(
                title=f"🚀 Override: {trade.symbol}",
                message=message,
                level="info"
            )
        except Exception as e:
            logger.error(f"Failed to send override notification: {e}")

    async def _send_limit_cancelled_notification(self, trade: PaperTrade, current_price: float):
        """Send notification when limit order is cancelled"""
        try:
            # Calculate time pending
            time_pending = (datetime.now(timezone.utc) - trade.entry_time).total_seconds() / 60

            # Calculate price movement
            price_diff_pct = ((current_price - trade.limit_price) / trade.limit_price) * 100

            message = f"""
❌ **LIMIT ORDER CANCELLED** ❌

💰 **ORDER:**
   Symbol: {trade.symbol}
   Direction: {trade.direction.upper()}
   Limit Price: ${trade.limit_price:,.4f}
   Current Price: ${current_price:,.4f}
   Price Move: {price_diff_pct:+.2f}%
   Trade Type: {trade.trade_type}

⚠️ **REASON:**
   Setup invalidated - confluence dropped below threshold
   Better to skip than force a low-quality trade

⏱️ **TIMING:**
   Time Pending: {time_pending:.0f} minutes

💼 **ACCOUNT:**
   Current Capital: ${self.current_capital:,.2f}
   Open Trades: {len(self.open_trades)}
   Pending Limits: {len(self.pending_limits) - 1}
"""

            await self.telegram_bot.send_alert(
                title=f"❌ Cancelled: {trade.symbol}",
                message=message,
                level="warning"
            )
        except Exception as e:
            logger.error(f"Failed to send cancellation notification: {e}")

    async def _send_trade_closed_notification(self, trade: PaperTrade):
        """Send Telegram notification for closed trade"""
        try:
            emoji = "🟢" if trade.pnl > 0 else "🔴"
            result_emoji = "✅" if trade.pnl > 0 else "❌"

            # Calculate trade duration
            duration_hours = (trade.exit_time - trade.entry_time).total_seconds() / 3600

            win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
            total_return = ((self.current_capital - self.initial_capital) / self.initial_capital) * 100

            # Risk/Reward achieved
            risk_pct = abs((trade.stop_loss - trade.entry_price) / trade.entry_price * 100)
            actual_rr = abs(trade.pnl_pct / risk_pct) if risk_pct > 0 else 0

            message = f"""
{emoji} **TRADE CLOSED** {result_emoji}

💰 **RESULT:**
   Symbol: {trade.symbol}
   Direction: {trade.direction.upper()}
   Entry: ${trade.entry_price:,.4f}
   Exit: ${trade.exit_price:,.4f}

   P&L: ${trade.pnl:+,.2f} ({trade.pnl_pct:+.2f}%)
   R:R Achieved: {actual_rr:.2f}R
   Reason: {trade.reason}

⏱️ **TIMING:**
   Duration: {duration_hours:.1f} hours
   Order Type: {trade.order_type}
   Trade Type: {trade.trade_type}
   Confluence: {trade.confluence_score}/10

📊 **ACCOUNT PERFORMANCE:**
   Win Rate: {win_rate:.1f}% ({self.winning_trades}W / {self.losing_trades}L)
   Total Trades: {self.total_trades}
   Total P&L: ${self.total_pnl:+,.2f}
   Total Return: {total_return:+.1f}%
   Current Capital: ${self.current_capital:,.2f}

💼 **ACTIVE POSITIONS:**
   Open Trades: {len(self.open_trades)}
   Pending Limits: {len(self.pending_limits)}
"""

            await self.telegram_bot.send_alert(
                title=f"{result_emoji} Closed: {trade.symbol} ({trade.pnl_pct:+.1f}%)",
                message=message,
                level="info"
            )
        except Exception as e:
            logger.error(f"Failed to send Telegram notification: {e}")

    def display_performance(self):
        """Display current performance metrics"""

        print(f"\n{'='*80}")
        print(f"{'📊 PAPER TRADING PERFORMANCE':^80}")
        print(f"{'='*80}\n")

        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        total_return = ((self.current_capital - self.initial_capital) / self.initial_capital) * 100

        print(f"  💼 Account:")
        print(f"     Initial Capital:  ${self.initial_capital:>12,.2f}")
        print(f"     Current Capital:  ${self.current_capital:>12,.2f}")
        print(f"     Total P&L:        ${self.total_pnl:>+12,.2f}")
        print(f"     Return:           {total_return:>12,.1f}%")

        print(f"\n  📈 Trades:")
        print(f"     Total:            {self.total_trades:>12}")
        print(f"     Open:             {len(self.open_trades):>12}")
        print(f"     Pending Limits:   {len(self.pending_limits):>12}")
        print(f"     Winners:          {self.winning_trades:>12}")
        print(f"     Losers:           {self.losing_trades:>12}")
        print(f"     Win Rate:         {win_rate:>12,.1f}%")

        if self.pending_limits:
            print(f"\n  📋 Pending Limit Orders:")
            for symbol, trade in self.pending_limits.items():
                print(f"     {symbol}: {trade.direction.upper()} @ ${trade.limit_price:,.4f} (Score: {trade.confluence_score}/10)")

        if self.open_trades:
            print(f"\n  🔓 Open Trades:")
            for symbol, trade in self.open_trades.items():
                order_emoji = "🚀" if trade.order_type == 'MARKET' else "📋"
                print(f"     {symbol}: {order_emoji} {trade.direction.upper()} @ ${trade.entry_price:,.4f}")

        print(f"\n{'='*80}\n")


async def main():
    """Run paper trading engine"""

    print(f"\n{'='*80}")
    print(f"{'🚀 HTF PAPER TRADING ENGINE':^80}")
    print(f"{'='*80}\n")
    print(f"  This mode simulates trades from live signals")
    print(f"  No real money is used - perfect for testing strategies!")
    print(f"\n{'='*80}\n")

    # Initialize engine
    engine = PaperTradingEngine(initial_capital=10000)

    # Start notification manager background tasks
    if engine.notification_manager:
        await engine.notification_manager.start()
        logger.info("✅ Notification manager tasks started")

    # Display current performance
    engine.display_performance()

    print("📡 Listening for signals from live scanner...")
    print("   (Signals will be picked up automatically when scanner runs)")
    print("\n⏳ Checking open trades every 60 seconds...\n")

    # Main loop
    while True:
        try:
            # Check pending limits (fills & overrides)
            engine.check_pending_limits()

            # Check and close open trades
            engine.check_and_close_trades()

            # Display performance every 10 minutes
            if datetime.now(timezone.utc).minute % 10 == 0:
                engine.display_performance()

            # Wait 60 seconds
            await asyncio.sleep(60)

        except KeyboardInterrupt:
            logger.info("\n\n⏹️  Stopping paper trading engine...")
            engine.display_performance()
            break
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            await asyncio.sleep(60)


if __name__ == "__main__":
    asyncio.run(main())
