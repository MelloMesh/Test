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
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from analysis.higher_timeframe_analyzer import HigherTimeframeAnalyzer, HTFContext
from data.binance_fetcher import BinanceFetcher
from signals.signal_discovery_htf import get_htf_aware_signals
from utils.telegram_bot import TelegramBot

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
    status: str = 'open'  # 'open', 'closed_win', 'closed_loss'
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
        self.closed_trades: List[PaperTrade] = []

        self.exchange = BinanceFetcher()
        self.htf_analyzer = HigherTimeframeAnalyzer()

        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0

        # Load Telegram bot
        self.telegram_bot = None
        try:
            self.telegram_bot = TelegramBot()
            logger.info("✅ Telegram bot initialized")
        except:
            logger.warning("⚠️ Telegram bot not configured")

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
                'last_updated': datetime.utcnow().isoformat()
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
        position_size_pct: float = 2.0  # 2% of capital per trade
    ) -> Optional[PaperTrade]:
        """Open a new paper trade"""

        # Check if we already have a trade open for this symbol
        if symbol in self.open_trades:
            logger.debug(f"Already have open trade for {symbol}, skipping")
            return None

        # Calculate position size
        position_size = self.current_capital * (position_size_pct / 100)

        # Calculate stop loss and take profit (2% risk, 4% reward = 1:2 R:R)
        risk_pct = 0.02
        reward_pct = 0.04

        if direction == 'long':
            stop_loss = entry_price * (1 - risk_pct)
            take_profit = entry_price * (1 + reward_pct)
        else:  # short
            stop_loss = entry_price * (1 + risk_pct)
            take_profit = entry_price * (1 - reward_pct)

        # Create trade
        trade = PaperTrade(
            symbol=symbol,
            signal_name=signal_name,
            direction=direction,
            entry_price=entry_price,
            entry_time=datetime.utcnow(),
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size,
            htf_bias=htf_context.primary_bias,
            htf_alignment=htf_context.alignment_score
        )

        self.open_trades[symbol] = trade
        self.total_trades += 1

        logger.info(f"\n{'='*80}")
        logger.info(f"📈 OPENED {direction.upper()} TRADE: {symbol}")
        logger.info(f"{'='*80}")
        logger.info(f"  Signal: {signal_name}")
        logger.info(f"  Entry: ${entry_price:,.4f}")
        logger.info(f"  Stop Loss: ${stop_loss:,.4f} ({-risk_pct*100:.1f}%)")
        logger.info(f"  Take Profit: ${take_profit:,.4f} (+{reward_pct*100:.1f}%)")
        logger.info(f"  Position Size: ${position_size:,.2f}")
        logger.info(f"  HTF Bias: {htf_context.primary_bias.upper()} ({htf_context.alignment_score:.0f}%)")
        logger.info(f"{'='*80}\n")

        # Send Telegram notification
        if self.telegram_bot:
            asyncio.create_task(self._send_trade_opened_notification(trade, htf_context))

        return trade

    def check_and_close_trades(self):
        """Check open trades and close if SL/TP hit"""

        if not self.open_trades:
            return

        symbols_to_close = []

        for symbol, trade in self.open_trades.items():
            # Get current price
            try:
                # Fetch latest candle
                end = datetime.utcnow()
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
        trade.exit_time = datetime.utcnow()
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

            message = f"""
{emoji} **PAPER TRADE OPENED** {emoji}

💰 **TRADE:**
   Symbol: {trade.symbol}
   Direction: **{trade.direction.upper()}**
   Entry: ${trade.entry_price:,.4f}

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
                title=f"Paper Trade Opened: {trade.symbol}",
                message=message,
                level="info"
            )
        except Exception as e:
            logger.error(f"Failed to send Telegram notification: {e}")

    async def _send_trade_closed_notification(self, trade: PaperTrade):
        """Send Telegram notification for closed trade"""
        try:
            emoji = "🟢" if trade.pnl > 0 else "🔴"

            win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
            total_return = ((self.current_capital - self.initial_capital) / self.initial_capital) * 100

            message = f"""
{emoji} **PAPER TRADE CLOSED** {emoji}

💰 **RESULT:**
   Symbol: {trade.symbol}
   Direction: {trade.direction.upper()}
   Entry: ${trade.entry_price:,.4f}
   Exit: ${trade.exit_price:,.4f}

   P&L: ${trade.pnl:+,.2f} ({trade.pnl_pct:+.2f}%)
   Reason: {trade.reason}

📊 **PERFORMANCE:**
   Win Rate: {win_rate:.1f}% ({self.winning_trades}W / {self.losing_trades}L)
   Total P&L: ${self.total_pnl:+,.2f}
   Total Return: {total_return:+.1f}%
   Current Capital: ${self.current_capital:,.2f}
"""

            await self.telegram_bot.send_alert(
                title=f"Paper Trade Closed: {trade.symbol}",
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
        print(f"     Winners:          {self.winning_trades:>12}")
        print(f"     Losers:           {self.losing_trades:>12}")
        print(f"     Win Rate:         {win_rate:>12,.1f}%")

        if self.open_trades:
            print(f"\n  🔓 Open Trades:")
            for symbol, trade in self.open_trades.items():
                print(f"     {symbol}: {trade.direction.upper()} @ ${trade.entry_price:,.4f}")

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

    # Display current performance
    engine.display_performance()

    print("📡 Listening for signals from live scanner...")
    print("   (Signals will be picked up automatically when scanner runs)")
    print("\n⏳ Checking open trades every 60 seconds...\n")

    # Main loop
    while True:
        try:
            # Check and close trades
            engine.check_and_close_trades()

            # Display performance every 10 minutes
            if datetime.utcnow().minute % 10 == 0:
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
