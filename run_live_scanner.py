"""
Live Trading Signal Scanner
Connects to Kraken WebSocket, scans for signals in real-time, sends to Telegram
"""

import asyncio
import logging
from pathlib import Path
import sys
from datetime import datetime
import pandas as pd

# Add project to path
sys.path.append(str(Path(__file__).parent))

import config
from src.data.realtime_data_stream import RealtimeDataStream
from src.trading.live_signal_generator import LiveSignalGenerator
from src.trading.telegram_bot import TradingTelegramBot, load_telegram_credentials

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LiveTradingScanner:
    """
    Real-time trading signal scanner with Telegram notifications
    """

    def __init__(self, config, telegram_bot=None):
        """
        Initialize scanner

        Args:
            config: Trading system config
            telegram_bot: Optional TradingTelegramBot instance
        """
        self.config = config
        self.telegram_bot = telegram_bot

        # Initialize components
        self.data_stream = RealtimeDataStream(config)
        self.signal_generator = LiveSignalGenerator(config)

        # Track sent signals to avoid duplicates
        self.sent_signals = set()

        # Performance tracking
        self.signals_sent = 0
        self.candles_processed = 0
        self.start_time = datetime.utcnow()

    async def on_candle_complete(self, instrument: str, timeframe: str, candle: dict):
        """
        Called when a candle completes - scan for signals

        Args:
            instrument: Trading instrument (e.g., 'BTCUSDT.P')
            timeframe: Candle timeframe (e.g., '5m')
            candle: OHLC candle data
        """
        self.candles_processed += 1

        logger.info(
            f"Candle complete: {instrument} {timeframe} | "
            f"Close: {candle['close']:.2f} | "
            f"Volume: {candle['volume']:.2f}"
        )

        # Check for signals on this timeframe
        try:
            # Get recent candle data as DataFrame
            recent_candles_df = self._get_recent_candles_as_dataframe(instrument, timeframe)

            if recent_candles_df.empty:
                logger.debug(f"No data available for {instrument} {timeframe}")
                return

            # Prepare data dict for signal generator (expects Dict[timeframe -> DataFrame])
            current_data = {timeframe: recent_candles_df}

            # Scan for signals
            signals = self.signal_generator.scan_for_signals(
                current_data=current_data,
                account_balance=self.config.INITIAL_CAPITAL
            )

            # Send any new signals
            for signal in signals:
                signal_key = f"{signal.signal_name}_{signal.instrument}_{signal.timestamp}"

                if signal_key not in self.sent_signals:
                    await self._send_signal(signal)
                    self.sent_signals.add(signal_key)
                    self.signals_sent += 1

        except Exception as e:
            logger.error(f"Error scanning for signals: {e}", exc_info=True)

    def _get_recent_candles_as_dataframe(self, instrument: str, timeframe: str, limit: int = 500):
        """
        Get recent candle data as DataFrame with OHLCV columns

        Returns:
            pd.DataFrame with columns: open, high, low, close, volume, timestamp
        """
        try:
            candles = self.data_stream.get_candles(instrument, timeframe, limit=limit)

            if not candles:
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(candles)

            # Ensure required columns exist
            required_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
            if not all(col in df.columns for col in required_cols):
                logger.warning(f"Missing required columns in candle data for {instrument} {timeframe}")
                return pd.DataFrame()

            # Set timestamp as index
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')

            return df

        except Exception as e:
            logger.error(f"Error converting candles to DataFrame: {e}")
            return pd.DataFrame()

    async def _send_signal(self, signal):
        """
        Send signal via Telegram and log locally

        Args:
            signal: LiveSignal object
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"🎯 NEW TRADING SIGNAL 🎯")
        logger.info(f"{'='*80}")
        logger.info(f"Signal: {signal.signal_name}")
        logger.info(f"Instrument: {signal.instrument} ({signal.timeframe})")
        logger.info(f"Direction: {signal.direction.upper()}")
        logger.info(f"Entry: ${signal.entry_price:,.2f}")
        logger.info(f"Stop Loss: ${signal.stop_loss:,.2f}")
        logger.info(f"Take Profit: ${signal.take_profit:,.2f}")
        logger.info(f"R:R Ratio: 1:{signal.risk_reward_ratio:.1f}")
        logger.info(f"Position Size: ${signal.position_size_usd:,.2f}")
        logger.info(f"Risk: ${signal.risk_amount_usd:.2f}")
        logger.info(f"Backtest WR: {signal.backtest_win_rate:.1%}")
        logger.info(f"{'='*80}\n")

        # Send to Telegram if configured
        if self.telegram_bot:
            try:
                await self.telegram_bot.send_signal_alert(signal)
                logger.info("✅ Signal sent to Telegram")
            except Exception as e:
                logger.error(f"Failed to send Telegram notification: {e}")

    async def start(self):
        """Start the live scanner"""
        logger.info(f"\n{'='*80}")
        logger.info(f"🚀 STARTING LIVE TRADING SCANNER 🚀")
        logger.info(f"{'='*80}")
        logger.info(f"Instruments: {', '.join(self.config.INSTRUMENTS)}")
        logger.info(f"Timeframes: {', '.join(self.config.TIMEFRAMES)}")
        logger.info(f"Telegram: {'✅ Enabled' if self.telegram_bot else '❌ Disabled'}")
        logger.info(f"{'='*80}\n")

        # Register callback for candle completion
        self.data_stream.register_candle_callback(self.on_candle_complete)

        # Start data stream
        await self.data_stream.start()

    async def stop(self):
        """Stop the scanner"""
        logger.info(f"\n{'='*80}")
        logger.info(f"Stopping scanner...")
        logger.info(f"Runtime: {(datetime.utcnow() - self.start_time).total_seconds():.0f}s")
        logger.info(f"Candles processed: {self.candles_processed}")
        logger.info(f"Signals sent: {self.signals_sent}")
        logger.info(f"{'='*80}\n")

        await self.data_stream.stop()

    async def print_status(self):
        """Print periodic status updates"""
        while True:
            await asyncio.sleep(300)  # Every 5 minutes

            runtime = (datetime.utcnow() - self.start_time).total_seconds()
            logger.info(f"\n📊 STATUS UPDATE")
            logger.info(f"Runtime: {runtime/60:.1f} minutes")
            logger.info(f"Candles processed: {self.candles_processed}")
            logger.info(f"Signals sent: {self.signals_sent}")
            logger.info(f"Processing rate: {self.candles_processed/runtime*60:.1f} candles/min\n")


async def main():
    """Main entry point"""

    # Load Telegram credentials (optional)
    telegram_bot = None
    try:
        token, chat_id = load_telegram_credentials()
        if token and chat_id:
            telegram_bot = TradingTelegramBot(token, chat_id, config)
            logger.info("✅ Telegram bot initialized")
        else:
            logger.warning("⚠️  Telegram credentials not found - running without notifications")
    except Exception as e:
        logger.warning(f"⚠️  Could not initialize Telegram bot: {e}")
        logger.warning("⚠️  Continuing without Telegram notifications")

    # Create scanner
    scanner = LiveTradingScanner(config, telegram_bot)

    # Send startup notification
    if telegram_bot:
        try:
            await telegram_bot.send_alert(
                title="🚀 Live Scanner Started",
                message=f"Monitoring {len(config.INSTRUMENTS)} instruments across {len(config.TIMEFRAMES)} timeframes.\n\nYou will receive signals when opportunities are detected.",
                level="info"
            )
        except Exception as e:
            logger.warning(f"Could not send startup notification: {e}")

    try:
        # Start scanner and status updates
        await asyncio.gather(
            scanner.start(),
            scanner.print_status()
        )
    except KeyboardInterrupt:
        logger.info("\n⚠️  Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Scanner error: {e}", exc_info=True)
    finally:
        await scanner.stop()

        # Send shutdown notification
        if telegram_bot:
            try:
                await telegram_bot.send_alert(
                    title="🛑 Live Scanner Stopped",
                    message=f"Scanner stopped after {scanner.signals_sent} signals sent.",
                    level="warning"
                )
            except Exception as e:
                logger.warning(f"Could not send shutdown notification: {e}")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("LIVE TRADING SIGNAL SCANNER")
    print("="*80)
    print("\nPress Ctrl+C to stop\n")
    print("="*80 + "\n")

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n✅ Scanner stopped by user\n")
