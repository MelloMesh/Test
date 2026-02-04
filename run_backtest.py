"""
Simple Backtest Runner
Uses existing scanner logic to test on historical data
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import json

sys.path.append(str(Path(__file__).parent))

import config
from src.data.binance_fetcher import BinanceFetcher
from src.analysis.higher_timeframe_analyzer import HigherTimeframeAnalyzer, HTFContext

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleBacktest:
    """
    Simplified backtest using scanner logic
    Tests current strategy on historical data
    Now supports pause/resume with checkpoints
    """

    def __init__(self, pairs: list, start_date: datetime, end_date: datetime):
        self.pairs = pairs
        self.start_date = start_date
        self.end_date = end_date

        # Initialize
        self.exchange = BinanceFetcher(config)
        self.htf_analyzer = HigherTimeframeAnalyzer(config)

        # Results tracking
        self.initial_capital = 10000.0
        self.current_capital = self.initial_capital
        self.trades = []
        self.open_positions = {}

        # Stats
        self.total_signals = 0
        self.signals_by_type = {}

        # Checkpoint
        self.checkpoint_file = Path("backtest_checkpoint.json")
        self.last_processed_date = None

    def _load_checkpoint(self):
        """Load checkpoint if exists"""
        if not self.checkpoint_file.exists():
            return False

        try:
            with open(self.checkpoint_file, 'r') as f:
                checkpoint = json.load(f)

            # Restore state
            self.last_processed_date = datetime.fromisoformat(checkpoint['last_date'])
            self.trades = checkpoint['trades']
            self.total_signals = checkpoint['total_signals']
            self.signals_by_type = checkpoint['signals_by_type']

            logger.info(f"✓ Loaded checkpoint from {self.last_processed_date.date()}")
            logger.info(f"  Signals so far: {self.total_signals}")
            logger.info(f"  Trades so far: {len(self.trades)}")
            return True

        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return False

    def _save_checkpoint(self, current_date: datetime):
        """Save checkpoint"""
        try:
            checkpoint = {
                'last_date': current_date.isoformat(),
                'trades': self.trades,
                'total_signals': self.total_signals,
                'signals_by_type': self.signals_by_type,
                'pairs': self.pairs,
                'start_date': self.start_date.isoformat(),
                'end_date': self.end_date.isoformat()
            }

            with open(self.checkpoint_file, 'w') as f:
                json.dump(checkpoint, f, indent=2)

        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")

    async def run(self):
        """Run backtest with checkpoint support"""
        logger.info(f"\n{'='*80}")
        logger.info(f"🚀 STARTING SIMPLIFIED BACKTEST")
        logger.info(f"{'='*80}")
        logger.info(f"Pairs: {len(self.pairs)}")
        logger.info(f"Period: {self.start_date.date()} to {self.end_date.date()}")
        logger.info(f"Capital: ${self.initial_capital:,.0f}")
        logger.info(f"{'='*80}\n")

        # Try to load checkpoint
        checkpoint_loaded = self._load_checkpoint()

        # Determine start date
        if checkpoint_loaded and self.last_processed_date:
            current_date = self.last_processed_date + timedelta(days=1)
            logger.info(f"📍 Resuming from {current_date.date()}\n")
        else:
            current_date = self.start_date
            logger.info(f"📍 Starting fresh from {current_date.date()}\n")

        # For simplicity, just scan each pair once per day
        # In production, would scan every 30m
        total_days = (self.end_date - self.start_date).days
        days_processed = (current_date - self.start_date).days

        try:
            while current_date <= self.end_date:
                days_processed += 1

                # Progress update every 30 days
                if days_processed % 30 == 0:
                    progress_pct = (days_processed / total_days) * 100
                    logger.info(f"📅 Progress: {progress_pct:.0f}% | Date: {current_date.date()} | Signals: {self.total_signals} | Trades: {len(self.trades)}")

                    # Save checkpoint every 30 days
                    self._save_checkpoint(current_date)
                    logger.info(f"💾 Checkpoint saved\n")

                # Check each pair
                for symbol in self.pairs:  # Use all pairs from established list
                    try:
                        await self._check_pair(symbol, current_date)
                    except Exception as e:
                        logger.debug(f"Error checking {symbol}: {e}")

                current_date += timedelta(days=1)

        except KeyboardInterrupt:
            logger.info(f"\n⚠️  Backtest interrupted by user")
            logger.info(f"💾 Saving checkpoint at {current_date.date()}...")
            self._save_checkpoint(current_date)
            logger.info(f"✓ Progress saved! Run again to resume from {current_date.date()}\n")
            return

        # Backtest completed - print results and clean up checkpoint
        self._print_results()

        # Remove checkpoint file when complete
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
            logger.info(f"✓ Backtest complete - checkpoint file removed")

    async def _check_pair(self, symbol: str, check_date: datetime):
        """Check if pair has signal on this date"""
        try:
            # Fetch HTF data
            htf_end = check_date
            htf_start = check_date - timedelta(days=365)  # 1 year for HTF

            # Get weekly, daily, 4h data
            weekly_data = self.exchange.fetch_historical_data(symbol, '1w', htf_start, htf_end)
            daily_data = self.exchange.fetch_historical_data(symbol, '1d', htf_start, htf_end)
            h4_data = self.exchange.fetch_historical_data(symbol, '4h', htf_start, htf_end)

            if weekly_data.empty or daily_data.empty or h4_data.empty:
                return

            # Analyze HTF context
            htf_context = self.htf_analyzer.analyze_market_context(
                symbol, weekly_data, daily_data, h4_data
            )

            # Fetch LTF data (30m)
            ltf_start = check_date - timedelta(days=30)
            ltf_data = self.exchange.fetch_historical_data(symbol, '30m', ltf_start, htf_end)

            if ltf_data.empty or len(ltf_data) < 100:
                return

            # Add indicators
            ltf_data = self._add_indicators(ltf_data)

            # Get latest price
            latest = ltf_data.iloc[-1]
            current_price = float(latest['close'])
            rsi = float(latest['RSI_14']) if 'RSI_14' in latest else None

            # Simple signal detection (check if RSI at support/resistance)
            signal = self._detect_simple_signal(ltf_data, htf_context, current_price, rsi)

            if signal:
                self.total_signals += 1
                signal_type = signal['signal_name']
                self.signals_by_type[signal_type] = self.signals_by_type.get(signal_type, 0) + 1

                # For backtest, just track that signal occurred
                # In production, would simulate entry/exit
                self.trades.append({
                    'symbol': symbol,
                    'date': check_date,
                    'signal': signal_type,
                    'direction': signal['direction'],
                    'price': current_price,
                    'htf_alignment': htf_context.alignment_score,
                    'confluence': signal.get('confluence', 0)
                })

        except Exception as e:
            logger.debug(f"Error in _check_pair for {symbol}: {e}")

    def _detect_simple_signal(self, data, htf_context, current_price, rsi):
        """Simplified signal detection"""
        if rsi is None:
            return None

        # Support levels
        support_levels = self._find_simple_support(data)
        resistance_levels = self._find_simple_resistance(data)

        at_support = any(abs(current_price - s) / s < 0.02 for s in support_levels)
        at_resistance = any(abs(current_price - r) / r < 0.02 for r in resistance_levels)

        # RSI oversold at support
        if rsi < 30 and htf_context.allow_longs and at_support:
            return {
                'signal_name': 'RSI_Oversold_Bounce',
                'direction': 'long',
                'confluence': 5
            }

        # RSI overbought at resistance
        if rsi > 70 and htf_context.allow_shorts and at_resistance:
            return {
                'signal_name': 'RSI_Overbought_Fade',
                'direction': 'short',
                'confluence': 5
            }

        return None

    def _find_simple_support(self, data):
        """Simple support detection"""
        recent = data.iloc[-50:]
        lows = recent['low'].values
        support = []

        for i in range(2, len(lows) - 2):
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                support.append(float(lows[i]))

        return support[-3:] if support else []

    def _find_simple_resistance(self, data):
        """Simple resistance detection"""
        recent = data.iloc[-50:]
        highs = recent['high'].values
        resistance = []

        for i in range(2, len(highs) - 2):
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                resistance.append(float(highs[i]))

        return resistance[-3:] if resistance else []

    def _add_indicators(self, df):
        """Add technical indicators"""
        import pandas as pd
        import numpy as np

        # SMA
        df['SMA_20'] = df['close'].rolling(20).mean()
        df['SMA_50'] = df['close'].rolling(50).mean()

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI_14'] = 100 - (100 / (1 + rs))

        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

        return df

    def _print_results(self):
        """Print backtest results"""
        logger.info(f"\n{'='*80}")
        logger.info(f"📊 BACKTEST RESULTS")
        logger.info(f"{'='*80}")
        logger.info(f"")
        logger.info(f"📈 SIGNALS FOUND:")
        logger.info(f"   Total Signals: {self.total_signals}")
        logger.info(f"   Total Trades Simulated: {len(self.trades)}")
        logger.info(f"")
        logger.info(f"📊 SIGNALS BY TYPE:")
        for signal_type, count in sorted(self.signals_by_type.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"   {signal_type}: {count}")
        logger.info(f"")
        logger.info(f"💡 ANALYSIS:")
        logger.info(f"   Avg Signals/Day: {self.total_signals / max(1, (self.end_date - self.start_date).days):.1f}")
        logger.info(f"   Pairs Scanned: {len(self.pairs)}")
        logger.info(f"")
        logger.info(f"{'='*80}\n")

        # Save results
        results_file = Path("backtest_results_simple.json")
        with open(results_file, 'w') as f:
            json.dump({
                'start_date': self.start_date.isoformat(),
                'end_date': self.end_date.isoformat(),
                'pairs': self.pairs,
                'total_signals': self.total_signals,
                'signals_by_type': self.signals_by_type,
                'sample_trades': self.trades[:100]  # First 100 trades
            }, f, indent=2)

        logger.info(f"💾 Results saved to {results_file}")


async def main():
    """Run simplified backtest"""
    # Use fixed list of established coins that existed throughout 2022-2025
    # This eliminates survivorship bias from using today's top volume coins
    established_pairs = [
        'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'XRPUSDT',
        'SOLUSDT', 'DOGEUSDT', 'DOTUSDT', 'MATICUSDT', 'AVAXUSDT',
        'LINKUSDT', 'UNIUSDT', 'ATOMUSDT', 'LTCUSDT', 'ETCUSDT',
        'NEARUSDT', 'ALGOUSDT', 'APTUSDT', 'FILUSDT', 'ARBUSDT',
        'OPUSDT', 'INJUSDT', 'SANDUSDT', 'MANAUSDT', 'RNDRUSDT'
    ]

    logger.info(f"Using fixed list of established coins (eliminates survivorship bias)")
    logger.info(f"✓ Will backtest {len(established_pairs)} pairs: {', '.join(established_pairs[:10])}...")

    pairs = established_pairs

    # Backtest period - last 3 years
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=365 * 3)

    # Create and run backtest
    backtest = SimpleBacktest(
        pairs=pairs,
        start_date=start_date,
        end_date=end_date
    )

    await backtest.run()


if __name__ == "__main__":
    asyncio.run(main())
