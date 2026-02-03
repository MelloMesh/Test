"""
Real-Time Data Stream for Kraken Futures
WebSocket connection for live price data across all timeframes
"""

import asyncio
import websockets
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Callable, Optional
from collections import deque
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class RealtimeDataStream:
    """
    Real-time data stream using Kraken Futures WebSocket API
    Maintains live candle data for all timeframes
    """

    def __init__(self, config):
        self.config = config
        self.ws_url = "wss://futures.kraken.com/ws/v1"

        # Store live candles for each timeframe
        self.candle_data: Dict[str, Dict[str, deque]] = {
            tf: {} for tf in config.TIMEFRAMES
        }

        # Callbacks for new candle completion
        self.candle_callbacks: List[Callable] = []

        # Connection state
        self.connected = False
        self.ws = None

        # Buffer for partial candles
        self.current_candles: Dict[str, Dict] = {}

    def add_callback(self, callback: Callable):
        """Add callback function to be called on new candle"""
        self.candle_callbacks.append(callback)

    def register_candle_callback(self, callback: Callable):
        """Alias for add_callback - register callback for new candles"""
        self.add_callback(callback)

    def _init_candle_buffer(self, instrument: str, timeframe: str, max_candles: int = 500):
        """Initialize buffer for storing candles"""
        if instrument not in self.candle_data[timeframe]:
            self.candle_data[timeframe][instrument] = deque(maxlen=max_candles)

    async def connect(self):
        """Establish WebSocket connection"""
        try:
            self.ws = await websockets.connect(self.ws_url)
            self.connected = True
            logger.info(f"✓ Connected to Kraken Futures WebSocket")
            return True
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            self.connected = False
            return False

    async def subscribe_tickers(self, instruments: List[str]):
        """
        Subscribe to ticker updates for instruments
        Note: Kraken uses XBT for Bitcoin, not BTC
        """
        # Convert instrument names to Kraken format
        kraken_instruments = []
        for inst in instruments:
            if inst == "BTCUSDT.P":
                kraken_instruments.append("PI_XBTUSD")  # Kraken perpetual inverse
            elif inst == "ETHUSDT.P":
                kraken_instruments.append("PI_ETHUSD")
            else:
                # Convert other instruments as needed
                kraken_instruments.append(inst)

        subscribe_message = {
            "event": "subscribe",
            "feed": "ticker",
            "product_ids": kraken_instruments
        }

        await self.ws.send(json.dumps(subscribe_message))
        logger.info(f"Subscribed to ticker for: {', '.join(kraken_instruments)}")

    async def subscribe_trades(self, instruments: List[str]):
        """Subscribe to trade feed for building candles"""
        kraken_instruments = []
        for inst in instruments:
            if inst == "BTCUSDT.P":
                kraken_instruments.append("PI_XBTUSD")
            elif inst == "ETHUSDT.P":
                kraken_instruments.append("PI_ETHUSD")

        subscribe_message = {
            "event": "subscribe",
            "feed": "trade",
            "product_ids": kraken_instruments
        }

        await self.ws.send(json.dumps(subscribe_message))
        logger.info(f"Subscribed to trades for: {', '.join(kraken_instruments)}")

    def _build_candle_from_trades(self, trades: List[Dict], timeframe: str) -> Optional[Dict]:
        """
        Build OHLC candle from list of trades

        Args:
            trades: List of trade dicts with 'price', 'time', 'qty'
            timeframe: Candle timeframe (2m, 5m, 15m, 30m)

        Returns:
            Candle dict or None
        """
        if not trades:
            return None

        timeframe_minutes = int(timeframe.replace('m', ''))

        # Get candle boundaries
        first_trade_time = datetime.fromtimestamp(trades[0]['time'] / 1000)
        candle_start = first_trade_time.replace(
            minute=(first_trade_time.minute // timeframe_minutes) * timeframe_minutes,
            second=0,
            microsecond=0
        )

        prices = [t['price'] for t in trades]
        volumes = [t['qty'] for t in trades]

        candle = {
            'timestamp': candle_start,
            'open': prices[0],
            'high': max(prices),
            'low': min(prices),
            'close': prices[-1],
            'volume': sum(volumes)
        }

        return candle

    async def _process_ticker(self, data: Dict):
        """Process ticker message"""
        if 'product_id' not in data:
            return

        instrument = data['product_id']

        # Convert back to our format
        if instrument == "PI_XBTUSD":
            instrument = "BTCUSDT.P"
        elif instrument == "PI_ETHUSD":
            instrument = "ETHUSDT.P"

        # Extract price data
        if 'last' in data:
            price = float(data['last'])
            timestamp = datetime.fromtimestamp(data.get('time', 0) / 1000)

            # Update current candles for all timeframes
            for timeframe in self.config.TIMEFRAMES:
                self._update_current_candle(instrument, timeframe, price, timestamp)

    def _update_current_candle(self, instrument: str, timeframe: str, price: float, timestamp: datetime):
        """Update the current (incomplete) candle for a timeframe"""
        timeframe_minutes = int(timeframe.replace('m', ''))

        # Determine candle start time
        candle_start = timestamp.replace(
            minute=(timestamp.minute // timeframe_minutes) * timeframe_minutes,
            second=0,
            microsecond=0
        )

        key = f"{instrument}_{timeframe}"

        # Check if we need a new candle
        if key not in self.current_candles or self.current_candles[key]['timestamp'] != candle_start:
            # Complete previous candle if exists
            if key in self.current_candles:
                completed_candle = self.current_candles[key]
                self._on_candle_complete(instrument, timeframe, completed_candle)

            # Start new candle
            self.current_candles[key] = {
                'timestamp': candle_start,
                'open': price,
                'high': price,
                'low': price,
                'close': price,
                'volume': 0
            }
        else:
            # Update existing candle
            candle = self.current_candles[key]
            candle['high'] = max(candle['high'], price)
            candle['low'] = min(candle['low'], price)
            candle['close'] = price

    def _on_candle_complete(self, instrument: str, timeframe: str, candle: Dict):
        """Called when a candle completes"""
        # Initialize buffer if needed
        self._init_candle_buffer(instrument, timeframe)

        # Add to buffer
        self.candle_data[timeframe][instrument].append(candle)

        logger.info(f"✓ Completed {instrument} {timeframe} candle: "
                   f"O={candle['open']:.2f} H={candle['high']:.2f} "
                   f"L={candle['low']:.2f} C={candle['close']:.2f}")

        # Call registered callbacks (handle both sync and async)
        for callback in self.candle_callbacks:
            try:
                result = callback(instrument, timeframe, candle)
                # If callback is async, schedule it
                if asyncio.iscoroutine(result):
                    asyncio.create_task(result)
            except Exception as e:
                logger.error(f"Error in callback: {e}")

    async def listen(self):
        """Main listening loop for WebSocket messages"""
        try:
            async for message in self.ws:
                data = json.loads(message)

                # Handle different message types
                if 'event' in data:
                    if data['event'] == 'subscribed':
                        logger.info(f"✓ Subscription confirmed: {data.get('feed')}")
                    elif data['event'] == 'error':
                        logger.error(f"WebSocket error: {data.get('message')}")

                # Handle ticker updates
                elif 'feed' in data and data['feed'] == 'ticker':
                    await self._process_ticker(data)

        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket connection closed")
            self.connected = False
        except Exception as e:
            logger.error(f"Error in listen loop: {e}")
            self.connected = False

    def get_current_data(self, instrument: str, timeframe: str, num_candles: int = 200) -> pd.DataFrame:
        """
        Get current candle data as DataFrame

        Args:
            instrument: Instrument symbol
            timeframe: Timeframe
            num_candles: Number of most recent candles

        Returns:
            DataFrame with OHLC data
        """
        if timeframe not in self.candle_data:
            return pd.DataFrame()

        if instrument not in self.candle_data[timeframe]:
            return pd.DataFrame()

        candles = list(self.candle_data[timeframe][instrument])

        if not candles:
            return pd.DataFrame()

        # Get last N candles
        recent_candles = candles[-num_candles:]

        df = pd.DataFrame(recent_candles)

        # Add technical indicators (from backtest engine)
        df = self._add_indicators(df)

        return df

    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to DataFrame"""
        if len(df) < 50:
            return df

        # Simple Moving Averages
        df['SMA_20'] = df['close'].rolling(window=20).mean()
        df['SMA_50'] = df['close'].rolling(window=50).mean()

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI_14'] = 100 - (100 / (1 + rs))

        # MACD
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema_12 - ema_26
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

        # ATR
        high = df['high']
        low = df['low']
        close = df['close']
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['ATR_14'] = tr.rolling(window=14).mean()

        return df

    def get_candles(self, instrument: str, timeframe: str, limit: int = 500) -> List[Dict]:
        """
        Get candle data for instrument and timeframe

        Args:
            instrument: Instrument symbol
            timeframe: Timeframe (e.g., '5m')
            limit: Maximum number of candles to return

        Returns:
            List of candle dictionaries
        """
        if timeframe not in self.candle_data:
            return []

        if instrument not in self.candle_data[timeframe]:
            return []

        candles = list(self.candle_data[timeframe][instrument])
        return candles[-limit:] if len(candles) > limit else candles

    async def start(self):
        """Start the data stream with configured instruments"""
        instruments = self.config.INSTRUMENTS
        await self.run(instruments)

    async def stop(self):
        """Stop the data stream and close WebSocket connection"""
        self.connected = False
        if self.ws:
            await self.ws.close()
            logger.info("WebSocket connection closed")

    async def run(self, instruments: List[str]):
        """
        Main run loop - connect, subscribe, and listen

        Args:
            instruments: List of instruments to track
        """
        # Connect
        connected = await self.connect()
        if not connected:
            return

        # Subscribe to data feeds
        await self.subscribe_tickers(instruments)

        # Listen for updates
        await self.listen()

    def save_snapshot(self, output_dir: str = "data/realtime"):
        """Save current data snapshot to disk"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for timeframe in self.config.TIMEFRAMES:
            for instrument, candles in self.candle_data[timeframe].items():
                if not candles:
                    continue

                df = pd.DataFrame(list(candles))
                filename = f"{instrument}_{timeframe}_{timestamp}.csv"
                filepath = output_path / filename
                df.to_csv(filepath, index=False)

        logger.info(f"Saved data snapshot to {output_dir}")


# Signal scanner that uses real-time data
class RealtimeSignalScanner:
    """
    Scans real-time data for trading signals
    Uses the live signal generator with streaming data
    """

    def __init__(self, config, signal_generator):
        self.config = config
        self.signal_generator = signal_generator
        self.data_stream = RealtimeDataStream(config)

        # Register callback for new candles
        self.data_stream.add_callback(self.on_new_candle)

    def on_new_candle(self, instrument: str, timeframe: str, candle: Dict):
        """Called when a new candle completes"""
        logger.info(f"🔔 New {instrument} {timeframe} candle completed")

        # Get current data for this timeframe
        current_data = self.data_stream.get_current_data(instrument, timeframe)

        if current_data.empty or len(current_data) < 50:
            return

        # Scan for signals
        try:
            data_dict = {timeframe: current_data}
            new_signals = self.signal_generator.scan_for_signals(
                current_data=data_dict,
                account_balance=10000.0  # Would get from account API
            )

            if new_signals:
                logger.info(f"🚨 Generated {len(new_signals)} new signals!")
                for signal in new_signals:
                    self.signal_generator.print_signal_summary(signal)

        except Exception as e:
            logger.error(f"Error scanning for signals: {e}")

    async def run(self, instruments: List[str]):
        """Start real-time signal scanning"""
        logger.info("Starting real-time signal scanner...")
        await self.data_stream.run(instruments)


async def main():
    """Example usage"""
    import sys
    sys.path.append("../..")
    import config
    from src.trading.live_signal_generator import LiveSignalGenerator

    # Initialize signal generator
    signal_generator = LiveSignalGenerator(config)

    # Initialize scanner
    scanner = RealtimeSignalScanner(config, signal_generator)

    # Run with Bitcoin and Ethereum
    instruments = ["BTCUSDT.P", "ETHUSDT.P"]

    print(f"\n{'='*80}")
    print("REAL-TIME SIGNAL SCANNER")
    print(f"{'='*80}")
    print(f"\nMonitoring: {', '.join(instruments)}")
    print(f"Timeframes: {', '.join(config.TIMEFRAMES)}")
    print(f"Scanning for signals every time a candle completes...")
    print(f"\nPress Ctrl+C to stop\n")

    try:
        await scanner.run(instruments)
    except KeyboardInterrupt:
        print("\n\nStopping scanner...")
        scanner.data_stream.save_snapshot()
        print("✓ Data snapshot saved")


if __name__ == "__main__":
    asyncio.run(main())
