"""
Momentum Agent - Computes RSI and OBV indicators.
"""

import asyncio
from datetime import datetime, timezone
from typing import List, Dict, Any
import statistics

from .base_agent import BaseAgent
from ..exchange.base import BaseExchange
from ..schemas import MomentumSignal
from ..config import MomentumConfig


class MomentumAgent(BaseAgent):
    """
    Agent that computes momentum indicators (RSI and OBV).

    Identifies:
    - Overbought conditions (RSI > 70)
    - Oversold conditions (RSI < 30)
    - On-Balance Volume trends
    """

    def __init__(
        self,
        exchange: BaseExchange,
        config: MomentumConfig
    ):
        """
        Initialize Momentum Agent.

        Args:
            exchange: Exchange adapter
            config: Momentum configuration
        """
        super().__init__(
            name="Momentum",
            exchange=exchange,
            update_interval=config.update_interval
        )
        self.config = config
        self.symbols: List[str] = []
        self.candle_cache: Dict[str, Dict[str, List[Dict]]] = {}

    async def execute(self):
        """Execute momentum analysis."""
        if not self.symbols:
            self.symbols = await self.exchange.get_trading_symbols()
            self.logger.info(f"Monitoring {len(self.symbols)} symbols")

        if not self.symbols:
            self.logger.warning("No symbols to monitor")
            return

        # Analyze symbols in batches to avoid overwhelming the API
        batch_size = 50
        all_signals = []

        for i in range(0, len(self.symbols), batch_size):
            batch = self.symbols[i:i + batch_size]
            tasks = [self._analyze_symbol(symbol) for symbol in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    self.logger.error(f"Error in momentum analysis: {result}")
                elif result:
                    all_signals.extend(result)

        # Update signals
        self.latest_signals = all_signals
        self.signals_generated += len(all_signals)

        # Log statistics
        overbought = sum(1 for s in all_signals if s.status == "overbought")
        oversold = sum(1 for s in all_signals if s.status == "oversold")

        if all_signals:
            self.logger.info(
                f"Generated {len(all_signals)} momentum signals. "
                f"Overbought: {overbought}, Oversold: {oversold}"
            )

    async def _analyze_symbol(self, symbol: str) -> List[MomentumSignal]:
        """
        Analyze a single symbol for momentum indicators.

        Args:
            symbol: Trading symbol

        Returns:
            List of momentum signals for different timeframes
        """
        signals = []

        for timeframe in self.config.timeframes:
            try:
                signal = await self._compute_momentum(symbol, timeframe)
                if signal:
                    signals.append(signal)
            except Exception as e:
                self.logger.error(f"Error analyzing {symbol} on {timeframe}m: {e}")

        return signals

    async def _compute_momentum(self, symbol: str, timeframe: str) -> MomentumSignal:
        """
        Compute RSI and OBV for a symbol and timeframe.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe in minutes

        Returns:
            Momentum signal
        """
        # Get candlestick data
        limit = max(self.config.rsi_period * 2, self.config.obv_lookback, 100)
        candles = await self.exchange.get_klines(symbol, timeframe, limit=limit)

        if not candles or len(candles) < self.config.rsi_period + 1:
            return None

        # Compute RSI
        rsi = self._calculate_rsi(candles, self.config.rsi_period)

        # Compute OBV
        obv, obv_change_pct = self._calculate_obv(candles, self.config.obv_lookback)

        # Determine status
        if rsi >= self.config.rsi_overbought:
            status = "overbought"
            strength_score = min(100, (rsi - 50) * 2)
        elif rsi <= self.config.rsi_oversold:
            status = "oversold"
            strength_score = min(100, (50 - rsi) * 2)
        else:
            status = "neutral"
            strength_score = abs(rsi - 50) * 2

        return MomentumSignal(
            symbol=symbol,
            rsi=rsi,
            obv=obv,
            obv_change_pct=obv_change_pct,
            status=status,
            strength_score=strength_score,
            timeframe=f"{timeframe}m",
            timestamp=datetime.now(timezone.utc)
        )

    def _calculate_rsi(self, candles: List[Dict], period: int) -> float:
        """
        Calculate RSI (Relative Strength Index).

        Args:
            candles: List of candlestick data
            period: RSI period

        Returns:
            RSI value (0-100)
        """
        if len(candles) < period + 1:
            return 50.0

        # Calculate price changes
        closes = [c["close"] for c in candles]
        changes = [closes[i] - closes[i - 1] for i in range(1, len(closes))]

        # Separate gains and losses
        gains = [max(0, change) for change in changes[-period:]]
        losses = [abs(min(0, change)) for change in changes[-period:]]

        # Calculate average gain and loss
        avg_gain = sum(gains) / period if gains else 0
        avg_loss = sum(losses) / period if losses else 0

        # Calculate RSI
        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return round(rsi, 2)

    def _calculate_obv(self, candles: List[Dict], lookback: int) -> tuple:
        """
        Calculate OBV (On-Balance Volume) and its change.

        Args:
            candles: List of candlestick data
            lookback: Lookback period for change calculation

        Returns:
            Tuple of (current OBV, OBV change percentage)
        """
        if len(candles) < 2:
            return 0.0, 0.0

        obv = 0.0
        obv_values = [0.0]

        for i in range(1, len(candles)):
            volume = candles[i]["volume"]
            close_change = candles[i]["close"] - candles[i - 1]["close"]

            if close_change > 0:
                obv += volume
            elif close_change < 0:
                obv -= volume

            obv_values.append(obv)

        # Calculate OBV change
        obv_change_pct = 0.0
        if len(obv_values) > lookback:
            old_obv = obv_values[-lookback]
            current_obv = obv_values[-1]

            if old_obv != 0:
                obv_change_pct = ((current_obv - old_obv) / abs(old_obv)) * 100

        return round(obv, 2), round(obv_change_pct, 2)
