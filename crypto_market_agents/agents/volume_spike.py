"""
Volume Spike Agent - Monitors volume changes and detects whale activity.
"""

import asyncio
from datetime import datetime, timezone
from typing import List, Dict, Any
import statistics

from .base_agent import BaseAgent
from ..exchange.base import BaseExchange
from ..schemas import VolumeSignal
from ..config import VolumeConfig


class VolumeSpikeAgent(BaseAgent):
    """
    Agent that monitors volume spikes across tradable assets.

    Detects:
    - Statistically significant volume spikes
    - Potential whale accumulation/distribution
    - High-liquidity trading opportunities
    """

    def __init__(
        self,
        exchange: BaseExchange,
        config: VolumeConfig
    ):
        """
        Initialize Volume Spike Agent.

        Args:
            exchange: Exchange adapter
            config: Volume configuration
        """
        super().__init__(
            name="VolumeSpike",
            exchange=exchange,
            update_interval=config.update_interval
        )
        self.config = config
        self.symbols: List[str] = []
        self.volume_history: Dict[str, List[float]] = {}

    async def execute(self):
        """Execute volume spike analysis."""
        if not self.symbols:
            self.symbols = await self.exchange.get_trading_symbols()
            self.logger.info(f"Monitoring {len(self.symbols)} symbols")

        if not self.symbols:
            self.logger.warning("No symbols to monitor")
            return

        # Get tickers for all symbols
        tickers = await self.exchange.get_tickers(self.symbols)

        signals = []
        for ticker in tickers:
            try:
                # Filter by minimum liquidity
                volume_usd = ticker["last_price"] * ticker["volume_24h"]
                if volume_usd < self.config.min_liquidity_usd:
                    continue

                signal = await self._analyze_volume(ticker, volume_usd)
                if signal:
                    signals.append(signal)

            except Exception as e:
                self.logger.error(f"Error analyzing {ticker.get('symbol')}: {e}")

        # Update signals
        self.latest_signals = signals
        self.signals_generated += len(signals)

        # Log statistics
        spikes = sum(1 for s in signals if s.spike_detected)

        if signals:
            self.logger.info(
                f"Generated {len(signals)} volume signals. "
                f"Spikes detected: {spikes}"
            )

    async def _analyze_volume(
        self,
        ticker: Dict[str, Any],
        liquidity_usd: float
    ) -> VolumeSignal:
        """
        Analyze volume for a single ticker.

        Args:
            ticker: Ticker data
            liquidity_usd: Liquidity in USD

        Returns:
            Volume signal or None
        """
        symbol = ticker["symbol"]
        volume_24h = ticker["volume_24h"]

        # Track volume history
        if symbol not in self.volume_history:
            self.volume_history[symbol] = []

        self.volume_history[symbol].append(volume_24h)

        # Keep history limited
        if len(self.volume_history[symbol]) > self.config.lookback_periods:
            self.volume_history[symbol].pop(0)

        # Need enough history for statistical analysis
        if len(self.volume_history[symbol]) < 10:
            return None

        # Calculate baseline and statistics
        history = self.volume_history[symbol]
        baseline_volume = statistics.mean(history[:-1]) if len(history) > 1 else history[0]
        volume_change_pct = ((volume_24h - baseline_volume) / baseline_volume * 100) if baseline_volume > 0 else 0

        # Calculate z-score
        volume_zscore = 0.0
        spike_detected = False

        if len(history) >= 20:
            mean_vol = statistics.mean(history)
            stdev_vol = statistics.stdev(history)

            if stdev_vol > 0:
                volume_zscore = (volume_24h - mean_vol) / stdev_vol

            # Detect spike using z-score or percentile
            if abs(volume_zscore) >= self.config.spike_threshold_zscore:
                spike_detected = True
            else:
                # Check percentile threshold
                sorted_history = sorted(history)
                percentile_idx = int(len(sorted_history) * self.config.spike_threshold_percentile / 100)
                percentile_value = sorted_history[min(percentile_idx, len(sorted_history) - 1)]

                if volume_24h >= percentile_value:
                    spike_detected = True

        return VolumeSignal(
            symbol=symbol,
            volume_24h=volume_24h,
            volume_change_pct=volume_change_pct,
            volume_zscore=volume_zscore,
            spike_detected=spike_detected,
            baseline_volume=baseline_volume,
            liquidity_usd=liquidity_usd,
            timestamp=datetime.now(timezone.utc)
        )
