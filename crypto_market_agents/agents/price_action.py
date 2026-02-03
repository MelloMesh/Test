"""
Price Action Agent - Monitors price movements and detects breakouts.
"""

import asyncio
from datetime import datetime, timezone
from typing import List, Dict, Any
import statistics

from .base_agent import BaseAgent
from ..exchange.base import BaseExchange
from ..schemas import PriceActionSignal
from ..config import PriceActionConfig


class PriceActionAgent(BaseAgent):
    """
    Agent that monitors price action across all tradable assets.

    Detects:
    - Breakout patterns
    - Rapid percentage changes
    - Abnormal volatility
    """

    def __init__(
        self,
        exchange: BaseExchange,
        config: PriceActionConfig
    ):
        """
        Initialize Price Action Agent.

        Args:
            exchange: Exchange adapter
            config: Price action configuration
        """
        super().__init__(
            name="PriceAction",
            exchange=exchange,
            update_interval=config.update_interval
        )
        self.config = config
        self.symbols: List[str] = []
        self.price_history: Dict[str, List[float]] = {}

    async def execute(self):
        """Execute price action analysis."""
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
                signal = await self._analyze_ticker(ticker)
                if signal:
                    signals.append(signal)
            except Exception as e:
                self.logger.error(f"Error analyzing {ticker.get('symbol')}: {e}")

        # Update signals
        self.latest_signals = signals
        self.signals_generated += len(signals)

        if signals:
            self.logger.info(
                f"Generated {len(signals)} price action signals. "
                f"Breakouts detected: {sum(1 for s in signals if s.breakout_detected)}"
            )

    async def _analyze_ticker(self, ticker: Dict[str, Any]) -> PriceActionSignal:
        """
        Analyze a single ticker for price action signals.

        Args:
            ticker: Ticker data

        Returns:
            Price action signal or None
        """
        symbol = ticker["symbol"]
        price = ticker["last_price"]
        high_24h = ticker["high_24h"]
        low_24h = ticker["low_24h"]
        price_change_pct = ticker["price_change_24h_pct"]

        # Track price history
        if symbol not in self.price_history:
            self.price_history[symbol] = []

        self.price_history[symbol].append(price)

        # Keep last 100 prices
        if len(self.price_history[symbol]) > 100:
            self.price_history[symbol].pop(0)

        # Calculate intraday range
        if low_24h > 0:
            intraday_range_pct = ((high_24h - low_24h) / low_24h) * 100
        else:
            intraday_range_pct = 0

        # Calculate volatility ratio
        volatility_ratio = 1.0
        if len(self.price_history[symbol]) >= 20:
            recent_prices = self.price_history[symbol][-20:]
            if len(recent_prices) > 1:
                volatility = statistics.stdev(recent_prices)
                mean_price = statistics.mean(recent_prices)
                if mean_price > 0:
                    current_volatility = abs(price - recent_prices[-2]) / recent_prices[-2]
                    avg_volatility = volatility / mean_price
                    if avg_volatility > 0:
                        volatility_ratio = current_volatility / avg_volatility

        # Detect breakout
        breakout_detected = (
            abs(price_change_pct) >= self.config.breakout_threshold * 100
            or volatility_ratio >= self.config.volatility_threshold
        )

        return PriceActionSignal(
            symbol=symbol,
            price=price,
            price_change_pct=price_change_pct,
            intraday_range_pct=intraday_range_pct,
            volatility_ratio=volatility_ratio,
            breakout_detected=breakout_detected,
            timeframe="24h",
            timestamp=datetime.now(timezone.utc)
        )
