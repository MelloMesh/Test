"""
HTF-Aware Live Scanner for Bybit (200+ USDT Perpetuals)
Scans with Higher Timeframe context filtering
Only generates signals aligned with W/D/4H bias
"""

import asyncio
import logging
from pathlib import Path
import sys
from datetime import datetime, timedelta, timezone
import pandas as pd
from typing import Dict, List, Optional
import json

# Add project to path
sys.path.append(str(Path(__file__).parent))

import config
from src.data.bybit_fetcher import BybitFetcher
from src.data.binance_fetcher import BinanceFetcher
from src.analysis.higher_timeframe_analyzer import HigherTimeframeAnalyzer, HTFContext
from src.signals.signal_discovery_htf import HTFAwareSignalDiscovery
from src.trading.telegram_bot import TradingTelegramBot, load_telegram_credentials
from src.signals.signal_deduplication import get_deduplicator
from src.signals.confluence_scorer import get_confluence_scorer
from src.signals.limit_order_manager import get_limit_manager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HTFLiveScannerBybit:
    """
    Multi-instrument live scanner with HTF filtering for Bybit
    Scans 100+ USDT perpetuals every 5 minutes
    """

    def __init__(self, config, telegram_bot=None, top_n_pairs: int = 100):
        """
        Initialize HTF live scanner

        Args:
            config: Trading config
            telegram_bot: Optional Telegram bot for notifications
            top_n_pairs: Number of top pairs to scan (default 100)
        """
        self.config = config
        self.telegram_bot = telegram_bot
        self.top_n_pairs = top_n_pairs

        # Initialize components - use Binance (more reliable API)
        self.exchange = BinanceFetcher(config)
        self.exchange_name = "Binance"
        self.htf_analyzer = HigherTimeframeAnalyzer(config)
        self.signal_discovery = HTFAwareSignalDiscovery(config)

        # Initialize intelligent systems
        self.deduplicator = get_deduplicator()
        self.confluence_scorer = get_confluence_scorer()
        self.limit_manager = get_limit_manager()

        # HTF context cache
        self.htf_cache: Dict[str, HTFContext] = {}
        self.htf_cache_time: Optional[datetime] = None
        self.htf_cache_duration_hours = 4  # Refresh every 4 hours

        # LTF data cache (refreshed more frequently)
        self.ltf_cache: Dict[str, Dict[str, pd.DataFrame]] = {}

        # Trading pairs to scan
        self.active_pairs: List[str] = []

        # Performance tracking
        self.scans_completed = 0
        self.signals_found = 0
        self.signals_sent = 0
        self.signals_market = 0
        self.signals_limit = 0
        self.signals_skipped = 0
        self.htf_blocked_count = 0
        self.start_time = datetime.now(timezone.utc)

    async def initialize(self):
        """Initialize scanner - fetch instruments and HTF data"""
        logger.info(f"Initializing HTF Live Scanner for {self.exchange_name}...")

        # Get top trading pairs by volume
        logger.info(f"Fetching top {self.top_n_pairs} pairs by 24h volume...")
        self.active_pairs = self.exchange.get_top_volume_pairs(top_n=self.top_n_pairs)

        if not self.active_pairs:
            logger.error("Failed to fetch trading pairs")
            return False

        logger.info(f"✓ Will scan {len(self.active_pairs)} pairs")
        logger.info(f"Top 10: {', '.join(self.active_pairs[:10])}")

        # Fetch initial HTF data
        await self.refresh_htf_data()

        logger.info("✓ Scanner initialized successfully")
        return True

    async def refresh_htf_data(self):
        """Refresh HTF context for all pairs (cached for 4 hours)"""
        logger.info("Refreshing HTF data for all pairs...")

        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=180)  # 6 months history

        # Fetch HTF data for all pairs
        htf_timeframes = self.config.HTF_TIMEFRAMES  # ['1w', '1d', '4h']

        for i, symbol in enumerate(self.active_pairs, 1):
            try:
                logger.info(f"[{i}/{len(self.active_pairs)}] Fetching HTF for {symbol}...")

                # Fetch each HTF timeframe
                weekly_data = self.exchange.fetch_historical_data(
                    symbol, '1w', start_date, end_date
                )
                daily_data = self.exchange.fetch_historical_data(
                    symbol, '1d', start_date, end_date
                )
                h4_data = self.exchange.fetch_historical_data(
                    symbol, '4h', start_date, end_date
                )

                # Add technical indicators
                weekly_data = self._add_indicators(weekly_data)
                daily_data = self._add_indicators(daily_data)
                h4_data = self._add_indicators(h4_data)

                # Analyze HTF context
                if not weekly_data.empty and not daily_data.empty and not h4_data.empty:
                    context = self.htf_analyzer.analyze_market_context(
                        instrument=symbol,
                        weekly_data=weekly_data,
                        daily_data=daily_data,
                        h4_data=h4_data
                    )

                    self.htf_cache[symbol] = context

                # Rate limiting
                await asyncio.sleep(0.15)  # ~6-7 pairs/second

            except Exception as e:
                logger.error(f"Error fetching HTF for {symbol}: {e}")
                continue

        self.htf_cache_time = datetime.now(timezone.utc)
        logger.info(f"✓ HTF data refreshed for {len(self.htf_cache)} pairs")

    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to DataFrame"""
        if len(df) < 50:
            return df

        # Moving Averages
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

        # ADX
        plus_dm = high.diff()
        minus_dm = low.diff().abs()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        tr_smooth = tr.rolling(window=14).sum()
        plus_di = 100 * (plus_dm.rolling(window=14).sum() / tr_smooth)
        minus_di = 100 * (minus_dm.rolling(window=14).sum() / tr_smooth)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        df['ADX_14'] = dx.rolling(window=14).mean()

        return df

    async def scan_ltf_signals(self):
        """Scan all pairs for LTF entry signals"""
        logger.info(f"\n{'='*80}")
        logger.info(f"Starting LTF signal scan ({len(self.active_pairs)} pairs)...")
        logger.info(f"{'='*80}")

        scan_start = datetime.now(timezone.utc)
        signals_this_scan = 0

        # Get LTF signals for HTF-approved pairs
        for symbol in self.active_pairs:
            # Check if HTF context exists
            if symbol not in self.htf_cache:
                continue

            htf_context = self.htf_cache[symbol]

            # Skip if HTF doesn't allow any trading
            if not htf_context.allow_longs and not htf_context.allow_shorts:
                self.htf_blocked_count += 1
                logger.debug(f"Skipped {symbol}: No HTF permission")
                continue

            # Skip if choppy regime
            if htf_context.regime == 'choppy':
                self.htf_blocked_count += 1
                logger.debug(f"Skipped {symbol}: Choppy regime")
                continue

            # Fetch latest LTF data
            ltf_signal = await self._check_ltf_entry(symbol, htf_context)

            if ltf_signal:
                signals_this_scan += 1
                self.signals_found += 1

                # Send to Telegram
                await self._send_htf_signal(symbol, ltf_signal, htf_context)

        scan_duration = (datetime.now(timezone.utc) - scan_start).total_seconds()
        self.scans_completed += 1

        logger.info(f"\n{'='*80}")
        logger.info(f"Scan complete in {scan_duration:.1f}s")
        logger.info(f"Signals found: {signals_this_scan}")
        logger.info(f"HTF blocked: {self.htf_blocked_count}")
        logger.info(f"{'='*80}\n")

    async def _check_ltf_entry(self, symbol: str, htf_context: HTFContext) -> Optional[Dict]:
        """
        Check for LTF entry signal with confluence scoring

        Returns:
            Signal dict with order type if found, None otherwise
        """
        try:
            # Fetch latest LTF data (30m for now)
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=7)

            ltf_data = self.exchange.fetch_historical_data(
                symbol, '30m', start_date, end_date
            )

            if ltf_data.empty or len(ltf_data) < 100:
                return None

            # Add indicators
            ltf_data = self._add_indicators(ltf_data)

            # Get latest price action
            latest = ltf_data.iloc[-1]
            current_price = float(latest['close'])
            rsi = float(latest['RSI_14']) if 'RSI_14' in latest else None

            # Detect signals
            signals = self._detect_ltf_signals(ltf_data, htf_context)

            if not signals:
                return None

            # Evaluate each signal with confluence scorer
            best_signal = None
            best_score = 0

            for signal in signals:
                # Check deduplication first
                should_alert = self.deduplicator.should_alert(
                    symbol=symbol,
                    signal_name=signal['signal_name'],
                    direction=signal['direction'],
                    current_price=current_price,
                    htf_alignment=htf_context.alignment_score,
                    htf_bias=htf_context.primary_bias
                )

                if not should_alert:
                    continue  # Skip duplicate

                # Evaluate confluence
                confluence = self.confluence_scorer.evaluate_signal(
                    symbol=symbol,
                    signal_name=signal['signal_name'],
                    direction=signal['direction'],
                    current_price=current_price,
                    htf_context=htf_context,
                    has_divergence=signal.get('has_divergence', False),
                    has_volume_confirmation=signal.get('has_volume', False),
                    multi_tf_confirmation=signal.get('multi_tf', False),
                    rsi_value=rsi
                )

                # Skip low-quality signals
                if confluence.order_type == 'SKIP':
                    self.signals_skipped += 1
                    continue

                # Track best signal
                if confluence.total_score > best_score:
                    best_score = confluence.total_score
                    best_signal = {
                        **signal,
                        'entry_price': current_price,
                        'confluence_score': confluence.total_score,
                        'order_type': confluence.order_type,
                        'trade_type': confluence.trade_type,
                        'position_size_multiplier': confluence.position_size_multiplier,
                        'limit_price': confluence.limit_price,
                        'reasoning': confluence.reasoning
                    }

            # Return best signal found
            if best_signal:
                # Track order types
                if best_signal['order_type'] == 'MARKET':
                    self.signals_market += 1
                elif best_signal['order_type'] == 'LIMIT':
                    self.signals_limit += 1

                    # Add to limit manager
                    self.limit_manager.add_limit_order(
                        symbol=symbol,
                        signal_name=best_signal['signal_name'],
                        direction=best_signal['direction'],
                        limit_price=best_signal['limit_price'],
                        current_price=current_price,
                        confluence_score=best_signal['confluence_score'],
                        trade_type=best_signal['trade_type'],
                        htf_bias=htf_context.primary_bias,
                        htf_alignment=htf_context.alignment_score
                    )

                return best_signal

            return None

        except Exception as e:
            logger.error(f"Error checking LTF for {symbol}: {e}")
            return None

    def _detect_ltf_signals(self, data: pd.DataFrame, htf_context: HTFContext) -> List[Dict]:
        """
        Detect all possible LTF signals with proper support/resistance checks

        Returns list of potential signals for evaluation
        """
        signals = []
        latest = data.iloc[-1]
        current_price = float(latest['close'])

        # Find support and resistance levels
        support_levels = self._find_support_levels(data)
        resistance_levels = self._find_resistance_levels(data)

        # Add HTF support/resistance if available
        if htf_context.daily_support:
            support_levels.append(htf_context.daily_support)
        if htf_context.h4_support:
            support_levels.append(htf_context.h4_support)
        if htf_context.daily_resistance:
            resistance_levels.append(htf_context.daily_resistance)
        if htf_context.h4_resistance:
            resistance_levels.append(htf_context.h4_resistance)

        # Check if at key levels
        at_support = self._is_near_support(current_price, support_levels)
        at_resistance = self._is_near_resistance(current_price, resistance_levels)

        # Check for trend-following pullback
        if htf_context.allow_longs and self._is_bullish_pullback(data):
            signals.append({
                'direction': 'long',
                'signal_name': 'HTF_Bullish_Pullback',
                'timeframe': '30m',
                'reason': 'Pullback to 20-EMA in bullish HTF',
                'has_divergence': False,
                'has_volume': False,
                'multi_tf': False
            })

        if htf_context.allow_shorts and self._is_bearish_rejection(data):
            signals.append({
                'direction': 'short',
                'signal_name': 'HTF_Bearish_Rejection',
                'timeframe': '30m',
                'reason': 'Rally to 20-EMA in bearish HTF',
                'has_divergence': False,
                'has_volume': False,
                'multi_tf': False
            })

        # Check for mean reversion - CRITICAL FIX: Only signal if at correct level
        rsi = latest['RSI_14'] if 'RSI_14' in latest else None

        if rsi is not None:
            # Oversold bounce - ONLY if price is at support
            if rsi < 30 and htf_context.allow_longs and at_support:
                signals.append({
                    'direction': 'long',
                    'signal_name': 'RSI_Oversold_Bounce',
                    'timeframe': '30m',
                    'reason': f'RSI oversold ({rsi:.0f}) AT SUPPORT - bounce setup',
                    'has_divergence': self._check_bullish_divergence(data),
                    'has_volume': self._check_volume_spike(data),
                    'multi_tf': False
                })

            # Overbought fade - ONLY if price is at resistance
            if rsi > 70 and htf_context.allow_shorts and at_resistance:
                signals.append({
                    'direction': 'short',
                    'signal_name': 'RSI_Overbought_Fade',
                    'timeframe': '30m',
                    'reason': f'RSI overbought ({rsi:.0f}) AT RESISTANCE - rejection setup',
                    'has_divergence': self._check_bearish_divergence(data),
                    'has_volume': self._check_volume_spike(data),
                    'multi_tf': False
                })

        return signals

    def _check_bullish_divergence(self, data: pd.DataFrame) -> bool:
        """Check for bullish RSI divergence"""
        if len(data) < 20 or 'RSI_14' not in data.columns:
            return False

        recent = data.iloc[-20:]
        price_low = recent['low'].min()
        rsi_low = recent['RSI_14'].min()

        # Simple check: is latest low lower in price but higher in RSI?
        if recent['low'].iloc[-1] <= price_low * 1.01:  # At or near lowest price
            if recent['RSI_14'].iloc[-1] > rsi_low * 1.05:  # But RSI higher
                return True
        return False

    def _check_bearish_divergence(self, data: pd.DataFrame) -> bool:
        """Check for bearish RSI divergence"""
        if len(data) < 20 or 'RSI_14' not in data.columns:
            return False

        recent = data.iloc[-20:]
        price_high = recent['high'].max()
        rsi_high = recent['RSI_14'].max()

        # Simple check: is latest high higher in price but lower in RSI?
        if recent['high'].iloc[-1] >= price_high * 0.99:  # At or near highest price
            if recent['RSI_14'].iloc[-1] < rsi_high * 0.95:  # But RSI lower
                return True
        return False

    def _check_volume_spike(self, data: pd.DataFrame) -> bool:
        """Check for volume confirmation"""
        if len(data) < 20 or 'volume' not in data.columns:
            return False

        avg_volume = data['volume'].iloc[-20:-1].mean()
        latest_volume = data['volume'].iloc[-1]

        return latest_volume > avg_volume * 1.5  # 50% above average

    def _find_support_levels(self, data: pd.DataFrame, lookback: int = 50) -> List[float]:
        """Find support levels from recent swing lows"""
        if len(data) < lookback:
            return []

        recent_data = data.iloc[-lookback:]
        support_levels = []

        # Find local lows (swing lows)
        for i in range(2, len(recent_data) - 2):
            current_low = recent_data['low'].iloc[i]
            prev_low = recent_data['low'].iloc[i-1]
            prev_prev_low = recent_data['low'].iloc[i-2]
            next_low = recent_data['low'].iloc[i+1]
            next_next_low = recent_data['low'].iloc[i+2]

            # Is this a swing low?
            if (current_low < prev_low and current_low < prev_prev_low and
                current_low < next_low and current_low < next_next_low):
                support_levels.append(current_low)

        # Also add recent lows
        support_levels.append(recent_data['low'].min())

        return support_levels

    def _find_resistance_levels(self, data: pd.DataFrame, lookback: int = 50) -> List[float]:
        """Find resistance levels from recent swing highs"""
        if len(data) < lookback:
            return []

        recent_data = data.iloc[-lookback:]
        resistance_levels = []

        # Find local highs (swing highs)
        for i in range(2, len(recent_data) - 2):
            current_high = recent_data['high'].iloc[i]
            prev_high = recent_data['high'].iloc[i-1]
            prev_prev_high = recent_data['high'].iloc[i-2]
            next_high = recent_data['high'].iloc[i+1]
            next_next_high = recent_data['high'].iloc[i+2]

            # Is this a swing high?
            if (current_high > prev_high and current_high > prev_prev_high and
                current_high > next_high and current_high > next_next_high):
                resistance_levels.append(current_high)

        # Also add recent highs
        resistance_levels.append(recent_data['high'].max())

        return resistance_levels

    def _is_near_support(self, price: float, support_levels: List[float], tolerance_pct: float = 0.015) -> bool:
        """Check if price is near any support level (within tolerance)"""
        if not support_levels:
            return False

        for support in support_levels:
            distance = abs(price - support) / support
            if distance <= tolerance_pct:  # Within 1.5% of support
                return True
        return False

    def _is_near_resistance(self, price: float, resistance_levels: List[float], tolerance_pct: float = 0.015) -> bool:
        """Check if price is near any resistance level (within tolerance)"""
        if not resistance_levels:
            return False

        for resistance in resistance_levels:
            distance = abs(price - resistance) / resistance
            if distance <= tolerance_pct:  # Within 1.5% of resistance
                return True
        return False

    def _is_bullish_pullback(self, data: pd.DataFrame) -> bool:
        """Detect bullish pullback pattern"""
        if len(data) < 20:
            return False

        latest = data.iloc[-1]

        # Price near 20-EMA
        if 'SMA_20' not in data.columns:
            return False

        sma_20 = latest['SMA_20']
        price = latest['close']

        # Within 1% of EMA
        if abs(price - sma_20) / sma_20 > 0.01:
            return False

        # RSI in pullback zone (40-55)
        if 'RSI_14' not in data.columns:
            return False

        rsi = latest['RSI_14']
        if not (40 <= rsi <= 55):
            return False

        # MACD still positive (trend intact)
        if 'MACD' not in data.columns:
            return False

        macd = latest['MACD']
        if macd <= 0:
            return False

        return True

    def _is_bearish_rejection(self, data: pd.DataFrame) -> bool:
        """Detect bearish rejection pattern"""
        if len(data) < 20:
            return False

        latest = data.iloc[-1]

        # Price near 20-EMA
        if 'SMA_20' not in data.columns:
            return False

        sma_20 = latest['SMA_20']
        price = latest['close']

        # Within 1% of EMA
        if abs(price - sma_20) / sma_20 > 0.01:
            return False

        # RSI in rejection zone (45-60)
        if 'RSI_14' not in data.columns:
            return False

        rsi = latest['RSI_14']
        if not (45 <= rsi <= 60):
            return False

        # MACD still negative (downtrend intact)
        if 'MACD' not in data.columns:
            return False

        macd = latest['MACD']
        if macd >= 0:
            return False

        return True

    def _generate_signal_analysis(self, signal: Dict, htf_context: HTFContext) -> str:
        """Generate AI analysis for a signal"""
        try:
            analysis_parts = []

            # Analyze trade type
            if "TREND_CONTINUATION" in signal['trade_type']:
                analysis_parts.append(f"Trend continuation setup on {htf_context.primary_bias} HTF bias.")
            elif "MEAN_REVERSION" in signal['trade_type']:
                analysis_parts.append(f"Mean reversion play from {htf_context.regime} market conditions.")

            # Analyze confluence
            score = signal['confluence_score']
            if score >= 8:
                analysis_parts.append("Exceptional confluence with multiple confirmations.")
            elif score >= 7:
                analysis_parts.append("Strong confluence - high-probability setup.")
            elif score >= 5:
                analysis_parts.append("Decent confluence - standard setup.")
            else:
                analysis_parts.append("Lower confluence - reduced position sizing.")

            # Analyze HTF alignment
            if htf_context.alignment_score >= 80:
                analysis_parts.append(f"Perfect HTF alignment ({htf_context.alignment_score:.0f}%) - all timeframes agree.")
            elif htf_context.alignment_score >= 60:
                analysis_parts.append(f"Good HTF alignment ({htf_context.alignment_score:.0f}%) - majority of timeframes aligned.")
            else:
                analysis_parts.append(f"Mixed HTF signals ({htf_context.alignment_score:.0f}%) - proceed with caution.")

            # Add trade rationale based on direction and HTF
            if signal['direction'] == 'long':
                if htf_context.primary_bias == 'bullish':
                    analysis_parts.append("Long entry with HTF support - trading with institutional flow.")
                else:
                    analysis_parts.append("Counter-trend long - looking for short-term bounce.")
            else:
                if htf_context.primary_bias == 'bearish':
                    analysis_parts.append("Short entry with HTF support - trading with institutional flow.")
                else:
                    analysis_parts.append("Counter-trend short - looking for short-term pullback.")

            return " ".join(analysis_parts)
        except Exception as e:
            logger.error(f"Error generating signal analysis: {e}")
            return "Setup meets entry criteria with acceptable risk/reward profile."

    async def _send_htf_signal(self, symbol: str, signal: Dict, htf_context: HTFContext):
        """Send signal with HTF context and confluence info to Telegram"""
        self.signals_sent += 1

        # Calculate entry price (limit or market)
        entry_price = signal['limit_price'] if signal['order_type'] == 'LIMIT' else signal['entry_price']

        # Calculate stop loss and take profit (2% risk, 4% reward)
        risk_pct = 0.02
        reward_pct = 0.04

        if signal['direction'] == 'long':
            stop_loss = entry_price * (1 - risk_pct)
            take_profit = entry_price * (1 + reward_pct)
        else:
            stop_loss = entry_price * (1 + risk_pct)
            take_profit = entry_price * (1 - reward_pct)

        # Calculate percentages for display
        if signal['direction'] == 'long':
            sl_pct = ((stop_loss - entry_price) / entry_price) * 100
            tp_pct = ((take_profit - entry_price) / entry_price) * 100
        else:
            sl_pct = ((entry_price - stop_loss) / entry_price) * 100
            tp_pct = ((entry_price - take_profit) / entry_price) * 100

        risk = abs(sl_pct)
        reward = abs(tp_pct)
        rr_ratio = reward / risk if risk > 0 else 0

        # Calculate position sizing (example with $100 base)
        base_position = 100.0
        position_size = base_position * signal['position_size_multiplier']
        max_risk = position_size * (risk / 100)
        potential_profit = position_size * (reward / 100)

        # Log signal
        order_type_emoji = "🚀" if signal['order_type'] == 'MARKET' else "📋"
        logger.info(f"\n{'='*80}")
        logger.info(f"{order_type_emoji} {signal['order_type']} SIGNAL: {symbol}")
        logger.info(f"{'='*80}")
        logger.info(f"Signal: {signal['signal_name']}")
        logger.info(f"Direction: {signal['direction'].upper()}")
        logger.info(f"Trade Type: {signal['trade_type']}")
        logger.info(f"Confluence Score: {signal['confluence_score']}/10")
        if signal['order_type'] == 'MARKET':
            logger.info(f"Entry: ${signal['entry_price']:,.2f}")
        else:
            logger.info(f"Current Price: ${signal['entry_price']:,.2f}")
            logger.info(f"Limit Price: ${signal['limit_price']:,.2f}")
        logger.info(f"Stop Loss: ${stop_loss:,.2f} ({sl_pct:.2f}%)")
        logger.info(f"Take Profit: ${take_profit:,.2f} ({tp_pct:+.2f}%)")
        logger.info(f"R:R Ratio: 1:{rr_ratio:.2f}")
        logger.info(f"Position Size: {signal['position_size_multiplier']:.2f}x")
        logger.info(f"HTF Bias: {htf_context.primary_bias.upper()} ({htf_context.bias_strength:.0f}%)")
        logger.info(f"HTF Alignment: {htf_context.alignment_score:.0f}%")
        logger.info(f"Reasoning: {signal['reasoning']}")
        logger.info(f"{'='*80}\n")

        # Send to Telegram if configured
        if self.telegram_bot:
            try:
                direction_emoji = "🟢" if signal['direction'] == 'long' else "🔴"
                order_type_emoji = "🚀" if signal['order_type'] == 'MARKET' else "📋"

                # Build entry display
                if signal['order_type'] == 'MARKET':
                    entry_display = f"${entry_price:,.4f}"
                    title = f"🎯 Signal: {symbol} {signal['direction'].upper()}"
                else:
                    current_vs_limit_pct = ((signal['limit_price'] - signal['entry_price']) / signal['entry_price']) * 100
                    entry_display = f"${signal['limit_price']:,.4f} (Limit at {current_vs_limit_pct:+.2f}% from current)"
                    title = f"📋 Limit Signal: {symbol} {signal['direction'].upper()}"

                # Confluence bar (visual score)
                score = signal['confluence_score']
                filled_bars = "█" * score
                empty_bars = "░" * (10 - score)
                confluence_bar = filled_bars + empty_bars

                # Generate AI analysis
                ai_analysis = self._generate_signal_analysis(signal, htf_context)

                message = f"""
{direction_emoji} **{signal['order_type']} {signal['direction'].upper()} SIGNAL** {order_type_emoji}

📊 **SETUP: {symbol}**
   Type: {signal['trade_type']}
   Confluence: {confluence_bar} {score}/10
   HTF Bias: {htf_context.primary_bias.upper()} ({htf_context.alignment_score:.0f}%)

💡 **ANALYSIS:**
   {ai_analysis}

📍 **ENTRY:**
   Price: {entry_display}

🛡️ **STOP LOSS:**
   Price: ${stop_loss:,.4f}
   Risk: {sl_pct:.2f}%

🎯 **TAKE PROFIT:**
   Price: ${take_profit:,.4f}
   Target: {tp_pct:+.2f}%

⚖️ **RISK/REWARD:**
   Risk: {risk:.2f}%
   Reward: {reward:.2f}%
   R:R Ratio: 1:{rr_ratio:.2f}

💰 **POSITION GUIDANCE:**
   Base Size: ${position_size:.2f} ({signal['position_size_multiplier']:.1f}x)
   Max Risk: ${max_risk:.2f}
   Potential Profit: ${potential_profit:.2f}

📊 **HTF CONTEXT:**
   Weekly: {htf_context.weekly_trend.upper()}
   Daily: {htf_context.daily_trend.upper()}
   4H: {htf_context.h4_trend.upper()}
   Regime: {htf_context.regime.upper()}
"""

                await self.telegram_bot.send_alert(
                    title=title,
                    message=message,
                    level="info"
                )

                logger.info("✅ Signal sent to Telegram")

            except Exception as e:
                logger.error(f"Failed to send Telegram notification: {e}")

    async def run(self):
        """Main scanner loop"""
        logger.info(f"\n{'='*80}")
        logger.info(f"🚀 HTF LIVE SCANNER RUNNING")
        logger.info(f"{'='*80}")
        logger.info(f"Pairs: {len(self.active_pairs)}")
        logger.info(f"HTF Cache: {len(self.htf_cache)} pairs")
        logger.info(f"Scan Interval: 5 minutes")
        logger.info(f"Telegram: {'✅ Enabled' if self.telegram_bot else '❌ Disabled'}")
        logger.info(f"{'='*80}\n")

        # Send startup notification
        if self.telegram_bot:
            try:
                await self.telegram_bot.send_alert(
                    title="🚀 HTF Live Scanner Started",
                    message=f"Monitoring {len(self.active_pairs)} {self.exchange_name} USDT perpetuals with HTF filtering.\n\nScan interval: 5 minutes\nHTF timeframes: W/D/4H\nLTF execution: 30m/15m/5m",
                    level="info"
                )
            except:
                pass

        try:
            while True:
                # Check if HTF cache needs refresh
                if self.htf_cache_time is None or \
                   (datetime.now(timezone.utc) - self.htf_cache_time).total_seconds() > self.htf_cache_duration_hours * 3600:
                    logger.info("HTF cache expired - refreshing...")
                    await self.refresh_htf_data()

                # Scan for LTF signals
                await self.scan_ltf_signals()

                # Print status
                self._print_status()

                # Wait 5 minutes
                logger.info("Waiting 5 minutes until next scan...")
                await asyncio.sleep(300)

        except KeyboardInterrupt:
            logger.info("\n⚠️  Stopping scanner...")
        except Exception as e:
            logger.error(f"Scanner error: {e}", exc_info=True)

    def _print_status(self):
        """Print scanner status"""
        runtime = (datetime.now(timezone.utc) - self.start_time).total_seconds() / 60

        logger.info(f"\n📊 SCANNER STATUS")
        logger.info(f"Runtime: {runtime:.1f} minutes")
        logger.info(f"Scans completed: {self.scans_completed}")
        logger.info(f"Signals found: {self.signals_found}")
        logger.info(f"Signals sent: {self.signals_sent}")
        logger.info(f"HTF blocked: {self.htf_blocked_count}")
        logger.info(f"")


async def main():
    """Main entry point"""

    # Load Telegram credentials
    telegram_bot = None
    try:
        token, chat_id = load_telegram_credentials()
        if token and chat_id:
            telegram_bot = TradingTelegramBot(token, chat_id, config)
            logger.info("✅ Telegram bot initialized")
        else:
            logger.warning("⚠️  Telegram not configured")
    except Exception as e:
        logger.warning(f"⚠️  Telegram initialization failed: {e}")

    # Create scanner
    scanner = HTFLiveScannerBybit(
        config,
        telegram_bot=telegram_bot,
        top_n_pairs=200  # Scan top 200 pairs by volume
    )

    # Initialize
    success = await scanner.initialize()
    if not success:
        logger.error("Failed to initialize scanner")
        return

    # Run scanner
    await scanner.run()


if __name__ == "__main__":
    print("\n" + "="*80)
    print("HTF LIVE SCANNER - BINANCE 200+ USDT PERPETUALS")
    print("="*80)
    print("\nPress Ctrl+C to stop\n")
    print("="*80 + "\n")

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n✅ Scanner stopped by user\n")
