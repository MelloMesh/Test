"""
HTF-Aware Backtesting Engine
Tests signals with Higher Timeframe context filtering
Only executes trades that align with W/D/4H bias
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
import json
from pathlib import Path

from src.backtest.backtest_engine import BacktestEngine, Trade, BacktestResult
from src.analysis.higher_timeframe_analyzer import HigherTimeframeAnalyzer, HTFContext

logger = logging.getLogger(__name__)


@dataclass
class HTFAwareTrade(Trade):
    """Trade with HTF context information"""
    # HTF Context at entry
    htf_primary_bias: str  # bullish, bearish, neutral
    htf_bias_strength: float  # 0-100
    htf_alignment_score: float  # 0-100
    htf_regime: str  # trending, ranging, choppy
    htf_aligned: bool  # True if signal matched HTF bias


@dataclass
class HTFAwareBacktestResult(BacktestResult):
    """Backtest results with HTF analysis"""
    # HTF Filtering Stats
    htf_aligned_trades: int
    htf_misaligned_trades_skipped: int
    htf_aligned_win_rate: float

    # HTF Regime Performance
    trending_win_rate: float
    ranging_win_rate: float
    choppy_win_rate: float


class HTFAwareBacktestEngine(BacktestEngine):
    """
    Backtesting engine with HTF context filtering
    Simulates real trading where signals are filtered by HTF bias
    """

    def __init__(self, config):
        super().__init__(config)
        self.htf_analyzer = HigherTimeframeAnalyzer(config)

        # HTF data cache
        self.htf_data_cache: Dict[str, Dict[str, pd.DataFrame]] = {}

    def load_htf_data(self, instrument: str, data_dir: str = "data/raw") -> Dict[str, pd.DataFrame]:
        """
        Load Higher Timeframe data (W, D, 4H)

        Returns:
            Dict with keys '1w', '1d', '4h' -> DataFrames
        """
        cache_key = f"{instrument}_htf"
        if cache_key in self.htf_data_cache:
            return self.htf_data_cache[cache_key]

        htf_data = {}

        # For now, use available timeframes as proxies
        # In production with Bybit/Binance, we'll fetch real W/D/4H data

        # Weekly proxy: Use 30m data aggregated (TODO: fetch real weekly)
        try:
            weekly_proxy = self.load_data(instrument, "30m", data_dir)
            if not weekly_proxy.empty:
                # Resample to weekly for now
                weekly_data = weekly_proxy.resample('W').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()

                # Add indicators
                weekly_data = self._add_technical_indicators(weekly_data)
                htf_data['1w'] = weekly_data
                logger.info(f"Loaded weekly proxy data: {len(weekly_data)} candles")
        except Exception as e:
            logger.warning(f"Could not load weekly data: {e}")
            htf_data['1w'] = pd.DataFrame()

        # Daily proxy: Use 15m data aggregated
        try:
            daily_proxy = self.load_data(instrument, "15m", data_dir)
            if not daily_proxy.empty:
                daily_data = daily_proxy.resample('D').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()

                daily_data = self._add_technical_indicators(daily_data)
                htf_data['1d'] = daily_data
                logger.info(f"Loaded daily proxy data: {len(daily_data)} candles")
        except Exception as e:
            logger.warning(f"Could not load daily data: {e}")
            htf_data['1d'] = pd.DataFrame()

        # 4H proxy: Use 5m data aggregated
        try:
            h4_proxy = self.load_data(instrument, "5m", data_dir)
            if not h4_proxy.empty:
                h4_data = h4_proxy.resample('4H').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()

                h4_data = self._add_technical_indicators(h4_data)
                htf_data['4h'] = h4_data
                logger.info(f"Loaded 4H proxy data: {len(h4_data)} candles")
        except Exception as e:
            logger.warning(f"Could not load 4H data: {e}")
            htf_data['4h'] = pd.DataFrame()

        self.htf_data_cache[cache_key] = htf_data
        return htf_data

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators required for HTF analysis"""
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

        # ADX for trend strength
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

    def get_htf_context_at_time(
        self,
        instrument: str,
        timestamp: datetime,
        htf_data: Dict[str, pd.DataFrame]
    ) -> Optional[HTFContext]:
        """
        Get HTF context at a specific point in time

        Args:
            instrument: Trading instrument
            timestamp: Time to analyze
            htf_data: Dict of HTF dataframes

        Returns:
            HTFContext or None if insufficient data
        """
        try:
            # Get HTF data up to this timestamp
            weekly_data = htf_data['1w'][htf_data['1w'].index <= timestamp].tail(100)
            daily_data = htf_data['1d'][htf_data['1d'].index <= timestamp].tail(100)
            h4_data = htf_data['4h'][htf_data['4h'].index <= timestamp].tail(100)

            # Need sufficient data for analysis
            if len(weekly_data) < 20 or len(daily_data) < 20 or len(h4_data) < 20:
                return None

            # Analyze HTF context
            context = self.htf_analyzer.analyze_market_context(
                instrument=instrument,
                weekly_data=weekly_data,
                daily_data=daily_data,
                h4_data=h4_data
            )

            return context

        except Exception as e:
            logger.error(f"Error getting HTF context: {e}")
            return None

    def backtest_signal_with_htf(
        self,
        signal_hypothesis: Dict,
        instrument: str,
        data: pd.DataFrame,
        htf_data: Dict[str, pd.DataFrame],
        funding_rates: Optional[pd.DataFrame] = None
    ) -> HTFAwareBacktestResult:
        """
        Backtest a signal with HTF filtering

        Only executes trades when signal direction aligns with HTF bias

        Args:
            signal_hypothesis: Signal definition with direction and HTF requirements
            instrument: Trading instrument
            data: LTF price data
            htf_data: HTF data (W, D, 4H)
            funding_rates: Funding rate data

        Returns:
            HTFAwareBacktestResult with HTF analysis
        """
        logger.info(f"Backtesting {signal_hypothesis['name']} with HTF filtering")

        # Get signal requirements
        signal_direction = signal_hypothesis.get('direction', 'long')
        htf_required_bias = signal_hypothesis.get('htf_required_bias', 'any')
        htf_min_alignment = signal_hypothesis.get('htf_min_alignment', 50.0)
        htf_min_strength = signal_hypothesis.get('htf_min_strength', 40.0)

        trades: List[HTFAwareTrade] = []
        htf_aligned_count = 0
        htf_misaligned_skipped = 0

        # Scan through data for entry signals
        for i in range(100, len(data) - 50):  # Need history and forward data
            current_time = data.index[i]
            current_price = data['close'].iloc[i]

            # Check if entry conditions are met (simplified)
            if not self._check_entry_conditions(signal_hypothesis, data.iloc[max(0, i-50):i+1]):
                continue

            # Get HTF context at this time
            htf_context = self.get_htf_context_at_time(instrument, current_time, htf_data)

            if htf_context is None:
                continue

            # Check HTF alignment
            aligned, reason = self.htf_analyzer.check_signal_alignment(signal_direction, htf_context)

            # Skip trade if not aligned (this is the key filter!)
            if not aligned:
                htf_misaligned_skipped += 1
                logger.debug(f"Skipped trade at {current_time}: {reason}")
                continue

            # Check minimum alignment and strength requirements
            if htf_context.alignment_score < htf_min_alignment:
                htf_misaligned_skipped += 1
                continue

            if htf_context.bias_strength < htf_min_strength:
                htf_misaligned_skipped += 1
                continue

            # HTF approved - execute trade
            htf_aligned_count += 1

            # Calculate stop loss and target
            atr = data['ATR_14'].iloc[i] if 'ATR_14' in data.columns else current_price * 0.02

            if signal_direction == 'long':
                stop_loss = current_price - (self.config.STOP_LOSS_ATR_MULTIPLIER * atr)
                target = current_price + (self.config.TARGET_ATR_MULTIPLIER * atr)
            else:  # short
                stop_loss = current_price + (self.config.STOP_LOSS_ATR_MULTIPLIER * atr)
                target = current_price - (self.config.TARGET_ATR_MULTIPLIER * atr)

            # Simulate trade execution
            exit_idx, exit_price, exit_reason = self._simulate_trade_exit(
                data.iloc[i:i+200],
                signal_direction,
                current_price,
                stop_loss,
                target
            )

            if exit_idx is None:
                continue

            # Calculate P&L
            if signal_direction == 'long':
                pnl_pct = (exit_price - current_price) / current_price
            else:
                pnl_pct = (current_price - exit_price) / current_price

            pnl_usd = pnl_pct * self.config.STARTING_CAPITAL_USDT * self.config.POSITION_SIZE_PCT

            # Create HTF-aware trade record
            trade = HTFAwareTrade(
                entry_time=current_time,
                exit_time=data.index[i + exit_idx],
                instrument=instrument,
                timeframe=signal_hypothesis['timeframe'],
                signal_id=signal_hypothesis['id'],
                signal_name=signal_hypothesis['name'],
                direction=signal_direction,
                entry_price=current_price,
                exit_price=exit_price,
                stop_loss=stop_loss,
                target=target,
                position_size=self.config.STARTING_CAPITAL_USDT * self.config.POSITION_SIZE_PCT,
                pnl_pips=(exit_price - current_price) / current_price * 10000,
                pnl_pct=pnl_pct,
                pnl_usd=pnl_usd,
                exit_reason=exit_reason,
                hold_duration_minutes=(data.index[i + exit_idx] - current_time).total_seconds() / 60,
                slippage_cost=signal_hypothesis.get('slippage_pips', 1.0),
                funding_cost=0.0,  # Simplified
                total_cost=signal_hypothesis.get('slippage_pips', 1.0),
                regime=htf_context.regime,
                # HTF fields
                htf_primary_bias=htf_context.primary_bias,
                htf_bias_strength=htf_context.bias_strength,
                htf_alignment_score=htf_context.alignment_score,
                htf_regime=htf_context.regime,
                htf_aligned=True  # Only aligned trades are executed
            )

            trades.append(trade)

        # Calculate results
        if not trades:
            logger.warning(f"No HTF-aligned trades found for {signal_hypothesis['name']}")
            return self._empty_htf_result(signal_hypothesis, instrument, htf_misaligned_skipped)

        return self._calculate_htf_results(signal_hypothesis, instrument, trades, htf_misaligned_skipped)

    def _check_entry_conditions(self, signal: Dict, data: pd.DataFrame) -> bool:
        """Simplified entry condition check"""
        # Randomly trigger ~5% of the time to simulate signal frequency
        return np.random.random() < 0.05

    def _simulate_trade_exit(
        self,
        future_data: pd.DataFrame,
        direction: str,
        entry_price: float,
        stop_loss: float,
        target: float
    ) -> Tuple[Optional[int], Optional[float], str]:
        """Simulate how trade would exit"""
        for i in range(len(future_data)):
            high = future_data['high'].iloc[i]
            low = future_data['low'].iloc[i]

            if direction == 'long':
                if low <= stop_loss:
                    return i, stop_loss, 'stop_hit'
                if high >= target:
                    return i, target, 'target_hit'
            else:  # short
                if high >= stop_loss:
                    return i, stop_loss, 'stop_hit'
                if low <= target:
                    return i, target, 'target_hit'

        # Timeout
        return len(future_data) - 1, future_data['close'].iloc[-1], 'timeout'

    def _calculate_htf_results(
        self,
        signal: Dict,
        instrument: str,
        trades: List[HTFAwareTrade],
        skipped_count: int
    ) -> HTFAwareBacktestResult:
        """Calculate backtest results with HTF stats"""
        wins = [t for t in trades if t.pnl_usd > 0]
        losses = [t for t in trades if t.pnl_usd <= 0]

        win_rate = len(wins) / len(trades) if trades else 0

        total_pnl = sum(t.pnl_usd for t in trades)
        avg_win = np.mean([t.pnl_usd for t in wins]) if wins else 0
        avg_loss = abs(np.mean([t.pnl_usd for t in losses])) if losses else 0

        profit_factor = abs(sum(t.pnl_usd for t in wins) / sum(t.pnl_usd for t in losses)) if losses else 0

        # HTF regime performance
        trending_trades = [t for t in trades if t.htf_regime == 'trending']
        ranging_trades = [t for t in trades if t.htf_regime == 'ranging']
        choppy_trades = [t for t in trades if t.htf_regime == 'choppy']

        trending_wr = len([t for t in trending_trades if t.pnl_usd > 0]) / len(trending_trades) if trending_trades else 0
        ranging_wr = len([t for t in ranging_trades if t.pnl_usd > 0]) / len(ranging_trades) if ranging_trades else 0
        choppy_wr = len([t for t in choppy_trades if t.pnl_usd > 0]) / len(choppy_trades) if choppy_trades else 0

        return HTFAwareBacktestResult(
            signal_id=signal['id'],
            signal_name=signal['name'],
            timeframe=signal['timeframe'],
            instrument=instrument,
            total_trades=len(trades),
            winning_trades=len(wins),
            losing_trades=len(losses),
            win_rate=win_rate,
            total_pnl_pips=sum(t.pnl_pips for t in trades),
            total_pnl_usd=total_pnl,
            avg_win_pips=np.mean([t.pnl_pips for t in wins]) if wins else 0,
            avg_loss_pips=abs(np.mean([t.pnl_pips for t in losses])) if losses else 0,
            largest_win_pips=max([t.pnl_pips for t in wins]) if wins else 0,
            largest_loss_pips=min([t.pnl_pips for t in losses]) if losses else 0,
            profit_factor=profit_factor,
            max_drawdown_pct=0.0,  # Simplified
            max_drawdown_usd=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            avg_hold_duration_minutes=np.mean([t.hold_duration_minutes for t in trades]),
            total_slippage_cost=sum(t.slippage_cost for t in trades),
            total_funding_cost=0.0,
            total_transaction_cost=sum(t.total_cost for t in trades),
            win_rate_by_regime={},
            trades_by_regime={},
            multi_tf_confirmed_trades=0,
            multi_tf_win_rate=0.0,
            trades=trades,
            # HTF stats
            htf_aligned_trades=len(trades),
            htf_misaligned_trades_skipped=skipped_count,
            htf_aligned_win_rate=win_rate,
            trending_win_rate=trending_wr,
            ranging_win_rate=ranging_wr,
            choppy_win_rate=choppy_wr
        )

    def _empty_htf_result(self, signal: Dict, instrument: str, skipped: int) -> HTFAwareBacktestResult:
        """Return empty result when no trades"""
        return HTFAwareBacktestResult(
            signal_id=signal['id'],
            signal_name=signal['name'],
            timeframe=signal['timeframe'],
            instrument=instrument,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            total_pnl_pips=0.0,
            total_pnl_usd=0.0,
            avg_win_pips=0.0,
            avg_loss_pips=0.0,
            largest_win_pips=0.0,
            largest_loss_pips=0.0,
            profit_factor=0.0,
            max_drawdown_pct=0.0,
            max_drawdown_usd=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            avg_hold_duration_minutes=0.0,
            total_slippage_cost=0.0,
            total_funding_cost=0.0,
            total_transaction_cost=0.0,
            win_rate_by_regime={},
            trades_by_regime={},
            multi_tf_confirmed_trades=0,
            multi_tf_win_rate=0.0,
            trades=[],
            htf_aligned_trades=0,
            htf_misaligned_trades_skipped=skipped,
            htf_aligned_win_rate=0.0,
            trending_win_rate=0.0,
            ranging_win_rate=0.0,
            choppy_win_rate=0.0
        )


def main():
    """Test HTF-aware backtesting"""
    import sys
    sys.path.append("../..")
    import config
    from src.signals.signal_discovery_htf import HTFAwareSignalDiscovery

    engine = HTFAwareBacktestEngine(config)
    signal_discovery = HTFAwareSignalDiscovery(config)

    # Generate HTF-aware signals
    signals = signal_discovery.generate_all_htf_aware_signals()

    # Load data
    instrument = "BTCUSDT.P"

    print(f"\nLoading data for {instrument}...")
    htf_data = engine.load_htf_data(instrument)
    ltf_data = engine.load_data(instrument, "30m")

    if ltf_data.empty:
        print("No LTF data available")
        return

    # Test backtest with first signal
    test_signal = asdict(signals[0]) if signals else None

    if test_signal:
        print(f"\nBacktesting: {test_signal['name']}")
        result = engine.backtest_signal_with_htf(
            test_signal,
            instrument,
            ltf_data,
            htf_data
        )

        print(f"\nResults:")
        print(f"  HTF-Aligned Trades: {result.htf_aligned_trades}")
        print(f"  Misaligned Skipped: {result.htf_misaligned_trades_skipped}")
        print(f"  Win Rate: {result.win_rate:.1%}")
        print(f"  Profit Factor: {result.profit_factor:.2f}x")
        print(f"  Total P&L: ${result.total_pnl_usd:.2f}")


if __name__ == "__main__":
    main()
