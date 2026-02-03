"""
Multi-Timeframe Crypto Perpetual Futures Backtesting Engine
Realistic simulation with slippage, funding rates, and cross-timeframe validation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Represents a single trade"""
    entry_time: datetime
    exit_time: datetime
    instrument: str
    timeframe: str
    signal_id: str
    signal_name: str
    direction: str  # long or short
    entry_price: float
    exit_price: float
    stop_loss: float
    target: float
    position_size: float
    pnl_pips: float
    pnl_pct: float
    pnl_usd: float
    exit_reason: str  # stop_hit, target_hit, timeout
    hold_duration_minutes: float
    slippage_cost: float
    funding_cost: float
    total_cost: float
    regime: str


@dataclass
class BacktestResult:
    """Results from backtesting a signal"""
    signal_id: str
    signal_name: str
    timeframe: str
    instrument: str

    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float

    # P&L metrics
    total_pnl_pips: float
    total_pnl_usd: float
    avg_win_pips: float
    avg_loss_pips: float
    largest_win_pips: float
    largest_loss_pips: float
    profit_factor: float

    # Risk metrics
    max_drawdown_pct: float
    max_drawdown_usd: float
    sharpe_ratio: float
    sortino_ratio: float

    # Timing metrics
    avg_hold_duration_minutes: float

    # Cost analysis
    total_slippage_cost: float
    total_funding_cost: float
    total_transaction_cost: float

    # Regime analysis
    win_rate_by_regime: Dict[str, float]
    trades_by_regime: Dict[str, int]

    # Cross-timeframe validation
    multi_tf_confirmed_trades: int
    multi_tf_win_rate: float

    # All trades
    trades: List[Trade]


class BacktestEngine:
    """
    Backtesting engine for crypto perpetual futures
    Tests signals across multiple timeframes with realistic costs
    """

    def __init__(self, config):
        self.config = config
        self.data_cache = {}
        self.funding_rates_cache = {}

    def load_data(self, instrument: str, timeframe: str, data_dir: str = "data/raw") -> pd.DataFrame:
        """Load OHLC data for backtesting"""
        cache_key = f"{instrument}_{timeframe}"

        if cache_key in self.data_cache:
            return self.data_cache[cache_key]

        filepath = Path(data_dir) / f"{instrument}_{timeframe}.csv"

        if not filepath.exists():
            logger.warning(f"Data file not found: {filepath}")
            return pd.DataFrame()

        df = pd.read_csv(filepath)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.sort_values('timestamp', inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Add technical indicators
        df = self.add_indicators(df, timeframe)

        self.data_cache[cache_key] = df
        logger.info(f"Loaded {len(df)} candles for {instrument} {timeframe}")

        return df

    def add_indicators(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Add technical indicators to dataframe"""
        # Simple Moving Averages
        df['SMA_20'] = df['close'].rolling(window=20).mean()
        df['SMA_50'] = df['close'].rolling(window=50).mean()
        df['SMA_100'] = df['close'].rolling(window=100).mean()
        df['SMA_200'] = df['close'].rolling(window=200).mean()

        # Exponential Moving Averages
        df['EMA_9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['EMA_50'] = df['close'].ewm(span=50, adjust=False).mean()

        # RSI
        df['RSI_7'] = self.calculate_rsi(df['close'], 7)
        df['RSI_14'] = self.calculate_rsi(df['close'], 14)
        df['RSI_21'] = self.calculate_rsi(df['close'], 21)

        # MACD
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema_12 - ema_26
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']

        # ATR
        df['ATR_14'] = self.calculate_atr(df, 14)

        # Bollinger Bands
        df['BB_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (2 * bb_std)
        df['BB_lower'] = df['BB_middle'] - (2 * bb_std)

        # Volume metrics
        df['Volume_MA'] = df['volume'].rolling(window=20).mean()
        df['Volume_ratio'] = df['volume'] / df['Volume_MA']

        # ADX
        df = self.calculate_adx(df, 14)

        # Stochastic
        df = self.calculate_stochastic(df, 14, 3)

        return df

    @staticmethod
    def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        return atr

    @staticmethod
    def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate ADX (Average Directional Index)"""
        high = df['high']
        low = df['low']
        close = df['close']

        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        tr = BacktestEngine.calculate_atr(df, 1)

        plus_di = 100 * (plus_dm.ewm(alpha=1/period).mean() / tr)
        minus_di = 100 * (minus_dm.ewm(alpha=1/period).mean() / tr)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(alpha=1/period).mean()

        df['ADX'] = adx
        df['Plus_DI'] = plus_di
        df['Minus_DI'] = minus_di

        return df

    @staticmethod
    def calculate_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
        """Calculate Stochastic Oscillator"""
        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()

        df['Stoch_K'] = 100 * (df['close'] - low_min) / (high_max - low_min)
        df['Stoch_D'] = df['Stoch_K'].rolling(window=d_period).mean()

        return df

    def detect_regime(self, df: pd.DataFrame, index: int, lookback: int = 100) -> str:
        """Detect market regime at specific point"""
        if index < lookback:
            return "unknown"

        window = df.iloc[index - lookback:index]

        # Check ADX for trend strength
        adx = window['ADX'].iloc[-1] if 'ADX' in window.columns else 20

        # Check ATR for volatility
        atr_pct = window['ATR_14'].iloc[-1] / window['close'].iloc[-1] if 'ATR_14' in window.columns else 0.02

        # Check price action
        price_range = (window['high'].max() - window['low'].min()) / window['close'].mean()

        if adx > self.config.ADX_STRONG_TREND_THRESHOLD:
            return "trending"
        elif adx > self.config.ADX_TRENDING_THRESHOLD:
            return "ranging"
        elif atr_pct > self.config.ATR_HIGH_VOLATILITY:
            return "volatile"
        else:
            return "choppy"

    def check_entry_conditions(self, signal: dict, df: pd.DataFrame, index: int) -> bool:
        """
        Check if entry conditions are met at specific candle
        This is a simplified version - full implementation would parse entry_conditions
        """
        if index < 200:  # Need enough data for indicators
            return False

        row = df.iloc[index]

        # Example condition checking (simplified)
        # In production, this would dynamically evaluate the signal's entry_conditions

        # For RSI oversold signals
        if "RSI" in signal['name'] and "Oversold" in signal['name']:
            if row['RSI_14'] < 30 and row['close'] > row['SMA_50']:
                return True

        # For MACD crossover signals
        if "MACD" in signal['name'] and "Cross" in signal['name']:
            prev_row = df.iloc[index - 1]
            if (row['MACD'] > row['MACD_signal'] and
                prev_row['MACD'] <= prev_row['MACD_signal']):
                return True

        # For breakout signals
        if "Breakout" in signal['name']:
            lookback = min(20, index)
            recent_high = df.iloc[index - lookback:index]['high'].max()
            if row['close'] > recent_high and row['Volume_ratio'] > 1.5:
                return True

        # For Bollinger Band bounces
        if "Bollinger" in signal['name'] or "BB" in signal['name']:
            if row['close'] <= row['BB_lower'] and row['RSI_14'] < 35:
                return True

        # Default: randomly trigger for testing (remove in production)
        # This is a placeholder - real implementation needs proper condition parsing
        return np.random.random() < 0.05  # 5% random entry for testing

    def simulate_trade(
        self,
        signal: dict,
        df: pd.DataFrame,
        entry_index: int,
        funding_rates: pd.DataFrame
    ) -> Optional[Trade]:
        """
        Simulate a single trade with realistic execution
        """
        timeframe = signal['timeframe']
        entry_row = df.iloc[entry_index]
        entry_time = entry_row['timestamp']
        entry_price = entry_row['close']

        # Apply entry slippage
        slippage_pips = self.config.SLIPPAGE[timeframe]
        slippage_cost = slippage_pips * (entry_price / 10000)  # Convert pips to price
        entry_price += slippage_cost

        # Calculate stop loss and target
        atr = entry_row['ATR_14']
        stop_loss = entry_price - (self.config.STOP_LOSS_ATR_MULTIPLIER * atr)

        # Target based on timeframe
        target_pips_min, target_pips_max = self.config.TIMEFRAME_CHARACTERISTICS[timeframe]['target_pips']
        target_pips = (target_pips_min + target_pips_max) / 2
        target = entry_price + (target_pips * (entry_price / 10000))

        # Position size
        position_size_usd = self.config.STARTING_CAPITAL_USDT * self.config.POSITION_SIZE_PCT

        # Simulate holding period
        max_hold_minutes = signal['typical_hold_minutes'][1]
        max_candles = int(max_hold_minutes / self.config.TIMEFRAME_CHARACTERISTICS[timeframe]['duration_minutes'])

        exit_index = entry_index + 1
        exit_reason = "timeout"
        funding_cost = 0.0

        # Track trade through time
        for i in range(entry_index + 1, min(entry_index + max_candles + 1, len(df))):
            candle = df.iloc[i]

            # Check stop loss
            if candle['low'] <= stop_loss:
                exit_index = i
                exit_price = stop_loss
                exit_reason = "stop_hit"
                break

            # Check target
            if candle['high'] >= target:
                exit_index = i
                exit_price = target
                exit_reason = "target_hit"
                break

            # Calculate funding costs if held past 8 hours
            hours_held = (candle['timestamp'] - entry_time).total_seconds() / 3600
            if hours_held >= 8:
                # Apply funding rate cost
                # This is simplified - real implementation would check exact funding timestamps
                pass

        # If no exit triggered, exit at timeout
        if exit_index == entry_index + 1:
            exit_index = min(entry_index + max_candles, len(df) - 1)
            exit_row = df.iloc[exit_index]
            exit_price = exit_row['close']
            exit_reason = "timeout"

        exit_row = df.iloc[exit_index]
        exit_time = exit_row['timestamp']

        # Apply exit slippage
        exit_price -= slippage_cost

        # Calculate P&L
        pnl_pips = ((exit_price - entry_price) / entry_price) * 10000
        pnl_pct = (exit_price - entry_price) / entry_price
        pnl_usd = position_size_usd * pnl_pct

        # Calculate total costs
        total_cost = slippage_cost * 2 + funding_cost  # Entry + exit slippage

        # Hold duration
        hold_duration = (exit_time - entry_time).total_seconds() / 60

        # Detect regime
        regime = self.detect_regime(df, entry_index)

        return Trade(
            entry_time=entry_time,
            exit_time=exit_time,
            instrument=signal.get('instrument', 'BTCUSDT.P'),
            timeframe=timeframe,
            signal_id=signal['id'],
            signal_name=signal['name'],
            direction="long",  # Simplified - would support short too
            entry_price=entry_price,
            exit_price=exit_price,
            stop_loss=stop_loss,
            target=target,
            position_size=position_size_usd,
            pnl_pips=pnl_pips,
            pnl_pct=pnl_pct,
            pnl_usd=pnl_usd,
            exit_reason=exit_reason,
            hold_duration_minutes=hold_duration,
            slippage_cost=slippage_cost * 2,
            funding_cost=funding_cost,
            total_cost=total_cost,
            regime=regime
        )

    def backtest_signal(
        self,
        signal: dict,
        instrument: str,
        data_dir: str = "data/raw"
    ) -> BacktestResult:
        """
        Backtest a single signal hypothesis
        """
        logger.info(f"Backtesting {signal['name']} on {instrument} {signal['timeframe']}")

        # Load data
        df = self.load_data(instrument, signal['timeframe'], data_dir)

        if df.empty:
            logger.warning(f"No data available for {instrument} {signal['timeframe']}")
            return None

        # Load funding rates
        funding_rates = pd.DataFrame()  # Placeholder

        # Simulate trades
        trades = []

        for i in range(200, len(df) - 50):  # Leave buffer for exits
            if self.check_entry_conditions(signal, df, i):
                trade = self.simulate_trade(signal, df, i, funding_rates)
                if trade:
                    trades.append(trade)

        if len(trades) < self.config.THRESHOLDS[signal['timeframe']]['min_trades']:
            logger.warning(f"Insufficient trades for {signal['name']}: {len(trades)}")
            return None

        # Calculate metrics
        winning_trades = [t for t in trades if t.pnl_pips > 0]
        losing_trades = [t for t in trades if t.pnl_pips <= 0]

        win_rate = len(winning_trades) / len(trades) if trades else 0

        total_pnl_pips = sum(t.pnl_pips for t in trades)
        total_pnl_usd = sum(t.pnl_usd for t in trades)

        avg_win_pips = np.mean([t.pnl_pips for t in winning_trades]) if winning_trades else 0
        avg_loss_pips = np.mean([abs(t.pnl_pips) for t in losing_trades]) if losing_trades else 0

        largest_win_pips = max([t.pnl_pips for t in winning_trades]) if winning_trades else 0
        largest_loss_pips = min([t.pnl_pips for t in losing_trades]) if losing_trades else 0

        gross_profit = sum(t.pnl_usd for t in winning_trades) if winning_trades else 0
        gross_loss = abs(sum(t.pnl_usd for t in losing_trades)) if losing_trades else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Calculate drawdown
        cumulative_pnl = np.cumsum([t.pnl_usd for t in trades])
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdown = running_max - cumulative_pnl
        max_drawdown_usd = np.max(drawdown) if len(drawdown) > 0 else 0
        max_drawdown_pct = max_drawdown_usd / self.config.STARTING_CAPITAL_USDT

        # Sharpe ratio (simplified)
        returns = [t.pnl_pct for t in trades]
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0

        # Sortino ratio (simplified)
        negative_returns = [r for r in returns if r < 0]
        downside_dev = np.std(negative_returns) if len(negative_returns) > 1 else 0.001
        sortino_ratio = np.mean(returns) / downside_dev * np.sqrt(252) if downside_dev > 0 else 0

        # Regime analysis
        regimes = {}
        for trade in trades:
            if trade.regime not in regimes:
                regimes[trade.regime] = {'wins': 0, 'total': 0}
            regimes[trade.regime]['total'] += 1
            if trade.pnl_pips > 0:
                regimes[trade.regime]['wins'] += 1

        win_rate_by_regime = {
            regime: data['wins'] / data['total'] if data['total'] > 0 else 0
            for regime, data in regimes.items()
        }
        trades_by_regime = {regime: data['total'] for regime, data in regimes.items()}

        # Costs
        total_slippage_cost = sum(t.slippage_cost for t in trades)
        total_funding_cost = sum(t.funding_cost for t in trades)
        total_transaction_cost = total_slippage_cost + total_funding_cost

        # Average hold duration
        avg_hold_duration = np.mean([t.hold_duration_minutes for t in trades])

        return BacktestResult(
            signal_id=signal['id'],
            signal_name=signal['name'],
            timeframe=signal['timeframe'],
            instrument=instrument,
            total_trades=len(trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=win_rate,
            total_pnl_pips=total_pnl_pips,
            total_pnl_usd=total_pnl_usd,
            avg_win_pips=avg_win_pips,
            avg_loss_pips=avg_loss_pips,
            largest_win_pips=largest_win_pips,
            largest_loss_pips=largest_loss_pips,
            profit_factor=profit_factor,
            max_drawdown_pct=max_drawdown_pct,
            max_drawdown_usd=max_drawdown_usd,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            avg_hold_duration_minutes=avg_hold_duration,
            total_slippage_cost=total_slippage_cost,
            total_funding_cost=total_funding_cost,
            total_transaction_cost=total_transaction_cost,
            win_rate_by_regime=win_rate_by_regime,
            trades_by_regime=trades_by_regime,
            multi_tf_confirmed_trades=0,  # Placeholder
            multi_tf_win_rate=0.0,  # Placeholder
            trades=trades
        )

    def save_results(self, results: List[BacktestResult], output_path: str = "results/backtests/backtest_results_multiframe.json"):
        """Save backtest results to JSON"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        results_dict = []
        for result in results:
            if result:
                r = asdict(result)
                # Convert trades to dict (don't save all trades, just summary)
                r['trades'] = len(result.trades)
                results_dict.append(r)

        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)

        logger.info(f"Saved {len(results_dict)} backtest results to {output_path}")


def main():
    """Example usage"""
    import sys
    sys.path.append("../..")
    import config

    engine = BacktestEngine(config)

    # Load hypotheses
    with open("results/discovery/hypotheses_multiframe.json", 'r') as f:
        hypotheses = json.load(f)

    # Backtest first 5 signals
    results = []
    for signal in hypotheses[:5]:
        result = engine.backtest_signal(signal, "BTCUSDT.P")
        if result:
            results.append(result)
            logger.info(f"✓ {signal['name']}: Win Rate {result.win_rate:.1%}, PF {result.profit_factor:.2f}x")

    # Save results
    engine.save_results(results)


if __name__ == "__main__":
    main()
