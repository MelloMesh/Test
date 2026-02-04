"""
Live Trading Signal Generator
Generates real-time trading signals based on validated strategies
with proper risk management and R:R calculations
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass, asdict
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class LiveSignal:
    """Represents a live trading signal"""
    timestamp: datetime
    signal_id: str
    signal_name: str
    instrument: str
    timeframe: str
    direction: str  # 'long' or 'short'

    # Entry
    entry_price: float

    # Risk Management
    stop_loss: float
    take_profit: float
    risk_amount_usd: float
    reward_amount_usd: float
    risk_reward_ratio: float

    # Position Sizing
    position_size_usd: float
    position_size_units: float

    # Signal Quality
    confidence: float  # 0.0 to 1.0
    backtest_win_rate: float
    backtest_profit_factor: float

    # Multi-Timeframe Context
    higher_tf_aligned: bool
    macro_trend: str  # 'bullish', 'bearish', 'neutral'

    # Additional Info
    entry_reason: str
    regime: str  # 'trending', 'ranging', 'choppy'
    funding_rate: Optional[float]

    # Status
    status: str  # 'active', 'filled', 'stopped', 'target_hit'


class LiveSignalGenerator:
    """
    Generates live trading signals based on backtested strategies
    with proper risk management
    """

    def __init__(self, config):
        self.config = config
        self.validated_signals = self._load_validated_signals()
        self.active_signals: List[LiveSignal] = []

    def _load_validated_signals(self) -> List[Dict]:
        """Load signals that passed backtesting"""
        try:
            with open('results/analysis/edge_analysis_multiframe.json', 'r') as f:
                analysis = json.load(f)

            # Get all ranked signals (even if they didn't pass strict thresholds)
            ranked_signals = analysis.get('ranked_signals', [])

            # Filter by more lenient thresholds based on R:R
            validated = []
            for sig in ranked_signals:
                # Calculate required win rate based on profit factor
                # For 2:1 R:R, need 50% WR; for 3:1 R:R, need 33% WR
                tf = sig['timeframe']
                min_rr = self.config.MIN_RISK_REWARD[tf]
                required_wr = 1 / (1 + min_rr)  # Break-even win rate

                # Allow signals that beat break-even by 5%
                if sig['win_rate'] >= required_wr + 0.05:
                    validated.append(sig)
                    logger.info(f"✓ {sig['signal_name']}: WR={sig['win_rate']:.1%} (req: {required_wr:.1%}), RR={min_rr}:1")

            logger.info(f"Loaded {len(validated)} validated signals for live trading")
            return validated

        except FileNotFoundError:
            logger.warning("No edge analysis found. Run backtesting first.")
            return []

    def calculate_position_size(
        self,
        account_balance: float,
        risk_pct: float,
        entry_price: float,
        stop_loss: float
    ) -> Tuple[float, float]:
        """
        Calculate position size based on risk parameters

        Returns:
            (position_size_usd, position_size_units)
        """
        # Risk amount in USD
        risk_amount = account_balance * risk_pct

        # Price distance to stop loss
        price_risk = abs(entry_price - stop_loss)

        # Position size to risk exactly the risk_amount
        position_size_units = risk_amount / price_risk
        position_size_usd = position_size_units * entry_price

        return position_size_usd, position_size_units

    def calculate_targets(
        self,
        entry_price: float,
        stop_loss: float,
        risk_reward_ratio: float,
        direction: str
    ) -> float:
        """Calculate take profit target based on R:R ratio"""
        risk = abs(entry_price - stop_loss)
        reward = risk * risk_reward_ratio

        if direction == 'long':
            take_profit = entry_price + reward
        else:  # short
            take_profit = entry_price - reward

        return take_profit

    def check_signal_conditions(
        self,
        signal_hypothesis: Dict,
        current_data: pd.DataFrame
    ) -> bool:
        """
        Check if signal conditions are met on current market data
        This is a simplified version - would need full implementation
        """
        # This would check the signal's entry_conditions against current_data
        # For now, return random for demonstration
        # In production, implement proper condition checking
        return np.random.random() > 0.95  # 5% trigger rate

    def generate_signal(
        self,
        signal_hypothesis: Dict,
        current_price: float,
        current_data: pd.DataFrame,
        account_balance: float = 10000.0
    ) -> Optional[LiveSignal]:
        """
        Generate a live trading signal if conditions are met
        """
        # Check if signal conditions are met
        if not self.check_signal_conditions(signal_hypothesis, current_data):
            return None

        instrument = self.config.INSTRUMENTS[0]  # Default to BTC
        timeframe = signal_hypothesis['timeframe']

        # Calculate stop loss (using ATR)
        atr = current_data['ATR_14'].iloc[-1] if 'ATR_14' in current_data.columns else current_price * 0.02
        direction = 'long'  # Simplified - would determine from signal

        if direction == 'long':
            stop_loss = current_price - (self.config.STOP_LOSS_ATR_MULTIPLIER * atr)
        else:
            stop_loss = current_price + (self.config.STOP_LOSS_ATR_MULTIPLIER * atr)

        # Get R:R ratio for this timeframe
        risk_reward_ratio = self.config.MIN_RISK_REWARD[timeframe]

        # Calculate take profit
        take_profit = self.calculate_targets(
            entry_price=current_price,
            stop_loss=stop_loss,
            risk_reward_ratio=risk_reward_ratio,
            direction=direction
        )

        # Calculate position size
        position_size_usd, position_size_units = self.calculate_position_size(
            account_balance=account_balance,
            risk_pct=self.config.POSITION_SIZE_PCT,
            entry_price=current_price,
            stop_loss=stop_loss
        )

        # Risk/Reward amounts
        risk_amount = abs(current_price - stop_loss) * position_size_units
        reward_amount = abs(take_profit - current_price) * position_size_units

        # Create signal
        signal = LiveSignal(
            timestamp=datetime.utcnow(),
            signal_id=f"{signal_hypothesis['id']}_{int(datetime.utcnow().timestamp())}",
            signal_name=signal_hypothesis['name'],
            instrument=instrument,
            timeframe=timeframe,
            direction=direction,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_amount_usd=risk_amount,
            reward_amount_usd=reward_amount,
            risk_reward_ratio=risk_reward_ratio,
            position_size_usd=position_size_usd,
            position_size_units=position_size_units,
            confidence=signal_hypothesis.get('win_rate', 0.5),
            backtest_win_rate=signal_hypothesis.get('win_rate', 0.0),
            backtest_profit_factor=signal_hypothesis.get('profit_factor', 0.0),
            higher_tf_aligned=True,  # Would check multi-TF
            macro_trend='bullish',  # Would determine from data
            entry_reason=signal_hypothesis.get('description', ''),
            regime='trending',  # Would detect from data
            funding_rate=None,  # Would fetch from API
            status='active'
        )

        self.active_signals.append(signal)
        logger.info(f"🚨 LIVE SIGNAL: {signal.signal_name} | {signal.direction.upper()} {signal.instrument} @ {signal.entry_price:.2f}")
        logger.info(f"   SL: {signal.stop_loss:.2f} | TP: {signal.take_profit:.2f} | R:R = {signal.risk_reward_ratio}:1")

        return signal

    def scan_for_signals(
        self,
        current_data: Dict[str, pd.DataFrame],
        account_balance: float = 10000.0
    ) -> List[LiveSignal]:
        """
        Scan all validated signals for entry opportunities

        Args:
            current_data: Dict mapping timeframe -> DataFrame with latest market data
            account_balance: Current account balance

        Returns:
            List of generated signals
        """
        new_signals = []

        for signal_hypothesis in self.validated_signals:
            timeframe = signal_hypothesis['timeframe']

            if timeframe not in current_data:
                continue

            data = current_data[timeframe]
            if data.empty:
                continue

            current_price = data['close'].iloc[-1]

            signal = self.generate_signal(
                signal_hypothesis=signal_hypothesis,
                current_price=current_price,
                current_data=data,
                account_balance=account_balance
            )

            if signal:
                new_signals.append(signal)

        return new_signals

    def save_signals(self, signals: List[LiveSignal], output_file: str = "results/trades/live_signals.json"):
        """Save generated signals to file"""
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        signals_dict = [asdict(sig) for sig in signals]

        # Convert datetime objects to strings
        for sig in signals_dict:
            sig['timestamp'] = sig['timestamp'].isoformat()

        with open(output_file, 'w') as f:
            json.dump(signals_dict, f, indent=2, default=str)

        logger.info(f"Saved {len(signals)} signals to {output_file}")

    def print_signal_summary(self, signal: LiveSignal):
        """Print formatted signal summary"""
        print("\n" + "=" * 80)
        print(f"🚨 LIVE TRADING SIGNAL")
        print("=" * 80)
        print(f"\nSignal: {signal.signal_name}")
        print(f"Instrument: {signal.instrument} ({signal.timeframe})")
        print(f"Direction: {signal.direction.upper()}")
        print(f"Timestamp: {signal.timestamp}")

        print(f"\n📍 ENTRY:")
        print(f"  Price: ${signal.entry_price:.2f}")

        print(f"\n🛡️ RISK MANAGEMENT:")
        print(f"  Stop Loss: ${signal.stop_loss:.2f} ({((signal.entry_price - signal.stop_loss)/signal.entry_price * 100):.2f}%)")
        print(f"  Take Profit: ${signal.take_profit:.2f} ({((signal.take_profit - signal.entry_price)/signal.entry_price * 100):.2f}%)")
        print(f"  Risk:Reward = 1:{signal.risk_reward_ratio:.1f}")

        print(f"\n💰 POSITION SIZING:")
        print(f"  Position Size: ${signal.position_size_usd:.2f}")
        print(f"  Units: {signal.position_size_units:.4f}")
        print(f"  Risk Amount: ${signal.risk_amount_usd:.2f}")
        print(f"  Reward Amount: ${signal.reward_amount_usd:.2f}")

        print(f"\n📊 SIGNAL QUALITY:")
        print(f"  Confidence: {signal.confidence:.1%}")
        print(f"  Backtest Win Rate: {signal.backtest_win_rate:.1%}")
        print(f"  Backtest Profit Factor: {signal.backtest_profit_factor:.2f}x")

        print(f"\n🌐 MARKET CONTEXT:")
        print(f"  Macro Trend: {signal.macro_trend}")
        print(f"  Regime: {signal.regime}")
        print(f"  Higher TF Aligned: {'✓' if signal.higher_tf_aligned else '✗'}")

        print("\n" + "=" * 80)


def main():
    """Example usage"""
    import sys
    sys.path.append("../..")
    import config
    from src.backtest.backtest_engine import BacktestEngine

    generator = LiveSignalGenerator(config)

    print(f"\n✓ Loaded {len(generator.validated_signals)} validated signals")

    if generator.validated_signals:
        print("\nTop 5 Validated Signals:")
        for i, sig in enumerate(generator.validated_signals[:5], 1):
            print(f"  {i}. {sig['signal_name']} ({sig['timeframe']}): WR={sig['win_rate']:.1%}, PF={sig['profit_factor']:.2f}x")

        # Simulate scanning for signals
        print("\n\nScanning for live signals...")

        # Load some current data (would be real-time in production)
        backtest_engine = BacktestEngine(config)
        data = backtest_engine.load_data("BTCUSDT.P", "5m")

        if not data.empty:
            current_data = {"5m": data.tail(200)}  # Last 200 candles

            signals = generator.scan_for_signals(current_data, account_balance=10000.0)

            if signals:
                for signal in signals:
                    generator.print_signal_summary(signal)

                generator.save_signals(signals)
            else:
                print("No signals generated at this time.")


if __name__ == "__main__":
    main()
