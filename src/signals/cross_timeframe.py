"""
Cross-Timeframe Validation Logic
Validates trading signals across multiple timeframes for higher confidence
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class CrossTimeframeValidator:
    """
    Validates trading signals across multiple timeframes
    Implements multi-timeframe confirmation logic
    """

    def __init__(self, config):
        self.config = config
        self.data_cache = {}

    def load_timeframe_data(self, instrument: str, timeframes: List[str], data_dir: str = "data/raw") -> Dict[str, pd.DataFrame]:
        """Load data for multiple timeframes"""
        data = {}

        for tf in timeframes:
            cache_key = f"{instrument}_{tf}"

            if cache_key in self.data_cache:
                data[tf] = self.data_cache[cache_key]
            else:
                from pathlib import Path
                filepath = Path(data_dir) / f"{instrument}_{tf}.csv"

                if filepath.exists():
                    df = pd.read_csv(filepath)
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    data[tf] = df
                    self.data_cache[cache_key] = df

        return data

    def align_timeframes(self, data: Dict[str, pd.DataFrame], target_time: datetime) -> Dict[str, pd.Series]:
        """
        Align data from different timeframes to a specific timestamp
        Returns the most recent candle from each timeframe at or before target_time
        """
        aligned = {}

        for tf, df in data.items():
            # Find the most recent candle at or before target_time
            valid_candles = df[df['timestamp'] <= target_time]

            if not valid_candles.empty:
                aligned[tf] = valid_candles.iloc[-1]

        return aligned

    def check_trend_alignment(self, aligned_data: Dict[str, pd.Series]) -> Dict[str, str]:
        """
        Check trend direction on each timeframe
        Returns: Dict mapping timeframe -> trend ('bullish', 'bearish', 'neutral')
        """
        trends = {}

        for tf, row in aligned_data.items():
            # Simple trend determination using price vs moving averages
            if 'SMA_50' in row and not pd.isna(row['SMA_50']):
                if row['close'] > row['SMA_50']:
                    trends[tf] = 'bullish'
                elif row['close'] < row['SMA_50']:
                    trends[tf] = 'bearish'
                else:
                    trends[tf] = 'neutral'
            else:
                trends[tf] = 'unknown'

        return trends

    def validate_entry_with_higher_timeframes(
        self,
        entry_timeframe: str,
        entry_signal: str,  # 'bullish' or 'bearish'
        aligned_data: Dict[str, pd.Series]
    ) -> Tuple[bool, float, str]:
        """
        Validate an entry signal against higher timeframes

        Returns:
            - is_valid: bool (True if higher TFs confirm)
            - confidence: float (0.0 to 1.0)
            - reason: str (explanation)
        """
        # Get higher timeframes
        timeframe_hierarchy = ['2m', '5m', '15m', '30m']
        entry_tf_index = timeframe_hierarchy.index(entry_timeframe)
        higher_timeframes = timeframe_hierarchy[entry_tf_index + 1:]

        if not higher_timeframes:
            # No higher timeframes to validate against
            return True, 1.0, "No higher timeframes available"

        # Check trends on higher timeframes
        trends = self.check_trend_alignment(aligned_data)

        confirmations = 0
        contradictions = 0
        neutral = 0

        for htf in higher_timeframes:
            if htf not in trends:
                continue

            htf_trend = trends[htf]

            if htf_trend == entry_signal:
                confirmations += 1
            elif htf_trend == 'neutral' or htf_trend == 'unknown':
                neutral += 1
            else:
                contradictions += 1

        total_checked = confirmations + contradictions + neutral

        if total_checked == 0:
            return True, 0.5, "No higher timeframe data available"

        # Calculate confidence
        confidence = (confirmations + (neutral * 0.5)) / total_checked

        # Validation logic
        is_valid = contradictions == 0 and confirmations >= 1

        if is_valid:
            reason = f"Confirmed by {confirmations} higher TF(s)"
        else:
            reason = f"Contradicted by {contradictions} higher TF(s)"

        return is_valid, confidence, reason

    def check_multi_timeframe_confluence(
        self,
        instrument: str,
        target_time: datetime,
        entry_timeframe: str,
        signal_direction: str,
        data_dir: str = "data/raw"
    ) -> Dict:
        """
        Complete multi-timeframe analysis for a potential entry

        Returns comprehensive validation result
        """
        # Load all timeframe data
        all_timeframes = self.config.TIMEFRAMES
        data = self.load_timeframe_data(instrument, all_timeframes, data_dir)

        if not data:
            return {
                'valid': False,
                'confidence': 0.0,
                'reason': 'No data available',
                'trends': {},
                'confirmations': 0
            }

        # Align timeframes to target time
        aligned_data = self.align_timeframes(data, target_time)

        # Check trend alignment
        trends = self.check_trend_alignment(aligned_data)

        # Validate with higher timeframes
        is_valid, confidence, reason = self.validate_entry_with_higher_timeframes(
            entry_timeframe,
            signal_direction,
            aligned_data
        )

        # Count confirmations
        confirmations = sum(1 for t in trends.values() if t == signal_direction)

        return {
            'valid': is_valid,
            'confidence': confidence,
            'reason': reason,
            'trends': trends,
            'confirmations': confirmations,
            'total_timeframes_checked': len(trends)
        }

    def apply_timeframe_weights(
        self,
        position_size: float,
        market_regime: str
    ) -> Dict[str, float]:
        """
        Apply dynamic position sizing based on market regime and timeframe

        Args:
            position_size: Base position size in USD
            market_regime: 'trending_market', 'choppy_market', or 'default'

        Returns:
            Dict mapping timeframe -> position size multiplier
        """
        weights = self.config.TIMEFRAME_WEIGHTS.get(
            market_regime,
            self.config.TIMEFRAME_WEIGHTS['default']
        )

        sized_positions = {
            tf: position_size * weight
            for tf, weight in weights.items()
        }

        return sized_positions

    def get_optimal_entry_timeframe(
        self,
        trends: Dict[str, str],
        target_direction: str
    ) -> Optional[str]:
        """
        Determine the optimal timeframe for entry based on trend alignment

        Returns the smallest timeframe where all higher TFs are aligned
        """
        timeframe_hierarchy = ['2m', '5m', '15m', '30m']

        for i, tf in enumerate(timeframe_hierarchy):
            # Check if this TF and all higher TFs are aligned
            higher_tfs = timeframe_hierarchy[i:]

            all_aligned = all(
                trends.get(htf, 'unknown') == target_direction
                for htf in higher_tfs
            )

            if all_aligned:
                return tf

        return None

    def backtest_with_multi_tf_validation(
        self,
        trades: List,
        data_dir: str = "data/raw"
    ) -> Tuple[List, Dict]:
        """
        Re-analyze historical trades with multi-TF validation
        Filter trades that would have been validated

        Returns:
            - validated_trades: List of trades that pass multi-TF validation
            - stats: Validation statistics
        """
        validated_trades = []
        validation_stats = {
            'total': len(trades),
            'validated': 0,
            'rejected': 0,
            'avg_confidence': 0.0
        }

        confidences = []

        for trade in trades:
            validation = self.check_multi_timeframe_confluence(
                instrument=trade.instrument,
                target_time=trade.entry_time,
                entry_timeframe=trade.timeframe,
                signal_direction='bullish',  # Simplified
                data_dir=data_dir
            )

            if validation['valid']:
                validated_trades.append(trade)
                validation_stats['validated'] += 1
                confidences.append(validation['confidence'])
            else:
                validation_stats['rejected'] += 1

        if confidences:
            validation_stats['avg_confidence'] = np.mean(confidences)

        logger.info(f"Multi-TF Validation: {validation_stats['validated']}/{validation_stats['total']} trades validated")

        return validated_trades, validation_stats


class MultiTimeframeStrategy:
    """
    Implements complete multi-timeframe trading strategy
    - 30m: Macro trend
    - 15m: Confirmation
    - 5m: Entry
    - 2m: Execution
    """

    def __init__(self, config):
        self.config = config
        self.validator = CrossTimeframeValidator(config)

    def analyze_market_structure(
        self,
        instrument: str,
        current_time: datetime,
        data_dir: str = "data/raw"
    ) -> Dict:
        """
        Complete market structure analysis across all timeframes
        """
        # Load all timeframe data
        data = self.validator.load_timeframe_data(instrument, self.config.TIMEFRAMES, data_dir)

        # Align to current time
        aligned = self.validator.align_timeframes(data, current_time)

        # Get trends
        trends = self.validator.check_trend_alignment(aligned)

        # Determine market regime
        if '30m' in aligned:
            row_30m = aligned['30m']
            if 'ADX' in row_30m and row_30m['ADX'] > 30:
                regime = 'trending_market'
            elif 'ATR_14' in row_30m and row_30m['ATR_14'] / row_30m['close'] > 0.03:
                regime = 'volatile_market'
            else:
                regime = 'choppy_market'
        else:
            regime = 'default'

        # Identify confluence opportunities
        all_bullish = all(t == 'bullish' for t in trends.values() if t != 'unknown')
        all_bearish = all(t == 'bearish' for t in trends.values() if t != 'unknown')

        confluence = 'strong_bullish' if all_bullish else ('strong_bearish' if all_bearish else 'mixed')

        return {
            'instrument': instrument,
            'timestamp': current_time,
            'trends': trends,
            'regime': regime,
            'confluence': confluence,
            'aligned_data': aligned
        }

    def generate_entry_signal(
        self,
        market_structure: Dict
    ) -> Optional[Dict]:
        """
        Generate entry signal based on multi-timeframe analysis
        """
        trends = market_structure['trends']

        # Check 30m trend (macro)
        trend_30m = trends.get('30m', 'unknown')

        # Check 15m confirmation
        trend_15m = trends.get('15m', 'unknown')

        # Check 5m entry
        trend_5m = trends.get('5m', 'unknown')

        # Check 2m execution
        trend_2m = trends.get('2m', 'unknown')

        # Entry logic: 30m + 15m aligned, 5m shows entry opportunity
        if trend_30m == 'bullish' and trend_15m == 'bullish' and trend_5m == 'bullish':
            # Strong bullish setup
            return {
                'direction': 'long',
                'entry_timeframe': '5m',
                'execution_timeframe': '2m',
                'confidence': 0.9,
                'reason': 'All higher timeframes aligned bullish',
                'trends': trends
            }

        elif trend_30m == 'bearish' and trend_15m == 'bearish' and trend_5m == 'bearish':
            # Strong bearish setup
            return {
                'direction': 'short',
                'entry_timeframe': '5m',
                'execution_timeframe': '2m',
                'confidence': 0.9,
                'reason': 'All higher timeframes aligned bearish',
                'trends': trends
            }

        # Pullback entry in uptrend
        elif trend_30m == 'bullish' and trend_15m == 'bullish' and trend_5m == 'bearish':
            # Pullback in uptrend - potential entry
            return {
                'direction': 'long',
                'entry_timeframe': '5m',
                'execution_timeframe': '2m',
                'confidence': 0.7,
                'reason': 'Pullback in 30m/15m uptrend',
                'trends': trends
            }

        return None


def main():
    """Example usage"""
    import sys
    sys.path.append("../..")
    import config

    validator = CrossTimeframeValidator(config)

    # Example validation
    result = validator.check_multi_timeframe_confluence(
        instrument="BTCUSDT.P",
        target_time=datetime(2024, 1, 15, 12, 0),
        entry_timeframe="5m",
        signal_direction="bullish"
    )

    print("\nCross-Timeframe Validation Result:")
    print(f"Valid: {result['valid']}")
    print(f"Confidence: {result['confidence']:.1%}")
    print(f"Reason: {result['reason']}")
    print(f"Trends: {result['trends']}")


if __name__ == "__main__":
    main()
