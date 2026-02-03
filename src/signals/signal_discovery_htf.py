"""
HTF-Aware Signal Discovery System
Generates trading signals on LTF (30m, 15m, 5m) that align with HTF context (W, D, 4H)
Only generates signals that trade WITH the higher timeframe bias
"""

import json
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class HTFAwareSignalHypothesis:
    """Trading signal hypothesis with HTF alignment requirements"""
    id: str
    name: str
    timeframe: str  # LTF execution timeframe (30m, 15m, 5m)
    direction: str  # 'long' or 'short' (each signal has ONE direction)
    description: str

    # Entry conditions
    entry_conditions: List[str]
    stop_loss: str
    target: str
    typical_hold_minutes: tuple
    slippage_pips: float
    expected_accuracy: float

    # Signal classification
    signal_type: str  # momentum, mean_reversion, breakout, reversal, confluence
    indicators_used: List[str]
    regime_best: str  # trending, ranging, choppy

    # HTF Requirements
    htf_required_bias: str  # 'bullish', 'bearish', 'neutral', 'any'
    htf_min_alignment: float  # Minimum alignment score (0-100)
    htf_min_strength: float  # Minimum bias strength (0-100)

    # Additional context
    funding_rate_consideration: bool
    volume_required: bool


class HTFAwareSignalDiscovery:
    """
    Generates LTF execution signals that align with HTF context
    Top-Down Approach: HTF determines bias, LTF finds entries
    """

    def __init__(self, config):
        self.config = config
        self.hypotheses: List[HTFAwareSignalHypothesis] = []
        self.hypothesis_counter = 0

        # Focus on LTF execution timeframes
        self.ltf_timeframes = config.LTF_TIMEFRAMES  # ['30m', '15m', '5m']

    def _generate_id(self, timeframe: str, direction: str, signal_type: str) -> str:
        """Generate unique hypothesis ID"""
        self.hypothesis_counter += 1
        return f"{timeframe}_{direction}_{signal_type}_{self.hypothesis_counter:03d}"

    def generate_30m_long_signals(self) -> List[HTFAwareSignalHypothesis]:
        """Generate LONG signals for 30m timeframe (requires bullish HTF)"""
        signals = []

        # 30M-LONG-001: HTF Trend Continuation Pullback
        signals.append(HTFAwareSignalHypothesis(
            id=self._generate_id("30m", "long", "trend"),
            name="HTF_Bullish_Pullback_30m_Long",
            timeframe="30m",
            direction="long",
            description="Enter long on pullback to support in confirmed bullish HTF trend",
            entry_conditions=[
                "HTF (W/D/4H) shows bullish bias",
                "Price pulls back to 20-EMA or 50-EMA",
                "RSI(14) dips to 40-50 (healthy pullback, not oversold)",
                "Volume decreasing on pullback (weakness)",
                "MACD still positive (trend intact)",
                "Price holding above HTF support level"
            ],
            stop_loss="Below 50-EMA or HTF support level (-1.5 ATR)",
            target="Recent high or HTF resistance (+3-4 ATR)",
            typical_hold_minutes=(30, 180),
            slippage_pips=2.0,
            expected_accuracy=0.68,
            signal_type="trend",
            indicators_used=["EMA", "RSI", "MACD", "ATR", "Volume"],
            regime_best="trending",
            htf_required_bias="bullish",
            htf_min_alignment=60.0,
            htf_min_strength=50.0,
            funding_rate_consideration=False,
            volume_required=True
        ))

        # 30M-LONG-002: HTF Support Bounce
        signals.append(HTFAwareSignalHypothesis(
            id=self._generate_id("30m", "long", "reversal"),
            name="HTF_Support_Bounce_30m_Long",
            timeframe="30m",
            direction="long",
            description="Long at HTF support level with reversal confirmation",
            entry_conditions=[
                "HTF shows bullish or neutral bias (not bearish)",
                "Price reaches HTF daily or 4H support zone",
                "Bullish engulfing or hammer candle",
                "RSI(14) < 35 (oversold at support)",
                "Volume spike on bounce candle",
                "MACD histogram turning up"
            ],
            stop_loss="Below support level (-1 ATR)",
            target="HTF resistance or +2.5:1 R:R minimum",
            typical_hold_minutes=(30, 120),
            slippage_pips=2.0,
            expected_accuracy=0.65,
            signal_type="reversal",
            indicators_used=["Support_Levels", "RSI", "MACD", "Volume", "Candlesticks"],
            regime_best="ranging",
            htf_required_bias="bullish",
            htf_min_alignment=50.0,
            htf_min_strength=40.0,
            funding_rate_consideration=True,
            volume_required=True
        ))

        # 30M-LONG-003: Breakout With HTF Confirmation
        signals.append(HTFAwareSignalHypothesis(
            id=self._generate_id("30m", "long", "breakout"),
            name="HTF_Confirmed_Breakout_30m_Long",
            timeframe="30m",
            direction="long",
            description="Breakout above resistance when HTF is bullish",
            entry_conditions=[
                "HTF shows strong bullish bias (alignment > 70%)",
                "Price breaks above LTF resistance with volume",
                "Resistance aligns with HTF 4H level",
                "RSI(14) > 55 but < 70 (strong but not overbought)",
                "Volume > 2x average on breakout",
                "Close above resistance level"
            ],
            stop_loss="Below breakout level or recent consolidation",
            target="Measured move or next HTF resistance",
            typical_hold_minutes=(30, 180),
            slippage_pips=2.5,
            expected_accuracy=0.62,
            signal_type="breakout",
            indicators_used=["Resistance_Levels", "Volume", "RSI"],
            regime_best="trending",
            htf_required_bias="bullish",
            htf_min_alignment=70.0,
            htf_min_strength=60.0,
            funding_rate_consideration=False,
            volume_required=True
        ))

        return signals

    def generate_30m_short_signals(self) -> List[HTFAwareSignalHypothesis]:
        """Generate SHORT signals for 30m timeframe (requires bearish HTF)"""
        signals = []

        # 30M-SHORT-001: HTF Trend Continuation Rejection
        signals.append(HTFAwareSignalHypothesis(
            id=self._generate_id("30m", "short", "trend"),
            name="HTF_Bearish_Rejection_30m_Short",
            timeframe="30m",
            direction="short",
            description="Enter short on rally to resistance in confirmed bearish HTF trend",
            entry_conditions=[
                "HTF (W/D/4H) shows bearish bias",
                "Price rallies to 20-EMA or 50-EMA",
                "RSI(14) rises to 50-60 (weak rally, not overbought)",
                "Volume decreasing on rally (weakness)",
                "MACD still negative (downtrend intact)",
                "Price rejected at HTF resistance level"
            ],
            stop_loss="Above 50-EMA or HTF resistance (+1.5 ATR)",
            target="Recent low or HTF support (-3-4 ATR)",
            typical_hold_minutes=(30, 180),
            slippage_pips=2.0,
            expected_accuracy=0.68,
            signal_type="trend",
            indicators_used=["EMA", "RSI", "MACD", "ATR", "Volume"],
            regime_best="trending",
            htf_required_bias="bearish",
            htf_min_alignment=60.0,
            htf_min_strength=50.0,
            funding_rate_consideration=False,
            volume_required=True
        ))

        # 30M-SHORT-002: HTF Resistance Rejection
        signals.append(HTFAwareSignalHypothesis(
            id=self._generate_id("30m", "short", "reversal"),
            name="HTF_Resistance_Rejection_30m_Short",
            timeframe="30m",
            direction="short",
            description="Short at HTF resistance with rejection confirmation",
            entry_conditions=[
                "HTF shows bearish or neutral bias (not bullish)",
                "Price reaches HTF daily or 4H resistance zone",
                "Bearish engulfing or shooting star candle",
                "RSI(14) > 65 (overbought at resistance)",
                "Volume spike on rejection candle",
                "MACD histogram turning down"
            ],
            stop_loss="Above resistance level (+1 ATR)",
            target="HTF support or +2.5:1 R:R minimum",
            typical_hold_minutes=(30, 120),
            slippage_pips=2.0,
            expected_accuracy=0.65,
            signal_type="reversal",
            indicators_used=["Resistance_Levels", "RSI", "MACD", "Volume", "Candlesticks"],
            regime_best="ranging",
            htf_required_bias="bearish",
            htf_min_alignment=50.0,
            htf_min_strength=40.0,
            funding_rate_consideration=True,
            volume_required=True
        ))

        return signals

    def generate_15m_long_signals(self) -> List[HTFAwareSignalHypothesis]:
        """Generate LONG signals for 15m timeframe"""
        signals = []

        # 15M-LONG-001: Quick HTF Pullback Entry
        signals.append(HTFAwareSignalHypothesis(
            id=self._generate_id("15m", "long", "momentum"),
            name="HTF_Quick_Pullback_15m_Long",
            timeframe="15m",
            direction="long",
            description="Fast entry on dip in strong bullish HTF trend",
            entry_conditions=[
                "HTF strongly bullish (alignment > 70%)",
                "Price dips to 20-EMA",
                "RSI(14) touches 45-50 (shallow pullback)",
                "MACD positive and curling up",
                "3-5 candle pullback completed"
            ],
            stop_loss="Below 50-EMA (-1 ATR)",
            target="Recent high or +2:1 R:R",
            typical_hold_minutes=(15, 60),
            slippage_pips=1.5,
            expected_accuracy=0.64,
            signal_type="momentum",
            indicators_used=["EMA", "RSI", "MACD", "ATR"],
            regime_best="trending",
            htf_required_bias="bullish",
            htf_min_alignment=70.0,
            htf_min_strength=60.0,
            funding_rate_consideration=False,
            volume_required=False
        ))

        # 15M-LONG-002: HTF Zone Scalp
        signals.append(HTFAwareSignalHypothesis(
            id=self._generate_id("15m", "long", "mean_reversion"),
            name="HTF_Zone_Bounce_15m_Long",
            timeframe="15m",
            direction="long",
            description="Scalp bounce from HTF support zone",
            entry_conditions=[
                "HTF bullish or neutral",
                "Price at HTF 4H support level",
                "RSI(14) < 30 (oversold)",
                "Bullish divergence on MACD",
                "Volume declining into support"
            ],
            stop_loss="Below support (-0.75 ATR)",
            target="Midpoint to resistance or +2:1",
            typical_hold_minutes=(15, 45),
            slippage_pips=1.5,
            expected_accuracy=0.61,
            signal_type="mean_reversion",
            indicators_used=["Support_Levels", "RSI", "MACD", "Volume"],
            regime_best="ranging",
            htf_required_bias="bullish",
            htf_min_alignment=50.0,
            htf_min_strength=40.0,
            funding_rate_consideration=False,
            volume_required=False
        ))

        return signals

    def generate_15m_short_signals(self) -> List[HTFAwareSignalHypothesis]:
        """Generate SHORT signals for 15m timeframe"""
        signals = []

        # 15M-SHORT-001: Quick HTF Rally Fade
        signals.append(HTFAwareSignalHypothesis(
            id=self._generate_id("15m", "short", "momentum"),
            name="HTF_Quick_Rally_Fade_15m_Short",
            timeframe="15m",
            direction="short",
            description="Fast short on rally in strong bearish HTF trend",
            entry_conditions=[
                "HTF strongly bearish (alignment > 70%)",
                "Price rallies to 20-EMA",
                "RSI(14) reaches 50-55 (weak rally)",
                "MACD negative and curling down",
                "3-5 candle rally completed"
            ],
            stop_loss="Above 50-EMA (+1 ATR)",
            target="Recent low or +2:1 R:R",
            typical_hold_minutes=(15, 60),
            slippage_pips=1.5,
            expected_accuracy=0.64,
            signal_type="momentum",
            indicators_used=["EMA", "RSI", "MACD", "ATR"],
            regime_best="trending",
            htf_required_bias="bearish",
            htf_min_alignment=70.0,
            htf_min_strength=60.0,
            funding_rate_consideration=False,
            volume_required=False
        ))

        return signals

    def generate_5m_long_signals(self) -> List[HTFAwareSignalHypothesis]:
        """Generate LONG signals for 5m timeframe (precision entries)"""
        signals = []

        # 5M-LONG-001: Precision HTF Pullback Entry
        signals.append(HTFAwareSignalHypothesis(
            id=self._generate_id("5m", "long", "momentum"),
            name="HTF_Precision_Entry_5m_Long",
            timeframe="5m",
            direction="long",
            description="Precision long entry on micro pullback in bullish HTF",
            entry_conditions=[
                "HTF bullish (all timeframes aligned)",
                "30m showing bullish setup",
                "Price touches 10-EMA or 20-EMA",
                "RSI(14) dips to 45-50",
                "MACD still positive",
                "Bullish hammer or engulfing candle at EMA"
            ],
            stop_loss="Below recent swing low (-0.5 ATR)",
            target="+1.5-2:1 R:R or 30m target",
            typical_hold_minutes=(5, 30),
            slippage_pips=1.0,
            expected_accuracy=0.62,
            signal_type="momentum",
            indicators_used=["EMA", "RSI", "MACD", "Candlesticks"],
            regime_best="trending",
            htf_required_bias="bullish",
            htf_min_alignment=70.0,
            htf_min_strength=60.0,
            funding_rate_consideration=False,
            volume_required=False
        ))

        # 5M-LONG-002: Micro Bounce at HTF Level
        signals.append(HTFAwareSignalHypothesis(
            id=self._generate_id("5m", "long", "reversal"),
            name="HTF_Micro_Bounce_5m_Long",
            timeframe="5m",
            direction="long",
            description="Quick bounce at precise HTF support level",
            entry_conditions=[
                "HTF support level reached on 5m",
                "Price wicks below and closes above support",
                "RSI(14) < 30",
                "Bullish candle pattern",
                "Volume spike on bounce"
            ],
            stop_loss="Below wick low (-0.5 ATR)",
            target="Back to EMA or +2:1",
            typical_hold_minutes=(5, 20),
            slippage_pips=1.0,
            expected_accuracy=0.59,
            signal_type="reversal",
            indicators_used=["Support_Levels", "RSI", "Volume", "Candlesticks"],
            regime_best="ranging",
            htf_required_bias="bullish",
            htf_min_alignment=50.0,
            htf_min_strength=40.0,
            funding_rate_consideration=False,
            volume_required=True
        ))

        return signals

    def generate_5m_short_signals(self) -> List[HTFAwareSignalHypothesis]:
        """Generate SHORT signals for 5m timeframe"""
        signals = []

        # 5M-SHORT-001: Precision HTF Rally Fade
        signals.append(HTFAwareSignalHypothesis(
            id=self._generate_id("5m", "short", "momentum"),
            name="HTF_Precision_Fade_5m_Short",
            timeframe="5m",
            direction="short",
            description="Precision short on micro rally in bearish HTF",
            entry_conditions=[
                "HTF bearish (all timeframes aligned)",
                "30m showing bearish setup",
                "Price touches 10-EMA or 20-EMA",
                "RSI(14) rises to 50-55",
                "MACD still negative",
                "Bearish shooting star or engulfing at EMA"
            ],
            stop_loss="Above recent swing high (+0.5 ATR)",
            target="+1.5-2:1 R:R or 30m target",
            typical_hold_minutes=(5, 30),
            slippage_pips=1.0,
            expected_accuracy=0.62,
            signal_type="momentum",
            indicators_used=["EMA", "RSI", "MACD", "Candlesticks"],
            regime_best="trending",
            htf_required_bias="bearish",
            htf_min_alignment=70.0,
            htf_min_strength=60.0,
            funding_rate_consideration=False,
            volume_required=False
        ))

        return signals

    def generate_all_htf_aware_signals(self) -> List[HTFAwareSignalHypothesis]:
        """Generate all HTF-aware signals across LTF execution timeframes"""
        all_signals = []

        logger.info("Generating HTF-aware signals for LTF execution...")

        # 30m signals
        logger.info("Generating 30m LONG signals (bullish HTF required)...")
        all_signals.extend(self.generate_30m_long_signals())

        logger.info("Generating 30m SHORT signals (bearish HTF required)...")
        all_signals.extend(self.generate_30m_short_signals())

        # 15m signals
        logger.info("Generating 15m LONG signals (bullish HTF required)...")
        all_signals.extend(self.generate_15m_long_signals())

        logger.info("Generating 15m SHORT signals (bearish HTF required)...")
        all_signals.extend(self.generate_15m_short_signals())

        # 5m signals
        logger.info("Generating 5m LONG signals (bullish HTF required)...")
        all_signals.extend(self.generate_5m_long_signals())

        logger.info("Generating 5m SHORT signals (bearish HTF required)...")
        all_signals.extend(self.generate_5m_short_signals())

        self.hypotheses = all_signals
        logger.info(f"Generated {len(all_signals)} HTF-aware signal hypotheses")

        return all_signals

    def save_hypotheses(self, output_path: str = "results/discovery/hypotheses_htf_aware.json"):
        """Save HTF-aware hypotheses to JSON"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        hypotheses_dict = [asdict(h) for h in self.hypotheses]

        with open(output_path, 'w') as f:
            json.dump(hypotheses_dict, f, indent=2)

        logger.info(f"Saved {len(self.hypotheses)} HTF-aware hypotheses to {output_path}")

    def print_summary(self):
        """Print summary of HTF-aware signals"""
        print("\n" + "=" * 80)
        print("HTF-AWARE SIGNAL DISCOVERY - TOP-DOWN ANALYSIS")
        print("=" * 80)

        timeframe_counts = {}
        direction_counts = {'long': 0, 'short': 0}
        htf_bias_counts = {}

        for h in self.hypotheses:
            timeframe_counts[h.timeframe] = timeframe_counts.get(h.timeframe, 0) + 1
            direction_counts[h.direction] += 1
            htf_bias_counts[h.htf_required_bias] = htf_bias_counts.get(h.htf_required_bias, 0) + 1

        print(f"\nTotal HTF-Aware Hypotheses: {len(self.hypotheses)}")

        print("\nBy Execution Timeframe (LTF):")
        for tf, count in sorted(timeframe_counts.items()):
            print(f"  {tf}: {count} signals")

        print("\nBy Direction:")
        for direction, count in direction_counts.items():
            print(f"  {direction.upper()}: {count} signals")

        print("\nBy HTF Bias Requirement:")
        for bias, count in htf_bias_counts.items():
            print(f"  {bias}: {count} signals")

        print("\nSample HTF-Aware Signals:")
        for i, h in enumerate(self.hypotheses[:5], 1):
            print(f"\n{i}. {h.name}")
            print(f"   Timeframe: {h.timeframe} | Direction: {h.direction.upper()}")
            print(f"   HTF Required: {h.htf_required_bias} (alignment ≥ {h.htf_min_alignment}%)")
            print(f"   Expected Accuracy: {h.expected_accuracy * 100:.1f}%")
            print(f"   Type: {h.signal_type} | Best Regime: {h.regime_best}")

        print("\n" + "=" * 80)
        print("NOTE: All signals require HTF alignment check before execution")
        print("HTF Context = W/D/4H | LTF Execution = 30m/15m/5m")
        print("=" * 80 + "\n")


def main():
    """Example usage"""
    import sys
    sys.path.append("../..")
    import config

    engine = HTFAwareSignalDiscovery(config)

    # Generate all HTF-aware hypotheses
    hypotheses = engine.generate_all_htf_aware_signals()

    # Save to file
    engine.save_hypotheses()

    # Print summary
    engine.print_summary()


if __name__ == "__main__":
    main()
