"""
Autonomous Signal Discovery System
Generates 30-50+ trading signal hypotheses across multiple timeframes (2m, 5m, 15m, 30m)
for crypto perpetual futures
"""

import json
from typing import List, Dict
from dataclasses import dataclass, asdict
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class SignalHypothesis:
    """Represents a trading signal hypothesis"""
    id: str
    name: str
    timeframe: str
    description: str
    entry_conditions: List[str]
    stop_loss: str
    target: str
    typical_hold_minutes: tuple
    slippage_pips: float
    expected_accuracy: float
    signal_type: str  # momentum, mean_reversion, breakout, reversal, confluence
    indicators_used: List[str]
    regime_best: str  # trending, ranging, choppy, all
    funding_rate_consideration: bool


class SignalDiscoveryEngine:
    """
    Autonomous signal discovery engine
    Generates diverse trading hypotheses across all timeframes
    """

    def __init__(self, config):
        self.config = config
        self.hypotheses: List[SignalHypothesis] = []
        self.hypothesis_counter = 0

    def _generate_id(self, timeframe: str, signal_type: str) -> str:
        """Generate unique hypothesis ID"""
        self.hypothesis_counter += 1
        return f"{timeframe}_{signal_type}_{self.hypothesis_counter:03d}"

    def generate_2m_hypotheses(self) -> List[SignalHypothesis]:
        """Generate signal hypotheses for 2-minute timeframe"""
        signals = []

        # 2M-001: Momentum Spike Reversal
        signals.append(SignalHypothesis(
            id=self._generate_id("2m", "momentum"),
            name="Momentum_Spike_Reversal_2m",
            timeframe="2m",
            description="Enter on sharp momentum reversal after 5-candle extreme",
            entry_conditions=[
                "Price closes above 5-candle high",
                "MACD histogram > 0 (bullish momentum)",
                "Volume spike > 1.5x average",
                "RSI(7) crosses above 50"
            ],
            stop_loss="1 ATR below entry (tight stop for 2m)",
            target="Next resistance level or +15 pips (quick scalp)",
            typical_hold_minutes=(2, 5),
            slippage_pips=1.0,
            expected_accuracy=0.62,
            signal_type="momentum",
            indicators_used=["MACD", "RSI", "ATR", "Volume"],
            regime_best="trending",
            funding_rate_consideration=False
        ))

        # 2M-002: Micro Mean Reversion
        signals.append(SignalHypothesis(
            id=self._generate_id("2m", "mean_reversion"),
            name="Micro_Mean_Reversion_2m",
            timeframe="2m",
            description="Quick bounce off Bollinger Band extremes with RSI confirmation",
            entry_conditions=[
                "Price touches lower Bollinger Band (2 std dev)",
                "RSI(7) < 25 (oversold on micro timeframe)",
                "Price > 20-EMA (overall uptrend)",
                "MACD starting to curl up"
            ],
            stop_loss="Below recent swing low - 0.5 ATR",
            target="+10-20 pips or middle Bollinger Band",
            typical_hold_minutes=(2, 7),
            slippage_pips=1.0,
            expected_accuracy=0.58,
            signal_type="mean_reversion",
            indicators_used=["Bollinger_Bands", "RSI", "EMA", "MACD"],
            regime_best="ranging",
            funding_rate_consideration=False
        ))

        # 2M-003: Volume Breakout Micro
        signals.append(SignalHypothesis(
            id=self._generate_id("2m", "breakout"),
            name="Volume_Breakout_Micro_2m",
            timeframe="2m",
            description="Immediate entry on volume spike breakout",
            entry_conditions=[
                "Price breaks above recent high (last 10 candles)",
                "Volume > 2x average volume",
                "ATR expanding (volatility increasing)",
                "No immediate resistance within 20 pips"
            ],
            stop_loss="Below breakout level - 1 ATR",
            target="Next resistance or +15 pips",
            typical_hold_minutes=(3, 8),
            slippage_pips=1.0,
            expected_accuracy=0.56,
            signal_type="breakout",
            indicators_used=["Volume", "ATR"],
            regime_best="volatile",
            funding_rate_consideration=False
        ))

        # 2M-004: MACD Divergence Quick
        signals.append(SignalHypothesis(
            id=self._generate_id("2m", "divergence"),
            name="MACD_Divergence_Quick_2m",
            timeframe="2m",
            description="Bullish divergence on 2m with immediate reversal",
            entry_conditions=[
                "Price makes lower low",
                "MACD makes higher low (bullish divergence)",
                "MACD crosses above signal line",
                "Volume confirms"
            ],
            stop_loss="Below divergence low - 0.75 ATR",
            target="+12 pips or next resistance",
            typical_hold_minutes=(2, 6),
            slippage_pips=1.0,
            expected_accuracy=0.60,
            signal_type="reversal",
            indicators_used=["MACD", "Volume"],
            regime_best="choppy",
            funding_rate_consideration=False
        ))

        # 2M-005: Scalp Stochastic Cross
        signals.append(SignalHypothesis(
            id=self._generate_id("2m", "oscillator"),
            name="Scalp_Stochastic_Cross_2m",
            timeframe="2m",
            description="Ultra-fast stochastic oversold cross",
            entry_conditions=[
                "Stochastic %K crosses above %D",
                "Both below 20 (oversold zone)",
                "Price finding support at key level",
                "Confirm with 5m trend direction"
            ],
            stop_loss="Below support - 1 ATR",
            target="+10 pips or stochastic reaches 80",
            typical_hold_minutes=(3, 5),
            slippage_pips=1.0,
            expected_accuracy=0.57,
            signal_type="mean_reversion",
            indicators_used=["Stochastic", "Support_Resistance"],
            regime_best="ranging",
            funding_rate_consideration=False
        ))

        # 2M-006: EMA Crossover Micro
        signals.append(SignalHypothesis(
            id=self._generate_id("2m", "trend"),
            name="EMA_Crossover_Micro_2m",
            timeframe="2m",
            description="Fast EMA crossover for quick trend capture",
            entry_conditions=[
                "9-EMA crosses above 20-EMA",
                "Price above both EMAs",
                "ADX > 20 (some trend strength)",
                "Volume increasing"
            ],
            stop_loss="Below 20-EMA - 0.5 ATR",
            target="+15 pips or EMA crossover fails",
            typical_hold_minutes=(4, 8),
            slippage_pips=1.0,
            expected_accuracy=0.59,
            signal_type="trend",
            indicators_used=["EMA", "ADX", "Volume"],
            regime_best="trending",
            funding_rate_consideration=False
        ))

        # 2M-007: Momentum Thrust
        signals.append(SignalHypothesis(
            id=self._generate_id("2m", "momentum"),
            name="Momentum_Thrust_2m",
            timeframe="2m",
            description="Enter on explosive momentum thrust with high RSI",
            entry_conditions=[
                "RSI(7) > 70 (strong momentum)",
                "Price breaks above resistance",
                "MACD histogram expanding",
                "Confirm with 5m also bullish"
            ],
            stop_loss="Recent swing low - 1 ATR",
            target="+20 pips (ride momentum)",
            typical_hold_minutes=(2, 5),
            slippage_pips=1.0,
            expected_accuracy=0.54,
            signal_type="momentum",
            indicators_used=["RSI", "MACD", "Support_Resistance"],
            regime_best="trending",
            funding_rate_consideration=False
        ))

        # 2M-008: Support Bounce Tight
        signals.append(SignalHypothesis(
            id=self._generate_id("2m", "support"),
            name="Support_Bounce_Tight_2m",
            timeframe="2m",
            description="Tight entry at key support with multiple confirmations",
            entry_conditions=[
                "Price tests key support level",
                "Support held 3+ times recently",
                "RSI showing bullish divergence",
                "Volume spike on bounce"
            ],
            stop_loss="Below support - 0.75 ATR (tight)",
            target="+12 pips or next resistance",
            typical_hold_minutes=(3, 6),
            slippage_pips=1.0,
            expected_accuracy=0.61,
            signal_type="reversal",
            indicators_used=["Support_Resistance", "RSI", "Volume"],
            regime_best="ranging",
            funding_rate_consideration=False
        ))

        return signals

    def generate_5m_hypotheses(self) -> List[SignalHypothesis]:
        """Generate signal hypotheses for 5-minute timeframe"""
        signals = []

        # 5M-001: RSI Oversold Reversal
        signals.append(SignalHypothesis(
            id=self._generate_id("5m", "reversal"),
            name="RSI_Oversold_5m_Reversal",
            timeframe="5m",
            description="Classic RSI oversold reversal with trend confirmation",
            entry_conditions=[
                "RSI(14) < 30 (oversold)",
                "Price > 50-SMA (overall uptrend)",
                "Funding rate < 0.0001 (neutral or negative)",
                "Price finding support at previous swing low"
            ],
            stop_loss="Lowest of last 5 candles - 1 ATR",
            target="Next resistance or +30 pips",
            typical_hold_minutes=(5, 15),
            slippage_pips=1.5,
            expected_accuracy=0.65,
            signal_type="mean_reversion",
            indicators_used=["RSI", "SMA", "Funding_Rate"],
            regime_best="ranging",
            funding_rate_consideration=True
        ))

        # 5M-002: MACD Bullish Cross
        signals.append(SignalHypothesis(
            id=self._generate_id("5m", "momentum"),
            name="MACD_Bullish_Cross_5m",
            timeframe="5m",
            description="MACD bullish crossover with volume confirmation",
            entry_conditions=[
                "MACD line crosses above signal line",
                "MACD histogram turns positive",
                "Volume above 20-period average",
                "Price above 20-EMA"
            ],
            stop_loss="Below recent swing low - 1 ATR",
            target="+25 pips or MACD weakens",
            typical_hold_minutes=(7, 20),
            slippage_pips=1.5,
            expected_accuracy=0.61,
            signal_type="momentum",
            indicators_used=["MACD", "Volume", "EMA"],
            regime_best="trending",
            funding_rate_consideration=False
        ))

        # 5M-003: Bollinger Bounce
        signals.append(SignalHypothesis(
            id=self._generate_id("5m", "mean_reversion"),
            name="Bollinger_Bounce_5m",
            timeframe="5m",
            description="Mean reversion from lower Bollinger Band",
            entry_conditions=[
                "Price touches or breaks below lower BB (2 std)",
                "RSI(14) < 35",
                "Previous bounce from BB successful",
                "15m timeframe not oversold (higher TF confirmation)"
            ],
            stop_loss="Below BB - 1.5 ATR",
            target="Middle BB or +30 pips",
            typical_hold_minutes=(8, 18),
            slippage_pips=1.5,
            expected_accuracy=0.63,
            signal_type="mean_reversion",
            indicators_used=["Bollinger_Bands", "RSI"],
            regime_best="ranging",
            funding_rate_consideration=False
        ))

        # 5M-004: Breakout Retest
        signals.append(SignalHypothesis(
            id=self._generate_id("5m", "breakout"),
            name="Breakout_Retest_5m",
            timeframe="5m",
            description="Enter on successful retest of broken resistance",
            entry_conditions=[
                "Price breaks above resistance with volume",
                "Price pulls back to test broken resistance (now support)",
                "Volume on retest lower than breakout (healthy pullback)",
                "RSI > 50 (maintain bullish momentum)"
            ],
            stop_loss="Below retested support - 1.5 ATR",
            target="Next resistance level or +35 pips",
            typical_hold_minutes=(10, 25),
            slippage_pips=1.5,
            expected_accuracy=0.59,
            signal_type="breakout",
            indicators_used=["Support_Resistance", "Volume", "RSI"],
            regime_best="trending",
            funding_rate_consideration=False
        ))

        # 5M-005: Confluence Entry
        signals.append(SignalHypothesis(
            id=self._generate_id("5m", "confluence"),
            name="Confluence_Entry_5m",
            timeframe="5m",
            description="Multiple indicator alignment for high probability setup",
            entry_conditions=[
                "Price at key Fibonacci retracement (61.8% or 50%)",
                "RSI showing bullish divergence",
                "MACD curling up",
                "Volume confirming reversal",
                "15m timeframe aligned"
            ],
            stop_loss="Below Fib level - 1 ATR",
            target="+40 pips or next Fib extension",
            typical_hold_minutes=(10, 20),
            slippage_pips=1.5,
            expected_accuracy=0.67,
            signal_type="confluence",
            indicators_used=["Fibonacci", "RSI", "MACD", "Volume"],
            regime_best="all",
            funding_rate_consideration=False
        ))

        # 5M-006: Trend Continuation
        signals.append(SignalHypothesis(
            id=self._generate_id("5m", "trend"),
            name="Trend_Continuation_5m",
            timeframe="5m",
            description="Pullback entry in established 15m/30m trend",
            entry_conditions=[
                "30m trend is bullish (ADX > 25, price > 50-SMA)",
                "5m pullback to 20-EMA",
                "Stochastic oversold on 5m",
                "Volume declining on pullback (healthy)"
            ],
            stop_loss="Below 20-EMA - 1.5 ATR",
            target="+30 pips or 5m trend breaks",
            typical_hold_minutes=(12, 30),
            slippage_pips=1.5,
            expected_accuracy=0.64,
            signal_type="trend",
            indicators_used=["EMA", "Stochastic", "ADX", "Volume"],
            regime_best="trending",
            funding_rate_consideration=False
        ))

        # 5M-007: Funding Rate Reversal
        signals.append(SignalHypothesis(
            id=self._generate_id("5m", "funding"),
            name="Funding_Rate_Reversal_5m",
            timeframe="5m",
            description="Enter short when funding rate extremely positive",
            entry_conditions=[
                "Funding rate > 0.0003 (extreme bullish)",
                "RSI(14) > 65 (overbought)",
                "Price at resistance",
                "MACD showing bearish divergence"
            ],
            stop_loss="Above resistance + 1.5 ATR",
            target="-30 pips or funding normalizes",
            typical_hold_minutes=(15, 45),
            slippage_pips=1.5,
            expected_accuracy=0.70,
            signal_type="reversal",
            indicators_used=["Funding_Rate", "RSI", "MACD", "Support_Resistance"],
            regime_best="all",
            funding_rate_consideration=True
        ))

        # 5M-008: Volume Surge Entry
        signals.append(SignalHypothesis(
            id=self._generate_id("5m", "volume"),
            name="Volume_Surge_Entry_5m",
            timeframe="5m",
            description="Enter on volume surge with momentum confirmation",
            entry_conditions=[
                "Volume > 3x average (significant surge)",
                "Price breaking key level",
                "RSI > 55 (momentum)",
                "ATR expanding"
            ],
            stop_loss="Below breakout - 1.5 ATR",
            target="+28 pips or volume dries up",
            typical_hold_minutes=(6, 15),
            slippage_pips=1.5,
            expected_accuracy=0.58,
            signal_type="breakout",
            indicators_used=["Volume", "RSI", "ATR"],
            regime_best="volatile",
            funding_rate_consideration=False
        ))

        # 5M-009: Double Bottom
        signals.append(SignalHypothesis(
            id=self._generate_id("5m", "pattern"),
            name="Double_Bottom_5m",
            timeframe="5m",
            description="Classic double bottom pattern with neckline break",
            entry_conditions=[
                "Two lows at similar price level",
                "Price breaks above neckline (resistance between bottoms)",
                "Volume increasing on breakout",
                "RSI showing bullish divergence between bottoms"
            ],
            stop_loss="Below second bottom - 1 ATR",
            target="+35 pips (measure height of pattern)",
            typical_hold_minutes=(10, 25),
            slippage_pips=1.5,
            expected_accuracy=0.62,
            signal_type="reversal",
            indicators_used=["Price_Pattern", "Volume", "RSI"],
            regime_best="ranging",
            funding_rate_consideration=False
        ))

        # 5M-010: ADX Trend Strength
        signals.append(SignalHypothesis(
            id=self._generate_id("5m", "trend"),
            name="ADX_Trend_Strength_5m",
            timeframe="5m",
            description="Enter when ADX confirms strong trend development",
            entry_conditions=[
                "ADX crosses above 25 (trend developing)",
                "+DI above -DI (bullish)",
                "Price above 50-SMA",
                "MACD confirming"
            ],
            stop_loss="Below recent swing low - 1.5 ATR",
            target="+32 pips or ADX weakens",
            typical_hold_minutes=(15, 35),
            slippage_pips=1.5,
            expected_accuracy=0.60,
            signal_type="trend",
            indicators_used=["ADX", "MACD", "SMA"],
            regime_best="trending",
            funding_rate_consideration=False
        ))

        return signals

    def generate_15m_hypotheses(self) -> List[SignalHypothesis]:
        """Generate signal hypotheses for 15-minute timeframe"""
        signals = []

        # 15M-001: Breakout Confluence
        signals.append(SignalHypothesis(
            id=self._generate_id("15m", "confluence"),
            name="Breakout_Confluence_15m",
            timeframe="15m",
            description="Multi-indicator confluence at key breakout level",
            entry_conditions=[
                "Price breaks above major resistance",
                "Volume surge > 2x average",
                "MACD bullish crossover",
                "RSI > 60 (momentum)",
                "30m trend aligned"
            ],
            stop_loss="Below broken resistance - 1.5 ATR",
            target="Next major resistance or +80 pips",
            typical_hold_minutes=(20, 45),
            slippage_pips=2.5,
            expected_accuracy=0.58,
            signal_type="breakout",
            indicators_used=["Support_Resistance", "Volume", "MACD", "RSI"],
            regime_best="trending",
            funding_rate_consideration=False
        ))

        # 15M-002: EMA Ribbon Reversal
        signals.append(SignalHypothesis(
            id=self._generate_id("15m", "trend"),
            name="EMA_Ribbon_Reversal_15m",
            timeframe="15m",
            description="EMA ribbon flip with momentum confirmation",
            entry_conditions=[
                "Price crosses above 20-50-100 EMA ribbon",
                "EMAs beginning to align bullishly",
                "MACD histogram expanding",
                "Volume confirming"
            ],
            stop_loss="Below EMA ribbon - 2 ATR",
            target="+90 pips or ribbon breaks",
            typical_hold_minutes=(25, 60),
            slippage_pips=2.5,
            expected_accuracy=0.61,
            signal_type="trend",
            indicators_used=["EMA", "MACD", "Volume"],
            regime_best="trending",
            funding_rate_consideration=False
        ))

        # 15M-003: Support Resistance Bounce
        signals.append(SignalHypothesis(
            id=self._generate_id("15m", "reversal"),
            name="Support_Resistance_Bounce_15m",
            timeframe="15m",
            description="Strong bounce from key support with institutional volume",
            entry_conditions=[
                "Price at major support (tested 5+ times)",
                "RSI showing bullish divergence",
                "Large volume spike on support test",
                "30m trend supportive"
            ],
            stop_loss="Below support - 2 ATR",
            target="Next resistance or +75 pips",
            typical_hold_minutes=(20, 50),
            slippage_pips=2.5,
            expected_accuracy=0.63,
            signal_type="reversal",
            indicators_used=["Support_Resistance", "RSI", "Volume"],
            regime_best="ranging",
            funding_rate_consideration=False
        ))

        # 15M-004: Ichimoku Cloud Break
        signals.append(SignalHypothesis(
            id=self._generate_id("15m", "trend"),
            name="Ichimoku_Cloud_Break_15m",
            timeframe="15m",
            description="Price breaks above Ichimoku cloud with all elements aligned",
            entry_conditions=[
                "Price breaks above cloud",
                "Tenkan crosses above Kijun",
                "Cloud ahead is green (bullish)",
                "Price > both Tenkan and Kijun"
            ],
            stop_loss="Below cloud - 2 ATR",
            target="+100 pips or cloud break fails",
            typical_hold_minutes=(30, 70),
            slippage_pips=2.5,
            expected_accuracy=0.59,
            signal_type="trend",
            indicators_used=["Ichimoku"],
            regime_best="trending",
            funding_rate_consideration=False
        ))

        # 15M-005: RSI Divergence Major
        signals.append(SignalHypothesis(
            id=self._generate_id("15m", "divergence"),
            name="RSI_Divergence_Major_15m",
            timeframe="15m",
            description="Clear RSI divergence at major level",
            entry_conditions=[
                "Price makes lower low",
                "RSI makes higher low (strong divergence)",
                "Price at key support level",
                "Volume confirming reversal",
                "30m not in strong downtrend"
            ],
            stop_loss="Below divergence low - 2 ATR",
            target="+85 pips or divergence plays out",
            typical_hold_minutes=(25, 55),
            slippage_pips=2.5,
            expected_accuracy=0.64,
            signal_type="reversal",
            indicators_used=["RSI", "Support_Resistance", "Volume"],
            regime_best="choppy",
            funding_rate_consideration=False
        ))

        # 15M-006: Funding Rate Extreme
        signals.append(SignalHypothesis(
            id=self._generate_id("15m", "funding"),
            name="Funding_Rate_Extreme_15m",
            timeframe="15m",
            description="Extreme funding rate mean reversion setup",
            entry_conditions=[
                "Funding rate > 0.0003 (extreme)",
                "Price at resistance",
                "Multiple bearish indicators (RSI > 70, MACD divergence)",
                "Volume suggesting exhaustion"
            ],
            stop_loss="Above resistance + 2.5 ATR",
            target="-80 pips or funding normalizes",
            typical_hold_minutes=(45, 120),
            slippage_pips=2.5,
            expected_accuracy=0.72,
            signal_type="reversal",
            indicators_used=["Funding_Rate", "RSI", "MACD", "Volume"],
            regime_best="all",
            funding_rate_consideration=True
        ))

        # 15M-007: Triple Indicator Alignment
        signals.append(SignalHypothesis(
            id=self._generate_id("15m", "confluence"),
            name="Triple_Indicator_Alignment_15m",
            timeframe="15m",
            description="RSI, MACD, and Stochastic all aligned bullishly",
            entry_conditions=[
                "RSI crosses above 50",
                "MACD bullish crossover",
                "Stochastic crossing up from oversold",
                "All three within 2 candles"
            ],
            stop_loss="Below alignment point - 2 ATR",
            target="+95 pips",
            typical_hold_minutes=(30, 60),
            slippage_pips=2.5,
            expected_accuracy=0.66,
            signal_type="confluence",
            indicators_used=["RSI", "MACD", "Stochastic"],
            regime_best="all",
            funding_rate_consideration=False
        ))

        # 15M-008: Trend Channel Bounce
        signals.append(SignalHypothesis(
            id=self._generate_id("15m", "channel"),
            name="Trend_Channel_Bounce_15m",
            timeframe="15m",
            description="Price bounces from lower channel line in uptrend",
            entry_conditions=[
                "Clear uptrend channel established (30m)",
                "Price tests lower channel line",
                "RSI > 40 (not oversold)",
                "MACD still positive"
            ],
            stop_loss="Below channel - 2 ATR",
            target="Upper channel line or +70 pips",
            typical_hold_minutes=(20, 50),
            slippage_pips=2.5,
            expected_accuracy=0.60,
            signal_type="trend",
            indicators_used=["Trend_Channel", "RSI", "MACD"],
            regime_best="trending",
            funding_rate_consideration=False
        ))

        return signals

    def generate_30m_hypotheses(self) -> List[SignalHypothesis]:
        """Generate signal hypotheses for 30-minute timeframe"""
        signals = []

        # 30M-001: Major Level Reversal
        signals.append(SignalHypothesis(
            id=self._generate_id("30m", "reversal"),
            name="Major_Level_Reversal_30m",
            timeframe="30m",
            description="Reversal at major support/resistance with multiple confirmations",
            entry_conditions=[
                "Price tests 200-SMA (major level)",
                "RSI showing strong bullish divergence",
                "Funding rate extreme (> 0.0003)",
                "Volume spike indicating institutional interest",
                "Multiple timeframes (5m, 15m) showing reversal signals"
            ],
            stop_loss="Below 200-SMA - 2.5 ATR",
            target="Next major resistance or +150 pips",
            typical_hold_minutes=(45, 120),
            slippage_pips=4.0,
            expected_accuracy=0.55,
            signal_type="reversal",
            indicators_used=["SMA", "RSI", "Funding_Rate", "Volume"],
            regime_best="all",
            funding_rate_consideration=True
        ))

        # 30M-002: Trend Momentum Breakout
        signals.append(SignalHypothesis(
            id=self._generate_id("30m", "breakout"),
            name="Trend_Momentum_Breakout_30m",
            timeframe="30m",
            description="Strong breakout with sustained momentum",
            entry_conditions=[
                "Price breaks major resistance with gap",
                "ADX > 30 (strong trend)",
                "MACD histogram expanding rapidly",
                "Volume 3x+ average",
                "15m and 5m both confirming"
            ],
            stop_loss="Below breakout level - 3 ATR",
            target="+180 pips or momentum weakens",
            typical_hold_minutes=(60, 150),
            slippage_pips=4.0,
            expected_accuracy=0.57,
            signal_type="breakout",
            indicators_used=["Support_Resistance", "ADX", "MACD", "Volume"],
            regime_best="trending",
            funding_rate_consideration=False
        ))

        # 30M-003: Moving Average Crossover
        signals.append(SignalHypothesis(
            id=self._generate_id("30m", "trend"),
            name="Moving_Average_Crossover_30m",
            timeframe="30m",
            description="Golden cross type setup on 30m timeframe",
            entry_conditions=[
                "50-SMA crosses above 200-SMA (golden cross)",
                "Price above both SMAs",
                "MACD confirming",
                "Volume increasing"
            ],
            stop_loss="Below 50-SMA - 3 ATR",
            target="+160 pips or crossover fails",
            typical_hold_minutes=(60, 180),
            slippage_pips=4.0,
            expected_accuracy=0.58,
            signal_type="trend",
            indicators_used=["SMA", "MACD", "Volume"],
            regime_best="trending",
            funding_rate_consideration=False
        ))

        # 30M-004: Institutional Volume Entry
        signals.append(SignalHypothesis(
            id=self._generate_id("30m", "volume"),
            name="Institutional_Volume_Entry_30m",
            timeframe="30m",
            description="Large institutional volume at key level",
            entry_conditions=[
                "Massive volume spike (5x+ average)",
                "Price at major Fibonacci level (50% or 61.8%)",
                "All lower timeframes confirming direction",
                "OBV showing accumulation"
            ],
            stop_loss="Below volume spike low - 3 ATR",
            target="+140 pips",
            typical_hold_minutes=(50, 130),
            slippage_pips=4.0,
            expected_accuracy=0.60,
            signal_type="confluence",
            indicators_used=["Volume", "OBV", "Fibonacci"],
            regime_best="all",
            funding_rate_consideration=False
        ))

        # 30M-005: Bollinger Squeeze Breakout
        signals.append(SignalHypothesis(
            id=self._generate_id("30m", "volatility"),
            name="Bollinger_Squeeze_Breakout_30m",
            timeframe="30m",
            description="Volatility contraction followed by expansion",
            entry_conditions=[
                "Bollinger Bands squeezed (low volatility)",
                "ATR at local minimum",
                "Price breaks above upper BB with volume",
                "MACD starting to expand"
            ],
            stop_loss="Below squeeze point - 3 ATR",
            target="+170 pips (measure band width)",
            typical_hold_minutes=(40, 100),
            slippage_pips=4.0,
            expected_accuracy=0.56,
            signal_type="breakout",
            indicators_used=["Bollinger_Bands", "ATR", "MACD", "Volume"],
            regime_best="choppy",
            funding_rate_consideration=False
        ))

        # 30M-006: Multi-Timeframe Trend Alignment
        signals.append(SignalHypothesis(
            id=self._generate_id("30m", "confluence"),
            name="Multi_Timeframe_Alignment_30m",
            timeframe="30m",
            description="All timeframes aligned for strong directional move",
            entry_conditions=[
                "30m trend bullish (ADX > 25)",
                "15m trend bullish",
                "5m showing entry signal",
                "2m momentum confirming",
                "Funding rate supportive"
            ],
            stop_loss="Below 30m swing low - 3 ATR",
            target="+200 pips",
            typical_hold_minutes=(90, 180),
            slippage_pips=4.0,
            expected_accuracy=0.68,
            signal_type="confluence",
            indicators_used=["ADX", "Multi_TF_Analysis"],
            regime_best="trending",
            funding_rate_consideration=True
        ))

        # 30M-007: Funding Rate Mean Reversion Major
        signals.append(SignalHypothesis(
            id=self._generate_id("30m", "funding"),
            name="Funding_Rate_Mean_Reversion_Major_30m",
            timeframe="30m",
            description="Major mean reversion from extreme funding rates",
            entry_conditions=[
                "Funding rate > 0.0004 (very extreme)",
                "Price at major resistance (tested 10+ times)",
                "RSI > 75 (extreme overbought)",
                "Multiple bearish divergences",
                "Volume showing distribution"
            ],
            stop_loss="Above resistance + 4 ATR",
            target="-150 pips or funding < 0.0001",
            typical_hold_minutes=(120, 240),
            slippage_pips=4.0,
            expected_accuracy=0.75,
            signal_type="reversal",
            indicators_used=["Funding_Rate", "RSI", "Volume", "Support_Resistance"],
            regime_best="all",
            funding_rate_consideration=True
        ))

        # 30M-008: Head and Shoulders
        signals.append(SignalHypothesis(
            id=self._generate_id("30m", "pattern"),
            name="Head_Shoulders_30m",
            timeframe="30m",
            description="Classic head and shoulders pattern completion",
            entry_conditions=[
                "Clear head and shoulders pattern formed",
                "Price breaks below neckline",
                "Volume confirming breakdown",
                "RSI confirming weakness"
            ],
            stop_loss="Above right shoulder + 3 ATR",
            target="-160 pips (pattern height)",
            typical_hold_minutes=(60, 150),
            slippage_pips=4.0,
            expected_accuracy=0.62,
            signal_type="reversal",
            indicators_used=["Price_Pattern", "Volume", "RSI"],
            regime_best="ranging",
            funding_rate_consideration=False
        ))

        return signals

    def generate_all_hypotheses(self) -> List[SignalHypothesis]:
        """Generate all signal hypotheses across all timeframes"""
        all_signals = []

        logger.info("Generating 2m signal hypotheses...")
        all_signals.extend(self.generate_2m_hypotheses())

        logger.info("Generating 5m signal hypotheses...")
        all_signals.extend(self.generate_5m_hypotheses())

        logger.info("Generating 15m signal hypotheses...")
        all_signals.extend(self.generate_15m_hypotheses())

        logger.info("Generating 30m signal hypotheses...")
        all_signals.extend(self.generate_30m_hypotheses())

        self.hypotheses = all_signals
        logger.info(f"Generated {len(all_signals)} total signal hypotheses")

        return all_signals

    def save_hypotheses(self, output_path: str = "results/discovery/hypotheses_multiframe.json"):
        """Save hypotheses to JSON file"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        hypotheses_dict = [asdict(h) for h in self.hypotheses]

        with open(output_path, 'w') as f:
            json.dump(hypotheses_dict, f, indent=2)

        logger.info(f"Saved {len(self.hypotheses)} hypotheses to {output_path}")

    def print_summary(self):
        """Print summary of generated hypotheses"""
        print("\n" + "=" * 80)
        print("SIGNAL DISCOVERY SUMMARY - MULTI-TIMEFRAME CRYPTO PERPETUAL FUTURES")
        print("=" * 80)

        timeframe_counts = {}
        signal_type_counts = {}

        for h in self.hypotheses:
            timeframe_counts[h.timeframe] = timeframe_counts.get(h.timeframe, 0) + 1
            signal_type_counts[h.signal_type] = signal_type_counts.get(h.signal_type, 0) + 1

        print(f"\nTotal Hypotheses: {len(self.hypotheses)}")
        print("\nBy Timeframe:")
        for tf, count in sorted(timeframe_counts.items()):
            print(f"  {tf}: {count} signals")

        print("\nBy Signal Type:")
        for st, count in sorted(signal_type_counts.items()):
            print(f"  {st}: {count} signals")

        print("\nSample Hypotheses:")
        for i, h in enumerate(self.hypotheses[:5], 1):
            print(f"\n{i}. {h.name} ({h.timeframe})")
            print(f"   Type: {h.signal_type}")
            print(f"   Expected Accuracy: {h.expected_accuracy * 100}%")
            print(f"   Best Regime: {h.regime_best}")

        print("\n" + "=" * 80)


def main():
    """Example usage"""
    import sys
    sys.path.append("../..")
    import config

    engine = SignalDiscoveryEngine(config)

    # Generate all hypotheses
    hypotheses = engine.generate_all_hypotheses()

    # Save to file
    engine.save_hypotheses()

    # Print summary
    engine.print_summary()


if __name__ == "__main__":
    main()
