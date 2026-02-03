"""
Higher Timeframe (HTF) Analyzer
Analyzes Weekly, Daily, and 4H timeframes to determine market context
Provides directional bias for lower timeframe signal execution
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class HTFContext:
    """Higher timeframe market context"""
    instrument: str

    # Trend Analysis
    weekly_trend: str  # 'bullish', 'bearish', 'neutral'
    daily_trend: str
    h4_trend: str

    # Overall bias (most important)
    primary_bias: str  # 'bullish', 'bearish', 'neutral'
    bias_strength: float  # 0-100 (how strong is the bias)

    # Market Regime
    regime: str  # 'trending', 'ranging', 'choppy', 'volatile'

    # Key Levels
    weekly_support: Optional[float]
    weekly_resistance: Optional[float]
    daily_support: Optional[float]
    daily_resistance: Optional[float]
    h4_support: Optional[float]
    h4_resistance: Optional[float]

    # Momentum Indicators
    weekly_rsi: Optional[float]
    daily_rsi: Optional[float]
    h4_rsi: Optional[float]

    # Trend Strength
    weekly_adx: Optional[float]
    daily_adx: Optional[float]
    h4_adx: Optional[float]

    # Alignment Score
    alignment_score: float  # 0-100 (all TFs agree = 100)

    # Trading Recommendation
    allow_longs: bool
    allow_shorts: bool

    timestamp: datetime


class HigherTimeframeAnalyzer:
    """
    Analyzes higher timeframes to determine market context and bias
    Uses Weekly, Daily, and 4H for top-down analysis
    """

    def __init__(self, config):
        self.config = config

        # HTF timeframes in order of importance
        self.htf_timeframes = ['1w', '1d', '4h']

        # Thresholds
        self.strong_trend_adx = 25
        self.ranging_adx = 20
        self.oversold_rsi = 30
        self.overbought_rsi = 70

    def analyze_market_context(
        self,
        instrument: str,
        weekly_data: pd.DataFrame,
        daily_data: pd.DataFrame,
        h4_data: pd.DataFrame
    ) -> HTFContext:
        """
        Main analysis function - determines overall market context

        Args:
            instrument: Trading instrument
            weekly_data: Weekly OHLCV data with indicators
            daily_data: Daily OHLCV data with indicators
            h4_data: 4H OHLCV data with indicators

        Returns:
            HTFContext with complete market analysis
        """
        # Analyze each timeframe independently
        weekly_trend = self._determine_trend(weekly_data, 'weekly')
        daily_trend = self._determine_trend(daily_data, 'daily')
        h4_trend = self._determine_trend(h4_data, '4h')

        # Determine primary bias (weighted: W=50%, D=30%, 4H=20%)
        primary_bias, bias_strength = self._calculate_primary_bias(
            weekly_trend, daily_trend, h4_trend
        )

        # Determine market regime
        regime = self._determine_regime(weekly_data, daily_data, h4_data)

        # Identify key support/resistance levels
        weekly_support, weekly_resistance = self._identify_key_levels(weekly_data)
        daily_support, daily_resistance = self._identify_key_levels(daily_data)
        h4_support, h4_resistance = self._identify_key_levels(h4_data)

        # Get momentum indicators
        weekly_rsi = weekly_data['RSI_14'].iloc[-1] if 'RSI_14' in weekly_data.columns else None
        daily_rsi = daily_data['RSI_14'].iloc[-1] if 'RSI_14' in daily_data.columns else None
        h4_rsi = h4_data['RSI_14'].iloc[-1] if 'RSI_14' in h4_data.columns else None

        weekly_adx = weekly_data['ADX_14'].iloc[-1] if 'ADX_14' in weekly_data.columns else None
        daily_adx = daily_data['ADX_14'].iloc[-1] if 'ADX_14' in daily_data.columns else None
        h4_adx = h4_data['ADX_14'].iloc[-1] if 'ADX_14' in h4_data.columns else None

        # Calculate alignment score (0-100)
        alignment_score = self._calculate_alignment_score(
            weekly_trend, daily_trend, h4_trend
        )

        # Get current price for location-aware logic
        current_price = h4_data['close'].iloc[-1]

        # Determine what types of trades to allow (location-aware)
        allow_longs, allow_shorts = self._determine_trade_permissions(
            primary_bias, bias_strength, regime, current_price,
            weekly_support, weekly_resistance,
            daily_support, daily_resistance
        )

        context = HTFContext(
            instrument=instrument,
            weekly_trend=weekly_trend,
            daily_trend=daily_trend,
            h4_trend=h4_trend,
            primary_bias=primary_bias,
            bias_strength=bias_strength,
            regime=regime,
            weekly_support=weekly_support,
            weekly_resistance=weekly_resistance,
            daily_support=daily_support,
            daily_resistance=daily_resistance,
            h4_support=h4_support,
            h4_resistance=h4_resistance,
            weekly_rsi=weekly_rsi,
            daily_rsi=daily_rsi,
            h4_rsi=h4_rsi,
            weekly_adx=weekly_adx,
            daily_adx=daily_adx,
            h4_adx=h4_adx,
            alignment_score=alignment_score,
            allow_longs=allow_longs,
            allow_shorts=allow_shorts,
            timestamp=datetime.utcnow()
        )

        logger.info(f"HTF Context for {instrument}:")
        logger.info(f"  Primary Bias: {primary_bias.upper()} (strength: {bias_strength:.0f}%)")
        logger.info(f"  Regime: {regime}")
        logger.info(f"  Alignment: {alignment_score:.0f}%")
        logger.info(f"  Allow Longs: {allow_longs} | Allow Shorts: {allow_shorts}")

        return context

    def _determine_trend(self, data: pd.DataFrame, timeframe: str) -> str:
        """
        Determine trend direction using multiple indicators

        Returns: 'bullish', 'bearish', or 'neutral'
        """
        if len(data) < 50:
            return 'neutral'

        # Moving Average Analysis
        sma_20 = data['SMA_20'].iloc[-1] if 'SMA_20' in data.columns else data['close'].iloc[-20:].mean()
        sma_50 = data['SMA_50'].iloc[-1] if 'SMA_50' in data.columns else data['close'].iloc[-50:].mean()
        current_price = data['close'].iloc[-1]

        # MACD Analysis
        macd = data['MACD'].iloc[-1] if 'MACD' in data.columns else 0
        macd_signal = data['MACD_signal'].iloc[-1] if 'MACD_signal' in data.columns else 0

        # Score system (-3 to +3)
        score = 0

        # Price vs Moving Averages
        if current_price > sma_20:
            score += 1
        elif current_price < sma_20:
            score -= 1

        if current_price > sma_50:
            score += 1
        elif current_price < sma_50:
            score -= 1

        # Moving Average Slope
        if sma_20 > sma_50:
            score += 1
        elif sma_20 < sma_50:
            score -= 1

        # MACD
        if macd > macd_signal and macd > 0:
            score += 1
        elif macd < macd_signal and macd < 0:
            score -= 1

        # Recent price action (higher highs/lower lows)
        recent_high = data['high'].iloc[-10:].max()
        previous_high = data['high'].iloc[-20:-10].max()
        if recent_high > previous_high:
            score += 1
        elif recent_high < previous_high:
            score -= 1

        # Determine trend
        if score >= 2:
            return 'bullish'
        elif score <= -2:
            return 'bearish'
        else:
            return 'neutral'

    def _calculate_primary_bias(
        self,
        weekly_trend: str,
        daily_trend: str,
        h4_trend: str
    ) -> Tuple[str, float]:
        """
        Calculate primary bias with weighted voting
        Weekly = 50%, Daily = 30%, 4H = 20%

        Returns: (bias, strength)
        """
        # Convert to numeric scores
        trend_scores = {
            'bullish': 1,
            'neutral': 0,
            'bearish': -1
        }

        weekly_score = trend_scores[weekly_trend] * 0.50
        daily_score = trend_scores[daily_trend] * 0.30
        h4_score = trend_scores[h4_trend] * 0.20

        total_score = weekly_score + daily_score + h4_score

        # Determine bias
        if total_score >= 0.3:
            bias = 'bullish'
        elif total_score <= -0.3:
            bias = 'bearish'
        else:
            bias = 'neutral'

        # Calculate strength (0-100)
        strength = abs(total_score) * 100

        return bias, strength

    def _determine_regime(
        self,
        weekly_data: pd.DataFrame,
        daily_data: pd.DataFrame,
        h4_data: pd.DataFrame
    ) -> str:
        """
        Determine market regime
        Returns: 'trending', 'ranging', 'choppy', 'volatile'
        """
        # Use Daily ADX as primary regime indicator
        daily_adx = daily_data['ADX_14'].iloc[-1] if 'ADX_14' in daily_data.columns else 20

        # Calculate volatility (ATR relative to price)
        if 'ATR_14' in daily_data.columns:
            atr = daily_data['ATR_14'].iloc[-1]
            price = daily_data['close'].iloc[-1]
            volatility_pct = (atr / price) * 100
        else:
            volatility_pct = 2.0

        # Determine regime
        if daily_adx > 25:
            if volatility_pct > 3:
                return 'volatile'
            else:
                return 'trending'
        elif daily_adx > 20:
            return 'ranging'
        else:
            return 'choppy'

    def _identify_key_levels(self, data: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
        """
        Identify key support and resistance levels
        Uses swing highs/lows with pivot point detection

        Returns: (support, resistance)
        """
        if len(data) < 20:
            return None, None

        # Use larger lookback for more significant levels
        lookback = min(100, len(data))
        recent_data = data.iloc[-lookback:]

        # Find swing lows (support) - local minima
        swing_lows = []
        for i in range(2, len(recent_data) - 2):
            if (recent_data['low'].iloc[i] < recent_data['low'].iloc[i-1] and
                recent_data['low'].iloc[i] < recent_data['low'].iloc[i-2] and
                recent_data['low'].iloc[i] < recent_data['low'].iloc[i+1] and
                recent_data['low'].iloc[i] < recent_data['low'].iloc[i+2]):
                swing_lows.append(recent_data['low'].iloc[i])

        # Find swing highs (resistance) - local maxima
        swing_highs = []
        for i in range(2, len(recent_data) - 2):
            if (recent_data['high'].iloc[i] > recent_data['high'].iloc[i-1] and
                recent_data['high'].iloc[i] > recent_data['high'].iloc[i-2] and
                recent_data['high'].iloc[i] > recent_data['high'].iloc[i+1] and
                recent_data['high'].iloc[i] > recent_data['high'].iloc[i+2]):
                swing_highs.append(recent_data['high'].iloc[i])

        current_price = recent_data['close'].iloc[-1]

        # Find nearest support (below current price)
        support = None
        if swing_lows:
            below_price = [low for low in swing_lows if low < current_price]
            if below_price:
                support = max(below_price)  # Closest support below
            else:
                # No swing low below price, use percentile
                support = np.percentile(recent_data['low'].values, 5)
        else:
            # Fallback to percentile if no swings found
            support = np.percentile(recent_data['low'].values, 10)

        # Find nearest resistance (above current price)
        resistance = None
        if swing_highs:
            above_price = [high for high in swing_highs if high > current_price]
            if above_price:
                resistance = min(above_price)  # Closest resistance above
            else:
                # No swing high above price, use percentile
                resistance = np.percentile(recent_data['high'].values, 95)
        else:
            # Fallback to percentile
            resistance = np.percentile(recent_data['high'].values, 90)

        return float(support) if support else None, float(resistance) if resistance else None

    def _calculate_alignment_score(
        self,
        weekly_trend: str,
        daily_trend: str,
        h4_trend: str
    ) -> float:
        """
        Calculate how well all timeframes align (0-100)
        100 = perfect alignment, 0 = complete disagreement
        """
        trends = [weekly_trend, daily_trend, h4_trend]

        # Count how many agree
        bullish_count = trends.count('bullish')
        bearish_count = trends.count('bearish')
        neutral_count = trends.count('neutral')

        # Perfect alignment (all same)
        if bullish_count == 3 or bearish_count == 3:
            return 100.0

        # Strong alignment (2 out of 3 agree)
        if bullish_count == 2 or bearish_count == 2:
            return 70.0

        # Partial alignment
        if neutral_count == 0:
            return 30.0  # Mixed signals

        # Weak alignment (neutrals involved)
        return 50.0

    def check_price_location(
        self,
        current_price: float,
        support: Optional[float],
        resistance: Optional[float],
        threshold_pct: float = 0.015
    ) -> str:
        """
        Check if price is at support, resistance, or mid-range

        Args:
            current_price: Current market price
            support: Support level
            resistance: Resistance level
            threshold_pct: Percentage threshold (default 1.5%)

        Returns: 'at_support', 'at_resistance', 'mid_range'
        """
        if support and abs(current_price - support) / current_price < threshold_pct:
            return 'at_support'
        elif resistance and abs(current_price - resistance) / current_price < threshold_pct:
            return 'at_resistance'
        else:
            return 'mid_range'

    def _determine_trade_permissions(
        self,
        primary_bias: str,
        bias_strength: float,
        regime: str,
        current_price: float,
        weekly_support: Optional[float],
        weekly_resistance: Optional[float],
        daily_support: Optional[float],
        daily_resistance: Optional[float]
    ) -> Tuple[bool, bool]:
        """
        Determine what types of trades to allow based on HTF bias AND price location

        Key Logic:
        - Bullish HTF + at support = LONG only (trend continuation)
        - Bullish HTF + at resistance = SHORT allowed (mean reversion)
        - Bullish HTF + mid-range = LONG only (trend following)
        - Bearish HTF + at resistance = SHORT only (trend continuation)
        - Bearish HTF + at support = LONG allowed (mean reversion)
        - Bearish HTF + mid-range = SHORT only (trend following)

        Returns: (allow_longs, allow_shorts)
        """
        # Don't trade in very choppy markets
        if regime == 'choppy' and bias_strength < 30:
            return False, False

        # Check price location relative to weekly levels (most important)
        weekly_location = self.check_price_location(current_price, weekly_support, weekly_resistance)

        # Also check daily levels for finer granularity
        daily_location = self.check_price_location(current_price, daily_support, daily_resistance)

        # Default: no trades
        allow_longs = False
        allow_shorts = False

        # BULLISH HTF BIAS
        if primary_bias == 'bullish':

            # At weekly resistance - allow mean reversion shorts
            if weekly_location == 'at_resistance':
                allow_shorts = True  # Countertrend mean reversion
                allow_longs = False  # Don't buy at resistance

            # At weekly support - strong long signal
            elif weekly_location == 'at_support':
                allow_longs = True  # Trend continuation from support
                allow_shorts = False

            # Mid-range - check daily levels
            else:
                if daily_location == 'at_resistance':
                    # At daily resistance in bullish HTF
                    allow_shorts = True  # Scalp short for pullback
                    allow_longs = False
                elif daily_location == 'at_support':
                    # At daily support in bullish HTF
                    allow_longs = True  # Buy the dip
                    allow_shorts = False
                else:
                    # Mid-range on both - only trend-following longs
                    allow_longs = True
                    allow_shorts = False

        # BEARISH HTF BIAS
        elif primary_bias == 'bearish':

            # At weekly support - allow mean reversion longs
            if weekly_location == 'at_support':
                allow_longs = True  # Countertrend mean reversion
                allow_shorts = False  # Don't sell at support

            # At weekly resistance - strong short signal
            elif weekly_location == 'at_resistance':
                allow_shorts = True  # Trend continuation from resistance
                allow_longs = False

            # Mid-range - check daily levels
            else:
                if daily_location == 'at_support':
                    # At daily support in bearish HTF
                    allow_longs = True  # Scalp long for bounce
                    allow_shorts = False
                elif daily_location == 'at_resistance':
                    # At daily resistance in bearish HTF
                    allow_shorts = True  # Sell the rip
                    allow_longs = False
                else:
                    # Mid-range on both - only trend-following shorts
                    allow_shorts = True
                    allow_longs = False

        # NEUTRAL HTF BIAS - Range trading
        else:
            if regime == 'ranging':
                # In ranging neutral market, trade the range
                if weekly_location == 'at_support' or daily_location == 'at_support':
                    allow_longs = True  # Buy at support
                elif weekly_location == 'at_resistance' or daily_location == 'at_resistance':
                    allow_shorts = True  # Sell at resistance
                else:
                    # Mid-range in neutral ranging = wait
                    allow_longs = False
                    allow_shorts = False
            else:
                # Neutral but not ranging (choppy/volatile) - allow both with caution
                allow_longs = True
                allow_shorts = True

        return allow_longs, allow_shorts

    def _old_determine_trade_permissions(
        self,
        primary_bias: str,
        bias_strength: float,
        regime: str
    ) -> Tuple[bool, bool]:
        """
        OLD METHOD - Kept for reference
        Determine what types of trades to allow

        Returns: (allow_longs, allow_shorts)
        """
        # Don't trade in choppy markets
        if regime == 'choppy':
            return False, False

        # Strong bias - only trade with the bias
        if bias_strength >= 60:
            if primary_bias == 'bullish':
                return True, False  # Only longs
            elif primary_bias == 'bearish':
                return False, True  # Only shorts
            else:
                return True, True  # Neutral, allow both

        # Medium bias - prefer bias direction but allow counter-trend in ranging
        elif bias_strength >= 40:
            if regime == 'ranging':
                return True, True  # Range trading allows both
            elif primary_bias == 'bullish':
                return True, False
            elif primary_bias == 'bearish':
                return False, True
            else:
                return True, True

        # Weak bias - allow both directions
        else:
            return True, True

    def check_signal_alignment(
        self,
        signal_direction: str,
        htf_context: HTFContext
    ) -> Tuple[bool, str]:
        """
        Check if a signal is aligned with HTF context

        Args:
            signal_direction: 'long' or 'short'
            htf_context: Current HTF context

        Returns:
            (is_aligned, reason)
        """
        # Check if this direction is allowed
        if signal_direction == 'long' and not htf_context.allow_longs:
            return False, f"HTF bias is {htf_context.primary_bias} - longs not allowed"

        if signal_direction == 'short' and not htf_context.allow_shorts:
            return False, f"HTF bias is {htf_context.primary_bias} - shorts not allowed"

        # Check alignment score
        if htf_context.alignment_score < 50:
            return False, f"Low HTF alignment ({htf_context.alignment_score:.0f}%) - conflicting timeframes"

        # Check regime
        if htf_context.regime == 'choppy':
            return False, "Market regime is choppy - no clear direction"

        # Signal is aligned
        return True, f"Aligned with {htf_context.primary_bias} bias (strength: {htf_context.bias_strength:.0f}%)"

    def format_context_summary(self, context: HTFContext) -> str:
        """Format HTF context as readable summary"""
        summary = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ HIGHER TIMEFRAME CONTEXT: {context.instrument:<45} ║
╚══════════════════════════════════════════════════════════════════════════════╝

📊 TREND ANALYSIS:
   Weekly (W):  {context.weekly_trend.upper():>10} | ADX: {context.weekly_adx:.1f if context.weekly_adx else 0:.1f} | RSI: {context.weekly_rsi:.1f if context.weekly_rsi else 0:.1f}
   Daily (D):   {context.daily_trend.upper():>10} | ADX: {context.daily_adx:.1f if context.daily_adx else 0:.1f} | RSI: {context.daily_rsi:.1f if context.daily_rsi else 0:.1f}
   4-Hour (4H): {context.h4_trend.upper():>10} | ADX: {context.h4_adx:.1f if context.h4_adx else 0:.1f} | RSI: {context.h4_rsi:.1f if context.h4_rsi else 0:.1f}

🎯 PRIMARY BIAS: {context.primary_bias.upper()}
   Strength: {context.bias_strength:.0f}%
   Alignment: {context.alignment_score:.0f}%
   Regime: {context.regime.upper()}

📍 KEY LEVELS:
   Weekly:  Support ${context.weekly_support:,.2f if context.weekly_support else 0:,.2f} | Resistance ${context.weekly_resistance:,.2f if context.weekly_resistance else 0:,.2f}
   Daily:   Support ${context.daily_support:,.2f if context.daily_support else 0:,.2f} | Resistance ${context.daily_resistance:,.2f if context.daily_resistance else 0:,.2f}
   4-Hour:  Support ${context.h4_support:,.2f if context.h4_support else 0:,.2f} | Resistance ${context.h4_resistance:,.2f if context.h4_resistance else 0:,.2f}

✅ TRADE PERMISSIONS:
   Longs:  {'ALLOWED' if context.allow_longs else 'BLOCKED'}
   Shorts: {'ALLOWED' if context.allow_shorts else 'BLOCKED'}

═══════════════════════════════════════════════════════════════════════════════
"""
        return summary


def main():
    """Example usage"""
    import sys
    sys.path.append("../..")
    import config
    from src.backtest.backtest_engine import BacktestEngine

    analyzer = HigherTimeframeAnalyzer(config)
    engine = BacktestEngine(config)

    # Load data for different timeframes
    print("Loading HTF data...")

    # For testing, use 30m/15m/5m since we have that data
    # In production, would use W/D/4H
    weekly_data = engine.load_data("BTCUSDT.P", "30m")
    daily_data = engine.load_data("BTCUSDT.P", "15m")
    h4_data = engine.load_data("BTCUSDT.P", "5m")

    if not weekly_data.empty and not daily_data.empty and not h4_data.empty:
        # Analyze context
        context = analyzer.analyze_market_context(
            instrument="BTCUSDT.P",
            weekly_data=weekly_data,
            daily_data=daily_data,
            h4_data=h4_data
        )

        # Print summary
        print(analyzer.format_context_summary(context))

        # Test signal alignment
        print("\nTesting Signal Alignment:")
        for direction in ['long', 'short']:
            aligned, reason = analyzer.check_signal_alignment(direction, context)
            status = "✅ ALIGNED" if aligned else "❌ BLOCKED"
            print(f"  {direction.upper():5} {status}: {reason}")
    else:
        print("Could not load data for analysis")


if __name__ == "__main__":
    main()
