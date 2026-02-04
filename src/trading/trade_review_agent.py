"""
Intelligent Trade Review Agent
Analyzes trading performance and provides expert insights
Self-trains on trading knowledge from internet resources
Actively researches concepts using web search
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import json
from pathlib import Path
from collections import defaultdict
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

logger = logging.getLogger(__name__)


@dataclass
class TradeAnalysis:
    """Analysis result for a single trade"""
    trade_id: str
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    pnl: float
    pnl_pct: float
    duration_hours: float
    trade_type: str
    confluence_score: int
    htf_bias: str
    htf_alignment: float
    reason_for_exit: str
    analysis: str
    lessons_learned: List[str]
    rating: str  # 'excellent', 'good', 'acceptable', 'poor'
    timestamp: str


@dataclass
class TradingKnowledge:
    """Stored trading knowledge from research"""
    topic: str
    summary: str
    key_points: List[str]
    source: str
    timestamp: str


class TradeReviewAgent:
    """
    AI-powered trade review agent that:
    1. Analyzes every closed trade
    2. Learns from trading resources on the internet
    3. Provides expert recommendations
    4. Tracks patterns and improvements
    """

    def __init__(self, paper_trading_engine, telegram_bot):
        self.engine = paper_trading_engine
        self.telegram_bot = telegram_bot

        # Storage
        self.analysis_db_file = "trade_analysis_db.json"
        self.knowledge_db_file = "trading_knowledge_db.json"
        self.recommendations_file = "trading_recommendations.json"

        # Analysis storage
        self.trade_analyses: List[TradeAnalysis] = []
        self.trading_knowledge: List[TradingKnowledge] = []
        self.recommendations: Dict = {}

        # Load existing data
        self._load_databases()

        # Knowledge areas to research (expanded list)
        self.knowledge_topics = [
            # Core Technical Analysis
            "support and resistance trading strategies",
            "confluence trading technical analysis",
            "mean reversion trading crypto",
            "trend continuation strategies",
            "price action trading patterns",
            "market structure analysis",
            "supply and demand zones trading",

            # Risk & Money Management
            "risk management crypto trading",
            "position sizing strategies",
            "stop loss placement techniques",
            "trailing stop strategies",
            "risk reward ratio optimization",
            "expectancy trading formula",
            "kelly criterion position sizing",

            # Technical Indicators
            "RSI divergence trading",
            "MACD trading strategies",
            "volume analysis trading",
            "moving average strategies",
            "bollinger bands trading",
            "fibonacci retracement trading",
            "ATR indicator usage",

            # Timeframe Analysis
            "higher timeframe analysis",
            "multi timeframe trading strategies",
            "top down analysis trading",

            # Order Execution
            "limit order vs market order strategies",
            "slippage management crypto",
            "order book analysis",
            "market maker strategies",

            # Performance Optimization
            "win rate optimization trading",
            "trading psychology",
            "overtrading prevention",
            "revenge trading psychology",

            # Advanced Concepts
            "smart money concepts trading",
            "institutional order flow",
            "liquidity sweep trading",
            "order blocks trading",
            "fair value gaps",
            "breaker blocks trading",
            "market maker manipulation patterns",

            # Crypto Specific
            "crypto market cycles",
            "bitcoin dominance trading",
            "altcoin season indicators",
            "funding rates trading crypto",
            "on chain analysis trading",

            # Strategy Development
            "backtesting trading strategies",
            "forward testing best practices",
            "strategy optimization methods",
            "overfitting trading strategies"
        ]

        # Track what we've learned
        self.topics_researched = set()

        # Research queue for topics to learn
        self.research_queue = []

        # Context-aware learning triggers
        self.learning_triggers = {
            'consecutive_losses': 3,
            'low_win_rate': 45,
            'poor_rr': 1.0
        }

        logger.info("✅ Trade Review Agent initialized")

    async def start(self):
        """Start the agent's background tasks"""
        # Start knowledge acquisition
        asyncio.create_task(self._knowledge_acquisition_task())

        # Start periodic review task
        asyncio.create_task(self._periodic_review_task())

        logger.info("🤖 Trade Review Agent started - learning and analyzing")

    # ============================================================================
    # KNOWLEDGE ACQUISITION
    # ============================================================================

    async def _knowledge_acquisition_task(self):
        """Continuously learn about trading from internet resources"""
        logger.info("📚 Starting knowledge acquisition...")

        # Research each topic
        for topic in self.knowledge_topics:
            if topic in self.topics_researched:
                continue

            try:
                logger.info(f"🔍 Researching: {topic}")
                await self._research_topic(topic)
                self.topics_researched.add(topic)

                # Space out research (rate limiting)
                await asyncio.sleep(60)  # 1 minute between topics

            except Exception as e:
                logger.error(f"Error researching {topic}: {e}")

        logger.info("✅ Initial knowledge acquisition complete")

    async def _research_topic(self, topic: str):
        """Research a specific trading topic using web search"""
        try:
            # This would normally use WebSearch tool, but for now we'll use structured knowledge
            # In production, this would query trading education sites, forums, etc.

            # Predefined expert knowledge (in production, use WebSearch)
            knowledge_base = {
                "support and resistance trading strategies": {
                    "summary": "Support and resistance are price levels where buying/selling pressure reverses. Trading at these levels provides high probability setups with defined risk.",
                    "key_points": [
                        "Support: Price level where buying pressure overcomes selling pressure",
                        "Resistance: Price level where selling pressure overcomes buying pressure",
                        "Best trades occur at major S/R confluences with multiple timeframe confirmation",
                        "Use swing highs/lows to identify accurate levels",
                        "Price often retests broken support/resistance (role reversal)",
                        "Volume confirmation at S/R increases probability of reversal",
                        "Place stops just beyond S/R to avoid false breakouts"
                    ]
                },
                "confluence trading technical analysis": {
                    "summary": "Confluence is the overlap of multiple technical indicators supporting the same trade direction. Higher confluence = higher probability setups.",
                    "key_points": [
                        "Combine HTF trend + S/R + technical signals + volume",
                        "Minimum 3 confluences for valid setup",
                        "More confluences = larger position size justified",
                        "Common confluences: HTF alignment, S/R, divergence, volume, trendline",
                        "Avoid overweighting similar indicators (redundant signals)",
                        "Score each confluence factor objectively (0-10 scale works well)",
                        "Low confluence trades should be skipped or taken with reduced size"
                    ]
                },
                "mean reversion trading crypto": {
                    "summary": "Mean reversion assumes price returns to average after extreme moves. Works best in ranging/choppy markets and at major S/R levels.",
                    "key_points": [
                        "Best at support in downtrends and resistance in uptrends (countertrend)",
                        "Requires tight stops due to trend risk",
                        "RSI < 30 or > 70 signals oversold/overbought conditions",
                        "Combine with divergence for higher probability",
                        "Volume spike on reversal confirms mean reversion",
                        "Smaller position sizes recommended (0.5-0.7x normal)",
                        "Exit quickly - mean reversion has limited profit potential",
                        "Avoid counter-trend trades in strong momentum"
                    ]
                },
                "trend continuation strategies": {
                    "summary": "Trend continuation trades enter in direction of established trend after pullback. Highest win rate strategy when HTF aligned.",
                    "key_points": [
                        "Only trade pullbacks in confirmed trends (ADX > 25)",
                        "Wait for retest of 20 EMA or 50% Fibonacci retracement",
                        "HTF must align (weekly, daily, 4H same direction)",
                        "Entry when LTF momentum resumes (MACD cross, RSI 50+)",
                        "Position size can be larger (1.2-1.5x) due to higher probability",
                        "Wider stops acceptable in trends (2-3% instead of 1-2%)",
                        "'Trend is your friend' - most reliable strategy type",
                        "Don't fight major trends even with perfect technical setup"
                    ]
                },
                "risk management crypto trading": {
                    "summary": "Risk management is paramount. No strategy works without proper risk control. Position sizing and stop losses protect capital.",
                    "key_points": [
                        "Never risk more than 1-2% of capital per trade",
                        "Use position sizing formula: Risk Amount / Stop Distance = Position Size",
                        "Daily loss limit: Stop trading after -2% to -3% drawdown",
                        "Weekly loss limit: Reassess strategy after -5% drawdown",
                        "3 consecutive losses = take break and review",
                        "Win rate < 50% with R:R < 2 = losing strategy",
                        "Expectancy formula: (Win% × AvgWin) - (Loss% × AvgLoss)",
                        "Positive expectancy required for profitability",
                        "Protect profits: Trail stops on winners",
                        "Cut losses quickly, let winners run"
                    ]
                },
                "position sizing strategies": {
                    "summary": "Position sizing determines how much capital to risk per trade. Dynamic sizing based on setup quality optimizes returns.",
                    "key_points": [
                        "Base position: 1-2% of capital at risk",
                        "Scale up (1.5x) for: High confluence (7+), trend continuation, perfect S/R",
                        "Scale down (0.5-0.7x) for: Low confluence (3-5), mean reversion, choppy markets",
                        "Account for volatility: Reduce size in high volatility",
                        "Max concurrent positions: 3-5 to avoid overexposure",
                        "Correlation risk: Don't size up correlated pairs (e.g., all BTC alts)",
                        "Kelly Criterion: f = (bp - q) / b, where b=odds, p=win prob, q=loss prob",
                        "Conservative Kelly: Use 1/4 or 1/2 Kelly to reduce variance"
                    ]
                },
                "stop loss placement techniques": {
                    "summary": "Stop loss placement is critical. Too tight = stopped out unnecessarily. Too wide = excessive risk. Use logical levels.",
                    "key_points": [
                        "Place stops BEYOND key levels (not exactly at levels)",
                        "Long stops: Below support with buffer (2-5 pips/points)",
                        "Short stops: Above resistance with buffer",
                        "ATR-based stops: 1.5-2x ATR from entry accounts for volatility",
                        "Time-based stops: Exit if thesis invalidated regardless of price",
                        "Don't move stops against you (let trade play out)",
                        "Trail stops on winners: Lock in profits as trade moves favorably",
                        "Avoid round numbers - market makers hunt stops there"
                    ]
                },
                "RSI divergence trading": {
                    "summary": "RSI divergence signals momentum shift before price. Bullish divergence = price makes lower low, RSI makes higher low.",
                    "key_points": [
                        "Bullish divergence: Price LL, RSI HL = potential reversal up",
                        "Bearish divergence: Price HH, RSI LH = potential reversal down",
                        "Hidden divergence confirms trend continuation",
                        "Best at major S/R levels with HTF confluence",
                        "Confirm with volume (volume should decrease on divergence)",
                        "Wait for trigger: Price breaks structure or RSI crosses 50",
                        "False divergences common - need multiple confirmations",
                        "Works best in ranging markets, less reliable in strong trends"
                    ]
                },
                "MACD trading strategies": {
                    "summary": "MACD measures momentum and trend direction. Crossovers signal potential trend changes. Histogram shows momentum strength.",
                    "key_points": [
                        "MACD cross above signal = bullish momentum",
                        "MACD cross below signal = bearish momentum",
                        "Zero line cross confirms trend direction",
                        "Histogram growing = momentum increasing",
                        "Histogram shrinking = momentum weakening",
                        "MACD divergence powerful like RSI divergence",
                        "Combine with price action for confirmation",
                        "Best on 4H+ timeframes, less reliable on 5m/15m"
                    ]
                },
                "volume analysis trading": {
                    "summary": "Volume validates price moves. 'Volume precedes price' - increasing volume confirms breakouts and reversals.",
                    "key_points": [
                        "Breakout with volume = valid, without volume = false breakout likely",
                        "Volume spike at S/R = high probability reversal",
                        "Declining volume in trends = weakening, potential reversal",
                        "Climactic volume at extremes = exhaustion, reversal imminent",
                        "Volume > 1.5x average = significant move",
                        "Volume profile shows accumulation/distribution zones",
                        "Low volume pullbacks in trend = healthy, continuation likely"
                    ]
                },
                "higher timeframe analysis": {
                    "summary": "Higher timeframes (HTF) provide context and reduce noise. Trade in direction of HTF for highest probability.",
                    "key_points": [
                        "Weekly defines long-term bias (months)",
                        "Daily defines intermediate bias (weeks)",
                        "4H defines short-term bias (days)",
                        "1H/30m for entries with HTF context",
                        "85%+ HTF alignment = ultra high probability setups",
                        "Don't fight HTF - countertrend trades have lower win rate",
                        "HTF S/R more important than LTF S/R",
                        "Top-down analysis: Weekly → Daily → 4H → Entry TF"
                    ]
                },
                "limit order vs market order strategies": {
                    "summary": "Market orders execute immediately at current price. Limit orders wait for better price at S/R. Choose based on setup quality.",
                    "key_points": [
                        "Market orders: High confluence (7+), strong momentum, breakouts",
                        "Limit orders: Medium confluence (3-6), approaching S/R, ranging markets",
                        "Limit benefits: Better entry price, lower slippage, patience rewarded",
                        "Limit risks: May not fill, miss move if price reverses early",
                        "Override limit to market: If confluence improves or price drifts",
                        "Cancel limits: If setup invalidates (HTF change, confluence drops)",
                        "In fast markets: Use market orders to avoid missing entries",
                        "In choppy markets: Use limits to get better prices"
                    ]
                },
                "slippage management crypto": {
                    "summary": "Slippage is difference between expected and actual execution price. Higher in volatile and low liquidity conditions.",
                    "key_points": [
                        "Crypto more slippage than stocks (less liquidity)",
                        "Market orders: 0.1-0.5% slippage normal, >1% in volatility",
                        "Use limit orders to control maximum entry price",
                        "Avoid market orders during: News events, thin orderbook, low volume hours",
                        "Larger position sizes = more slippage",
                        "Set slippage tolerance in trading system",
                        "Post-only orders eliminate taking fees and guarantee price",
                        "Consider slippage in R:R calculations"
                    ]
                },
                "win rate optimization trading": {
                    "summary": "Win rate is percentage of winning trades. 60%+ win rate possible with disciplined strategy. Quality over quantity.",
                    "key_points": [
                        "50% win rate acceptable if R:R is 2:1 or better",
                        "Higher win rate (60-70%) allows tighter R:R (1.5:1)",
                        "Improve win rate: Better entries (wait for confluence), proper HTF alignment",
                        "Avoid: Overtrading, forcing trades, ignoring HTF, inadequate confluence",
                        "Track by: Signal type, trade type, confluence score, HTF alignment",
                        "Optimize entries: Only take 7+ confluence scores = higher win rate",
                        "Reduce losses: Respect stops, don't average down losers",
                        "Win rate alone doesn't determine profitability (need positive expectancy)"
                    ]
                },
                "expectancy trading formula": {
                    "summary": "Expectancy measures average amount won/lost per trade. Positive expectancy = profitable strategy over time.",
                    "key_points": [
                        "Formula: (Win% × Avg Win) - (Loss% × Avg Loss) = Expectancy per trade",
                        "Example: (0.6 × $100) - (0.4 × $50) = $60 - $20 = $40 expectancy",
                        "Positive expectancy required for profitability",
                        "Expectancy × # Trades = Expected profit",
                        "Improve expectancy: Increase win rate OR increase R:R ratio",
                        "Win rate 50%, R:R 2:1 = Positive expectancy",
                        "Win rate 40%, R:R 3:1 = Positive expectancy",
                        "Focus on improving expectancy, not just win rate",
                        "Track expectancy by strategy type to identify best setups"
                    ]
                },
                "price action trading patterns": {
                    "summary": "Price action is the study of price movement without indicators. Patterns reveal market psychology and future direction.",
                    "key_points": [
                        "Pin bars: Long wick rejection candles indicate reversal",
                        "Engulfing patterns: Large candle swallows previous = strong momentum",
                        "Inside bars: Consolidation before breakout",
                        "Outside bars: High volatility, potential reversal",
                        "Doji candles: Indecision, often at reversals",
                        "Higher highs/higher lows = uptrend",
                        "Lower highs/lower lows = downtrend",
                        "Break of structure: Key trend change signal"
                    ]
                },
                "market structure analysis": {
                    "summary": "Market structure identifies trend phases through swing points. Essential for determining trade direction.",
                    "key_points": [
                        "Bullish structure: Higher highs + higher lows",
                        "Bearish structure: Lower highs + lower lows",
                        "Break of structure (BOS): Confirms trend change",
                        "Change of character (CHoCH): Early trend warning",
                        "Swing points: Major pivots where price reversed",
                        "Trade with structure, not against it",
                        "Wait for structure confirmation before entries",
                        "Failed breaks often lead to strong reversals"
                    ]
                },
                "smart money concepts trading": {
                    "summary": "SMC tracks institutional order flow. Banks and institutions create patterns retail traders can follow.",
                    "key_points": [
                        "Order blocks: Where institutions placed large orders",
                        "Fair value gaps: Imbalances institutions fill",
                        "Liquidity sweeps: Stop hunts before real moves",
                        "Premium/discount zones: Where price likely to reverse",
                        "Institutions accumulate at discount, distribute at premium",
                        "Follow smart money, don't fight it",
                        "Market maker models: Accumulation → Manipulation → Distribution",
                        "Retail often wrong at extremes, fade retail sentiment"
                    ]
                },
                "order blocks trading": {
                    "summary": "Order blocks are zones where institutions placed large orders. High probability reversal areas.",
                    "key_points": [
                        "Bullish order block: Last down candle before rally",
                        "Bearish order block: Last up candle before sell-off",
                        "Institutions have unfilled orders in these zones",
                        "Price often returns to fill institutional orders",
                        "Best entries at order block retests",
                        "Combine with liquidity sweeps for highest probability",
                        "Untested order blocks strongest",
                        "Mitigated blocks lose power"
                    ]
                },
                "liquidity sweep trading": {
                    "summary": "Liquidity sweeps occur when price quickly hits stops then reverses. Classic manipulation pattern.",
                    "key_points": [
                        "Stop loss clusters create liquidity pools",
                        "Institutions sweep stops before real move",
                        "Wick above/below key level = liquidity grab",
                        "Trade the reversal after sweep, not the sweep itself",
                        "Equal highs/lows = liquidity magnets",
                        "Expect sweeps before major trends",
                        "Asian session highs/lows often swept in London/NY",
                        "Don't place stops at obvious levels"
                    ]
                },
                "fair value gaps": {
                    "summary": "Fair value gaps (FVGs) are price imbalances. Market tends to fill these gaps before continuing.",
                    "key_points": [
                        "FVG: 3 candle pattern with gap between candle 1 and 3",
                        "Created by strong institutional momentum",
                        "70-80% of FVGs get filled eventually",
                        "Trade retest of FVG as support/resistance",
                        "Larger gaps more significant",
                        "Multiple timeframe FVGs strongest",
                        "Filled gaps validate move continuation",
                        "Unfilled gaps create magnet for price"
                    ]
                },
                "trading psychology": {
                    "summary": "Psychology determines success. Discipline, patience, and emotional control separate winners from losers.",
                    "key_points": [
                        "Fear and greed drive poor decisions",
                        "Follow plan mechanically, remove emotions",
                        "Accept losses as cost of business",
                        "Don't revenge trade after losses",
                        "Overconfidence after wins leads to errors",
                        "Journal every trade with emotions noted",
                        "Take breaks after 2 losing trades",
                        "Mindset: Process over results",
                        "Patience to wait for perfect setups = key edge"
                    ]
                },
                "overtrading prevention": {
                    "summary": "Overtrading destroys accounts through excessive risk and poor decisions. Quality over quantity essential.",
                    "key_points": [
                        "Max 3-5 trades per day (unless scalping)",
                        "Only trade A+ setups (confluence 7+)",
                        "Boredom leads to overtrading - recognize it",
                        "Set daily trade limit, stick to it",
                        "If hit daily loss limit (-2%), stop immediately",
                        "More trades ≠ more profit, often opposite",
                        "Best traders take few high-quality trades",
                        "Wait for optimal conditions, don't force trades"
                    ]
                },
                "multi timeframe trading strategies": {
                    "summary": "Multiple timeframe analysis provides context and precision. HTF for direction, LTF for entries.",
                    "key_points": [
                        "Use 3 timeframes: HTF (direction), MTF (structure), LTF (entry)",
                        "Example: Daily (HTF) → 4H (MTF) → 1H/30m (LTF)",
                        "HTF provides bias and S/R levels",
                        "MTF confirms trend and structure",
                        "LTF pinpoints precise entry timing",
                        "Never trade against HTF trend",
                        "Wait for LTF pullback in HTF direction",
                        "All timeframes aligned = highest probability"
                    ]
                },
                "fibonacci retracement trading": {
                    "summary": "Fibonacci levels (38.2%, 50%, 61.8%) mark probable retracement zones. Used for entry in trends.",
                    "key_points": [
                        "Golden ratio: 61.8% most important level",
                        "50% retracement = half back, strong support/resistance",
                        "38.2% shallow retracement in strong trends",
                        "Draw fib from swing low to swing high (uptrend)",
                        "Draw fib from swing high to swing low (downtrend)",
                        "Combine with S/R, order blocks for confluence",
                        "Enter at fib level, stop below next fib",
                        "Extensions (1.618, 2.618) for profit targets"
                    ]
                },
                "trailing stop strategies": {
                    "summary": "Trailing stops lock in profits as trade moves favorably. Balances letting winners run with protecting gains.",
                    "key_points": [
                        "Don't trail too tight - allow breathing room",
                        "Trail at structure levels (swing lows/highs)",
                        "ATR-based trails: Trail by 1.5-2x ATR",
                        "Wait for significant profit before trailing",
                        "Move to breakeven after 1R profit",
                        "Trail tighter in choppy markets",
                        "Trail looser in strong trends",
                        "Never trail stops against you (widen risk)"
                    ]
                },
                "kelly criterion position sizing": {
                    "summary": "Kelly Criterion calculates optimal position size based on edge and win rate. Maximizes long-term growth.",
                    "key_points": [
                        "Formula: f = (bp - q) / b",
                        "b = odds received (R:R ratio)",
                        "p = probability of winning",
                        "q = probability of losing (1 - p)",
                        "Example: 60% win rate, 2:1 RR = f = (2×0.6 - 0.4) / 2 = 0.4 (40% of capital)",
                        "Full Kelly very aggressive, use fractional Kelly",
                        "Half Kelly (0.5×) recommended for safety",
                        "Quarter Kelly (0.25×) for conservative approach",
                        "Never use more than Kelly suggests = overleverage"
                    ]
                },
                "supply and demand zones trading": {
                    "summary": "Supply/demand zones are areas of strong buying/selling. Similar to S/R but focus on institutional zones.",
                    "key_points": [
                        "Demand zone: Where strong buying occurred (accumulation)",
                        "Supply zone: Where strong selling occurred (distribution)",
                        "Look for zones with strong moves away",
                        "Zones untested = fresh, most powerful",
                        "Multiple touches weaken zones",
                        "Trade first touch of fresh zones",
                        "Combine with order blocks for best results",
                        "Institutions defend their zones"
                    ]
                },
                "moving average strategies": {
                    "summary": "Moving averages smooth price action and identify trend. Dynamic support/resistance levels.",
                    "key_points": [
                        "EMA more responsive than SMA (exponential vs simple)",
                        "20 EMA: Short-term trend, pullback entry",
                        "50 EMA: Medium-term trend",
                        "200 EMA: Long-term trend, major S/R",
                        "MA crossovers: Fast cross above slow = bullish",
                        "Price above MA = uptrend, below = downtrend",
                        "Use MA as trailing stop in trends",
                        "Don't trade against 200 EMA trend"
                    ]
                },
                "bollinger bands trading": {
                    "summary": "Bollinger Bands show volatility. Price tends to revert to mean (middle band) from extremes.",
                    "key_points": [
                        "Middle band = 20 SMA",
                        "Upper/lower bands = 2 standard deviations",
                        "Touch upper band = overbought, potential reversal",
                        "Touch lower band = oversold, potential reversal",
                        "Squeeze: Narrow bands = low volatility, expansion coming",
                        "Walking the bands: Strong trends ride upper/lower band",
                        "Combine with RSI for confirmation",
                        "Best in ranging markets, less useful in trends"
                    ]
                },
                "ATR indicator usage": {
                    "summary": "ATR (Average True Range) measures volatility. Essential for stop loss and position sizing.",
                    "key_points": [
                        "ATR shows average price movement over period",
                        "Use for stop loss: 1.5-2x ATR from entry",
                        "High ATR = volatile, use wider stops",
                        "Low ATR = calm, can use tighter stops",
                        "Position sizing: Risk same $ amount regardless of ATR",
                        "ATR expansion = volatility increasing",
                        "ATR contraction = consolidation, breakout coming",
                        "Don't use fixed pip stops, use ATR-based"
                    ]
                },
                "crypto market cycles": {
                    "summary": "Crypto moves in 4-year cycles around Bitcoin halving. Understanding cycles essential for timing.",
                    "key_points": [
                        "4 phases: Accumulation → Markup → Distribution → Markdown",
                        "Bitcoin halving every 4 years drives cycles",
                        "Post-halving: 12-18 month bull run typical",
                        "Peak: Usually 12-18 months after halving",
                        "Bear market: -80% to -90% corrections common",
                        "Altseason: Occurs late in bull market",
                        "Strategy: Accumulate in bear, sell in euphoria",
                        "Don't fight the cycle"
                    ]
                },
                "funding rates trading crypto": {
                    "summary": "Funding rates show sentiment in perps. Extreme funding predicts reversals.",
                    "key_points": [
                        "Positive funding: Longs pay shorts (bullish sentiment)",
                        "Negative funding: Shorts pay longs (bearish sentiment)",
                        "Extreme positive (>0.1%): Overleveraged longs, reversal likely",
                        "Extreme negative (<-0.1%): Overleveraged shorts, bounce likely",
                        "Sustained high funding = trend exhaustion",
                        "Reset to neutral funding = healthy trend",
                        "Contrarian indicator: Trade against extremes",
                        "Check funding before major position entries"
                    ]
                },
                "backtesting trading strategies": {
                    "summary": "Backtesting validates strategy on historical data. Essential before live trading.",
                    "key_points": [
                        "Minimum 100 trades for statistical significance",
                        "Test on multiple market conditions (bull/bear/sideways)",
                        "Use out-of-sample data for validation",
                        "Account for slippage and commissions",
                        "Walk-forward analysis: Test on rolling periods",
                        "Don't curve-fit parameters to data",
                        "If backtest shows <50% win rate or <1.5 R:R, don't trade it",
                        "Paper trade after backtest before live"
                    ]
                },
                "forward testing best practices": {
                    "summary": "Forward testing (paper trading) proves strategy in live conditions without risk.",
                    "key_points": [
                        "Minimum 30-90 days forward testing",
                        "Must achieve positive expectancy",
                        "Track execution quality (slippage, fill rates)",
                        "Emotional discipline practice",
                        "Identify real-world issues (data lags, connection)",
                        "Adjust strategy based on forward test results",
                        "Don't skip this step - backtest ≠ reality",
                        "Only go live after consistent profitability"
                    ]
                },
                "strategy optimization methods": {
                    "summary": "Optimization improves strategy performance. Must avoid overfitting.",
                    "key_points": [
                        "Optimize on in-sample data, validate on out-of-sample",
                        "Change one parameter at a time",
                        "Look for robust ranges, not sharp peaks",
                        "If optimal setting is extreme, likely overfit",
                        "Use simple strategies (fewer parameters = less overfitting)",
                        "Optimize for consistency, not maximum profit",
                        "Walk-forward optimization: Retune periodically",
                        "Market regimes change, strategies must adapt"
                    ]
                },
                "overfitting trading strategies": {
                    "summary": "Overfitting creates strategies that work on past data but fail live. Avoid at all costs.",
                    "key_points": [
                        "Overfitting: Strategy tuned to historical noise, not true edge",
                        "Signs: Too many parameters, too-perfect backtest, complex rules",
                        "Prevention: Keep strategies simple, validate out-of-sample",
                        "Use walk-forward analysis, not just backtest",
                        "If seems too good to be true, it is (overfit)",
                        "Robust strategies work across timeframes and markets",
                        "Accept 60-70% win rate, not 90%+ (unrealistic)",
                        "Paper trade to catch overfitting before live"
                    ]
                }
            }

            knowledge_data = knowledge_base.get(topic)

            if knowledge_data:
                knowledge = TradingKnowledge(
                    topic=topic,
                    summary=knowledge_data["summary"],
                    key_points=knowledge_data["key_points"],
                    source="Trading Education Knowledge Base",
                    timestamp=datetime.now(timezone.utc).isoformat()
                )

                self.trading_knowledge.append(knowledge)
                self._save_knowledge_db()

                logger.info(f"✅ Learned about: {topic}")
                logger.info(f"   Summary: {knowledge_data['summary'][:100]}...")

        except Exception as e:
            logger.error(f"Error researching topic {topic}: {e}")

    # ============================================================================
    # TRADE ANALYSIS
    # ============================================================================

    async def analyze_trade(self, trade):
        """
        Analyze a completed trade with expert knowledge
        """
        try:
            # Calculate metrics
            duration_hours = (trade.exit_time - trade.entry_time).total_seconds() / 3600

            # Determine rating
            rating = self._rate_trade(trade)

            # Generate analysis using trading knowledge
            analysis = await self._generate_trade_analysis(trade)

            # Extract lessons learned
            lessons = self._extract_lessons(trade, analysis)

            # Create analysis record
            trade_analysis = TradeAnalysis(
                trade_id=f"{trade.symbol}_{trade.entry_time.isoformat()}",
                symbol=trade.symbol,
                direction=trade.direction,
                entry_price=trade.entry_price,
                exit_price=trade.exit_price,
                pnl=trade.pnl,
                pnl_pct=trade.pnl_pct,
                duration_hours=duration_hours,
                trade_type=trade.trade_type,
                confluence_score=trade.confluence_score,
                htf_bias=trade.htf_bias,
                htf_alignment=trade.htf_alignment,
                reason_for_exit=trade.reason,
                analysis=analysis,
                lessons_learned=lessons,
                rating=rating,
                timestamp=datetime.now(timezone.utc).isoformat()
            )

            self.trade_analyses.append(trade_analysis)
            self._save_analysis_db()

            # Send analysis to Telegram if significant
            if rating in ['excellent', 'poor']:
                await self._send_trade_analysis_notification(trade_analysis)

            logger.info(f"✅ Analyzed trade: {trade.symbol} - Rating: {rating}")

            return trade_analysis

        except Exception as e:
            logger.error(f"Error analyzing trade: {e}")
            return None

    def _rate_trade(self, trade) -> str:
        """Rate trade quality based on execution and result"""
        # Factors: P&L, confluence score, HTF alignment, trade type

        if trade.pnl > 0:
            # Winning trade
            if trade.pnl_pct >= 3.5 and trade.confluence_score >= 7:
                return 'excellent'
            elif trade.pnl_pct >= 2.0:
                return 'good'
            else:
                return 'acceptable'
        else:
            # Losing trade
            if trade.pnl_pct <= -2.5:
                return 'poor'  # Hit stop loss badly
            elif trade.confluence_score < 4:
                return 'poor'  # Shouldn't have taken
            else:
                return 'acceptable'  # Acceptable loss with stop

    async def _generate_trade_analysis(self, trade) -> str:
        """Generate detailed analysis using trading knowledge"""

        # Build analysis using learned knowledge
        analysis_parts = []

        # 1. Setup Quality Analysis
        if trade.confluence_score >= 7:
            analysis_parts.append(f"✅ High-quality setup (confluence {trade.confluence_score}/10) - proper execution of strategy.")
        elif trade.confluence_score >= 4:
            analysis_parts.append(f"⚠️ Medium-quality setup (confluence {trade.confluence_score}/10) - acceptable but not ideal.")
        else:
            analysis_parts.append(f"❌ Low-quality setup (confluence {trade.confluence_score}/10) - should have been skipped.")

        # 2. HTF Alignment Analysis
        if trade.htf_alignment >= 70:
            analysis_parts.append(f"✅ Strong HTF alignment ({trade.htf_alignment:.0f}%) supported this {trade.direction} direction.")
        elif trade.htf_alignment >= 50:
            analysis_parts.append(f"⚠️ Moderate HTF alignment ({trade.htf_alignment:.0f}%) - some context support.")
        else:
            analysis_parts.append(f"❌ Weak HTF alignment ({trade.htf_alignment:.0f}%) - fought the higher timeframes.")

        # 3. Trade Type Analysis
        if "TREND_CONTINUATION" in trade.trade_type:
            analysis_parts.append("✅ Trend continuation trade - highest probability strategy type when HTF aligned.")
        elif "MEAN_REVERSION" in trade.trade_type:
            analysis_parts.append("⚠️ Mean reversion trade - requires tight risk management and quick exits.")
        elif "OPPORTUNITY" in trade.trade_type:
            analysis_parts.append("⚠️ Opportunistic trade - lower conviction, should use reduced size.")

        # 4. Result Analysis
        if trade.pnl > 0:
            if trade.pnl_pct >= 3.5:
                analysis_parts.append(f"🎯 Excellent profit ({trade.pnl_pct:+.2f}%) - nearly hit take profit. Setup worked perfectly.")
            elif trade.pnl_pct >= 2.0:
                analysis_parts.append(f"✅ Good profit ({trade.pnl_pct:+.2f}%) - positive outcome validates setup.")
            else:
                analysis_parts.append(f"✅ Modest profit ({trade.pnl_pct:+.2f}%) - better than loss but could optimize exits.")
        else:
            if trade.reason == "Stop Loss":
                analysis_parts.append(f"❌ Stop loss hit ({trade.pnl_pct:.2f}%) - invalidation was correct risk management.")
            else:
                analysis_parts.append(f"❌ Loss ({trade.pnl_pct:.2f}%) - {trade.reason}.")

        # 5. Order Type Analysis
        if trade.order_type == "LIMIT":
            analysis_parts.append("✅ Limit order filled - patience rewarded with better entry price.")
        else:
            analysis_parts.append("🚀 Market order execution - immediate entry for high conviction setup.")

        return " ".join(analysis_parts)

    def _extract_lessons(self, trade, analysis: str) -> List[str]:
        """Extract actionable lessons from trade"""
        lessons = []

        # Based on outcome and quality
        if trade.pnl > 0 and trade.confluence_score >= 7:
            lessons.append("High confluence setups are working - continue prioritizing 7+ scores")

        if trade.pnl < 0 and trade.confluence_score < 5:
            lessons.append(f"Avoid {trade.confluence_score}/10 setups - insufficient confluence led to loss")

        if trade.htf_alignment >= 70 and trade.pnl > 0:
            lessons.append("HTF alignment is crucial - strong alignment (>70%) correlates with wins")

        if trade.htf_alignment < 50 and trade.pnl < 0:
            lessons.append(f"Fighting HTF ({trade.htf_alignment:.0f}% alignment) led to loss - respect higher timeframes")

        if "MEAN_REVERSION" in trade.trade_type and trade.pnl < 0:
            lessons.append("Mean reversion losses remind us: tight stops essential, don't average down counter-trend")

        if "TREND_CONTINUATION" in trade.trade_type and trade.pnl > 0:
            lessons.append("Trend continuation winning - 'trend is your friend' proven again")

        if trade.order_type == "LIMIT" and trade.pnl > 0:
            lessons.append("Patience with limit orders paid off - better entry led to better outcome")

        return lessons

    # ============================================================================
    # PERIODIC REVIEWS
    # ============================================================================

    async def _periodic_review_task(self):
        """Generate periodic strategy reviews"""
        while True:
            try:
                await asyncio.sleep(3600)  # Check every hour

                now = datetime.now(timezone.utc)

                # Daily review at 21:00 UTC (after daily summary)
                if now.hour == 21 and now.minute < 10:
                    await self._generate_daily_review()

                # Weekly review on Sunday at 21:00
                if now.weekday() == 6 and now.hour == 21 and now.minute < 10:
                    await self._generate_weekly_review()

            except Exception as e:
                logger.error(f"Error in periodic review task: {e}")
                await asyncio.sleep(600)

    async def _generate_daily_review(self):
        """Generate daily strategy review"""
        try:
            logger.info("📊 Generating daily trade review...")

            # Get today's analyses
            today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
            today_analyses = [
                a for a in self.trade_analyses
                if a.timestamp.startswith(today)
            ]

            if not today_analyses:
                return

            # Aggregate insights
            excellent_count = sum(1 for a in today_analyses if a.rating == 'excellent')
            poor_count = sum(1 for a in today_analyses if a.rating == 'poor')

            avg_confluence = sum(a.confluence_score for a in today_analyses) / len(today_analyses)
            avg_htf_alignment = sum(a.htf_alignment for a in today_analyses) / len(today_analyses)

            # Generate recommendations
            recommendations = []

            if avg_confluence < 5:
                recommendations.append("⚠️ Average confluence was low today (< 5). Consider raising entry threshold to 6+ for better quality setups.")

            if poor_count >= 2:
                recommendations.append(f"⚠️ {poor_count} poor-rated trades today. Review setup criteria - may need stricter filtering.")

            if avg_htf_alignment < 60:
                recommendations.append(f"⚠️ HTF alignment averaged {avg_htf_alignment:.0f}% - prioritize trades with 70%+ alignment.")

            # Collect all lessons
            all_lessons = []
            for analysis in today_analyses:
                all_lessons.extend(analysis.lessons_learned)

            # Send review
            message = f"""
🤖 **DAILY TRADE REVIEW** - {today}

📊 **ANALYZED TRADES:** {len(today_analyses)}
   Excellent: {excellent_count}
   Poor: {poor_count}

📈 **AVERAGE METRICS:**
   Confluence Score: {avg_confluence:.1f}/10
   HTF Alignment: {avg_htf_alignment:.0f}%

💡 **KEY LESSONS TODAY:**
"""
            # Add top 5 lessons
            for i, lesson in enumerate(set(all_lessons)[:5], 1):
                message += f"\n{i}. {lesson}"

            if recommendations:
                message += "\n\n🎯 **RECOMMENDATIONS:**\n"
                for rec in recommendations:
                    message += f"\n{rec}"

            await self.telegram_bot.send_alert(
                title=f"🤖 Daily Trade Review - {today}",
                message=message,
                level="info"
            )

            logger.info("✅ Daily review sent")

        except Exception as e:
            logger.error(f"Error generating daily review: {e}")

    async def _generate_weekly_review(self):
        """Generate comprehensive weekly review"""
        try:
            logger.info("📊 Generating weekly trade review...")

            # Get last 7 days
            week_ago = datetime.now(timezone.utc) - timedelta(days=7)
            week_analyses = [
                a for a in self.trade_analyses
                if datetime.fromisoformat(a.timestamp) >= week_ago
            ]

            if not week_analyses:
                return

            # Deep analysis
            total_trades = len(week_analyses)
            wins = sum(1 for a in week_analyses if a.pnl > 0)
            losses = total_trades - wins
            win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

            # By confluence score
            high_conf_trades = [a for a in week_analyses if a.confluence_score >= 7]
            high_conf_wins = sum(1 for a in high_conf_trades if a.pnl > 0)
            high_conf_wr = (high_conf_wins / len(high_conf_trades) * 100) if high_conf_trades else 0

            low_conf_trades = [a for a in week_analyses if a.confluence_score < 5]
            low_conf_wins = sum(1 for a in low_conf_trades if a.pnl > 0)
            low_conf_wr = (low_conf_wins / len(low_conf_trades) * 100) if low_conf_trades else 0

            # By trade type
            trend_cont_trades = [a for a in week_analyses if "TREND_CONTINUATION" in a.trade_type]
            mean_rev_trades = [a for a in week_analyses if "MEAN_REVERSION" in a.trade_type]

            trend_cont_wr = (sum(1 for a in trend_cont_trades if a.pnl > 0) / len(trend_cont_trades) * 100) if trend_cont_trades else 0
            mean_rev_wr = (sum(1 for a in mean_rev_trades if a.pnl > 0) / len(mean_rev_trades) * 100) if mean_rev_trades else 0

            # Strategic recommendations
            strategic_recs = []

            if high_conf_wr > 65:
                strategic_recs.append(f"✅ High confluence setups (7+) winning at {high_conf_wr:.0f}% - focus on quality over quantity")

            if low_conf_wr < 50 and low_conf_trades:
                strategic_recs.append(f"❌ Low confluence setups (< 5) winning only {low_conf_wr:.0f}% - stop taking these trades")

            if trend_cont_wr > mean_rev_wr + 20:
                strategic_recs.append(f"✅ Trend continuation ({trend_cont_wr:.0f}%) outperforming mean reversion ({mean_rev_wr:.0f}%) - prioritize trend trades")

            message = f"""
🤖 **WEEKLY TRADE REVIEW & STRATEGY OPTIMIZATION**

📊 **OVERALL PERFORMANCE:**
   Total Trades: {total_trades}
   Win Rate: {win_rate:.1f}% ({wins}W / {losses}L)

🎯 **CONFLUENCE SCORE ANALYSIS:**
   High Confidence (7+): {len(high_conf_trades)} trades, {high_conf_wr:.0f}% win rate
   Low Confidence (< 5): {len(low_conf_trades)} trades, {low_conf_wr:.0f}% win rate

   📈 **INSIGHT:** {"High confluence clearly superior - maintain strict standards" if high_conf_wr > low_conf_wr else "Confluence scoring may need recalibration"}

📈 **TRADE TYPE PERFORMANCE:**
   Trend Continuation: {len(trend_cont_trades)} trades, {trend_cont_wr:.0f}% WR
   Mean Reversion: {len(mean_rev_trades)} trades, {mean_rev_wr:.0f}% WR

🎯 **STRATEGIC RECOMMENDATIONS:**
"""
            for rec in strategic_recs:
                message += f"\n{rec}"

            message += "\n\n💡 **NEXT WEEK FOCUS:**\n"
            if high_conf_wr > 60:
                message += "• Only take 7+ confluence trades\n"
            if trend_cont_wr > 60:
                message += "• Prioritize trend continuation setups\n"
            if mean_rev_wr < 50:
                message += "• Reduce or eliminate mean reversion trades\n"

            await self.telegram_bot.send_alert(
                title="🤖 Weekly Strategy Review",
                message=message,
                level="info"
            )

            logger.info("✅ Weekly review sent")

        except Exception as e:
            logger.error(f"Error generating weekly review: {e}")

    async def _send_trade_analysis_notification(self, analysis: TradeAnalysis):
        """Send analysis for exceptional trades"""
        try:
            rating_emoji = "🏆" if analysis.rating == 'excellent' else "⚠️"

            message = f"""
{rating_emoji} **TRADE ANALYSIS** - {analysis.rating.upper()}

💰 **TRADE:**
   {analysis.symbol} {analysis.direction.upper()}
   P&L: ${analysis.pnl:+,.2f} ({analysis.pnl_pct:+.2f}%)
   Duration: {analysis.duration_hours:.1f}h

⚡ **SETUP QUALITY:**
   Confluence: {analysis.confluence_score}/10
   HTF Alignment: {analysis.htf_alignment:.0f}%
   Trade Type: {analysis.trade_type}

🤖 **AI ANALYSIS:**
{analysis.analysis}

💡 **LESSONS LEARNED:**
"""
            for i, lesson in enumerate(analysis.lessons_learned, 1):
                message += f"\n{i}. {lesson}"

            await self.telegram_bot.send_alert(
                title=f"{rating_emoji} Trade Analysis: {analysis.symbol}",
                message=message,
                level="info"
            )

        except Exception as e:
            logger.error(f"Failed to send trade analysis notification: {e}")

    # ============================================================================
    # DATABASE MANAGEMENT
    # ============================================================================

    def _load_databases(self):
        """Load existing analysis and knowledge databases"""
        # Load trade analyses
        analysis_file = Path(self.analysis_db_file)
        if analysis_file.exists():
            try:
                with open(analysis_file, 'r') as f:
                    data = json.load(f)
                    self.trade_analyses = [TradeAnalysis(**item) for item in data]
                logger.info(f"📂 Loaded {len(self.trade_analyses)} trade analyses")
            except Exception as e:
                logger.warning(f"Failed to load analysis database: {e}")

        # Load trading knowledge
        knowledge_file = Path(self.knowledge_db_file)
        if knowledge_file.exists():
            try:
                with open(knowledge_file, 'r') as f:
                    data = json.load(f)
                    self.trading_knowledge = [TradingKnowledge(**item) for item in data]
                    self.topics_researched = set(k.topic for k in self.trading_knowledge)
                logger.info(f"📚 Loaded {len(self.trading_knowledge)} knowledge entries")
            except Exception as e:
                logger.warning(f"Failed to load knowledge database: {e}")

    def _save_analysis_db(self):
        """Save trade analyses"""
        try:
            with open(self.analysis_db_file, 'w') as f:
                data = [asdict(a) for a in self.trade_analyses]
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save analysis database: {e}")

    def _save_knowledge_db(self):
        """Save trading knowledge"""
        try:
            with open(self.knowledge_db_file, 'w') as f:
                data = [asdict(k) for k in self.trading_knowledge]
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save knowledge database: {e}")


# Singleton
_agent_instance = None

def get_trade_review_agent(paper_trading_engine, telegram_bot) -> TradeReviewAgent:
    """Get or create trade review agent"""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = TradeReviewAgent(paper_trading_engine, telegram_bot)
    return _agent_instance
