"""
Learning Agent for paper trading and strategy optimization.

This agent tracks trading signals, executes paper trades, analyzes outcomes,
and continuously learns to improve strategy performance.
"""

import asyncio
import json
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from pathlib import Path
import logging

from .base_agent import BaseAgent
from ..exchange.base import BaseExchange
from ..schemas import (
    TradingSignal,
    PaperTrade,
    StrategyMetrics,
    LearningInsights
)


class LearningAgent(BaseAgent):
    """
    Agent for paper trading and continuous learning.

    Tracks all trading signals, executes paper trades, analyzes outcomes,
    and builds a knowledge base to optimize strategy over time.
    """

    def __init__(
        self,
        exchange: BaseExchange,
        data_dir: str = "data",
        paper_trading: bool = True,
        min_trades_before_learning: int = 20,
        auto_optimize: bool = True,
        update_interval: int = 300
    ):
        """
        Initialize Learning Agent.

        Args:
            exchange: Exchange adapter instance
            data_dir: Directory for storing learning data
            paper_trading: Enable paper trading
            min_trades_before_learning: Minimum trades before optimization
            auto_optimize: Auto-adjust parameters based on performance
            update_interval: Seconds between updates
        """
        super().__init__(
            name="Learning Agent",
            exchange=exchange,
            update_interval=update_interval
        )

        self.paper_trading = paper_trading
        self.min_trades_before_learning = min_trades_before_learning
        self.auto_optimize = auto_optimize

        # Data storage paths
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        self.trades_file = self.data_dir / "paper_trades.json"
        self.metrics_file = self.data_dir / "strategy_metrics.json"
        self.knowledge_file = self.data_dir / "agent_knowledge.json"

        # In-memory storage
        self.paper_trades: List[PaperTrade] = []
        self.closed_trades: List[PaperTrade] = []
        self.open_trades: List[PaperTrade] = []

        # Strategy metrics
        self.current_metrics: Optional[StrategyMetrics] = None

        # Knowledge base
        self.knowledge_base: Dict[str, Any] = {
            "order_flow": {
                "concepts": [],
                "patterns": [],
                "observations": []
            },
            "price_action": {
                "breakouts": {},
                "reversals": {},
                "consolidations": {}
            },
            "liquidity": {
                "swept_levels": [],
                "high_liquidity_zones": [],
                "thin_zones": []
            },
            "indicators": {
                "rsi": {"optimal_settings": {}, "failure_modes": []},
                "obv": {"optimal_settings": {}, "failure_modes": []},
                "macd": {"optimal_settings": {}, "failure_modes": []}
            },
            "support_resistance": {
                "daily_levels": {"success_rate": 0.0, "trades": 0},
                "h4_levels": {"success_rate": 0.0, "trades": 0},
                "confluence_zones": {"success_rate": 0.0, "trades": 0}
            },
            "poc": {
                "observations": [],
                "trading_rules": []
            }
        }

        # Insights cache
        self.insights_cache: Dict[str, LearningInsights] = {}

        self.logger = logging.getLogger(self.__class__.__name__)

        # Load existing data
        self._load_data()

    async def execute(self):
        """Main execution loop for learning agent."""
        try:
            # Update open paper trades
            if self.paper_trading:
                await self._update_open_trades()

            # Analyze performance if we have enough trades
            if len(self.closed_trades) >= self.min_trades_before_learning:
                self._analyze_performance()

                # Learn from trades
                self._learn_from_trades()

                # Optimize parameters if enabled
                if self.auto_optimize:
                    self._optimize_parameters()

            # Save data periodically
            self._save_data()

        except Exception as e:
            self.logger.error(f"Error in learning agent: {e}")

    async def open_paper_trade(
        self,
        signal: TradingSignal,
        sr_levels: Optional[List[Dict[str, Any]]] = None,
        volume_data: Optional[Dict[str, Any]] = None,
        momentum_data: Optional[Dict[str, Any]] = None
    ) -> Optional[PaperTrade]:
        """
        Open a new paper trade based on a signal.

        Args:
            signal: Trading signal to execute
            sr_levels: S/R levels at entry
            volume_data: Volume analysis data
            momentum_data: Momentum indicator data

        Returns:
            PaperTrade object or None
        """
        if not self.paper_trading:
            return None

        try:
            trade = PaperTrade(
                trade_id=str(uuid.uuid4()),
                symbol=signal.asset,
                direction=signal.direction,
                entry_price=signal.entry,
                stop_loss=signal.stop,
                take_profit=signal.target,
                entry_time=datetime.now(timezone.utc),
                exit_time=None,
                exit_price=None,
                pnl=None,
                pnl_percent=None,
                outcome='OPEN',
                signal_data=signal.to_dict(),
                sr_levels=sr_levels,
                volume_data=volume_data,
                momentum_data=momentum_data,
                notes="",
                confidence_at_entry=signal.confidence
            )

            self.paper_trades.append(trade)
            self.open_trades.append(trade)

            self.logger.info(
                f"Opened paper trade {trade.trade_id[:8]} for {signal.asset} "
                f"{signal.direction} @ ${signal.entry:,.2f}"
            )

            return trade

        except Exception as e:
            self.logger.error(f"Error opening paper trade: {e}")
            return None

    async def _update_open_trades(self):
        """Update all open paper trades."""
        if not self.open_trades:
            return

        for trade in self.open_trades[:]:  # Copy list to allow modification
            try:
                # Get current price
                ticker = await self.exchange.get_ticker(trade.symbol)
                if not ticker or 'last_price' not in ticker:
                    continue

                current_price = ticker['last_price']

                # Check if stop loss or take profit hit
                should_close = False
                outcome = 'OPEN'

                if trade.direction == 'LONG':
                    if current_price <= trade.stop_loss:
                        should_close = True
                        outcome = 'LOSS'
                    elif current_price >= trade.take_profit:
                        should_close = True
                        outcome = 'WIN'
                else:  # SHORT
                    if current_price >= trade.stop_loss:
                        should_close = True
                        outcome = 'LOSS'
                    elif current_price <= trade.take_profit:
                        should_close = True
                        outcome = 'WIN'

                if should_close:
                    await self._close_paper_trade(trade, current_price, outcome)

            except Exception as e:
                self.logger.error(f"Error updating trade {trade.trade_id[:8]}: {e}")

    async def _close_paper_trade(
        self,
        trade: PaperTrade,
        exit_price: float,
        outcome: str
    ):
        """
        Close a paper trade.

        Args:
            trade: Trade to close
            exit_price: Exit price
            outcome: 'WIN', 'LOSS', or 'BREAKEVEN'
        """
        try:
            # Calculate P&L
            if trade.direction == 'LONG':
                pnl_percent = ((exit_price - trade.entry_price) / trade.entry_price) * 100
            else:  # SHORT
                pnl_percent = ((trade.entry_price - exit_price) / trade.entry_price) * 100

            # Update trade
            trade.exit_time = datetime.now(timezone.utc)
            trade.exit_price = exit_price
            trade.pnl_percent = pnl_percent
            trade.outcome = outcome

            # Determine if breakeven
            if abs(pnl_percent) < 0.1:
                trade.outcome = 'BREAKEVEN'

            # Move to closed trades
            self.open_trades.remove(trade)
            self.closed_trades.append(trade)

            self.logger.info(
                f"Closed paper trade {trade.trade_id[:8]} for {trade.symbol} "
                f"{trade.direction}: {trade.outcome} ({pnl_percent:+.2f}%)"
            )

        except Exception as e:
            self.logger.error(f"Error closing trade: {e}")

    def _analyze_performance(self):
        """Analyze strategy performance from closed trades."""
        if not self.closed_trades:
            return

        wins = [t for t in self.closed_trades if t.outcome == 'WIN']
        losses = [t for t in self.closed_trades if t.outcome == 'LOSS']
        breakeven = [t for t in self.closed_trades if t.outcome == 'BREAKEVEN']

        total_trades = len(self.closed_trades)
        num_wins = len(wins)
        num_losses = len(losses)
        num_breakeven = len(breakeven)

        win_rate = (num_wins / total_trades * 100) if total_trades > 0 else 0

        # Calculate average R:R
        avg_win = sum(t.pnl_percent for t in wins) / num_wins if num_wins > 0 else 0
        avg_loss = sum(abs(t.pnl_percent) for t in losses) / num_losses if num_losses > 0 else 0
        avg_rr = abs(avg_win / avg_loss) if avg_loss > 0 else 0

        # Calculate profit factor
        total_wins = sum(t.pnl_percent for t in wins)
        total_losses = abs(sum(t.pnl_percent for t in losses))
        profit_factor = (total_wins / total_losses) if total_losses > 0 else 0

        # Total P&L
        total_pnl = sum(t.pnl_percent for t in self.closed_trades if t.pnl_percent)

        # Largest win/loss
        largest_win = max((t.pnl_percent for t in wins), default=0)
        largest_loss = min((t.pnl_percent for t in losses), default=0)

        self.current_metrics = StrategyMetrics(
            total_trades=total_trades,
            wins=num_wins,
            losses=num_losses,
            breakeven=num_breakeven,
            open_trades=len(self.open_trades),
            win_rate=win_rate,
            avg_rr=avg_rr,
            profit_factor=profit_factor,
            total_pnl_percent=total_pnl,
            avg_win_percent=avg_win,
            avg_loss_percent=avg_loss,
            largest_win_percent=largest_win,
            largest_loss_percent=largest_loss,
            timestamp=datetime.now(timezone.utc)
        )

        self.logger.info(
            f"Strategy Metrics: {num_wins}W/{num_losses}L ({win_rate:.1f}% win rate), "
            f"R:R {avg_rr:.2f}:1, PF {profit_factor:.2f}"
        )

    def _learn_from_trades(self):
        """Learn from trade outcomes and update knowledge base."""
        if not self.closed_trades:
            return

        try:
            # Analyze S/R level effectiveness
            self._analyze_sr_effectiveness()

            # Analyze indicator performance
            self._analyze_indicators()

            # Identify patterns in winners vs losers
            self._identify_patterns()

            # Update knowledge base
            self._update_knowledge_base()

        except Exception as e:
            self.logger.error(f"Error learning from trades: {e}")

    def _analyze_sr_effectiveness(self):
        """Analyze effectiveness of S/R levels in trades."""
        trades_with_sr = [
            t for t in self.closed_trades
            if t.sr_levels and len(t.sr_levels) > 0
        ]

        if not trades_with_sr:
            return

        # Daily levels
        daily_trades = [
            t for t in trades_with_sr
            if any(level.get('timeframe') == '1d' for level in t.sr_levels)
        ]
        daily_wins = len([t for t in daily_trades if t.outcome == 'WIN'])
        daily_win_rate = (daily_wins / len(daily_trades) * 100) if daily_trades else 0

        # 4h levels
        h4_trades = [
            t for t in trades_with_sr
            if any(level.get('timeframe') == '4h' for level in t.sr_levels)
        ]
        h4_wins = len([t for t in h4_trades if t.outcome == 'WIN'])
        h4_win_rate = (h4_wins / len(h4_trades) * 100) if h4_trades else 0

        # Confluence zones (multiple levels)
        confluence_trades = [
            t for t in trades_with_sr
            if len(t.sr_levels) >= 2
        ]
        confluence_wins = len([t for t in confluence_trades if t.outcome == 'WIN'])
        confluence_win_rate = (confluence_wins / len(confluence_trades) * 100) if confluence_trades else 0

        # Update knowledge base
        self.knowledge_base['support_resistance']['daily_levels'] = {
            'success_rate': daily_win_rate,
            'trades': len(daily_trades)
        }
        self.knowledge_base['support_resistance']['h4_levels'] = {
            'success_rate': h4_win_rate,
            'trades': len(h4_trades)
        }
        self.knowledge_base['support_resistance']['confluence_zones'] = {
            'success_rate': confluence_win_rate,
            'trades': len(confluence_trades)
        }

    def _analyze_indicators(self):
        """Analyze indicator performance."""
        # This would analyze RSI, OBV, MACD effectiveness
        # For now, we'll implement basic tracking
        pass

    def _identify_patterns(self):
        """Identify patterns in winning vs losing trades."""
        wins = [t for t in self.closed_trades if t.outcome == 'WIN']
        losses = [t for t in self.closed_trades if t.outcome == 'LOSS']

        # Analyze what differentiates wins from losses
        # This is where we'd implement more sophisticated pattern recognition

        # For now, store basic observations
        if wins and losses:
            avg_win_confidence = sum(t.confidence_at_entry for t in wins) / len(wins)
            avg_loss_confidence = sum(t.confidence_at_entry for t in losses) / len(losses)

            observation = {
                "type": "confidence_analysis",
                "avg_win_confidence": avg_win_confidence,
                "avg_loss_confidence": avg_loss_confidence,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

            if "confidence" not in self.knowledge_base:
                self.knowledge_base["confidence"] = []
            self.knowledge_base["confidence"].append(observation)

    def _update_knowledge_base(self):
        """Update knowledge base with learned insights."""
        # This is where the agent would implement more sophisticated learning
        # For now, we're tracking basic metrics and patterns
        pass

    def _optimize_parameters(self):
        """Auto-optimize strategy parameters based on performance."""
        if not self.current_metrics:
            return

        # Example: If win rate is low, we might recommend being more selective
        # This would be expanded with more sophisticated optimization logic
        pass

    def get_insights(self, symbol: str) -> Optional[LearningInsights]:
        """
        Get learning insights for a specific symbol.

        Args:
            symbol: Trading symbol

        Returns:
            LearningInsights or None
        """
        # Check cache first
        if symbol in self.insights_cache:
            return self.insights_cache[symbol]

        if not self.current_metrics:
            return None

        # Calculate insights based on historical performance
        symbol_trades = [
            t for t in self.closed_trades
            if t.symbol == symbol
        ]

        if not symbol_trades:
            return None

        wins = len([t for t in symbol_trades if t.outcome == 'WIN'])
        total = len(symbol_trades)
        symbol_win_rate = (wins / total * 100) if total > 0 else 0

        # Get S/R effectiveness
        sr_data = self.knowledge_base.get('support_resistance', {})
        win_rate_at_sr = sr_data.get('confluence_zones', {}).get('success_rate', 0)

        # Determine confidence adjustment
        confidence_adjustment = 0.0
        if symbol_win_rate > self.current_metrics.win_rate:
            confidence_adjustment = 0.1  # Boost confidence for this symbol
        elif symbol_win_rate < self.current_metrics.win_rate:
            confidence_adjustment = -0.1  # Reduce confidence

        # Determine context multiplier
        context_multiplier = 1.0
        if win_rate_at_sr > 70:
            context_multiplier = 1.2  # Boost signals at S/R levels
        elif win_rate_at_sr < 40:
            context_multiplier = 0.8  # Reduce signals at S/R levels

        # Determine recommended action
        recommended_action = 'normal'
        if symbol_win_rate > 70:
            recommended_action = 'increase_position'
        elif symbol_win_rate < 40:
            recommended_action = 'decrease_position'

        insights = LearningInsights(
            symbol=symbol,
            confidence_adjustment=confidence_adjustment,
            context_multiplier=context_multiplier,
            win_rate_at_sr=win_rate_at_sr,
            win_rate_overall=symbol_win_rate,
            recommended_action=recommended_action,
            reasoning=f"Based on {total} trades with {symbol_win_rate:.1f}% win rate",
            timestamp=datetime.now(timezone.utc)
        )

        # Cache insights
        self.insights_cache[symbol] = insights

        return insights

    def get_metrics(self) -> Optional[StrategyMetrics]:
        """Get current strategy metrics."""
        return self.current_metrics

    def _load_data(self):
        """Load existing learning data from disk."""
        try:
            # Load paper trades
            if self.trades_file.exists():
                with open(self.trades_file, 'r') as f:
                    data = json.load(f)
                    for trade_data in data:
                        # Convert datetime strings back to datetime objects
                        trade_data['entry_time'] = datetime.fromisoformat(trade_data['entry_time'])
                        if trade_data.get('exit_time'):
                            trade_data['exit_time'] = datetime.fromisoformat(trade_data['exit_time'])

                        trade = PaperTrade(**trade_data)
                        self.paper_trades.append(trade)

                        if trade.outcome == 'OPEN':
                            self.open_trades.append(trade)
                        else:
                            self.closed_trades.append(trade)

                self.logger.info(f"Loaded {len(self.paper_trades)} paper trades from disk")

            # Load knowledge base
            if self.knowledge_file.exists():
                with open(self.knowledge_file, 'r') as f:
                    self.knowledge_base = json.load(f)

        except Exception as e:
            self.logger.error(f"Error loading data: {e}")

    def _save_data(self):
        """Save learning data to disk."""
        try:
            # Save paper trades
            with open(self.trades_file, 'w') as f:
                trades_data = [trade.to_dict() for trade in self.paper_trades]
                json.dump(trades_data, f, indent=2)

            # Save metrics
            if self.current_metrics:
                with open(self.metrics_file, 'w') as f:
                    json.dump(self.current_metrics.to_dict(), f, indent=2)

            # Save knowledge base
            with open(self.knowledge_file, 'w') as f:
                json.dump(self.knowledge_base, f, indent=2)

        except Exception as e:
            self.logger.error(f"Error saving data: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get agent status."""
        return {
            "name": self.name,
            "running": self.running,
            "paper_trading": self.paper_trading,
            "total_trades": len(self.paper_trades),
            "open_trades": len(self.open_trades),
            "closed_trades": len(self.closed_trades),
            "current_metrics": self.current_metrics.to_dict() if self.current_metrics else None,
            "last_update": self.last_update.isoformat() if self.last_update else None
        }
