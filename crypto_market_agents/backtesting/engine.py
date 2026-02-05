"""
Backtesting Engine - Replays historical data through all agents for strategy validation.
"""

import asyncio
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta, timezone
from pathlib import Path
import logging

from .historical_data import HistoricalDataFetcher
from ..config import SystemConfig
from ..exchange.base import BaseExchange
from ..agents.price_action import PriceActionAgent
from ..agents.momentum import MomentumAgent
from ..agents.volume_spike import VolumeSpikeAgent
from ..agents.sr_detector import SRDetectionAgent
from ..agents.fibonacci_agent import FibonacciAgent
from ..agents.learning_agent import LearningAgent
from ..agents.signal_synthesis import SignalSynthesisAgent
from ..schemas import TradingSignal, StrategyMetrics
from ..utils.risk_manager import RiskManager, RiskLimits


class MockExchange(BaseExchange):
    """
    Mock exchange for backtesting that serves historical data.
    """

    def __init__(self, historical_data: Dict[str, List[Dict[str, Any]]]):
        """
        Initialize mock exchange with historical data.

        Args:
            historical_data: Dict mapping symbol -> OHLCV candles
        """
        self.historical_data = historical_data
        self.current_time = None
        self.logger = logging.getLogger(self.__class__.__name__)

    def set_current_time(self, current_time: datetime):
        """Set the current backtesting time."""
        self.current_time = current_time

    async def connect(self) -> bool:
        """Mock connect."""
        return True

    async def disconnect(self):
        """Mock disconnect."""
        pass

    async def get_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get ticker data at current backtesting time.

        Args:
            symbol: Trading symbol

        Returns:
            Ticker dictionary with last_price, volume, etc.
        """
        if symbol not in self.historical_data:
            return None

        candles = self.historical_data[symbol]
        if not candles or self.current_time is None:
            return None

        # Find candle at current time
        for candle in candles:
            if candle['timestamp'] >= self.current_time:
                return {
                    'symbol': symbol,
                    'last_price': candle['close'],
                    'high_24h': candle['high'],
                    'low_24h': candle['low'],
                    'volume_24h': candle['volume'],
                    'timestamp': candle['timestamp']
                }

        # If no future candles, use last available
        if candles:
            last_candle = candles[-1]
            return {
                'symbol': symbol,
                'last_price': last_candle['close'],
                'high_24h': last_candle['high'],
                'low_24h': last_candle['low'],
                'volume_24h': last_candle['volume'],
                'timestamp': last_candle['timestamp']
            }

        return None

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        since: Optional[int] = None,
        limit: int = 100
    ) -> List[List]:
        """
        Fetch OHLCV data up to current backtesting time.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            since: Start timestamp in ms
            limit: Max candles to return

        Returns:
            OHLCV data in ccxt format [[timestamp_ms, o, h, l, c, v], ...]
        """
        if symbol not in self.historical_data:
            return []

        candles = self.historical_data[symbol]
        if not candles or self.current_time is None:
            return []

        # Filter candles up to current time
        filtered = [
            c for c in candles
            if c['timestamp'] <= self.current_time
        ]

        # Apply since filter
        if since:
            since_dt = datetime.fromtimestamp(since / 1000, tz=timezone.utc)
            filtered = [c for c in filtered if c['timestamp'] >= since_dt]

        # Apply limit
        filtered = filtered[-limit:]

        # Convert to ccxt format
        return [
            [
                int(c['timestamp'].timestamp() * 1000),
                c['open'],
                c['high'],
                c['low'],
                c['close'],
                c['volume']
            ]
            for c in filtered
        ]

    async def fetch_markets(self) -> List[Dict[str, Any]]:
        """Get available markets."""
        return [
            {
                'symbol': symbol,
                'base': symbol.split('/')[0] if '/' in symbol else symbol,
                'quote': symbol.split('/')[1] if '/' in symbol else 'USDT',
                'active': True
            }
            for symbol in self.historical_data.keys()
        ]

    async def get_trading_symbols(self, quote_currency: str = "USDT") -> List[str]:
        """Get list of available trading symbols."""
        return [
            symbol for symbol in self.historical_data.keys()
            if symbol.endswith(f"/{quote_currency}")
        ]

    async def get_tickers(self, symbols: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Get ticker data for multiple symbols."""
        target_symbols = symbols if symbols else list(self.historical_data.keys())
        tickers = []
        for symbol in target_symbols:
            ticker = await self.get_ticker(symbol)
            if ticker:
                tickers.append(ticker)
        return tickers

    async def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 100,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Get candlestick/kline data."""
        if symbol not in self.historical_data:
            return []

        candles = self.historical_data[symbol]
        if not candles:
            return []

        # Filter by time range
        filtered = candles
        if start_time:
            filtered = [c for c in filtered if c['timestamp'] >= start_time]
        if end_time:
            filtered = [c for c in filtered if c['timestamp'] <= end_time]

        # Apply limit (take most recent)
        filtered = filtered[-limit:]

        return filtered

    async def get_order_book(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        """Get order book depth (mocked for backtesting)."""
        ticker = await self.get_ticker(symbol)
        if not ticker:
            return {'bids': [], 'asks': [], 'timestamp': datetime.now(timezone.utc)}

        # Generate mock order book around current price
        price = ticker['last_price']
        return {
            'bids': [[price * 0.999, 100.0], [price * 0.998, 200.0]],
            'asks': [[price * 1.001, 100.0], [price * 1.002, 200.0]],
            'timestamp': ticker['timestamp']
        }

    async def get_recent_trades(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent trades (mocked for backtesting)."""
        return []  # Not needed for backtesting

    def check_us_accessibility(self) -> Dict[str, Any]:
        """Check US accessibility (mock exchange - always accessible)."""
        return {
            'accessible': True,
            'restrictions': [],
            'notes': 'Mock exchange for backtesting - no restrictions'
        }


class BacktestEngine:
    """
    Backtesting engine that replays historical data through all agents.
    """

    def __init__(
        self,
        config: SystemConfig,
        data_dir: str = "backtest_results"
    ):
        """
        Initialize backtest engine.

        Args:
            config: System configuration
            data_dir: Directory for storing backtest results
        """
        self.config = config
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        self.logger = logging.getLogger(self.__class__.__name__)

        # Will be set during run
        self.mock_exchange: Optional[MockExchange] = None
        self.agents: List = []
        self.learning_agent: Optional[LearningAgent] = None
        self.signal_synthesis_agent: Optional[SignalSynthesisAgent] = None
        self.historical_data: Dict[str, List[Dict[str, Any]]] = {}

    def set_historical_data(self, historical_data: Dict[str, List[Dict[str, Any]]]):
        """
        Set historical data for backtesting.

        Args:
            historical_data: Dict mapping symbol -> OHLCV candles
        """
        self.historical_data = historical_data
        self.mock_exchange = MockExchange(historical_data)
        self.logger.info(f"Loaded historical data for {len(historical_data)} symbols")

    async def run_backtest(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1h",
        initial_capital: float = 10000.0
    ) -> Dict[str, Any]:
        """
        Run backtest on historical data.

        Args:
            symbols: List of symbols to backtest
            start_date: Backtest start date
            end_date: Backtest end date
            timeframe: Candle timeframe
            initial_capital: Starting capital

        Returns:
            Backtest results dictionary
        """
        self.logger.info(f"Starting backtest from {start_date.date()} to {end_date.date()}")
        self.logger.info(f"Symbols: {', '.join(symbols)}")
        self.logger.info(f"Timeframe: {timeframe}, Initial Capital: ${initial_capital:,.2f}")

        # Ensure historical data is loaded
        if not self.historical_data or self.mock_exchange is None:
            raise ValueError(
                "Historical data not loaded. Call set_historical_data() "
                "with fetched data before running backtest."
            )

        # Initialize agents with mock exchange
        await self._initialize_agents(initial_capital)

        # Run backtest simulation
        results = await self._simulate_trading(
            symbols,
            start_date,
            end_date,
            timeframe
        )

        # Save results
        self._save_results(results)

        return results

    async def _initialize_agents(self, initial_capital: float):
        """Initialize all agents with mock exchange."""
        self.logger.info("Initializing agents...")

        # Price Action Agent
        if self.config.price_action.enabled:
            price_action_agent = PriceActionAgent(
                self.mock_exchange,
                self.config.price_action
            )
            self.agents.append(price_action_agent)

        # Momentum Agent
        if self.config.momentum.enabled:
            momentum_agent = MomentumAgent(
                self.mock_exchange,
                self.config.momentum
            )
            self.agents.append(momentum_agent)

        # Volume Spike Agent
        if self.config.volume.enabled:
            volume_spike_agent = VolumeSpikeAgent(
                self.mock_exchange,
                self.config.volume
            )
            self.agents.append(volume_spike_agent)

        # S/R Detection Agent
        sr_agent = None
        if hasattr(self.config, 'sr_detection') and self.config.sr_detection.enabled:
            sr_agent = SRDetectionAgent(
                self.mock_exchange,
                [],  # Symbols will be set during simulation
                timeframes=self.config.sr_detection.timeframes,
                lookback=self.config.sr_detection.lookback,
                min_touches=self.config.sr_detection.min_touches,
                confluence_tolerance=self.config.sr_detection.confluence_tolerance,
                update_interval=self.config.sr_detection.update_interval
            )
            self.agents.append(sr_agent)

        # Fibonacci Agent
        fibonacci_agent = None
        if hasattr(self.config, 'fibonacci') and self.config.fibonacci.enabled:
            fibonacci_agent = FibonacciAgent(
                self.mock_exchange,
                [],
                lookback=self.config.fibonacci.lookback,
                min_swing_size=self.config.fibonacci.min_swing_size,
                update_interval=self.config.fibonacci.update_interval
            )
            self.agents.append(fibonacci_agent)

        # Create risk manager
        risk_limits = RiskLimits(
            max_portfolio_risk_pct=self.config.risk_management.max_portfolio_risk_pct,
            max_drawdown_pct=self.config.risk_management.max_drawdown_pct,
            max_concurrent_positions=self.config.risk_management.max_concurrent_positions,
            max_single_position_pct=self.config.risk_management.max_single_position_pct,
            max_correlated_exposure_pct=self.config.risk_management.max_correlated_exposure_pct
        )
        risk_manager = RiskManager(initial_capital=initial_capital, risk_limits=risk_limits)

        # Learning Agent
        if hasattr(self.config, 'learning') and self.config.learning.enabled:
            self.learning_agent = LearningAgent(
                self.mock_exchange,
                data_dir=str(self.data_dir / "learning_data"),
                paper_trading=True,
                min_trades_before_learning=self.config.learning.min_trades_before_learning,
                auto_optimize=False,  # Disable auto-optimize during backtest
                update_interval=self.config.learning.update_interval,
                risk_manager=risk_manager
            )
            self.agents.append(self.learning_agent)

        # Signal Synthesis Agent
        if len(self.agents) >= 3:  # Need at least price, momentum, volume
            self.signal_synthesis_agent = SignalSynthesisAgent(
                self.mock_exchange,
                self.config.signal_synthesis,
                self.agents[0],  # price_action
                self.agents[1],  # momentum
                self.agents[2],  # volume
                sr_agent=sr_agent,
                fibonacci_agent=fibonacci_agent,
                learning_agent=self.learning_agent
            )
            self.agents.append(self.signal_synthesis_agent)

        self.logger.info(f"Initialized {len(self.agents)} agents")

    async def _simulate_trading(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        timeframe: str
    ) -> Dict[str, Any]:
        """
        Simulate trading by replaying historical data.

        Args:
            symbols: Symbols to trade
            start_date: Start date
            end_date: End date
            timeframe: Timeframe

        Returns:
            Simulation results
        """
        self.logger.info("Starting trading simulation...")

        # Start all agents
        for agent in self.agents:
            await agent.start()

        # Simulate time progression
        current_time = start_date
        step = self._get_time_step(timeframe)
        signals_generated = []

        while current_time <= end_date:
            # Update mock exchange time
            self.mock_exchange.set_current_time(current_time)

            # Execute all agents
            for agent in self.agents:
                try:
                    await agent.execute()
                except Exception as e:
                    self.logger.error(f"Error executing {agent.name}: {e}")

            # Collect signals from synthesis agent
            if self.signal_synthesis_agent:
                latest_signals = self.signal_synthesis_agent.get_latest_signals()
                if latest_signals:
                    signals_generated.extend(latest_signals[:3])  # Top 3 signals

                # Execute signals via learning agent
                if self.learning_agent and latest_signals:
                    for signal in latest_signals[:5]:  # Top 5 signals
                        await self.learning_agent.open_paper_trade(signal)

            # Update open trades
            if self.learning_agent:
                await self.learning_agent._update_open_trades()

            # Progress time
            current_time += step

            # Log progress every 100 steps
            if (current_time - start_date).total_seconds() % (step.total_seconds() * 100) == 0:
                self.logger.info(f"Backtest progress: {current_time.date()}")

        # Stop all agents
        for agent in self.agents:
            await agent.stop()

        # Collect final metrics
        final_metrics = None
        if self.learning_agent and self.learning_agent.current_metrics:
            final_metrics = self.learning_agent.current_metrics

        results = {
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'timeframe': timeframe,
            'symbols': symbols,
            'signals_generated': len(signals_generated),
            'metrics': final_metrics.to_dict() if final_metrics else None,
            'trades': [t.to_dict() for t in self.learning_agent.closed_trades] if self.learning_agent else []
        }

        self.logger.info(f"Backtest complete: {len(signals_generated)} signals, "
                        f"{len(results['trades'])} trades executed")

        return results

    def _get_time_step(self, timeframe: str) -> timedelta:
        """Get time step for simulation based on timeframe."""
        steps = {
            '1m': timedelta(minutes=1),
            '5m': timedelta(minutes=5),
            '15m': timedelta(minutes=15),
            '1h': timedelta(hours=1),
            '4h': timedelta(hours=4),
            '1d': timedelta(days=1)
        }
        return steps.get(timeframe, timedelta(hours=1))

    def _save_results(self, results: Dict[str, Any]):
        """Save backtest results to file."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = self.data_dir / f"backtest_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)

        self.logger.info(f"Results saved to {filename}")


    def analyze_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze backtest results and generate detailed report.

        Args:
            results: Backtest results dictionary

        Returns:
            Analysis report
        """
        metrics = results.get('metrics')
        if not metrics:
            return {'error': 'No metrics available'}

        trades = results.get('trades', [])

        analysis = {
            'summary': {
                'total_trades': metrics['total_trades'],
                'win_rate': metrics['win_rate'],
                'profit_factor': metrics['profit_factor'],
                'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                'max_drawdown': metrics.get('max_drawdown_percent', 0),
                'total_return': metrics['total_pnl_percent']
            },
            'trade_distribution': {
                'wins': metrics['wins'],
                'losses': metrics['losses'],
                'breakeven': metrics['breakeven']
            },
            'risk_metrics': {
                'avg_win': metrics['avg_win_percent'],
                'avg_loss': metrics['avg_loss_percent'],
                'largest_win': metrics['largest_win_percent'],
                'largest_loss': metrics['largest_loss_percent'],
                'avg_rr': metrics['avg_rr']
            },
            'streak_analysis': {
                'max_consecutive_wins': metrics.get('max_consecutive_wins', 0),
                'max_consecutive_losses': metrics.get('max_consecutive_losses', 0),
                'current_streak': metrics.get('current_streak', 0)
            }
        }

        return analysis
