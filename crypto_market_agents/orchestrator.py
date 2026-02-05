"""
Orchestrator - Manages all agents and generates consolidated reports.
"""

import asyncio
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Optional
import logging

from .config import SystemConfig
from .exchange.adapter import ExchangeFactory
from .exchange.base import BaseExchange
from .agents.price_action import PriceActionAgent
from .agents.momentum import MomentumAgent
from .agents.volume_spike import VolumeSpikeAgent
from .agents.sr_detector import SRDetectionAgent
from .agents.fibonacci_agent import FibonacciAgent
from .agents.learning_agent import LearningAgent
from .agents.signal_synthesis import SignalSynthesisAgent
from .schemas import SystemReport, TradingSignal
from .utils.logging import setup_logger
from .integrations.telegram_bot import TelegramBot


class AgentOrchestrator:
    """
    Orchestrator for managing multiple agents and generating consolidated reports.
    """

    def __init__(self, config: SystemConfig):
        """
        Initialize the orchestrator.

        Args:
            config: System configuration
        """
        self.config = config
        self.logger = setup_logger(
            "Orchestrator",
            level=config.log_level,
            log_file=config.log_file
        )

        self.exchange: Optional[BaseExchange] = None
        self.agents: List = []
        self.running = False

        # Agent instances
        self.price_action_agent: Optional[PriceActionAgent] = None
        self.momentum_agent: Optional[MomentumAgent] = None
        self.volume_spike_agent: Optional[VolumeSpikeAgent] = None
        self.sr_agent: Optional[SRDetectionAgent] = None
        self.fibonacci_agent: Optional[FibonacciAgent] = None
        self.learning_agent: Optional[LearningAgent] = None
        self.signal_synthesis_agent: Optional[SignalSynthesisAgent] = None

        # Telegram bot integration
        self.telegram_bot: Optional[TelegramBot] = None
        if config.telegram.enabled:
            self.telegram_bot = TelegramBot(
                bot_token=config.telegram.bot_token,
                chat_id=config.telegram.chat_id,
                enabled=True
            )

        # Signal deduplication tracking (asset+direction+entry -> timestamp)
        self.sent_signals: dict[str, float] = {}
        self.signal_dedupe_window = 1800  # 30 minutes in seconds

        # Report generation
        self.report_task: Optional[asyncio.Task] = None
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True)

    async def initialize(self) -> bool:
        """
        Initialize the orchestrator and all agents.

        Returns:
            True if initialization successful
        """
        self.logger.info("Initializing agent orchestrator...")

        try:
            # Create exchange adapter
            self.exchange = ExchangeFactory.create(self.config.exchange)

            # Check US accessibility
            accessibility = self.exchange.check_us_accessibility()
            if not accessibility["accessible"]:
                self.logger.warning(
                    f"Exchange accessibility warning: {accessibility['notes']}"
                )
                self.logger.info(
                    f"Recommended alternatives: {', '.join(accessibility.get('recommended_alternatives', []))}"
                )

            # Connect to exchange
            if not await self.exchange.connect():
                self.logger.error("Failed to connect to exchange")
                return False

            # Initialize agents
            if self.config.price_action.enabled:
                self.price_action_agent = PriceActionAgent(
                    self.exchange,
                    self.config.price_action
                )
                self.agents.append(self.price_action_agent)

            if self.config.momentum.enabled:
                self.momentum_agent = MomentumAgent(
                    self.exchange,
                    self.config.momentum
                )
                self.agents.append(self.momentum_agent)

            if self.config.volume.enabled:
                self.volume_spike_agent = VolumeSpikeAgent(
                    self.exchange,
                    self.config.volume
                )
                self.agents.append(self.volume_spike_agent)

            # Get symbols for S/R and Learning agents
            # Will be populated from exchange when agents execute
            symbols = self.config.symbols if self.config.symbols else []

            # S/R Detection Agent
            if hasattr(self.config, 'sr_detection') and self.config.sr_detection.enabled:
                self.sr_agent = SRDetectionAgent(
                    self.exchange,
                    symbols,
                    timeframes=self.config.sr_detection.timeframes,
                    lookback=self.config.sr_detection.lookback,
                    min_touches=self.config.sr_detection.min_touches,
                    confluence_tolerance=self.config.sr_detection.confluence_tolerance,
                    update_interval=self.config.sr_detection.update_interval
                )
                self.agents.append(self.sr_agent)
                self.logger.info("S/R Detection Agent initialized")

            # Fibonacci Agent
            if hasattr(self.config, 'fibonacci') and self.config.fibonacci.enabled:
                self.fibonacci_agent = FibonacciAgent(
                    self.exchange,
                    symbols,
                    lookback=self.config.fibonacci.lookback,
                    min_swing_size=self.config.fibonacci.min_swing_size,
                    update_interval=self.config.fibonacci.update_interval
                )
                self.agents.append(self.fibonacci_agent)
                self.logger.info("Fibonacci Agent initialized")

            # Learning Agent
            if hasattr(self.config, 'learning') and self.config.learning.enabled:
                self.learning_agent = LearningAgent(
                    self.exchange,
                    data_dir=self.config.learning.data_dir if hasattr(self.config.learning, 'data_dir') else "data",
                    paper_trading=self.config.learning.paper_trading,
                    min_trades_before_learning=self.config.learning.min_trades_before_learning,
                    auto_optimize=self.config.learning.auto_optimize,
                    update_interval=self.config.learning.update_interval
                )
                self.agents.append(self.learning_agent)
                self.logger.info("Learning Agent initialized")

            # Signal synthesis agent requires other agents
            if self.config.signal_synthesis.enabled:
                if not all([
                    self.price_action_agent,
                    self.momentum_agent,
                    self.volume_spike_agent
                ]):
                    self.logger.warning(
                        "Signal synthesis agent requires all other agents to be enabled"
                    )
                else:
                    self.signal_synthesis_agent = SignalSynthesisAgent(
                        self.exchange,
                        self.config.signal_synthesis,
                        self.price_action_agent,
                        self.momentum_agent,
                        self.volume_spike_agent,
                        sr_agent=self.sr_agent,
                        fibonacci_agent=self.fibonacci_agent,
                        learning_agent=self.learning_agent
                    )
                    self.agents.append(self.signal_synthesis_agent)

            self.logger.info(f"Initialized {len(self.agents)} agents")
            return True

        except Exception as e:
            self.logger.error(f"Initialization failed: {e}", exc_info=True)
            return False

    async def start(self):
        """Start all agents and the orchestrator."""
        if self.running:
            self.logger.warning("Orchestrator is already running")
            return

        self.running = True
        self.logger.info("Starting agent orchestrator...")

        # Start all agents in parallel
        start_tasks = [agent.start() for agent in self.agents]
        await asyncio.gather(*start_tasks)

        # Start report generation task
        self.report_task = asyncio.create_task(self._report_loop())

        self.logger.info("All agents started successfully")

        # Send startup notification to Telegram
        if self.telegram_bot and self.config.telegram.send_alerts:
            await self.telegram_bot.send_alert(
                "System Started",
                f"✅ {len(self.agents)} agents initialized and running\n"
                f"📊 Monitoring {len(self.config.symbols) if self.config.symbols else 'all'} symbols",
                critical=False
            )

    async def stop(self):
        """Stop all agents and the orchestrator."""
        if not self.running:
            return

        self.running = False
        self.logger.info("Stopping agent orchestrator...")

        # Stop report task
        if self.report_task:
            self.report_task.cancel()
            try:
                await self.report_task
            except asyncio.CancelledError:
                pass

        # Stop all agents in parallel
        stop_tasks = [agent.stop() for agent in self.agents]
        await asyncio.gather(*stop_tasks, return_exceptions=True)

        # Disconnect from exchange
        if self.exchange:
            await self.exchange.disconnect()

        # Close Telegram bot session
        if self.telegram_bot:
            await self.telegram_bot.close()

        self.logger.info("Orchestrator stopped")

    def _get_signal_id(self, signal: TradingSignal) -> str:
        """
        Generate a unique ID for a signal for deduplication.

        Args:
            signal: Trading signal

        Returns:
            Unique signal identifier
        """
        # Use asset, direction, and rounded entry price as unique ID
        return f"{signal.asset}_{signal.direction}_{signal.entry:.8f}"

    def _should_send_signal(self, signal: TradingSignal) -> bool:
        """
        Check if a signal should be sent (not sent recently).

        Args:
            signal: Trading signal to check

        Returns:
            True if signal should be sent
        """
        signal_id = self._get_signal_id(signal)
        current_time = datetime.now(timezone.utc).timestamp()

        # Clean up old entries (older than dedupe window)
        expired_ids = [
            sid for sid, timestamp in self.sent_signals.items()
            if current_time - timestamp > self.signal_dedupe_window
        ]
        for sid in expired_ids:
            del self.sent_signals[sid]

        # Check if signal was sent recently
        if signal_id in self.sent_signals:
            time_since_sent = current_time - self.sent_signals[signal_id]
            if time_since_sent < self.signal_dedupe_window:
                return False

        # Mark as sent
        self.sent_signals[signal_id] = current_time
        return True

    async def _report_loop(self):
        """Periodic report generation loop."""
        while self.running:
            try:
                await self._generate_report()
                await asyncio.sleep(self.config.signal_synthesis.update_interval)

            except asyncio.CancelledError:
                break

            except Exception as e:
                self.logger.error(f"Error generating report: {e}", exc_info=True)
                await asyncio.sleep(60)

    async def _generate_report(self):
        """Generate and save a consolidated system report."""
        try:
            # Collect trading signals
            trading_signals = []
            if self.signal_synthesis_agent:
                signals = self.signal_synthesis_agent.get_latest_signals()
                trading_signals = signals[:20]  # Top 20 signals

            # Collect agent statuses
            agent_statuses = [agent.get_status() for agent in self.agents]

            # Create system report
            report = SystemReport(
                timestamp=datetime.now(timezone.utc),
                active_agents=len([a for a in self.agents if a.running]),
                trading_signals=trading_signals,
                agent_statuses=agent_statuses
            )

            # Save report to file
            timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            report_file = self.output_dir / f"report_{timestamp_str}.json"

            with open(report_file, "w") as f:
                f.write(report.to_json())

            # Also save the latest report
            latest_file = self.output_dir / "latest_report.json"
            with open(latest_file, "w") as f:
                f.write(report.to_json())

            # Log summary
            self.logger.info(
                f"Generated report: {len(trading_signals)} trading signals, "
                f"{report.active_agents}/{len(self.agents)} agents active"
            )

            # Log top signals
            if trading_signals:
                self.logger.info("Top trading signals:")
                for i, signal in enumerate(trading_signals[:5], 1):
                    self.logger.info(
                        f"  {i}. {signal.asset} {signal.direction} @ {signal.entry:.8f} "
                        f"(confidence: {signal.confidence:.2f}) - {signal.rationale}"
                    )

            # Print real-time performance metrics
            if self.learning_agent and self.learning_agent.current_metrics:
                metrics = self.learning_agent.current_metrics
                print(f"\n{'='*80}")
                print(f"📊 REAL-TIME PERFORMANCE METRICS")
                print(f"{'='*80}")
                print(f"  Total Trades:      {metrics.total_trades} ({metrics.open_trades} open)")
                print(f"  Win Rate:          {metrics.win_rate:.1f}% ({metrics.wins}W / {metrics.losses}L)")
                print(f"  Avg R:R Ratio:     {metrics.avg_rr:.2f}:1")
                print(f"  Profit Factor:     {metrics.profit_factor:.2f}")
                print(f"  Total P&L:         {metrics.total_pnl_percent:+.2f}%")
                print(f"  Avg Win/Loss:      +{metrics.avg_win_percent:.2f}% / {metrics.avg_loss_percent:.2f}%")
                print(f"  Largest Win/Loss:  +{metrics.largest_win_percent:.2f}% / {metrics.largest_loss_percent:.2f}%")

                # Streak tracking
                streak_str = f"+{metrics.current_streak}" if metrics.current_streak > 0 else str(metrics.current_streak)
                print(f"  Current Streak:    {streak_str} (Max W: {metrics.max_consecutive_wins}, Max L: {metrics.max_consecutive_losses})")

                # Risk metrics
                print(f"  Sharpe Ratio:      {metrics.sharpe_ratio:.2f}")
                print(f"  Sortino Ratio:     {metrics.sortino_ratio:.2f}")
                print(f"  Max Drawdown:      {metrics.max_drawdown_percent:.2f}%")
                print(f"  Calmar Ratio:      {metrics.calmar_ratio:.2f}")

                # Performance status indicator
                if metrics.profit_factor > 1.5:
                    status = "✅ PROFITABLE"
                elif metrics.profit_factor > 1.0:
                    status = "🟡 BREAKEVEN+"
                elif metrics.profit_factor > 0.7:
                    status = "🟠 NEEDS IMPROVEMENT"
                else:
                    status = "🔴 LOSING"
                print(f"  Status:            {status}")
                print(f"{'='*80}\n")

            # Send top signals to Telegram (with deduplication)
            if trading_signals:
                if self.telegram_bot and self.config.telegram.send_signals:
                    # Filter out signals that were already sent recently
                    new_signals = [s for s in trading_signals if self._should_send_signal(s)]

                    if new_signals:
                        max_signals = self.config.telegram.max_signals_per_batch
                        sent = await self.telegram_bot.send_signals_batch(new_signals, max_signals)
                        if sent > 0:
                            self.logger.info(f"Sent {sent} new signals to Telegram (filtered {len(trading_signals) - len(new_signals)} duplicates)")
                    else:
                        self.logger.debug(f"No new signals to send (all {len(trading_signals)} were already sent recently)")

        except Exception as e:
            self.logger.error(f"Failed to generate report: {e}", exc_info=True)
            # Send error alert to Telegram (best effort, don't throw)
            if self.telegram_bot and self.config.telegram.send_alerts:
                try:
                    await self.telegram_bot.send_error(str(e), "Report generation")
                except Exception as telegram_error:
                    self.logger.debug(f"Could not send Telegram error notification: {telegram_error}")

    async def run_forever(self):
        """
        Run the orchestrator indefinitely.

        This method will block until stopped.
        """
        try:
            await self.start()

            # Wait for stop signal
            while self.running:
                await asyncio.sleep(1)

        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt")

        finally:
            await self.stop()

    def get_latest_signals(self) -> List[TradingSignal]:
        """
        Get the latest trading signals.

        Returns:
            List of trading signals
        """
        if self.signal_synthesis_agent:
            return self.signal_synthesis_agent.get_latest_signals()
        return []
