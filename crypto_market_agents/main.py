"""
Main entry point for the Crypto Market Agents system.
"""

import asyncio
import sys
import signal
from pathlib import Path

from .config import SystemConfig
from .orchestrator import AgentOrchestrator
from .utils.logging import setup_logger


async def main():
    """Main application entry point."""

    # Load configuration
    config = SystemConfig.from_env()

    # Setup root logger
    logger = setup_logger(
        "CryptoAgents",
        level=config.log_level,
        log_file=config.log_file
    )

    logger.info("=" * 80)
    logger.info("Crypto Market Agents System")
    logger.info("=" * 80)

    # Create orchestrator
    orchestrator = AgentOrchestrator(config)

    # Initialize
    if not await orchestrator.initialize():
        logger.error("Failed to initialize system")
        return 1

    # Setup signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}, shutting down...")
        orchestrator.running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run
    try:
        logger.info("System initialized successfully")
        logger.info(f"Output directory: {config.output_dir}")
        logger.info(f"Reports will be generated every {config.signal_synthesis.update_interval}s")
        logger.info("-" * 80)

        await orchestrator.run_forever()

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1

    finally:
        logger.info("Shutdown complete")

    return 0


def run():
    """Synchronous entry point."""
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nShutdown requested")
        sys.exit(0)


if __name__ == "__main__":
    run()
