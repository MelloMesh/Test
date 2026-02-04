"""
Test script for visual signal formatting.

Demonstrates the new confidence/confluence bar visualization.
"""

import asyncio
from datetime import datetime, timezone
from crypto_market_agents.schemas import TradingSignal
from crypto_market_agents.utils.signal_formatter import (
    format_signal_visual,
    format_signal_compact,
    format_signal_telegram
)


def create_sample_signal() -> TradingSignal:
    """Create a sample trading signal for testing."""
    return TradingSignal(
        asset="ETHUSDT",
        direction="LONG",
        entry=2450.00,
        stop=2400.00,
        target=2680.00,
        confidence=0.72,
        rationale="Bullish breakout (+3.2%) | Oversold RSI 28.4 on 60 | HTF support @ $2,445.00 (3 levels) | 🎯 Golden Pocket ($2,438-$2,453) in bullish swing | Learning: normal (62% win rate)",
        timestamp=datetime.now(timezone.utc),
        order_type="LIMIT",
        confluence_score=6
    )


def main():
    """Test signal formatting."""
    print("=" * 80)
    print("SIGNAL FORMATTING TEST")
    print("=" * 80)
    print()

    # Create sample signal
    signal = create_sample_signal()

    # Test visual format (full box)
    print("1. VISUAL FORMAT (Full)")
    print("-" * 80)
    print(format_signal_visual(signal))
    print()

    # Test compact format
    print("2. COMPACT FORMAT (One Line)")
    print("-" * 80)
    print(format_signal_compact(signal))
    print()

    # Test Telegram format
    print("3. TELEGRAM FORMAT")
    print("-" * 80)
    print(format_signal_telegram(signal))
    print()

    # Test with different confidence levels
    print("4. CONFIDENCE LEVELS TEST")
    print("-" * 80)
    for conf_level in [0.3, 0.5, 0.7, 0.9]:
        test_signal = TradingSignal(
            asset="BTCUSDT",
            direction="SHORT",
            entry=50000.00,
            stop=50500.00,
            target=48000.00,
            confidence=conf_level,
            rationale=f"Test signal with {conf_level:.0%} confidence",
            timestamp=datetime.now(timezone.utc),
            order_type="MARKET" if conf_level < 0.6 else "LIMIT",
            confluence_score=int(conf_level * 10)
        )
        print(format_signal_compact(test_signal))
    print()

    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
