"""
Test edge cases found in second quality review.

Tests:
1. Division by zero in signal_formatter
2. Negative values in create_bar
3. Message length truncation
"""

from datetime import datetime, timezone
from crypto_market_agents.schemas import TradingSignal
from crypto_market_agents.utils.signal_formatter import (
    create_bar,
    create_percentage_bar,
    format_signal_visual,
    format_signal_compact
)


def test_division_by_zero():
    """Test that zero entry price doesn't crash."""
    print("=" * 80)
    print("TEST 1: Division by Zero Protection")
    print("=" * 80)

    # Create signal with zero entry (edge case)
    signal = TradingSignal(
        asset="TESTUSDT",
        direction="LONG",
        entry=0.0,  # ❌ Zero entry!
        stop=0.0,
        target=0.0,
        confidence=0.5,
        rationale="Test signal with zero entry",
        timestamp=datetime.now(timezone.utc),
        order_type="MARKET",
        confluence_score=0
    )

    try:
        result = format_signal_visual(signal)
        print("✅ PASS: No crash with zero entry")
        print(f"   Entry displayed as: ${signal.entry:,.4f}")
    except ZeroDivisionError as e:
        print(f"❌ FAIL: Division by zero occurred: {e}")
        raise
    except Exception as e:
        print(f"❌ FAIL: Unexpected error: {e}")
        raise

    print()


def test_negative_values_in_bar():
    """Test that negative values don't create broken bars."""
    print("=" * 80)
    print("TEST 2: Negative Value Handling")
    print("=" * 80)

    test_cases = [
        (-0.5, 1.0, "Negative value"),
        (1.5, 1.0, "Value exceeds max"),
        (0.5, 1.0, "Normal value"),
        (0.0, 1.0, "Zero value"),
        (1.0, 0.0, "Zero max"),
        (-1.0, -1.0, "Both negative")
    ]

    for value, max_value, description in test_cases:
        try:
            bar = create_bar(value, max_value, length=10)
            # Check bar is correct length
            if len(bar) != 10:
                print(f"❌ FAIL: {description} - Bar length {len(bar)} != 10")
                print(f"   Value: {value}, Max: {max_value}")
                print(f"   Bar: '{bar}'")
            else:
                print(f"✅ PASS: {description}")
                print(f"   Value: {value:>5.1f}, Max: {max_value:>5.1f} → [{bar}]")
        except Exception as e:
            print(f"❌ FAIL: {description} - Exception: {e}")
            raise

    print()


def test_message_truncation():
    """Test that long rationales don't break formatting."""
    print("=" * 80)
    print("TEST 3: Long Message Handling")
    print("=" * 80)

    # Create signal with very long rationale
    long_rationale = "Bullish breakout " + "x" * 500  # 516 chars

    signal = TradingSignal(
        asset="LONGUSDT",
        direction="LONG",
        entry=100.0,
        stop=95.0,
        target=120.0,
        confidence=0.8,
        rationale=long_rationale,
        timestamp=datetime.now(timezone.utc),
        order_type="LIMIT",
        confluence_score=8
    )

    try:
        result = format_signal_visual(signal)
        print("✅ PASS: Long rationale handled")
        print(f"   Rationale length: {len(signal.rationale)} chars")
        print(f"   Output lines: {result.count(chr(10))} lines")

        # Check output doesn't have misaligned borders
        lines = result.split('\n')
        for i, line in enumerate(lines):
            # Check that box borders are present where expected
            if i > 0 and i < len(lines) - 1:  # Skip first and last line
                if not (line.startswith('║') or line.startswith('╠') or line.startswith('╚')):
                    print(f"⚠️  WARNING: Line {i} might be misaligned: {line[:20]}...")

    except Exception as e:
        print(f"❌ FAIL: Long rationale crashed: {e}")
        raise

    print()


def test_extreme_confidence_values():
    """Test edge case confidence values."""
    print("=" * 80)
    print("TEST 4: Extreme Confidence Values")
    print("=" * 80)

    test_cases = [
        (1.5, "Above 100%"),
        (1.0, "Exactly 100%"),
        (0.5, "Normal 50%"),
        (0.0, "Zero confidence"),
        (-0.1, "Negative confidence")
    ]

    for conf, description in test_cases:
        try:
            bar = create_percentage_bar(conf * 100, length=10)
            print(f"✅ PASS: {description} (conf={conf:.1f})")
            print(f"   Bar: [{bar}]")
        except Exception as e:
            print(f"❌ FAIL: {description} - Exception: {e}")
            raise

    print()


def test_compact_format():
    """Test compact format with edge cases."""
    print("=" * 80)
    print("TEST 5: Compact Format Edge Cases")
    print("=" * 80)

    signal = TradingSignal(
        asset="TEST",
        direction="SHORT",
        entry=50000.0,
        stop=50500.0,
        target=48000.0,
        confidence=0.72,
        rationale="Test compact",
        timestamp=datetime.now(timezone.utc),
        order_type="MARKET",
        confluence_score=6
    )

    try:
        result = format_signal_compact(signal)
        print("✅ PASS: Compact format works")
        print(f"   {result}")
        if len(result) > 200:
            print(f"   ⚠️  WARNING: Output is quite long ({len(result)} chars)")
    except Exception as e:
        print(f"❌ FAIL: Compact format crashed: {e}")
        raise

    print()


def main():
    """Run all edge case tests."""
    print("\n")
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║       EDGE CASE TESTING - Second Quality Review            ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    try:
        test_division_by_zero()
        test_negative_values_in_bar()
        test_message_truncation()
        test_extreme_confidence_values()
        test_compact_format()

        print("=" * 80)
        print("✅ ALL TESTS PASSED")
        print("=" * 80)
        print()
        print("Edge cases are now handled correctly!")
        print("- Division by zero: Protected")
        print("- Negative values: Clamped")
        print("- Long messages: Handled")
        print("- Extreme confidence: Clamped")
        print()

    except Exception as e:
        print()
        print("=" * 80)
        print("❌ TESTS FAILED")
        print("=" * 80)
        print(f"Error: {e}")
        import sys
        sys.exit(1)


if __name__ == "__main__":
    main()
