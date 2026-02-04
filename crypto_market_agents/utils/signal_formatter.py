"""
Signal Formatting Utilities - Creates visual representations of trading signals.
"""

from typing import Optional
from ..schemas import TradingSignal


def create_bar(value: float, max_value: float, length: int = 20, filled_char: str = "█", empty_char: str = "░") -> str:
    """
    Create a visual progress bar.

    Args:
        value: Current value
        max_value: Maximum value for the bar
        length: Length of the bar in characters
        filled_char: Character for filled portion
        empty_char: Character for empty portion

    Returns:
        String representing the bar
    """
    if max_value <= 0:
        return empty_char * length

    filled_length = int((value / max_value) * length)
    filled_length = max(0, min(filled_length, length))  # Clamp between 0 and length

    bar = filled_char * filled_length + empty_char * (length - filled_length)
    return bar


def create_percentage_bar(percentage: float, length: int = 20, filled_char: str = "█", empty_char: str = "░") -> str:
    """
    Create a visual progress bar from a percentage (0-100).

    Args:
        percentage: Percentage value (0-100)
        length: Length of the bar in characters
        filled_char: Character for filled portion
        empty_char: Character for empty portion

    Returns:
        String representing the bar
    """
    return create_bar(percentage, 100.0, length, filled_char, empty_char)


def get_confidence_color_emoji(confidence: float) -> str:
    """
    Get emoji based on confidence level.

    Args:
        confidence: Confidence value (0-1)

    Returns:
        Emoji representing confidence level
    """
    if confidence >= 0.8:
        return "🟢"  # Green - High confidence
    elif confidence >= 0.6:
        return "🟡"  # Yellow - Medium confidence
    elif confidence >= 0.4:
        return "🟠"  # Orange - Low-medium confidence
    else:
        return "🔴"  # Red - Low confidence


def get_confluence_badge(score: int) -> str:
    """
    Get badge/emoji for confluence score.

    Args:
        score: Confluence score (0-12+)

    Returns:
        Badge representing confluence level
    """
    if score >= 8:
        return "⭐⭐⭐"  # Triple star - Very high confluence
    elif score >= 6:
        return "⭐⭐"  # Double star - High confluence
    elif score >= 4:
        return "⭐"  # Single star - Medium confluence
    else:
        return "○"  # Circle - Low confluence


def format_signal_visual(signal: TradingSignal) -> str:
    """
    Format a trading signal with visual elements (bars, emojis).

    Args:
        signal: Trading signal to format

    Returns:
        Formatted string with visual elements
    """
    # Confidence as percentage
    confidence_pct = signal.confidence * 100

    # Confidence bar (20 characters)
    confidence_bar = create_percentage_bar(confidence_pct, length=20)
    confidence_emoji = get_confidence_color_emoji(signal.confidence)

    # Confluence badge
    confluence_badge = get_confluence_badge(signal.confluence_score)

    # Direction emoji
    direction_emoji = "🟢" if signal.direction == "LONG" else "🔴"

    # Order type badge
    order_type_badge = "⏱️ LIMIT" if signal.order_type == "LIMIT" else "⚡ MARKET"

    # Calculate risk/reward
    if signal.direction == "LONG":
        risk = signal.entry - signal.stop
        reward = signal.target - signal.entry
    else:
        risk = signal.stop - signal.entry
        reward = signal.entry - signal.target

    rr_ratio = reward / risk if risk > 0 else 0

    # Format output
    output = f"""
╔══════════════════════════════════════════════════════════════╗
║  {direction_emoji} {signal.asset:<12} {signal.direction:<6}  {order_type_badge:<12}  {confluence_badge}  ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Entry:  ${signal.entry:>12,.4f}                              ║
║  Stop:   ${signal.stop:>12,.4f}  ({abs((signal.stop - signal.entry) / signal.entry * 100):>5.2f}%)           ║
║  Target: ${signal.target:>12,.4f}  ({abs((signal.target - signal.entry) / signal.entry * 100):>5.2f}%)           ║
║  R:R     {rr_ratio:>5.2f}x                                              ║
║                                                              ║
╠══════════════════════════════════════════════════════════════╣
║  Confidence: {confidence_emoji} {confidence_pct:>5.1f}%                                      ║
║  [{confidence_bar}] {confidence_pct:>5.1f}%  ║
║                                                              ║
║  Confluence: {confluence_badge} {signal.confluence_score:>2} points                                    ║
║                                                              ║
╠══════════════════════════════════════════════════════════════╣
║  Rationale:                                                  ║
║  {signal.rationale[:60]:<60} ║
"""

    # Add continuation lines if rationale is longer
    if len(signal.rationale) > 60:
        remaining = signal.rationale[60:]
        while remaining:
            chunk = remaining[:60]
            output += f"║  {chunk:<60} ║\n"
            remaining = remaining[60:]

    output += "╚══════════════════════════════════════════════════════════════╝"

    return output


def format_signal_compact(signal: TradingSignal) -> str:
    """
    Format a trading signal in compact format (one line).

    Args:
        signal: Trading signal to format

    Returns:
        Compact formatted string
    """
    direction_emoji = "🟢" if signal.direction == "LONG" else "🔴"
    confidence_emoji = get_confidence_color_emoji(signal.confidence)
    confluence_badge = get_confluence_badge(signal.confluence_score)

    confidence_pct = signal.confidence * 100
    confidence_bar = create_percentage_bar(confidence_pct, length=10)

    return (
        f"{direction_emoji} {signal.asset:<10} {signal.direction:<5} | "
        f"Entry: ${signal.entry:>10,.2f} | Stop: ${signal.stop:>10,.2f} | Target: ${signal.target:>10,.2f} | "
        f"{confidence_emoji} [{confidence_bar}] {confidence_pct:>4.1f}% | {confluence_badge} {signal.confluence_score}"
    )


def format_signal_telegram(signal: TradingSignal) -> str:
    """
    Format a trading signal for Telegram (supports markdown).

    Args:
        signal: Trading signal to format

    Returns:
        Telegram-formatted string with markdown
    """
    # Confidence as percentage
    confidence_pct = signal.confidence * 100

    # Confidence bar using Unicode block elements
    confidence_bar = create_percentage_bar(confidence_pct, length=15, filled_char="▰", empty_char="▱")

    # Confluence bar
    confluence_max = 12
    confluence_bar = create_bar(signal.confluence_score, confluence_max, length=12, filled_char="▰", empty_char="▱")

    # Direction emoji
    direction_emoji = "🟢" if signal.direction == "LONG" else "🔴"

    # Calculate risk/reward
    if signal.direction == "LONG":
        risk = signal.entry - signal.stop
        reward = signal.target - signal.entry
        stop_pct = (signal.stop - signal.entry) / signal.entry * 100
        target_pct = (signal.target - signal.entry) / signal.entry * 100
    else:
        risk = signal.stop - signal.entry
        reward = signal.entry - signal.target
        stop_pct = (signal.stop - signal.entry) / signal.entry * 100
        target_pct = (signal.entry - signal.target) / signal.entry * 100

    rr_ratio = reward / risk if risk > 0 else 0

    # Confidence emoji
    confidence_emoji = get_confidence_color_emoji(signal.confidence)

    # Order type
    order_badge = "⏱ LIMIT" if signal.order_type == "LIMIT" else "⚡ MARKET"

    # Format for Telegram with markdown
    message = f"""
{direction_emoji} *{signal.asset}* {signal.direction} Signal

📊 *Trade Details:*
• Entry:  `${signal.entry:,.4f}`
• Stop:   `${signal.stop:,.4f}` ({abs(stop_pct):.2f}%)
• Target: `${signal.target:,.4f}` ({abs(target_pct):.2f}%)
• R:R:    `{rr_ratio:.2f}x`
• Type:   {order_badge}

📈 *Confidence:* {confidence_emoji} {confidence_pct:.1f}%
{confidence_bar} {confidence_pct:.1f}%

🎯 *Confluence:* {signal.confluence_score} points
{confluence_bar} {signal.confluence_score}/12

💡 *Rationale:*
{signal.rationale}

🤖 _Signal generated by Crypto Market Agents_
"""

    return message
