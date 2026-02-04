"""
Beautiful signal display formatter for trading signals.
"""

from typing import List
from datetime import datetime
from ..schemas import TradingSignal


class SignalDisplay:
    """Format trading signals for beautiful terminal display."""

    # Color codes for terminal
    GREEN = '\033[92m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    YELLOW = '\033[93m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

    @staticmethod
    def format_signal(signal: TradingSignal, index: int = 1) -> str:
        """
        Format a single trading signal for display.

        Args:
            signal: Trading signal to format
            index: Signal number

        Returns:
            Formatted string
        """
        # Direction color
        if signal.direction == "LONG":
            direction_color = SignalDisplay.GREEN
            direction_symbol = "📈"
        else:
            direction_color = SignalDisplay.RED
            direction_symbol = "📉"

        # Confidence color
        if signal.confidence >= 0.8:
            conf_color = SignalDisplay.GREEN
        elif signal.confidence >= 0.6:
            conf_color = SignalDisplay.YELLOW
        else:
            conf_color = SignalDisplay.RED

        # Calculate risk/reward
        if signal.direction == "LONG":
            risk = signal.entry - signal.stop
            reward = signal.target - signal.entry
        else:
            risk = signal.stop - signal.entry
            reward = signal.entry - signal.target

        rr_ratio = reward / risk if risk > 0 else 0

        # Format the signal
        output = []
        output.append(f"\n{SignalDisplay.BOLD}{SignalDisplay.CYAN}{'─' * 70}{SignalDisplay.RESET}")
        output.append(f"{SignalDisplay.BOLD}#{index} {direction_symbol} {signal.asset} - {direction_color}{signal.direction}{SignalDisplay.RESET}")
        output.append(f"{SignalDisplay.CYAN}{'─' * 70}{SignalDisplay.RESET}")

        output.append(f"  {SignalDisplay.BOLD}Entry:{SignalDisplay.RESET}      ${signal.entry:,.8g}")
        output.append(f"  {SignalDisplay.RED}Stop Loss:{SignalDisplay.RESET}  ${signal.stop:,.8g}  {SignalDisplay.RED}({abs((signal.stop - signal.entry) / signal.entry * 100):.2f}%){SignalDisplay.RESET}")
        output.append(f"  {SignalDisplay.GREEN}Target:{SignalDisplay.RESET}     ${signal.target:,.8g}  {SignalDisplay.GREEN}(+{abs((signal.target - signal.entry) / signal.entry * 100):.2f}%){SignalDisplay.RESET}")
        output.append(f"  {SignalDisplay.BLUE}R:R Ratio:{SignalDisplay.RESET}  {rr_ratio:.2f}:1")
        output.append(f"  {SignalDisplay.BOLD}Confidence:{SignalDisplay.RESET} {conf_color}{signal.confidence * 100:.0f}%{SignalDisplay.RESET}")
        output.append(f"\n  {SignalDisplay.YELLOW}💡 {signal.rationale}{SignalDisplay.RESET}")
        output.append(f"{SignalDisplay.CYAN}{'─' * 70}{SignalDisplay.RESET}\n")

        return "\n".join(output)

    @staticmethod
    def format_signal_simple(signal: TradingSignal, index: int = 1) -> str:
        """
        Format a single trading signal in simple mode (no colors).

        Args:
            signal: Trading signal to format
            index: Signal number

        Returns:
            Formatted string
        """
        # Calculate risk/reward
        if signal.direction == "LONG":
            risk = signal.entry - signal.stop
            reward = signal.target - signal.entry
        else:
            risk = signal.stop - signal.entry
            reward = signal.entry - signal.target

        rr_ratio = reward / risk if risk > 0 else 0

        # Direction symbol
        direction_symbol = "📈" if signal.direction == "LONG" else "📉"

        # Format the signal
        output = []
        output.append(f"\n{'─' * 70}")
        output.append(f"#{index} {direction_symbol} {signal.asset} - {signal.direction}")
        output.append(f"{'─' * 70}")
        output.append(f"  Entry:      ${signal.entry:,.8g}")
        output.append(f"  Stop Loss:  ${signal.stop:,.8g}  ({abs((signal.stop - signal.entry) / signal.entry * 100):.2f}%)")
        output.append(f"  Target:     ${signal.target:,.8g}  (+{abs((signal.target - signal.entry) / signal.entry * 100):.2f}%)")
        output.append(f"  R:R Ratio:  {rr_ratio:.2f}:1")
        output.append(f"  Confidence: {signal.confidence * 100:.0f}%")
        output.append(f"\n  💡 {signal.rationale}")
        output.append(f"{'─' * 70}\n")

        return "\n".join(output)

    @staticmethod
    def format_signals_table(signals: List[TradingSignal], max_signals: int = 5) -> str:
        """
        Format multiple signals as a compact table.

        Args:
            signals: List of trading signals
            max_signals: Maximum number of signals to display

        Returns:
            Formatted table string
        """
        if not signals:
            return f"\n{SignalDisplay.YELLOW}No trading signals at this time.{SignalDisplay.RESET}\n"

        output = []
        output.append(f"\n{SignalDisplay.BOLD}{SignalDisplay.CYAN}{'═' * 100}{SignalDisplay.RESET}")
        output.append(f"{SignalDisplay.BOLD}{'TRADING SIGNALS':^100}{SignalDisplay.RESET}")
        output.append(f"{SignalDisplay.CYAN}{'═' * 100}{SignalDisplay.RESET}")

        # Table header
        header = f"{SignalDisplay.BOLD}{'#':<3} {'ASSET':<12} {'DIR':<6} {'ENTRY':>12} {'STOP':>12} {'TARGET':>12} {'R:R':<6} {'CONF':<6}{SignalDisplay.RESET}"
        output.append(header)
        output.append(f"{SignalDisplay.CYAN}{'─' * 100}{SignalDisplay.RESET}")

        # Display signals
        for i, signal in enumerate(signals[:max_signals], 1):
            # Direction color and symbol
            if signal.direction == "LONG":
                dir_color = SignalDisplay.GREEN
                dir_text = "📈 LONG"
            else:
                dir_color = SignalDisplay.RED
                dir_text = "📉 SHORT"

            # Confidence color
            if signal.confidence >= 0.8:
                conf_color = SignalDisplay.GREEN
            elif signal.confidence >= 0.6:
                conf_color = SignalDisplay.YELLOW
            else:
                conf_color = SignalDisplay.RED

            # Calculate R:R
            if signal.direction == "LONG":
                risk = signal.entry - signal.stop
                reward = signal.target - signal.entry
            else:
                risk = signal.stop - signal.entry
                reward = signal.entry - signal.target

            rr_ratio = reward / risk if risk > 0 else 0

            row = (
                f"{i:<3} "
                f"{signal.asset:<12} "
                f"{dir_color}{dir_text:<6}{SignalDisplay.RESET} "
                f"${signal.entry:>11,.2f} "
                f"${signal.stop:>11,.2f} "
                f"${signal.target:>11,.2f} "
                f"{rr_ratio:>5.2f}:1 "
                f"{conf_color}{signal.confidence * 100:>4.0f}%{SignalDisplay.RESET}"
            )
            output.append(row)

        output.append(f"{SignalDisplay.CYAN}{'═' * 100}{SignalDisplay.RESET}\n")

        return "\n".join(output)

    @staticmethod
    def format_signal_card(signal: TradingSignal, index: int = 1) -> str:
        """
        Format signal as a clean card layout.

        Args:
            signal: Trading signal to format
            index: Signal number

        Returns:
            Formatted card string
        """
        # Direction styling
        if signal.direction == "LONG":
            dir_color = SignalDisplay.GREEN
            symbol = "🚀"
            action = "BUY"
        else:
            dir_color = SignalDisplay.RED
            symbol = "⚡"
            action = "SELL"

        # Confidence bar
        conf_bars = int(signal.confidence * 10)
        conf_bar = "█" * conf_bars + "░" * (10 - conf_bars)

        # Calculate percentages
        if signal.direction == "LONG":
            stop_pct = abs((signal.stop - signal.entry) / signal.entry * 100)
            target_pct = abs((signal.target - signal.entry) / signal.entry * 100)
            risk = signal.entry - signal.stop
            reward = signal.target - signal.entry
        else:
            stop_pct = abs((signal.stop - signal.entry) / signal.entry * 100)
            target_pct = abs((signal.entry - signal.target) / signal.entry * 100)
            risk = signal.stop - signal.entry
            reward = signal.entry - signal.target

        rr_ratio = reward / risk if risk > 0 else 0

        # Build card
        output = []
        output.append(f"\n{SignalDisplay.BOLD}{SignalDisplay.CYAN}┌{'─' * 68}┐{SignalDisplay.RESET}")
        output.append(f"{SignalDisplay.CYAN}│{SignalDisplay.RESET} {symbol} {SignalDisplay.BOLD}Signal #{index}: {signal.asset}{SignalDisplay.RESET}{' ' * (58 - len(signal.asset))} {SignalDisplay.CYAN}│{SignalDisplay.RESET}")
        output.append(f"{SignalDisplay.CYAN}├{'─' * 68}┤{SignalDisplay.RESET}")

        output.append(f"{SignalDisplay.CYAN}│{SignalDisplay.RESET}  {SignalDisplay.BOLD}ACTION:{SignalDisplay.RESET}  {dir_color}{action} {signal.direction}{SignalDisplay.RESET}{' ' * (55 - len(action) - len(signal.direction))} {SignalDisplay.CYAN}│{SignalDisplay.RESET}")
        output.append(f"{SignalDisplay.CYAN}│{SignalDisplay.RESET}{' ' * 68}{SignalDisplay.CYAN}│{SignalDisplay.RESET}")

        output.append(f"{SignalDisplay.CYAN}│{SignalDisplay.RESET}  📍 Entry:      {SignalDisplay.BOLD}${signal.entry:,.8g}{SignalDisplay.RESET}{' ' * (47 - len(f'${signal.entry:,.8g}'))} {SignalDisplay.CYAN}│{SignalDisplay.RESET}")
        output.append(f"{SignalDisplay.CYAN}│{SignalDisplay.RESET}  🛑 Stop Loss:  ${signal.stop:,.8g}  {SignalDisplay.RED}(-{stop_pct:.2f}%){SignalDisplay.RESET}{' ' * (35 - len(f'${signal.stop:,.8g}') - len(f'{stop_pct:.2f}'))} {SignalDisplay.CYAN}│{SignalDisplay.RESET}")
        output.append(f"{SignalDisplay.CYAN}│{SignalDisplay.RESET}  🎯 Target:     ${signal.target:,.8g}  {SignalDisplay.GREEN}(+{target_pct:.2f}%){SignalDisplay.RESET}{' ' * (34 - len(f'${signal.target:,.8g}') - len(f'{target_pct:.2f}'))} {SignalDisplay.CYAN}│{SignalDisplay.RESET}")
        output.append(f"{SignalDisplay.CYAN}│{SignalDisplay.RESET}{' ' * 68}{SignalDisplay.CYAN}│{SignalDisplay.RESET}")

        output.append(f"{SignalDisplay.CYAN}│{SignalDisplay.RESET}  ⚖️  Risk:Reward  {SignalDisplay.BOLD}{rr_ratio:.2f}:1{SignalDisplay.RESET}{' ' * (50 - len(f'{rr_ratio:.2f}'))} {SignalDisplay.CYAN}│{SignalDisplay.RESET}")
        output.append(f"{SignalDisplay.CYAN}│{SignalDisplay.RESET}  📊 Confidence:  {SignalDisplay.BOLD}{signal.confidence * 100:.0f}%{SignalDisplay.RESET}  {conf_bar}{' ' * (42 - len(f'{signal.confidence * 100:.0f}'))} {SignalDisplay.CYAN}│{SignalDisplay.RESET}")
        output.append(f"{SignalDisplay.CYAN}│{SignalDisplay.RESET}{' ' * 68}{SignalDisplay.CYAN}│{SignalDisplay.RESET}")

        # Wrap rationale
        rationale_lines = SignalDisplay._wrap_text(signal.rationale, 64)
        for line in rationale_lines:
            output.append(f"{SignalDisplay.CYAN}│{SignalDisplay.RESET}  {SignalDisplay.YELLOW}{line}{SignalDisplay.RESET}{' ' * (66 - len(line))}{SignalDisplay.CYAN}│{SignalDisplay.RESET}")

        output.append(f"{SignalDisplay.CYAN}└{'─' * 68}┘{SignalDisplay.RESET}\n")

        return "\n".join(output)

    @staticmethod
    def _wrap_text(text: str, width: int) -> List[str]:
        """Wrap text to specified width."""
        words = text.split()
        lines = []
        current_line = ""

        for word in words:
            if len(current_line) + len(word) + 1 <= width:
                current_line += (word + " ")
            else:
                if current_line:
                    lines.append(current_line.rstrip())
                current_line = word + " "

        if current_line:
            lines.append(current_line.rstrip())

        return lines or [""]
