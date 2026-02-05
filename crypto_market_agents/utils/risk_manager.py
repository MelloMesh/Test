"""
Risk Management Module - Portfolio risk, position sizing, and correlation tracking.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone
import logging


@dataclass
class PositionSize:
    """Position sizing recommendation."""
    symbol: str
    recommended_size_pct: float  # Percentage of portfolio
    kelly_fraction: float
    max_loss_pct: float  # Maximum loss as % of portfolio
    reasoning: str


@dataclass
class RiskLimits:
    """Portfolio-level risk limits."""
    max_portfolio_risk_pct: float = 10.0  # Max total risk as % of portfolio
    max_drawdown_pct: float = 20.0  # Auto-stop if exceeded
    max_concurrent_positions: int = 5
    max_single_position_pct: float = 5.0  # Max size per position
    max_correlated_exposure_pct: float = 15.0  # Max exposure to correlated assets


class RiskManager:
    """
    Manages portfolio-level risk including position sizing, correlation tracking,
    and risk limit enforcement.
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        risk_limits: Optional[RiskLimits] = None
    ):
        """
        Initialize risk manager.

        Args:
            initial_capital: Starting capital
            risk_limits: Risk limits configuration
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.risk_limits = risk_limits or RiskLimits()

        # Track open positions: symbol -> (size_pct, entry_price, stop_loss)
        self.open_positions: Dict[str, Tuple[float, float, float]] = {}

        # Correlation matrix: {(symbol1, symbol2): correlation}
        self.correlation_matrix: Dict[Tuple[str, str], float] = {}

        # Price history for correlation calculation: symbol -> [prices]
        self.price_history: Dict[str, List[float]] = {}
        self.max_price_history = 100  # Keep last 100 prices

        self.logger = logging.getLogger(self.__class__.__name__)

    def calculate_kelly_position_size(
        self,
        symbol: str,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        stop_loss_pct: float,
        kelly_fraction: float = 0.25
    ) -> PositionSize:
        """
        Calculate optimal position size using Kelly Criterion.

        Kelly% = (Win Rate * Avg Win - Loss Rate * Avg Loss) / Avg Win

        Args:
            symbol: Trading symbol
            win_rate: Historical win rate (0-1)
            avg_win: Average win percentage
            avg_loss: Average loss percentage (positive value)
            stop_loss_pct: Stop loss percentage for this trade
            kelly_fraction: Fraction of Kelly to use (0.25 = quarter Kelly, recommended)

        Returns:
            PositionSize recommendation
        """
        # Kelly formula
        loss_rate = 1 - win_rate

        # Kelly% = (p * W - q * L) / W
        # where p = win rate, q = loss rate, W = avg win, L = avg loss
        kelly_pct = (win_rate * avg_win - loss_rate * avg_loss) / avg_win if avg_win > 0 else 0

        # Apply Kelly fraction for safety (full Kelly is too aggressive)
        kelly_pct = kelly_pct * kelly_fraction

        # Cap at risk limits
        kelly_pct = max(0, min(kelly_pct, self.risk_limits.max_single_position_pct))

        # Calculate max loss percentage of portfolio
        max_loss_pct = kelly_pct * (stop_loss_pct / 100)

        reasoning = (
            f"Kelly: {kelly_pct:.2f}% (Fractional: {kelly_fraction}), "
            f"Win Rate: {win_rate*100:.1f}%, "
            f"Avg W/L: {avg_win:.2f}%/{avg_loss:.2f}%, "
            f"Max Loss: {max_loss_pct:.2f}% of portfolio"
        )

        return PositionSize(
            symbol=symbol,
            recommended_size_pct=kelly_pct,
            kelly_fraction=kelly_fraction,
            max_loss_pct=max_loss_pct,
            reasoning=reasoning
        )

    def check_portfolio_risk_limits(self) -> Tuple[bool, str]:
        """
        Check if opening a new position would violate portfolio risk limits.

        Returns:
            (can_trade, reason) tuple
        """
        # Check max concurrent positions
        if len(self.open_positions) >= self.risk_limits.max_concurrent_positions:
            return False, f"Max concurrent positions ({self.risk_limits.max_concurrent_positions}) reached"

        # Calculate current portfolio risk
        total_risk_pct = sum(
            size_pct * abs((stop - entry) / entry) * 100
            for size_pct, entry, stop in self.open_positions.values()
        )

        if total_risk_pct >= self.risk_limits.max_portfolio_risk_pct:
            return False, f"Portfolio risk ({total_risk_pct:.2f}%) exceeds limit ({self.risk_limits.max_portfolio_risk_pct}%)"

        return True, "OK"

    def can_open_position(
        self,
        symbol: str,
        position_size_pct: float,
        entry_price: float,
        stop_loss: float
    ) -> Tuple[bool, str]:
        """
        Check if a position can be opened without violating risk limits.

        Args:
            symbol: Trading symbol
            position_size_pct: Proposed position size as % of portfolio
            entry_price: Entry price
            stop_loss: Stop loss price

        Returns:
            (can_open, reason) tuple
        """
        # Check if position already exists
        if symbol in self.open_positions:
            return False, f"Position already open for {symbol}"

        # Check portfolio-level limits
        can_trade, reason = self.check_portfolio_risk_limits()
        if not can_trade:
            return False, reason

        # Check position size limit
        if position_size_pct > self.risk_limits.max_single_position_pct:
            return False, f"Position size ({position_size_pct:.2f}%) exceeds limit ({self.risk_limits.max_single_position_pct}%)"

        # Calculate risk for this position
        stop_loss_pct = abs((stop_loss - entry_price) / entry_price) * 100
        position_risk = position_size_pct * (stop_loss_pct / 100)

        # Check if adding this position would exceed portfolio risk
        current_risk = sum(
            size_pct * abs((stop - entry) / entry) * 100
            for size_pct, entry, stop in self.open_positions.values()
        )
        total_risk = current_risk + position_risk

        if total_risk > self.risk_limits.max_portfolio_risk_pct:
            return False, f"Adding position would exceed portfolio risk limit ({total_risk:.2f}% > {self.risk_limits.max_portfolio_risk_pct}%)"

        # Check correlation limits
        correlated_exposure = self._calculate_correlated_exposure(symbol, position_size_pct)
        if correlated_exposure > self.risk_limits.max_correlated_exposure_pct:
            return False, f"Correlated exposure ({correlated_exposure:.2f}%) exceeds limit ({self.risk_limits.max_correlated_exposure_pct}%)"

        return True, "OK"

    def add_position(
        self,
        symbol: str,
        size_pct: float,
        entry_price: float,
        stop_loss: float
    ):
        """
        Add a position to tracking.

        Args:
            symbol: Trading symbol
            size_pct: Position size as % of portfolio
            entry_price: Entry price
            stop_loss: Stop loss price
        """
        self.open_positions[symbol] = (size_pct, entry_price, stop_loss)
        self.logger.info(
            f"Added position: {symbol} @ {size_pct:.2f}% of portfolio, "
            f"Entry: ${entry_price:.8f}, Stop: ${stop_loss:.8f}"
        )

    def remove_position(self, symbol: str):
        """
        Remove a position from tracking.

        Args:
            symbol: Trading symbol
        """
        if symbol in self.open_positions:
            del self.open_positions[symbol]
            self.logger.info(f"Removed position: {symbol}")

    def update_capital(self, new_capital: float):
        """
        Update current capital (e.g., after trades close).

        Args:
            new_capital: New capital amount
        """
        old_capital = self.current_capital
        self.current_capital = new_capital
        drawdown = ((self.initial_capital - new_capital) / self.initial_capital) * 100

        self.logger.info(
            f"Capital updated: ${old_capital:,.2f} -> ${new_capital:,.2f}, "
            f"Drawdown: {drawdown:.2f}%"
        )

        # Check drawdown limit
        if drawdown > self.risk_limits.max_drawdown_pct:
            self.logger.critical(
                f"⚠️  MAX DRAWDOWN EXCEEDED: {drawdown:.2f}% > {self.risk_limits.max_drawdown_pct}%"
            )

    def update_price_history(self, symbol: str, price: float):
        """
        Update price history for correlation calculation.

        Args:
            symbol: Trading symbol
            price: Current price
        """
        if symbol not in self.price_history:
            self.price_history[symbol] = []

        self.price_history[symbol].append(price)

        # Trim to max length
        if len(self.price_history[symbol]) > self.max_price_history:
            self.price_history[symbol] = self.price_history[symbol][-self.max_price_history:]

    def calculate_correlation(self, symbol1: str, symbol2: str) -> Optional[float]:
        """
        Calculate price correlation between two symbols.

        Args:
            symbol1: First symbol
            symbol2: Second symbol

        Returns:
            Correlation coefficient (-1 to 1) or None if insufficient data
        """
        if symbol1 not in self.price_history or symbol2 not in self.price_history:
            return None

        prices1 = self.price_history[symbol1]
        prices2 = self.price_history[symbol2]

        # Need at least 30 data points for meaningful correlation
        min_length = min(len(prices1), len(prices2))
        if min_length < 30:
            return None

        # Use the last N common prices
        prices1 = prices1[-min_length:]
        prices2 = prices2[-min_length:]

        # Calculate returns
        returns1 = [(prices1[i] - prices1[i-1]) / prices1[i-1] for i in range(1, len(prices1))]
        returns2 = [(prices2[i] - prices2[i-1]) / prices2[i-1] for i in range(1, len(prices2))]

        # Calculate correlation coefficient
        n = len(returns1)
        if n == 0:
            return None

        mean1 = sum(returns1) / n
        mean2 = sum(returns2) / n

        numerator = sum((returns1[i] - mean1) * (returns2[i] - mean2) for i in range(n))

        std1 = (sum((r - mean1) ** 2 for r in returns1) / n) ** 0.5
        std2 = (sum((r - mean2) ** 2 for r in returns2) / n) ** 0.5

        if std1 == 0 or std2 == 0:
            return None

        correlation = numerator / (n * std1 * std2)

        # Cache correlation
        self.correlation_matrix[(symbol1, symbol2)] = correlation
        self.correlation_matrix[(symbol2, symbol1)] = correlation

        return correlation

    def _calculate_correlated_exposure(self, symbol: str, proposed_size_pct: float) -> float:
        """
        Calculate total exposure to assets correlated with the given symbol.

        Args:
            symbol: Symbol to check
            proposed_size_pct: Proposed position size

        Returns:
            Total correlated exposure as % of portfolio
        """
        # Start with the proposed position itself
        total_exposure = proposed_size_pct

        # Add exposure from correlated positions (correlation > 0.7)
        for open_symbol, (size_pct, _, _) in self.open_positions.items():
            if open_symbol == symbol:
                continue

            # Calculate correlation
            correlation = self.calculate_correlation(symbol, open_symbol)

            if correlation is not None and correlation > 0.7:
                # Add proportional exposure based on correlation strength
                total_exposure += size_pct * correlation
                self.logger.debug(
                    f"High correlation detected: {symbol} <-> {open_symbol} = {correlation:.2f}"
                )

        return total_exposure

    def get_risk_report(self) -> Dict:
        """
        Generate a comprehensive risk report.

        Returns:
            Risk metrics dictionary
        """
        total_exposure_pct = sum(size for size, _, _ in self.open_positions.values())

        total_risk_pct = sum(
            size_pct * abs((stop - entry) / entry) * 100
            for size_pct, entry, stop in self.open_positions.values()
        )

        drawdown_pct = ((self.initial_capital - self.current_capital) / self.initial_capital) * 100

        return {
            "capital": self.current_capital,
            "initial_capital": self.initial_capital,
            "drawdown_pct": drawdown_pct,
            "open_positions": len(self.open_positions),
            "max_positions": self.risk_limits.max_concurrent_positions,
            "total_exposure_pct": total_exposure_pct,
            "total_risk_pct": total_risk_pct,
            "max_portfolio_risk_pct": self.risk_limits.max_portfolio_risk_pct,
            "risk_utilization_pct": (total_risk_pct / self.risk_limits.max_portfolio_risk_pct) * 100,
            "positions": [
                {
                    "symbol": symbol,
                    "size_pct": size,
                    "entry": entry,
                    "stop": stop,
                    "risk_pct": size * abs((stop - entry) / entry) * 100
                }
                for symbol, (size, entry, stop) in self.open_positions.items()
            ]
        }
