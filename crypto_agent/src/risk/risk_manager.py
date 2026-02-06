"""
Risk manager — enforces all risk rules before any order is placed.

Every check must pass before a position can be opened. If ANY check fails,
the trade is rejected with a reason string.

This is the last line of defense. These rules are NON-NEGOTIABLE.
"""

from src.config import (
    CONSECUTIVE_LOSS_THRESHOLD,
    DAILY_LOSS_LIMIT,
    DEFAULT_RISK_PCT,
    MAX_CONCURRENT_POSITIONS,
    MAX_RISK_PCT,
    MAX_STOP_DISTANCE_PCT,
    MAX_TOTAL_EXPOSURE,
    REDUCED_RISK_PCT,
    WEEKLY_LOSS_LIMIT,
)
from src.risk.portfolio import Portfolio
from src.strategy.signals import Signal
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RiskManager:
    """
    Enforces all risk management rules.

    Call can_open_position() before every trade. If it returns (False, reason),
    the trade MUST be rejected.
    """

    def __init__(self, paper_mode: bool = True):
        self.paper_mode = paper_mode
        self._halted_until: float | None = None  # Unix timestamp
        self._manual_review_required: bool = False

    def can_open_position(
        self,
        signal: Signal,
        portfolio: Portfolio,
    ) -> tuple[bool, str]:
        """
        Run all risk checks. ALL must pass for a trade to be allowed.

        Args:
            signal: The trade signal to evaluate.
            portfolio: Current portfolio state.

        Returns:
            (True, "approved") if all checks pass.
            (False, "reason") if any check fails.
        """
        checks = [
            self._check_manual_review,
            self._check_halt,
            self._check_daily_loss_limit,
            self._check_weekly_loss_limit,
            self._check_max_concurrent_positions,
            self._check_total_exposure,
            self._check_stop_loss_exists,
            self._check_stop_distance,
            self._check_risk_pct_valid,
            self._check_take_profits_exist,
        ]

        for check in checks:
            allowed, reason = check(signal, portfolio)
            if not allowed:
                logger.warning(f"RISK REJECTED: {signal.symbol} {signal.direction} — {reason}")
                return False, reason

        # Apply consecutive loss override
        risk_pct = signal.risk_pct
        if portfolio.consecutive_losses >= CONSECUTIVE_LOSS_THRESHOLD:
            if risk_pct > REDUCED_RISK_PCT:
                logger.warning(
                    f"Consecutive loss override: {portfolio.consecutive_losses} losses, "
                    f"reducing risk from {risk_pct:.1%} to {REDUCED_RISK_PCT:.1%}"
                )
                signal.risk_pct = REDUCED_RISK_PCT

        logger.info(f"RISK APPROVED: {signal.symbol} {signal.direction} @ {signal.risk_pct:.1%} risk")
        return True, "approved"

    def _check_manual_review(self, signal: Signal, portfolio: Portfolio) -> tuple[bool, str]:
        if self._manual_review_required:
            return False, "Trading halted: manual review required (weekly loss limit hit)"
        return True, ""

    def _check_halt(self, signal: Signal, portfolio: Portfolio) -> tuple[bool, str]:
        if self._halted_until is not None:
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc).timestamp()
            if now < self._halted_until:
                remaining_hrs = (self._halted_until - now) / 3600
                return False, f"Trading halted for {remaining_hrs:.1f} more hours (daily loss limit)"
            else:
                self._halted_until = None
        return True, ""

    def _check_daily_loss_limit(self, signal: Signal, portfolio: Portfolio) -> tuple[bool, str]:
        if portfolio.daily_pnl_pct <= -DAILY_LOSS_LIMIT:
            from datetime import datetime, timezone, timedelta
            self._halted_until = (datetime.now(timezone.utc) + timedelta(hours=24)).timestamp()
            return False, (
                f"Daily loss limit hit: {portfolio.daily_pnl_pct:.2%} "
                f"(limit: -{DAILY_LOSS_LIMIT:.0%}). Halted for 24 hours."
            )
        return True, ""

    def _check_weekly_loss_limit(self, signal: Signal, portfolio: Portfolio) -> tuple[bool, str]:
        if portfolio.weekly_pnl_pct <= -WEEKLY_LOSS_LIMIT:
            self._manual_review_required = True
            return False, (
                f"Weekly loss limit hit: {portfolio.weekly_pnl_pct:.2%} "
                f"(limit: -{WEEKLY_LOSS_LIMIT:.0%}). Manual review required."
            )
        return True, ""

    def _check_max_concurrent_positions(self, signal: Signal, portfolio: Portfolio) -> tuple[bool, str]:
        if portfolio.open_position_count >= MAX_CONCURRENT_POSITIONS:
            return False, (
                f"Max concurrent positions reached: {portfolio.open_position_count} "
                f"(limit: {MAX_CONCURRENT_POSITIONS})"
            )
        return True, ""

    def _check_total_exposure(self, signal: Signal, portfolio: Portfolio) -> tuple[bool, str]:
        if self.paper_mode:
            return True, ""  # Paper mode can exceed for testing

        if portfolio.total_exposure_pct >= MAX_TOTAL_EXPOSURE:
            return False, (
                f"Max total exposure reached: {portfolio.total_exposure_pct:.2%} "
                f"(limit: {MAX_TOTAL_EXPOSURE:.0%})"
            )
        return True, ""

    def _check_stop_loss_exists(self, signal: Signal, portfolio: Portfolio) -> tuple[bool, str]:
        if signal.stop_loss <= 0:
            return False, "No stop loss defined — every trade MUST have a stop"
        return True, ""

    def _check_stop_distance(self, signal: Signal, portfolio: Portfolio) -> tuple[bool, str]:
        if signal.stop_distance_pct > MAX_STOP_DISTANCE_PCT + 0.001:  # Small tolerance
            return False, (
                f"Stop distance {signal.stop_distance_pct:.2%} exceeds "
                f"{MAX_STOP_DISTANCE_PCT:.0%} maximum"
            )
        return True, ""

    def _check_risk_pct_valid(self, signal: Signal, portfolio: Portfolio) -> tuple[bool, str]:
        if signal.risk_pct < DEFAULT_RISK_PCT or signal.risk_pct > MAX_RISK_PCT:
            # Allow reduced risk from consecutive loss override
            if signal.risk_pct != REDUCED_RISK_PCT:
                return False, (
                    f"Risk pct {signal.risk_pct:.2%} outside valid range "
                    f"({DEFAULT_RISK_PCT:.0%}-{MAX_RISK_PCT:.0%})"
                )
        return True, ""

    def _check_take_profits_exist(self, signal: Signal, portfolio: Portfolio) -> tuple[bool, str]:
        if not signal.take_profits:
            return False, "No take profit levels defined"
        return True, ""

    def reset_halt(self) -> None:
        """Manually reset trading halt (for manual review override)."""
        self._halted_until = None
        self._manual_review_required = False
        logger.info("Trading halt manually reset")
