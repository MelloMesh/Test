"""
Limit Order Manager
Tracks pending limit orders and determines when to override to market
"""

from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class PendingLimit:
    """Pending limit order"""
    symbol: str
    signal_name: str
    direction: str
    limit_price: float
    current_price_at_placement: float
    original_confluence_score: int
    trade_type: str
    htf_bias: str
    htf_alignment: float
    created_at: datetime
    expires_at: datetime


class LimitOrderManager:
    """
    Manages pending limit orders and override logic

    Override Conditions:
    1. Confluence score jumped significantly (weak signal became strong)
    2. Extreme LTF signals appeared (RSI < 25 + divergence + volume)
    3. Price moving away from limit (might miss the move)
    4. Time decay - limit pending too long but setup still valid
    """

    def __init__(self, cache_file: str = ".limit_orders.json"):
        self.cache_file = Path(cache_file)
        self.pending_limits: Dict[str, PendingLimit] = {}

        # Override thresholds
        self.score_jump_threshold = 3  # Score must increase by 3+ to override
        self.price_drift_pct = 0.02    # 2% price drift away from limit
        self.max_pending_hours = 4     # Cancel limits older than 4 hours

        self._load_limits()

    def add_limit_order(
        self,
        symbol: str,
        signal_name: str,
        direction: str,
        limit_price: float,
        current_price: float,
        confluence_score: int,
        trade_type: str,
        htf_bias: str,
        htf_alignment: float
    ) -> str:
        """
        Add a new pending limit order

        Returns: Order key
        """
        order_key = f"{symbol}_{signal_name}_{direction}"

        # Remove any existing limit for same signal
        if order_key in self.pending_limits:
            logger.info(f"Replacing existing limit order: {order_key}")

        # Create pending limit
        pending = PendingLimit(
            symbol=symbol,
            signal_name=signal_name,
            direction=direction,
            limit_price=limit_price,
            current_price_at_placement=current_price,
            original_confluence_score=confluence_score,
            trade_type=trade_type,
            htf_bias=htf_bias,
            htf_alignment=htf_alignment,
            created_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc) + timedelta(hours=self.max_pending_hours)
        )

        self.pending_limits[order_key] = pending
        self._save_limits()

        logger.info(
            f"📋 LIMIT ORDER: {symbol} {direction.upper()} @ ${limit_price:.2f} "
            f"(current: ${current_price:.2f}, score: {confluence_score})"
        )

        return order_key

    def check_limit_override(
        self,
        order_key: str,
        current_price: float,
        current_confluence_score: int,
        rsi_value: Optional[float] = None,
        has_divergence: bool = False,
        volume_spike: float = 1.0
    ) -> str:
        """
        Check if pending limit should be overridden to market

        Args:
            order_key: Limit order key
            current_price: Current market price
            current_confluence_score: Re-evaluated confluence score
            rsi_value: Current RSI (if available)
            has_divergence: New divergence appeared
            volume_spike: Volume multiple vs average (1.0 = normal)

        Returns: 'CANCEL_LIMIT_GO_MARKET', 'KEEP_LIMIT', or 'CANCEL_LIMIT'
        """
        if order_key not in self.pending_limits:
            return 'KEEP_LIMIT'

        pending = self.pending_limits[order_key]
        now = datetime.now(timezone.utc)
        time_pending = (now - pending.created_at).total_seconds() / 3600  # hours

        # === 1. CHECK IF LIMIT FILLED ===
        filled = self._check_if_filled(pending, current_price)
        if filled:
            self.remove_limit_order(order_key)
            return 'LIMIT_FILLED'

        # === 2. CHECK EXPIRATION ===
        if now > pending.expires_at:
            logger.info(f"⏰ Limit expired: {order_key} ({time_pending:.1f}h old)")
            # If still good confluence, override to market
            if current_confluence_score >= 5:
                logger.info(f"  → Still good setup (score: {current_confluence_score}), going MARKET")
                return 'CANCEL_LIMIT_GO_MARKET'
            else:
                logger.info(f"  → Setup degraded (score: {current_confluence_score}), canceling")
                self.remove_limit_order(order_key)
                return 'CANCEL_LIMIT'

        # === 3. STRONG CONFLUENCE APPEARED ===
        score_increase = current_confluence_score - pending.original_confluence_score
        if score_increase >= self.score_jump_threshold and current_confluence_score >= 7:
            logger.info(
                f"🚀 OVERRIDE: {order_key} - Confluence jumped "
                f"({pending.original_confluence_score} → {current_confluence_score}, +{score_increase})"
            )
            return 'CANCEL_LIMIT_GO_MARKET'

        # === 4. EXTREME LTF SIGNALS ===
        if rsi_value is not None and has_divergence and volume_spike > 2.5:
            if pending.direction == "long" and rsi_value < 25:
                logger.info(
                    f"🚀 OVERRIDE: {order_key} - Extreme oversold "
                    f"(RSI: {rsi_value:.0f}, divergence, volume spike {volume_spike:.1f}x)"
                )
                return 'CANCEL_LIMIT_GO_MARKET'
            elif pending.direction == "short" and rsi_value > 75:
                logger.info(
                    f"🚀 OVERRIDE: {order_key} - Extreme overbought "
                    f"(RSI: {rsi_value:.0f}, divergence, volume spike {volume_spike:.1f}x)"
                )
                return 'CANCEL_LIMIT_GO_MARKET'

        # === 5. PRICE MOVING AWAY FROM LIMIT ===
        if pending.direction == "long":
            # Long limit not filled, price moving up (away from limit)
            if current_price > pending.limit_price * (1 + self.price_drift_pct):
                drift_pct = ((current_price - pending.limit_price) / pending.limit_price) * 100
                # Check if trend is accelerating
                if current_confluence_score >= 5:
                    logger.info(
                        f"🚀 OVERRIDE: {order_key} - Price drifting away "
                        f"(limit: ${pending.limit_price:.2f}, current: ${current_price:.2f}, +{drift_pct:.1f}%)"
                    )
                    return 'CANCEL_LIMIT_GO_MARKET'

        elif pending.direction == "short":
            # Short limit not filled, price moving down (away from limit)
            if current_price < pending.limit_price * (1 - self.price_drift_pct):
                drift_pct = ((pending.limit_price - current_price) / pending.limit_price) * 100
                if current_confluence_score >= 5:
                    logger.info(
                        f"🚀 OVERRIDE: {order_key} - Price drifting away "
                        f"(limit: ${pending.limit_price:.2f}, current: ${current_price:.2f}, -{drift_pct:.1f}%)"
                    )
                    return 'CANCEL_LIMIT_GO_MARKET'

        # === 6. SETUP INVALIDATED ===
        if current_confluence_score < 2:
            logger.info(
                f"❌ CANCEL: {order_key} - Setup invalidated "
                f"(score dropped from {pending.original_confluence_score} to {current_confluence_score})"
            )
            self.remove_limit_order(order_key)
            return 'CANCEL_LIMIT'

        # Keep waiting for limit to fill
        return 'KEEP_LIMIT'

    def _check_if_filled(self, pending: PendingLimit, current_price: float) -> bool:
        """Check if limit would have filled"""
        if pending.direction == "long":
            # Long limit fills if price touched limit or below
            return current_price <= pending.limit_price
        else:  # short
            # Short limit fills if price touched limit or above
            return current_price >= pending.limit_price

    def remove_limit_order(self, order_key: str):
        """Remove a pending limit order"""
        if order_key in self.pending_limits:
            del self.pending_limits[order_key]
            self._save_limits()
            logger.info(f"🗑️  Removed limit: {order_key}")

    def get_pending_limits(self) -> List[PendingLimit]:
        """Get all pending limit orders"""
        return list(self.pending_limits.values())

    def get_limit(self, order_key: str) -> Optional[PendingLimit]:
        """Get specific pending limit"""
        return self.pending_limits.get(order_key)

    def clear_all(self):
        """Clear all pending limits"""
        self.pending_limits = {}
        self._save_limits()
        logger.info("🗑️  Cleared all pending limits")

    def _load_limits(self):
        """Load pending limits from file"""
        if not self.cache_file.exists():
            return

        try:
            with open(self.cache_file, 'r') as f:
                data = json.load(f)

            # Convert timestamps back to datetime
            for key, limit_data in data.items():
                limit_data['created_at'] = datetime.fromisoformat(limit_data['created_at'])
                limit_data['expires_at'] = datetime.fromisoformat(limit_data['expires_at'])
                self.pending_limits[key] = PendingLimit(**limit_data)

            logger.info(f"📂 Loaded {len(self.pending_limits)} pending limits")

            # Clean up expired limits
            self._cleanup_expired()

        except Exception as e:
            logger.warning(f"Failed to load pending limits: {e}")

    def _save_limits(self):
        """Save pending limits to file"""
        try:
            # Convert to JSON-serializable format
            data = {}
            for key, limit in self.pending_limits.items():
                limit_dict = asdict(limit)
                limit_dict['created_at'] = limit.created_at.isoformat()
                limit_dict['expires_at'] = limit.expires_at.isoformat()
                data[key] = limit_dict

            with open(self.cache_file, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.warning(f"Failed to save pending limits: {e}")

    def _cleanup_expired(self):
        """Remove expired limits"""
        now = datetime.now(timezone.utc)
        expired = [key for key, limit in self.pending_limits.items() if now > limit.expires_at]

        for key in expired:
            logger.info(f"⏰ Removing expired limit: {key}")
            del self.pending_limits[key]

        if expired:
            self._save_limits()

    def get_stats(self) -> Dict:
        """Get limit order statistics"""
        if not self.pending_limits:
            return {"total_pending": 0}

        now = datetime.now(timezone.utc)
        ages = [(now - limit.created_at).total_seconds() / 3600 for limit in self.pending_limits.values()]

        return {
            "total_pending": len(self.pending_limits),
            "avg_age_hours": sum(ages) / len(ages) if ages else 0,
            "oldest_hours": max(ages) if ages else 0
        }


# Singleton instance
_manager_instance = None

def get_limit_manager() -> LimitOrderManager:
    """Get global limit order manager instance"""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = LimitOrderManager()
    return _manager_instance
