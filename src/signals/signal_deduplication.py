"""
Signal Deduplication System
Prevents duplicate alerts for the same signal across multiple scanner cycles
"""

from datetime import datetime, timedelta, timezone
from typing import Dict, Optional
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class CachedSignal:
    """Cached signal information for deduplication"""
    signal_key: str
    timestamp: datetime
    price: float
    htf_alignment: float
    htf_bias: str
    alert_sent: bool


class SignalDeduplicator:
    """
    Intelligent signal deduplication with reset conditions

    Features:
    - Prevents spam (same signal every 5 minutes)
    - Allows re-alerts on significant price moves
    - Allows re-alerts on HTF context changes
    - Configurable cooldown periods per signal type
    """

    def __init__(self, cache_file: str = ".signal_cache.json"):
        self.cache_file = Path(cache_file)
        self.signal_cache: Dict[str, CachedSignal] = {}

        # Cooldown periods (in seconds)
        self.default_cooldown = 4 * 3600  # 4 hours

        # Custom cooldowns per signal type
        self.signal_cooldowns = {
            'RSI_DIVERGENCE': 4 * 3600,      # 4 hours
            'RSI_OVERSOLD': 2 * 3600,         # 2 hours
            'RSI_OVERBOUGHT': 2 * 3600,       # 2 hours
            'MACD_CROSS': 2 * 3600,           # 2 hours
            'MACD_DIVERGENCE': 4 * 3600,      # 4 hours
            'SUPPORT_BOUNCE': 6 * 3600,       # 6 hours (S/R doesn't move often)
            'RESISTANCE_REJECTION': 6 * 3600, # 6 hours
            'BOLLINGER_SQUEEZE': 8 * 3600,    # 8 hours (slower signal)
        }

        # Thresholds for resets
        self.price_change_threshold = 0.05  # 5% price change triggers reset
        self.htf_alignment_change_threshold = 20  # 20% alignment change triggers reset

        # Load existing cache
        self._load_cache()

    def should_alert(
        self,
        symbol: str,
        signal_name: str,
        direction: str,
        current_price: float,
        htf_alignment: float,
        htf_bias: str
    ) -> bool:
        """
        Determine if this signal should trigger an alert

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            signal_name: Signal type (e.g., 'RSI_DIVERGENCE')
            direction: 'long' or 'short'
            current_price: Current market price
            htf_alignment: HTF alignment score (0-100)
            htf_bias: HTF bias ('bullish', 'bearish', 'neutral')

        Returns:
            True if should alert, False if should skip (duplicate)
        """
        signal_key = f"{symbol}_{signal_name}_{direction}"

        # Never seen this signal before
        if signal_key not in self.signal_cache:
            self._cache_signal(
                signal_key, current_price, htf_alignment, htf_bias
            )
            logger.info(f"✅ NEW signal: {signal_key} at ${current_price:.2f}")
            return True  # ALERT

        cached = self.signal_cache[signal_key]
        time_since_last = (datetime.now(timezone.utc) - cached.timestamp).total_seconds()

        # Get cooldown for this signal type
        cooldown = self.signal_cooldowns.get(signal_name, self.default_cooldown)

        # CHECK RESET CONDITIONS (allow re-alert even within cooldown)

        # 1. Significant price move
        price_change = abs(current_price - cached.price) / cached.price
        if price_change > self.price_change_threshold:
            logger.info(
                f"✅ PRICE RESET: {signal_key} - Price moved {price_change*100:.1f}% "
                f"(${cached.price:.2f} → ${current_price:.2f})"
            )
            self._cache_signal(signal_key, current_price, htf_alignment, htf_bias)
            return True  # ALERT - significant price move

        # 2. HTF context changed meaningfully
        alignment_change = abs(htf_alignment - cached.htf_alignment)
        if alignment_change > self.htf_alignment_change_threshold:
            logger.info(
                f"✅ HTF RESET: {signal_key} - Alignment changed {alignment_change:.0f}% "
                f"({cached.htf_alignment:.0f}% → {htf_alignment:.0f}%)"
            )
            self._cache_signal(signal_key, current_price, htf_alignment, htf_bias)
            return True  # ALERT - HTF context shifted

        # 3. HTF bias changed
        if htf_bias != cached.htf_bias:
            logger.info(
                f"✅ BIAS RESET: {signal_key} - Bias changed "
                f"({cached.htf_bias} → {htf_bias})"
            )
            self._cache_signal(signal_key, current_price, htf_alignment, htf_bias)
            return True  # ALERT - bias flipped

        # 4. Cooldown period expired
        if time_since_last > cooldown:
            logger.info(
                f"✅ COOLDOWN EXPIRED: {signal_key} - {time_since_last/3600:.1f}h "
                f"since last alert (cooldown: {cooldown/3600:.1f}h)"
            )
            self._cache_signal(signal_key, current_price, htf_alignment, htf_bias)
            return True  # ALERT - enough time passed

        # WITHIN COOLDOWN - Don't spam
        logger.debug(
            f"⏭️  SKIP duplicate: {signal_key} - "
            f"Within cooldown ({time_since_last/60:.0f}m / {cooldown/3600:.1f}h)"
        )
        return False  # SKIP

    def _cache_signal(
        self,
        signal_key: str,
        price: float,
        htf_alignment: float,
        htf_bias: str
    ):
        """Update cache with new signal"""
        self.signal_cache[signal_key] = CachedSignal(
            signal_key=signal_key,
            timestamp=datetime.now(timezone.utc),
            price=price,
            htf_alignment=htf_alignment,
            htf_bias=htf_bias,
            alert_sent=True
        )
        self._save_cache()

    def clear_signal(self, symbol: str, signal_name: str, direction: str):
        """Manually clear a cached signal (e.g., when trade closes)"""
        signal_key = f"{symbol}_{signal_name}_{direction}"
        if signal_key in self.signal_cache:
            del self.signal_cache[signal_key]
            self._save_cache()
            logger.info(f"🗑️  Cleared cache: {signal_key}")

    def clear_all(self):
        """Clear all cached signals"""
        self.signal_cache = {}
        self._save_cache()
        logger.info("🗑️  Cleared all signal cache")

    def _load_cache(self):
        """Load signal cache from file"""
        if not self.cache_file.exists():
            return

        try:
            with open(self.cache_file, 'r') as f:
                data = json.load(f)

            # Convert timestamps back to datetime
            for key, cached_data in data.items():
                cached_data['timestamp'] = datetime.fromisoformat(cached_data['timestamp'])
                self.signal_cache[key] = CachedSignal(**cached_data)

            logger.info(f"📂 Loaded {len(self.signal_cache)} cached signals")
        except Exception as e:
            logger.warning(f"Failed to load signal cache: {e}")

    def _save_cache(self):
        """Save signal cache to file"""
        try:
            # Convert to JSON-serializable format
            data = {}
            for key, cached in self.signal_cache.items():
                cached_dict = asdict(cached)
                cached_dict['timestamp'] = cached.timestamp.isoformat()
                data[key] = cached_dict

            with open(self.cache_file, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.warning(f"Failed to save signal cache: {e}")

    def get_stats(self) -> Dict:
        """Get cache statistics"""
        if not self.signal_cache:
            return {"total_signals": 0}

        now = datetime.now(timezone.utc)
        active_count = sum(
            1 for cached in self.signal_cache.values()
            if (now - cached.timestamp).total_seconds() < self.default_cooldown
        )

        return {
            "total_signals": len(self.signal_cache),
            "active_cooldowns": active_count,
            "expired_cooldowns": len(self.signal_cache) - active_count
        }


# Singleton instance
_deduplicator_instance = None

def get_deduplicator() -> SignalDeduplicator:
    """Get global deduplicator instance"""
    global _deduplicator_instance
    if _deduplicator_instance is None:
        _deduplicator_instance = SignalDeduplicator()
    return _deduplicator_instance
