"""
Rate limiting utilities for API calls.
"""

import asyncio
import time
from collections import deque
from typing import Optional


class RateLimiter:
    """
    Token bucket rate limiter for API calls.
    """

    def __init__(
        self,
        max_requests: int,
        time_window: float = 1.0,
        burst_size: Optional[int] = None
    ):
        """
        Initialize rate limiter.

        Args:
            max_requests: Maximum number of requests allowed in time window
            time_window: Time window in seconds
            burst_size: Maximum burst size (defaults to max_requests)
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.burst_size = burst_size or max_requests
        self.tokens = self.burst_size
        self.last_update = time.time()
        self.request_times = deque()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1):
        """
        Acquire tokens, waiting if necessary.

        Args:
            tokens: Number of tokens to acquire
        """
        async with self._lock:
            while True:
                now = time.time()
                self._refill_tokens(now)

                if self.tokens >= tokens:
                    self.tokens -= tokens
                    self.request_times.append(now)
                    break

                # Calculate wait time
                if self.request_times:
                    oldest = self.request_times[0]
                    wait_time = self.time_window - (now - oldest)
                    if wait_time > 0:
                        await asyncio.sleep(wait_time)
                else:
                    await asyncio.sleep(0.1)

    def _refill_tokens(self, now: float):
        """Refill tokens based on elapsed time."""
        elapsed = now - self.last_update
        new_tokens = elapsed * (self.max_requests / self.time_window)
        self.tokens = min(self.burst_size, self.tokens + new_tokens)
        self.last_update = now

        # Remove old request times
        cutoff = now - self.time_window
        while self.request_times and self.request_times[0] < cutoff:
            self.request_times.popleft()


class ExponentialBackoff:
    """
    Exponential backoff for retry logic.
    """

    def __init__(
        self,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        multiplier: float = 2.0,
        jitter: bool = True
    ):
        """
        Initialize exponential backoff.

        Args:
            initial_delay: Initial delay in seconds
            max_delay: Maximum delay in seconds
            multiplier: Backoff multiplier
            jitter: Whether to add random jitter
        """
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.multiplier = multiplier
        self.jitter = jitter
        self.attempts = 0

    async def wait(self):
        """Wait for the current backoff duration."""
        import random

        delay = min(
            self.initial_delay * (self.multiplier ** self.attempts),
            self.max_delay
        )

        if self.jitter:
            delay *= (0.5 + random.random())

        await asyncio.sleep(delay)
        self.attempts += 1

    def reset(self):
        """Reset the backoff counter."""
        self.attempts = 0
