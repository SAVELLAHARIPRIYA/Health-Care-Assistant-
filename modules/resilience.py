"""
HealthTrack AI - Resilience & Fault Tolerance Module

Implements enterprise-grade reliability patterns for LLM integration:
1. Circuit Breaker (CLOSED -> OPEN -> HALF-OPEN state machine)
2. Exponential Backoff with Full Jitter for rate-limit / transient network handling
3. Client-side Rate Limiting (Token Bucket)
4. Graceful Degradation / Fallback orchestration
"""

import enum
import logging
import random
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple, Type

logger = logging.getLogger("HealthTrackAI.Resilience")


class CircuitState(enum.Enum):
    CLOSED = "CLOSED"        # Normal operation: remote requests allowed
    OPEN = "OPEN"            # Tripped: remote requests blocked, fail fast to fallback
    HALF_OPEN = "HALF_OPEN"  # Testing: trial request allowed to verify remote recovery


class CircuitBreakerOpenException(Exception):
    """Raised when an operation is attempted while the Circuit Breaker is OPEN."""
    pass


class CircuitBreaker:
    """
    Thread-safe Circuit Breaker state machine preventing cascading failures
    when downstream LLM APIs (Gemini) experience degradation or downtime.
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout_sec: float = 30.0,
        name: str = "GeminiCircuitBreaker",
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout_sec = recovery_timeout_sec

        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._last_failure_time: float = 0.0
        self._total_trips = 0
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        with self._lock:
            if self._state == CircuitState.OPEN:
                # Check if recovery timeout has elapsed to move to HALF_OPEN
                if time.time() - self._last_failure_time >= self.recovery_timeout_sec:
                    self._state = CircuitState.HALF_OPEN
                    logger.info(f"[{self.name}] Transitioned from OPEN to HALF_OPEN (trial probing).")
            return self._state

    def record_success(self) -> None:
        """Record a successful remote invocation."""
        with self._lock:
            if self._state in (CircuitState.HALF_OPEN, CircuitState.OPEN):
                logger.info(f"[{self.name}] Trial succeeded. Resetting circuit to CLOSED.")
            self._state = CircuitState.CLOSED
            self._consecutive_failures = 0

    def record_failure(self, error: Optional[Exception] = None) -> None:
        """Record a failed remote invocation."""
        with self._lock:
            self._consecutive_failures += 1
            self._last_failure_time = time.time()
            logger.warning(
                f"[{self.name}] Failure recorded ({self._consecutive_failures}/{self.failure_threshold}): {error}"
            )

            if self._consecutive_failures >= self.failure_threshold:
                if self._state != CircuitState.OPEN:
                    self._state = CircuitState.OPEN
                    self._total_trips += 1
                    logger.error(
                        f"[{self.name}] Failure threshold breached! Circuit TRIPPED to OPEN for {self.recovery_timeout_sec}s."
                    )

    def can_execute(self) -> bool:
        """Determine whether a request should be dispatched to the remote service."""
        return self.state in (CircuitState.CLOSED, CircuitState.HALF_OPEN)

    def reset(self) -> None:
        """Manually reset circuit breaker to normal operational state."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._consecutive_failures = 0
            self._last_failure_time = 0.0

    def get_status(self) -> Dict[str, Any]:
        """Return diagnostic metrics for UI dashboards."""
        curr_state = self.state
        with self._lock:
            remaining_cooldown = 0.0
            if curr_state == CircuitState.OPEN:
                elapsed = time.time() - self._last_failure_time
                remaining_cooldown = max(0.0, self.recovery_timeout_sec - elapsed)

            return {
                "name": self.name,
                "state": curr_state.value,
                "consecutive_failures": self._consecutive_failures,
                "failure_threshold": self.failure_threshold,
                "total_trips": self._total_trips,
                "remaining_cooldown_sec": round(remaining_cooldown, 1),
            }


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 8.0,
    jitter: bool = True,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
):
    """
    Decorator implementing Exponential Backoff with Full Jitter.
    Formula: Delay = min(max_delay, base_delay * (2 ^ attempt)) * Uniform(0.5, 1.5)
    Protects downstream LLMs against thundering herds and resolves transient 429/503 errors.
    """
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt == max_retries - 1:
                        logger.error(f"[Retry Exhausted] Max retries ({max_retries}) reached: {e}")
                        raise e

                    # Exponential backoff calculation
                    delay = min(max_delay, base_delay * (2 ** attempt))
                    if jitter:
                        delay = delay * random.uniform(0.5, 1.5)

                    logger.warning(
                        f"[Retry Attempt {attempt + 1}/{max_retries}] Retrying after {delay:.2f}s due to: {e}"
                    )
                    time.sleep(delay)
            raise last_exception
        return wrapper
    return decorator


class TokenBucketRateLimiter:
    """
    Client-side token bucket rate limiter to adhere to Gemini free-tier limits (e.g. 15 RPM).
    """

    def __init__(self, max_tokens: int = 15, refill_period_sec: float = 60.0):
        self.max_tokens = max_tokens
        self.refill_period_sec = refill_period_sec
        self.tokens = float(max_tokens)
        self.last_refill = time.time()
        self._lock = threading.Lock()

    def acquire(self) -> Tuple[bool, float]:
        """
        Attempt to acquire 1 token.
        Returns:
            Tuple of (allowed: bool, wait_time_sec: float)
        """
        with self._lock:
            now = time.time()
            elapsed = now - self.last_refill
            # Refill tokens proportionally
            refill = elapsed * (self.max_tokens / self.refill_period_sec)
            self.tokens = min(float(self.max_tokens), self.tokens + refill)
            self.last_refill = now

            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return True, 0.0

            # Calculate wait time needed for next token
            needed = 1.0 - self.tokens
            wait_time = needed * (self.refill_period_sec / self.max_tokens)
            return False, wait_time
