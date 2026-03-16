"""
Circuit Breaker — Fail-Fast Protection for External Service Calls

USER STORY
----------
As a Finance Tracker engineer, I want ERP calls to fail instantly when
Tripletex or SAP is down, so that a broken third-party service does not
exhaust our connection pool and take our entire API offline.

CONCEPT: THE THREE STATES
--------------------------
A circuit breaker wraps calls to an external service and tracks failures.
It has three states:

  CLOSED   → normal operation. All calls pass through. Failure counter ticks up.
             When failures >= threshold → trip to OPEN.

  OPEN     → service is considered down. No calls made. Returns CircuitOpenError
             immediately (0ms). After recovery_timeout seconds → try HALF-OPEN.

  HALF-OPEN → probe state. One test call is allowed through.
               Success → back to CLOSED (service recovered).
               Failure → back to OPEN (still broken, wait again).

CASCADING FAILURE (what this prevents)
----------------------------------------
Without a circuit breaker:
  POST /expenses/ → calls Tripletex → Tripletex down → hangs 30s (HTTP timeout)
  20 concurrent users × 30s = 600 connection-seconds blocked
  DB connection pool exhausted → GET /accounts/ fails → auth fails → system down
  One broken ERP takes down the whole Finance Tracker.

With a circuit breaker (after 5 failures, state = OPEN):
  POST /expenses/ → CircuitBreaker.call() → raises CircuitOpenError in 0ms
  Route catches it → returns 503 "ERP temporarily unavailable"
  All other endpoints (accounts, auth, budgets) keep working normally.

EXAMPLE TIMELINE
-----------------
  t=0:00  Tripletex goes down
  t=0:01  req 1 → CLOSED → call → timeout → failure_count=1
  t=0:02  req 2 → CLOSED → call → timeout → failure_count=2
  t=0:03  req 3 → CLOSED → call → timeout → failure_count=3
  t=0:04  req 4 → CLOSED → call → timeout → failure_count=4
  t=0:05  req 5 → CLOSED → call → timeout → failure_count=5 → OPEN
  t=0:06  req 6 → OPEN   → CircuitOpenError (0ms, no network call)
  t=0:07  req 7 → OPEN   → CircuitOpenError (0ms)
  ...
  t=1:05  60s elapsed → HALF-OPEN
  t=1:06  req N → HALF-OPEN → 1 test call → Tripletex recovered → SUCCESS → CLOSED
  t=1:07  req N+1 → CLOSED → normal operation resumes
"""

import asyncio
import logging
import time
from enum import Enum
from typing import Callable, Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State machine enum
# ---------------------------------------------------------------------------
class CircuitState(Enum):
    """
    The three possible states of a circuit breaker.

    We use an Enum (not plain strings) so that:
      - Invalid states are impossible (Python raises ValueError for unknown values)
      - IDE autocomplete works: CircuitState.OPEN instead of "open"
      - State comparisons are identity checks, not string comparisons
    """
    CLOSED    = "closed"     # normal — calls pass through
    OPEN      = "open"       # tripped — calls fail fast
    HALF_OPEN = "half_open"  # probing — one test call allowed


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------
class CircuitOpenError(Exception):
    """
    Raised when a call is attempted while the circuit is OPEN.

    We use a custom exception (not a generic RuntimeError) so callers can
    catch it specifically:

      try:
          result = await breaker.call(push_to_tripletex, expense)
      except CircuitOpenError:
          # ERP is known to be down — return 503 immediately
          return SyncResult(success=False, error="ERP temporarily unavailable")
      except Exception:
          # Unexpected error from the ERP call itself
          ...

    Using a generic exception would force callers to parse error messages,
    which is brittle. Specific exception types = clean control flow.
    """
    pass


# ---------------------------------------------------------------------------
# The Circuit Breaker class
# ---------------------------------------------------------------------------
class CircuitBreaker:
    """
    Wraps calls to a single external service and tracks its health.

    One instance per external service (not one per request).
    Instantiate at module level so state persists across all requests.

    Parameters
    ----------
    name               : Human-readable name for logs (e.g. "tripletex").
    failure_threshold  : How many consecutive failures before tripping OPEN.
    recovery_timeout   : Seconds to wait in OPEN before trying HALF-OPEN.
    half_open_successes: How many consecutive successes in HALF-OPEN before
                         returning to CLOSED (default 1 for simplicity).
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_successes: int = 1,
    ):
        # ── Identity ─────────────────────────────────────────────────────────
        # name is used in log messages so operators know which ERP tripped.
        # Example log: "[circuit_breaker] tripletex tripped OPEN after 5 failures"
        self.name = name

        # ── Configuration ─────────────────────────────────────────────────────
        # failure_threshold: how tolerant we are of transient errors.
        # Too low (1) = one slow response trips the breaker (false positive).
        # Too high (20) = too many slow requests before protection kicks in.
        # 5 is a common production default — tolerates brief blips.
        self.failure_threshold = failure_threshold

        # recovery_timeout: how long to wait in OPEN before probing.
        # Too short → we probe too aggressively while service is still down.
        # Too long  → we wait too long to recover after the service comes back.
        # 60s is a reasonable default for ERP APIs.
        self.recovery_timeout = recovery_timeout

        # half_open_successes: how many consecutive successes in HALF-OPEN
        # before declaring the service fully recovered.
        # 1 = fast recovery (one success and we're back to normal).
        # 2+ = more conservative (ensures one success wasn't a fluke).
        self.half_open_successes = half_open_successes

        # ── Mutable state ────────────────────────────────────────────────────
        # These change as the breaker transitions between states.

        # Current state — starts CLOSED (assume service is healthy at boot).
        self._state = CircuitState.CLOSED

        # Count of consecutive failures in CLOSED state.
        # Resets to 0 on any success or when transitioning to OPEN.
        self._failure_count = 0

        # Count of consecutive successes in HALF-OPEN state.
        # Resets to 0 when transitioning to HALF-OPEN.
        self._half_open_success_count = 0

        # Unix timestamp of the last failure that caused a state transition.
        # Used to calculate when OPEN → HALF-OPEN (now > last_failure + timeout).
        self._last_failure_time: float = 0.0

        # asyncio.Lock prevents race conditions when multiple concurrent requests
        # all try to read/write state simultaneously.
        # Without the lock: two requests could both see failure_count=4, both
        # increment it, both transition to OPEN, and both reset independently.
        # With the lock: state transitions are atomic.
        self._lock = asyncio.Lock()

    # ── Public interface ──────────────────────────────────────────────────────

    @property
    def state(self) -> CircuitState:
        """
        Current state, computed fresh on every access.

        We check the OPEN → HALF-OPEN transition here (time-based) rather than
        in a background task, because we want lazy evaluation:
          - No timer threads
          - No background asyncio tasks
          - State transition happens naturally when the next request arrives
            after recovery_timeout has elapsed

        This is called the "lazy state machine" pattern.
        """
        if (
            self._state == CircuitState.OPEN
            and time.monotonic() - self._last_failure_time >= self.recovery_timeout
        ):
            # Enough time has passed. Move to HALF-OPEN to probe the service.
            # We don't acquire the lock here (property can't be async), but the
            # actual state write happens inside call() under the lock.
            return CircuitState.HALF_OPEN
        return self._state

    async def call(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """
        Execute `func(*args, **kwargs)` through the circuit breaker.

        This is the ONLY entry point for protected calls. Never call the
        external service directly — always go through this method.

        Parameters
        ----------
        func   : The async function to call (e.g. adapter._do_push_expense).
        *args  : Positional arguments forwarded to func.
        **kwargs: Keyword arguments forwarded to func.

        Returns
        -------
        Whatever func returns on success.

        Raises
        ------
        CircuitOpenError : If state is OPEN (fail fast — no network call made).
        Exception        : Any exception raised by func itself (network error,
                           HTTP 500, timeout). The breaker records this as a
                           failure and re-raises so the caller can handle it.
        """
        async with self._lock:
            # ── Check current state ──────────────────────────────────────────
            current_state = self.state  # uses the property (checks timeout)

            if current_state == CircuitState.OPEN:
                # Service is known to be down. Don't even try. Fail instantly.
                # This is the core value proposition: 0ms response instead of
                # waiting for a 30s timeout on every request.
                logger.warning(
                    "[circuit_breaker] %s is OPEN — failing fast (no network call)",
                    self.name,
                )
                raise CircuitOpenError(
                    f"Circuit breaker '{self.name}' is OPEN. "
                    f"Service unavailable. "
                    f"Retry after {self.recovery_timeout}s."
                )

            # If we were OPEN and timeout has elapsed, the property returned
            # HALF-OPEN. Record the actual state transition now (under the lock).
            if current_state == CircuitState.HALF_OPEN and self._state == CircuitState.OPEN:
                self._state = CircuitState.HALF_OPEN
                self._half_open_success_count = 0
                logger.info(
                    "[circuit_breaker] %s → HALF-OPEN (probing after %ss)",
                    self.name,
                    self.recovery_timeout,
                )

        # ── Make the actual call (outside the lock) ──────────────────────────
        # We release the lock before calling func so other requests aren't
        # blocked while we wait for the network response (could be seconds).
        # The lock is re-acquired in _on_success / _on_failure for state writes.
        try:
            result = await func(*args, **kwargs)
            # Call succeeded — update state (may transition HALF-OPEN → CLOSED)
            await self._on_success()
            return result

        except CircuitOpenError:
            # Don't count CircuitOpenError as a failure — it's our own exception,
            # not an external service failure. Just re-raise.
            raise

        except Exception as exc:
            # The actual external call failed (timeout, HTTP 500, DNS failure).
            # Record the failure and potentially trip the breaker.
            await self._on_failure()
            # Re-raise so the caller can handle it (e.g. log it, return 503).
            raise

    # ── Private state transition methods ─────────────────────────────────────

    async def _on_success(self) -> None:
        """
        Called after a successful external call.

        CLOSED:    Reset failure_count. Service is healthy, no state change.
        HALF-OPEN: Increment success counter. If enough successes → CLOSED.
        OPEN:      Shouldn't happen (we don't allow calls in OPEN state).
        """
        async with self._lock:
            if self._state == CircuitState.CLOSED:
                # Consecutive successes clear the failure count.
                # This prevents a pattern like: 4 failures, 1 success, 4 more failures
                # from NOT tripping — the success resets the counter.
                self._failure_count = 0

            elif self._state == CircuitState.HALF_OPEN:
                self._half_open_success_count += 1
                logger.info(
                    "[circuit_breaker] %s HALF-OPEN success %d/%d",
                    self.name,
                    self._half_open_success_count,
                    self.half_open_successes,
                )
                if self._half_open_success_count >= self.half_open_successes:
                    # Service has recovered. Return to normal operation.
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._half_open_success_count = 0
                    logger.info(
                        "[circuit_breaker] %s → CLOSED (service recovered)",
                        self.name,
                    )

    async def _on_failure(self) -> None:
        """
        Called after a failed external call.

        CLOSED:    Increment failure_count. Trip to OPEN if threshold reached.
        HALF-OPEN: Service still broken. Go back to OPEN.
        OPEN:      Shouldn't reach here (calls are blocked in OPEN state).
        """
        async with self._lock:
            self._last_failure_time = time.monotonic()

            if self._state == CircuitState.CLOSED:
                self._failure_count += 1
                logger.warning(
                    "[circuit_breaker] %s failure %d/%d",
                    self.name,
                    self._failure_count,
                    self.failure_threshold,
                )
                if self._failure_count >= self.failure_threshold:
                    # Threshold reached. Trip the breaker.
                    self._state = CircuitState.OPEN
                    logger.error(
                        "[circuit_breaker] %s → OPEN after %d failures. "
                        "Blocking calls for %ss.",
                        self.name,
                        self._failure_count,
                        self.recovery_timeout,
                    )

            elif self._state == CircuitState.HALF_OPEN:
                # The probe call failed — service still broken. Back to OPEN.
                self._state = CircuitState.OPEN
                self._half_open_success_count = 0
                logger.error(
                    "[circuit_breaker] %s → OPEN (probe failed, waiting another %ss)",
                    self.name,
                    self.recovery_timeout,
                )

    def get_status(self) -> dict:
        """
        Return current breaker status for health check endpoint and monitoring.

        Called by GET /health/ready to include circuit breaker state in the
        readiness response. Operators can see at a glance which ERPs are down.

        Example output:
          {
            "state": "open",
            "failure_count": 5,
            "seconds_until_probe": 47.3
          }
        """
        current_state = self.state
        seconds_until_probe = None
        if current_state == CircuitState.OPEN:
            elapsed = time.monotonic() - self._last_failure_time
            seconds_until_probe = max(0.0, self.recovery_timeout - elapsed)

        return {
            "state": current_state.value,
            "failure_count": self._failure_count,
            "seconds_until_probe": round(seconds_until_probe, 1) if seconds_until_probe else None,
        }


# ---------------------------------------------------------------------------
# Registry — one breaker per external service, module-level singletons
# ---------------------------------------------------------------------------
# These are created once at import time. Their state persists for the lifetime
# of the process — across all requests, all workers (within one process).
#
# Why different thresholds per service?
#   tripletex: Norwegian ERP, reliable but occasional blips → threshold=5
#   sap:       Enterprise grade, failures usually mean serious problems → threshold=3
#   dynamics:  Microsoft, similar to SAP → threshold=3
#   adls:      Azure Data Lake — batch uploads, more tolerant → threshold=10
#                                faster recovery (30s) since uploads can retry
#
# In production you'd load these from environment variables so ops can tune
# without a code deploy.
BREAKERS: dict[str, CircuitBreaker] = {
    "tripletex": CircuitBreaker(
        name="tripletex",
        failure_threshold=5,
        recovery_timeout=60.0,
    ),
    "sap": CircuitBreaker(
        name="sap",
        failure_threshold=3,
        recovery_timeout=120.0,  # SAP is slow to recover, wait longer
    ),
    "dynamics": CircuitBreaker(
        name="dynamics",
        failure_threshold=3,
        recovery_timeout=120.0,
    ),
    "adls": CircuitBreaker(
        name="adls",
        failure_threshold=10,    # more tolerant — batch uploads can spike
        recovery_timeout=30.0,   # Azure recovers quickly
    ),
}
