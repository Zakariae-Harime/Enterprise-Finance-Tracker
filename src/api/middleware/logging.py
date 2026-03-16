"""
Structured Logging + Request ID Middleware

USER STORY
----------
As an on-call engineer, I want every log line for a single HTTP request to
share the same unique ID, so that when a bug is reported I can grep for
one UUID and see the complete request lifecycle across all services.

CONCEPT
-------
Every HTTP request gets a UUID assigned at the very first middleware layer.
This UUID is:
  1. Stored on `request.state.request_id` so route handlers can read it
  2. Echoed back to the client in the `X-Request-ID` response header
     (so the client can include it in a bug report: "request ID: abc-123")
  3. Included in every log line emitted by this middleware (method, path,
     status code, duration)

In production you'd also inject request_id into your database query logs,
Kafka event metadata, and any downstream HTTP calls (via headers), giving
you a single ID that traces one user action across every system.

EXAMPLE
-------
Request comes in with no X-Request-ID header:
  → middleware generates: "f47ac10b-58cc-4372-a567-0e02b2c3d479"
  → stores on request.state.request_id
  → after route finishes, logs:
       {"request_id": "f47ac10b...", "method": "POST", "path": "/api/v1/expenses/",
        "status_code": 201, "duration_ms": 42.7}
  → response contains header: X-Request-ID: f47ac10b-58cc-4372-a567-0e02b2c3d479

Client gets a 500 error?
  They send you the X-Request-ID from their browser DevTools.
  You grep logs for that ID → instant full timeline.

Request comes in with X-Request-ID already set (e.g. nginx generated it):
  → middleware reuses the existing ID instead of generating a new one
  → allows tracing across nginx → FastAPI boundary with the same ID
"""

import json
import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request, Response

# ---------------------------------------------------------------------------
# Logger setup
# ---------------------------------------------------------------------------
# logging.getLogger(__name__) creates a logger named after this module:
# "src.api.middleware.logging"
# This allows fine-grained control in logging config:
#   - You can set this logger to DEBUG while keeping other loggers at WARNING
#   - Log aggregators (Datadog, Loki) can filter by logger name
logger = logging.getLogger("finance_tracker.http")
# If no handlers are configured (common in tests), add a basic one to stdout.
# In production, configure handlers via logging.config.dictConfig() at startup.
if not logger.handlers:
    _handler = logging.StreamHandler()
    # %(message)s is the only format field we need — our message is already JSON.
    # Adding a prefix like "%(asctime)s" would double-encode the timestamp.
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware that assigns a request ID to every request and logs a
    structured JSON line after the response is sent.

    WHY LOG AFTER THE RESPONSE?
    ----------------------------
    We need to know the status_code and duration_ms, which are only available
    after the route handler finishes. So we:
      1. Assign the request_id and record start_time BEFORE calling call_next
      2. Await call_next (route handler runs here)
      3. Log AFTER we have the response object with status_code
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        """
        Wraps every request with ID assignment, timing, and structured logging.

        Parameters
        ----------
        request   : Incoming HTTP request (headers, method, URL, app state).
        call_next : Coroutine — call to forward to the next middleware/route.
        """

        # ── 1. Assign or inherit the request ID ─────────────────────────────
        # Check if the client (or nginx upstream) already sent an X-Request-ID.
        # This is important for distributed tracing: nginx can generate IDs
        # and forward them, so the same ID appears in nginx logs AND app logs.
        # If no ID was provided, we generate a fresh UUID4 (128 bits of entropy
        # → effectively zero collision probability even at millions of req/sec).
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())

        # ── 2. Store on request.state ────────────────────────────────────────
        # request.state is a SimpleNamespace attached to the request object.
        # Anything stored here is accessible anywhere the `request` object is
        # available — route handlers, other middleware, background tasks.
        #
        # Example usage in a route handler:
        #   async def create_expense(request: Request, ...):
        #       rid = request.state.request_id
        #       logger.info({"request_id": rid, "msg": "validating expense"})
        request.state.request_id = request_id

        # ── 3. Start the timer ───────────────────────────────────────────────
        # time.perf_counter() is the highest-resolution clock available in Python.
        # It's monotonic (never goes backward), unlike time.time() which can jump
        # if the system clock is adjusted (NTP sync, DST, etc.).
        # We use it for duration measurement only — not for timestamps.
        start_time = time.perf_counter()

        # ── 4. Forward to the route handler ─────────────────────────────────
        # Everything above runs BEFORE the route handler.
        # call_next suspends this coroutine, runs the route, then resumes here.
        # The `response` object now has status_code, headers, and body.
        response = await call_next(request)

        # ── 5. Calculate duration ────────────────────────────────────────────
        # (perf_counter() - start_time) gives elapsed seconds as a float.
        # Multiply by 1000 → milliseconds. Round to 2 decimal places.
        # Example: 0.047231... → 47.23ms
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # ── 6. Emit the structured log line ──────────────────────────────────
        # We build a dict and serialize it to JSON.
        # Why not use a logging formatter? Because we want the entire log line
        # to be a single JSON object (not "2026-03-15 INFO {json}") so log
        # aggregators parse it as one record, not nested strings.
        #
        # Fields explained:
        #   request_id   → the UUID — primary filter key in any log query
        #   method       → GET/POST/PUT/DELETE — useful for filtering writes
        #   path         → URL path without query string
        #   status_code  → HTTP status — 200/201/422/429/500
        #   duration_ms  → how long the route took — SLO monitoring
        #
        # In production you'd also add:
        #   user_id, org_id  → from request.state after auth middleware runs
        #   query_params     → for debugging filter issues
        #   body_size        → to catch large payloads
        log_entry = {
            "request_id":  request_id,
            "method":      request.method,
            "path":        request.url.path,
            "status_code": response.status_code,
            "duration_ms": duration_ms,
        }
        # json.dumps converts the dict to a JSON string.
        # logger.info writes it to stdout (or whatever handler is configured).
        logger.info(json.dumps(log_entry))

        # ── 7. Echo the request ID back to the client ────────────────────────
        # Adding the ID to the response header lets:
        #   - Browsers show it in DevTools (Network tab → Response Headers)
        #   - API clients include it in bug reports ("request ID: abc-123")
        #   - Frontend apps display it in error dialogs for support tickets
        response.headers["X-Request-ID"] = request_id

        return response
