"""
Sliding Window Rate Limiter Middleware

USER STORY
----------
As a Finance Tracker API operator, I want every authenticated user to be
limited to a configurable number of requests per minute, so that a
compromised token or a runaway client cannot take the API offline for
other tenants.

CONCEPT: SLIDING WINDOW LOG
----------------------------
We store one Redis Sorted Set (ZSET) per user per endpoint group.
  Key:   ratelimit:{user_id}:{endpoint_group}
  Score: Unix timestamp in milliseconds
  Value: same timestamp (must be unique, so we append a counter suffix)

On every request:
  1. Remove all scores older than (now - window_ms)  → ZREMRANGEBYSCORE
  2. Count remaining scores in the set               → ZCARD
  3. If count >= limit → reject with HTTP 429
  4. Otherwise, add current timestamp as a new score → ZADD
  5. Set a TTL so the key self-destructs if idle     → EXPIRE

All four Redis commands run in a single PIPELINE — atomic from the client's
perspective and only one network round-trip.

EXAMPLE
-------
Limit = 5 requests per 10 seconds (simplified for illustration).
Timeline for user "alice":

  t=0s   → req 1 → ZSET: [0]           count=1 ✓ allowed
  t=2s   → req 2 → ZSET: [0, 2000]     count=2 ✓ allowed
  t=4s   → req 3 → ZSET: [0,2000,4000] count=3 ✓ allowed
  t=6s   → req 4 → count=4 ✓ allowed
  t=8s   → req 5 → count=5 ✓ allowed
  t=9s   → req 6 → count=6 ✗ 429 (window still [0..9s], all 5 entries present)
  t=11s  → req 7 → ZREMRANGEBYSCORE removes t=0 (older than t=11-10=1s)
               ZSET: [2000,4000,6000,8000] count=4 ✓ allowed again

Compare this to FIXED WINDOW:
  A fixed-window user could send 5 req at t=59s and 5 more at t=61s — 10 req
  in 2 seconds. Sliding window prevents this because both batches fall inside
  the same 10-second lookback window.

RATE LIMIT GROUPS
-----------------
Different endpoints have different limits because they have different costs:
  - Auth endpoints (login, refresh): 10/min  — brute-force protection
  - Default (most GET/POST endpoints): 100/min
"""

import time
import uuid as _uuid
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

# ---------------------------------------------------------------------------
# Rate limit configuration table
# ---------------------------------------------------------------------------
# Maps a URL *prefix* to a (limit, window_seconds) pair.
# The FIRST matching prefix wins (checked top-to-bottom, most specific first).
# "default" is the catch-all applied when no prefix matches.
#
# Why different limits per endpoint group?
#   - /auth/login is expensive (bcrypt hash comparison) and a brute-force target.
#     10 attempts/min is plenty for a human; a bot would need 6 hours to try 600.
#   - Most read endpoints are cheap; 100/min supports normal dashboard usage.
#
RATE_LIMITS: dict[str, dict] = {
    "/api/v1/auth/login":   {"limit": 10,  "window_seconds": 60},  # brute-force guard
    "/api/v1/auth/refresh": {"limit": 10,  "window_seconds": 60},  # same: token refresh
    "default":              {"limit": 100, "window_seconds": 60},   # everything else
}


class SlidingWindowRateLimiter(BaseHTTPMiddleware):
    """
    Starlette middleware that enforces per-user sliding-window rate limits.

    HOW MIDDLEWARE WORKS IN FASTAPI / STARLETTE
    -------------------------------------------
    Middleware wraps every incoming HTTP request before it reaches any route.
    FastAPI uses Starlette's middleware stack internally. We inherit from
    BaseHTTPMiddleware, which requires us to implement `dispatch(request, call_next)`.

    `call_next` is a coroutine that passes the request down to the next layer
    (either another middleware or the actual route handler). By awaiting it we
    get the response back and can add headers to it before returning.

    The middleware stack runs like an onion:
      Request  →  [RateLimiter.dispatch]  →  [Route Handler]
      Response ←  [RateLimiter.dispatch]  ←  [Route Handler]
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        """
        Called on every HTTP request. Either blocks with 429 or passes through.

        Parameters
        ----------
        request  : The incoming HTTP request (method, headers, URL, body, app state).
        call_next: Coroutine — call this to forward the request to the route handler.
        """

        # ── 1. Exempt health checks ──────────────────────────────────────────
        # Health probes from Docker/Kubernetes fire every 5 seconds per instance.
        # Rate-limiting them would cause your own infrastructure to get 429s,
        # which would cause Kubernetes to think the pod is unhealthy and restart it.
        if request.url.path in ("/health", "/health/live", "/health/ready"):
            return await call_next(request)

        # ── 2. Identify the requester ────────────────────────────────────────
        # We prefer to rate-limit by user_id (from the JWT) rather than by IP
        # because:
        #   - Multiple users can share an IP (corporate NAT, office network)
        #   - A single attacker can rotate IPs but not forge a valid JWT
        # For unauthenticated endpoints (login), we fall back to the client IP.
        user_key = self._get_user_key(request)

        # ── 3. Look up the rate limit config for this endpoint ───────────────
        # We match the request path against our RATE_LIMITS table.
        # The first prefix that matches wins; "default" is always last.
        config = self._get_limit_config(request.url.path)
        limit          = config["limit"]           # e.g. 100 requests
        window_seconds = config["window_seconds"]  # e.g. 60 seconds

        # ── 4. Build the Redis key ────────────────────────────────────────────
        # Format: ratelimit:{user_id_or_ip}:{path}
        # We include the path so each endpoint group has its own counter.
        # Example: "ratelimit:user-uuid-123:/api/v1/expenses/"
        # Why not just one key per user? Because a user could legitimately send
        # 100 GET requests to /accounts while also hitting /auth/login 10 times.
        # They're independent budgets.
        redis_key = f"ratelimit:{user_key}:{request.url.path}"

        # ── 5. Current timestamp in milliseconds ─────────────────────────────
        # We use milliseconds (not seconds) as the ZSET score for sub-second
        # precision. This prevents two requests in the same second from getting
        # the same score (which would cause ZADD to overwrite instead of add).
        now_ms = int(time.time() * 1000)

        # ── 6. The start of our sliding window ───────────────────────────────
        # Any timestamp older than this is outside the window and should be
        # removed. E.g. if window=60s: window_start = now - 60000ms.
        window_start_ms = now_ms - (window_seconds * 1000)

        # ── 7. Redis pipeline: four commands in one round-trip ────────────────
        # A pipeline batches multiple Redis commands and sends them together.
        # This is NOT the same as a transaction (MULTI/EXEC), but for rate
        # limiting it's sufficient because:
        #   - We're doing a read-then-write, and slight races are acceptable
        #     (worst case: two requests from the same user both slip through
        #      right at the limit boundary — a tiny, acceptable imprecision)
        #   - One network round-trip instead of four = 3× faster
        redis = request.app.state.cache._redis

        # Create a pipeline context manager. All commands inside are queued
        # and sent to Redis in one batch when the `async with` block exits.
        async with redis.pipeline(transaction=False) as pipe:

            # Command 1: ZREMRANGEBYSCORE
            # Remove all entries in the ZSET whose score (timestamp) is
            # less than window_start_ms. These are "old" requests that no
            # longer count toward the current window.
            # Example: if now=10:00:60.000 and window=60s, remove everything
            # before 10:00:00.000. A request at 09:59:55 is evicted.
            pipe.zremrangebyscore(redis_key, 0, window_start_ms)

            # Command 2: ZCARD
            # Count how many entries remain in the ZSET *after* the cleanup.
            # This is the number of requests this user has made in the last
            # `window_seconds`. If this number >= limit, we reject.
            pipe.zcard(redis_key)

            # Command 3: ZADD
            # Add the current request's timestamp to the ZSET.
            # Key insight: we add BEFORE checking the count (above), so this
            # request itself is counted. The count from ZCARD reflects the
            # state BEFORE this request, which is what we want:
            #   count=99 → this is the 100th request → allowed (count < 100? no, 99 < 100 ✓)
            #   count=100 → this would be the 101st → blocked
            #
            # The value in the ZSET must be unique. Using the timestamp alone
            # would cause collisions if two requests arrive in the same millisecond
            # (ZADD NX would silently ignore duplicates). We append a UUID suffix.
            pipe.zadd(redis_key, {f"{now_ms}-{_uuid.uuid4()}": now_ms})

            # Command 4: EXPIRE
            # Set a Time-To-Live on the key so it auto-deletes when idle.
            # Without this, every user who ever made a request would accumulate
            # a permanent key in Redis. We add +1 to the window so the key lives
            # slightly longer than the window (prevents edge-case where a request
            # arrives at exactly TTL boundary and the key is already gone).
            pipe.expire(redis_key, window_seconds + 1)

            # Execute all four commands. Returns a list of results in order:
            # [zremrangebyscore_result, zcard_count, zadd_result, expire_result]
            results = await pipe.execute()

        # results[1] is the ZCARD output — the count of requests in the window
        # (before this request was added, so this request is "request #count+1")
        request_count = results[1]

        # ── 8. Build RFC 6585 rate limit response headers ────────────────────
        # These headers are a standard that clients (browsers, SDKs) understand.
        # They allow clients to self-throttle instead of hammering until 429.
        #
        # X-RateLimit-Limit     → the maximum allowed per window
        # X-RateLimit-Remaining → how many requests the client has left this window
        # X-RateLimit-Reset     → Unix timestamp (seconds) when the window resets
        #
        # "Remaining" uses max(0, ...) to avoid going negative (if count already
        # exceeded limit due to a race, remaining should show 0, not -1).
        rate_limit_headers = {
            "X-RateLimit-Limit":     str(limit),
            "X-RateLimit-Remaining": str(max(0, limit - request_count - 1)),
            "X-RateLimit-Reset":     str(int(time.time()) + window_seconds),
        }

        # ── 9. Block if over limit ────────────────────────────────────────────
        # request_count is the count BEFORE this request. So if request_count
        # equals limit (e.g. 100), this is the 101st request → blocked.
        if request_count >= limit:
            return JSONResponse(
                status_code=429,  # 429 Too Many Requests (RFC 6585)
                content={
                    "detail": (
                        f"Rate limit exceeded. Maximum {limit} requests per "
                        f"{window_seconds} seconds allowed."
                    )
                },
                headers={
                    **rate_limit_headers,
                    # Retry-After tells the client how long to wait (seconds).
                    # A well-behaved SDK will read this and back off automatically.
                    "Retry-After": str(window_seconds),
                },
            )

        # ── 10. Forward to the actual route handler ───────────────────────────
        # The request is within the rate limit. Pass it to the next layer.
        # `call_next` returns the response from the route handler.
        response = await call_next(request)

        # ── 11. Attach rate limit headers to the real response ────────────────
        # Even for allowed requests, we attach the headers so clients can see
        # how much budget they have left. This enables proactive throttling.
        for header_name, header_value in rate_limit_headers.items():
            response.headers[header_name] = header_value

        return response

    # ── Helper: identify the requester ──────────────────────────────────────

    def _get_user_key(self, request: Request) -> str:
        """
        Extract a stable identifier for the requester.

        Strategy:
          1. Try to read the user_id from request.state (set by auth middleware
             or by the route's Depends(get_current_user) — not available yet at
             middleware level, so we read the Authorization header directly).
          2. Fall back to the client IP address for unauthenticated endpoints.

        Why not decode the JWT here?
          Decoding requires the secret key and adds latency. For rate limiting
          purposes, we just need a stable string — even the raw token value
          works (same token = same user). We truncate to 64 chars so long tokens
          don't become huge Redis key names.

        Example output:
          "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI..."[:64]  → authenticated
          "192.168.1.42"                                  → unauthenticated
        """
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            # Use the first 64 characters of the token as the key.
            # Two users with different tokens → different keys ✓
            # Same user with same token across requests → same key ✓
            return auth_header[7:71]  # skip "Bearer ", take 64 chars

        # No token: fall back to the real client IP.
        # request.client is a (host, port) tuple set by Starlette from the
        # TCP connection. Behind a load balancer you'd read X-Forwarded-For.
        return request.client.host if request.client else "unknown"

    # ── Helper: match URL path to a limit config ─────────────────────────────

    def _get_limit_config(self, path: str) -> dict:
        """
        Return the rate limit config for the given URL path.

        Walks RATE_LIMITS in insertion order (Python 3.7+ dicts are ordered).
        Returns the first entry whose key is a prefix of `path`.
        Falls back to "default" if nothing matches.

        Example:
          path="/api/v1/auth/login"  → matches "/api/v1/auth/login" → limit=10
          path="/api/v1/expenses/"   → no match → "default"          → limit=100
        """
        for prefix, config in RATE_LIMITS.items():
            if prefix == "default":
                continue  # handled at the end
            if path.startswith(prefix):
                return config
        return RATE_LIMITS["default"]
