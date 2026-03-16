"""
Prometheus Metrics Middleware

USER STORY
----------
As a DevOps engineer, I want a /metrics endpoint that exposes real-time
request counts, error rates, and latency percentiles for every API endpoint,
so that Grafana can display a live dashboard and PagerDuty can alert when
error rate exceeds 1% or P99 latency crosses 500ms.

CONCEPT: RED METHOD
-------------------
Every service should expose exactly three categories of metrics:
  R - Rate:    how many requests per second           → Counter
  E - Errors:  what fraction of requests fail         → same Counter, filtered
  D - Duration: how long requests take (P50/P95/P99)  → Histogram

These three numbers answer: "Is my service healthy right now?"

HOW PROMETHEUS WORKS
---------------------
Prometheus uses a PULL model: it sends GET /metrics to your app every 15s.
Your app doesn't push anything — it just maintains in-memory counters and
returns them when asked. This means:
  - If the app dies, Prometheus sees a scrape failure → triggers alert
  - No network overhead except during scrapes (every 15s)
  - The /metrics response is plain text (Prometheus exposition format)

EXAMPLE METRICS OUTPUT (GET /metrics):
---------------------------------------
# HELP http_requests_total Total number of HTTP requests
# TYPE http_requests_total counter
http_requests_total{endpoint="/api/v1/expenses/",method="POST",status_code="201"} 342.0
http_requests_total{endpoint="/api/v1/expenses/",method="POST",status_code="422"} 12.0
http_requests_total{endpoint="/api/v1/auth/login",method="POST",status_code="429"} 87.0

# HELP http_request_duration_seconds HTTP request latency
# TYPE http_request_duration_seconds histogram
http_request_duration_seconds_bucket{endpoint="/api/v1/expenses/",le="0.05"} 280.0
http_request_duration_seconds_bucket{endpoint="/api/v1/expenses/",le="0.1"} 335.0
http_request_duration_seconds_bucket{endpoint="/api/v1/expenses/",le="+Inf"} 342.0
http_request_duration_seconds_sum{endpoint="/api/v1/expenses/"} 18.432
http_request_duration_seconds_count{endpoint="/api/v1/expenses/"} 342.0

GRAFANA QUERIES (PromQL) YOU CAN WRITE AFTER THIS:
---------------------------------------------------
  Request rate:    rate(http_requests_total[5m])
  Error rate:      rate(http_requests_total{status_code=~"5.."}[5m])
  P99 latency:     histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m]))
  P95 latency:     histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))
"""

import time
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request, Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# ---------------------------------------------------------------------------
# Metric definitions — module-level singletons
# ---------------------------------------------------------------------------
# These are created ONCE when the module is imported, not per-request.
# prometheus_client maintains their state in memory for the lifetime of the
# process. Each has a globally unique name — duplicate names raise an error,
# which is why they must be module-level (not inside a function or class).

# ── Counter: http_requests_total ────────────────────────────────────────────
# A Counter is a cumulative number that only increases (resets on restart).
# We never set it to a value — we only call .inc() or .labels(...).inc().
#
# Arguments:
#   "http_requests_total"         → the metric name (appears in /metrics output)
#   "Total number of HTTP requests" → help text (shown as # HELP comment)
#   ["method", "endpoint", "status_code"] → label names
#
# Labels are key=value pairs that let you slice the same counter into dimensions.
# Example: instead of one counter for all requests, you get one counter per
# (method, endpoint, status_code) combination:
#   http_requests_total{method="POST", endpoint="/expenses/", status_code="201"} 342
#   http_requests_total{method="GET",  endpoint="/accounts/", status_code="200"} 891
#   http_requests_total{method="POST", endpoint="/expenses/", status_code="500"} 3
#
# In PromQL you can then ask: sum(rate(http_requests_total{status_code=~"5.."}[5m]))
# → "total error rate across all endpoints in the last 5 minutes"
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total number of HTTP requests",
    ["method", "endpoint", "status_code"],
    # label dimensions ↑
)

# ── Histogram: http_request_duration_seconds ────────────────────────────────
# A Histogram records observations (individual request durations) into
# pre-defined buckets. Each bucket is a threshold in seconds:
#   le="0.005" → how many requests took ≤ 5ms?
#   le="0.05"  → how many requests took ≤ 50ms?
#   le="0.5"   → how many requests took ≤ 500ms?
#   le="+Inf"  → how many requests total? (all of them)
#
# Buckets are CUMULATIVE — each bucket includes all observations from smaller
# buckets. So if le="0.05" = 280 and le="0.1" = 335, it means:
#   280 requests took ≤ 50ms
#   335 - 280 = 55 requests took between 50ms and 100ms
#
# From buckets, PromQL can calculate any percentile:
#   histogram_quantile(0.99, ...) → P99: 99% of requests completed within X ms
#   histogram_quantile(0.95, ...) → P95: 95% of requests completed within X ms
#
# WHY NOT USE SUMMARY INSTEAD OF HISTOGRAM?
# Summary calculates quantiles in the app (client-side), which can't be
# aggregated across multiple instances. Histograms are calculated server-side
# by Prometheus, so you can aggregate across 3 app replicas correctly.
#
# Bucket design strategy:
#   Start with very fast (5ms, 10ms) to catch cache hits vs misses.
#   Include your SLO thresholds (100ms = fast, 500ms = acceptable, 1s = slow).
#   End with 5s and 10s to catch very slow queries (event replay, cold paths).
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
    # We don't include status_code as a label here because histograms already
    # have high cardinality from their buckets. Adding status_code would
    # multiply bucket count by number of status codes → memory explosion.
    buckets=[
        0.005,  #   5ms — Redis cache hit, trivial reads
        0.01,   #  10ms — fast DB query with index
        0.025,  #  25ms — typical indexed query + serialization
        0.05,   #  50ms — good response time
        0.1,    # 100ms — acceptable (SLO threshold for simple reads)
        0.25,   # 250ms — getting slow
        0.5,    # 500ms — SLO threshold for writes (acceptable max)
        1.0,    #   1s  — slow — event replay or complex joins
        2.5,    # 2.5s  — very slow — investigate
        5.0,    #   5s  — critical — likely a full table scan
        10.0,   #  10s  — unacceptable — probable connection pool exhaustion
    ],
)


class PrometheusMetricsMiddleware(BaseHTTPMiddleware):
    """
    Middleware that records request count and latency for every HTTP request.

    Sits in the middleware stack between RequestLoggingMiddleware (outer) and
    SlidingWindowRateLimiter (inner). Records metrics for ALL requests
    including 429s from the rate limiter and 404s from unknown routes.

    WHY MIDDLEWARE INSTEAD OF A DECORATOR ON EACH ROUTE?
    -----------------------------------------------------
    A per-route decorator would require adding @track_metrics to every single
    endpoint function. Middleware instruments everything automatically —
    including routes added by third-party libraries, and error responses
    generated by FastAPI itself (422 validation errors, 404 not found).
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        """
        Record metrics for every request.

        Parameters
        ----------
        request   : Incoming HTTP request.
        call_next : Coroutine — forwards request to inner layers/route handler.
        """

        # ── 1. Normalize the endpoint label ─────────────────────────────────
        # request.url.path gives the exact path including dynamic segments:
        #   "/api/v1/accounts/f47ac10b-58cc-4372-a567-0e02b2c3d479"
        #
        # If we used the raw path as a label, every unique UUID would create
        # a separate time series in Prometheus. With 10,000 accounts, you'd
        # have 10,000 time series for one endpoint → HIGH CARDINALITY PROBLEM.
        # Prometheus would run out of memory and crash.
        #
        # Solution: use the route template (with path parameters) instead:
        #   "/api/v1/accounts/{account_id}"
        #
        # request.scope["route"] is set by FastAPI after routing. At middleware
        # level, routing hasn't happened yet for some requests (e.g. 404s), so
        # we fall back to the raw path for those.
        route = request.scope.get("route")
        endpoint = route.path if route else request.url.path
        # Examples:
        #   route.path = "/api/v1/accounts/{account_id}"   → good label
        #   raw path   = "/api/v1/nonexistent"             → fallback for 404s

        # ── 2. Record the HTTP method ────────────────────────────────────────
        # request.method is "GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"
        # We store it as a label so we can split reads (GET) from writes (POST).
        # Example PromQL: rate(http_requests_total{method="POST"}[5m]) → write rate
        method = request.method

        # ── 3. Start the latency timer ───────────────────────────────────────
        # time.perf_counter() is a monotonic high-resolution clock.
        # We record start BEFORE call_next so we measure the full round-trip
        # including inner middleware and the route handler.
        start_time = time.perf_counter()

        # ── 4. Execute the request ───────────────────────────────────────────
        # Everything below this line runs AFTER the route handler returns.
        response = await call_next(request)

        # ── 5. Calculate duration ────────────────────────────────────────────
        # (now - start) in seconds. Histogram expects seconds (not ms).
        # Example: 0.047 = 47ms
        duration = time.perf_counter() - start_time

        # ── 6. Increment the request counter ────────────────────────────────
        # .labels(method=..., endpoint=..., status_code=...) selects the specific
        # time series for this combination. .inc() adds 1 to it.
        #
        # str(response.status_code) converts 201 → "201" (labels are strings).
        #
        # After 1000 requests to POST /expenses/ returning 201:
        #   http_requests_total{method="POST", endpoint="/api/v1/expenses/", status_code="201"} 1000
        REQUEST_COUNT.labels(
            method=method,
            endpoint=endpoint,
            status_code=str(response.status_code),
        ).inc()

        # ── 7. Record the latency observation ────────────────────────────────
        # .observe(duration) places this request's duration into the appropriate
        # histogram buckets. Prometheus figures out which buckets it falls into.
        #
        # Example: duration=0.047 (47ms)
        # Buckets updated: le="0.05" ✓, le="0.1" ✓, le="0.25" ✓ ... le="+Inf" ✓
        # Buckets NOT updated: le="0.005", le="0.01", le="0.025" (47ms > 25ms)
        REQUEST_LATENCY.labels(
            method=method,
            endpoint=endpoint,
        ).observe(duration)

        return response


# ---------------------------------------------------------------------------
# /metrics endpoint handler
# ---------------------------------------------------------------------------
# This is NOT a middleware — it's a plain function that the FastAPI route
# calls when someone hits GET /metrics.
#
# generate_latest() serializes ALL registered metrics to the Prometheus
# text exposition format — a plain-text format that Prometheus scrapes.
#
# CONTENT_TYPE_LATEST is "text/plain; version=0.0.4; charset=utf-8"
# Prometheus expects this exact Content-Type to parse the response correctly.
#
# Why not return JSON? Prometheus has its own text format optimized for
# fast parsing of millions of metric lines. JSON would be ~3x larger.
def metrics_endpoint():
    """
    Return all Prometheus metrics in the standard text exposition format.

    Prometheus scrapes this endpoint every 15 seconds (configurable in
    prometheus.yml). Grafana queries Prometheus, not this endpoint directly.

    Security note: in production, restrict /metrics to internal network only
    (nginx allow/deny, or Kubernetes NetworkPolicy) — metric names can reveal
    internal architecture details you don't want public.
    """
    return Response(
        content=generate_latest(),        # serialized metrics text
        media_type=CONTENT_TYPE_LATEST,   # "text/plain; version=0.0.4"
    )
