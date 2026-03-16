"""
  FastAPI Application Entry Point
    1. Creates the FastAPI app instance
    2. Sets up database connection pool on startup
    3. Registers API routes
    4. Handles graceful shutdown
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
import asyncpg
import os
from aiokafka import AIOKafkaProducer
from fastapi.middleware.cors import CORSMiddleware
from src.api.middleware.rate_limiter import SlidingWindowRateLimiter
from src.api.middleware.logging import RequestLoggingMiddleware
from src.api.middleware.metrics import PrometheusMetricsMiddleware, metrics_endpoint
from src.api.routes.auth import router as auth_router
from src.api.routes.accounts import router as accounts_router
from src.api.routes.transactions import router as transactions_router
from src.api.routes.budgets import router as budgets_router
from src.api.routes.organizations import router as organizations_router
from src.api.routes.expenses import router as expenses_router
from src.api.routes.integrations import router as integrations_router
from src.api.routes.debug import router as debug_router
from src.ml.categorizer import TransactionCategorizer
from src.infrastructure.cache import CacheClient

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages application lifecycle.

    Startup: Create database pool, Kafka producer, ML categorizer
    Shutdown: Close connections gracefully

    lifespan vs @app.on_event?
      - @app.on_event is deprecated in FastAPI
      - lifespan is the modern, recommended approach
    """
    # STARTUP
    # ── Primary pool — all writes, EventStore, outbox relay ──────────────────
    # Every INSERT/UPDATE/DELETE goes here. EventStore ALWAYS uses this pool
    # because event ordering requires the primary (replica lag would break
    # optimistic concurrency version checks).
    primary_url = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/finance_tracker")
    app.state.db_pool = await asyncpg.create_pool(
        dsn=primary_url,
        min_size=5,
        max_size=20,
    )

    # ── Replica pool — all GET route queries ─────────────────────────────────
    # DATABASE_REPLICA_URL defaults to primary_url in dev/CI (same DB, two pools).
    # In production: set DATABASE_REPLICA_URL to the PostgreSQL streaming replica.
    # The replica is read-only — asyncpg will raise if a write is attempted.
    #
    # Why more connections on the replica (max_size=30)?
    # Reads are the majority of traffic (dashboard loads, report queries).
    # The replica can handle more concurrent reads than the primary can handle
    # writes, because reads don't acquire row locks or write WAL entries.
    replica_url = os.environ.get("DATABASE_REPLICA_URL", primary_url)
    app.state.db_read_pool = await asyncpg.create_pool(
        dsn=replica_url,
        min_size=5,
        max_size=30,  # more connections: reads dominate traffic
    )
    # Create Kafka producer
    app.state.kafka_producer = AIOKafkaProducer(
        bootstrap_servers='localhost:9092'
    )
    await app.state.kafka_producer.start()

    # Redis cache (account projections, integrations, member lookups)
    app.state.cache = CacheClient(os.environ.get("REDIS_URL", "redis://localhost:6379"))

    # Load ML categorizer (3-layer: rules → TF-IDF → NB-BERT ONNX)
    # Loads once at startup — 170MB ONNX model stays in memory for all requests
    try:
        app.state.categorizer = TransactionCategorizer()
        print("[startup] ML categorizer ready (rules + TF-IDF + NB-BERT)")
    except Exception as e:
        app.state.categorizer = None
        print(f"[startup] ML categorizer unavailable: {e} — transactions will need manual categorization")

    print("Primary pool, replica pool, Kafka producer, and Redis cache initialized")
    yield
    # SHUTDOWN — close both pools independently so one failure doesn't skip the other
    await app.state.db_pool.close()
    await app.state.db_read_pool.close()
    await app.state.kafka_producer.stop()
    await app.state.cache.close()
    print("Primary pool, replica pool, Kafka producer, and Redis cache closed gracefully")

app = FastAPI(
    title="Entreprise Finance Tracker API",
    description="API for managing and tracking financial data within an enterprise.",
    version="1.0.0",
    lifespan=lifespan  # Connect our startup/shutdown handler
    )
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], # Allow frontend origin for development
    allow_methods=["*"], # Allow all HTTP methods
     allow_headers=["*"], # Allow all headers
     allow_credentials=True, # Allow cookies and auth headers
    )
# Rate limiter runs after CORS but before any route handler.
# Middleware stack is LIFO: the LAST middleware added is the FIRST to run.
# So SlidingWindowRateLimiter (added last) runs first on every request.
app.add_middleware(SlidingWindowRateLimiter)
# Metrics middleware — sits between logging (outer) and rate limiter (inner).
# Records request count + latency for every request including 429s and 500s.
# Added before RequestLoggingMiddleware so logging stays the true outermost layer.
app.add_middleware(PrometheusMetricsMiddleware)
# Logging middleware runs outermost (added last = runs first in Starlette's LIFO stack).
# This means it times the ENTIRE request including rate limiter overhead.
# Every request gets a JSON log line: {request_id, method, path, status_code, duration_ms}
app.add_middleware(RequestLoggingMiddleware)
#Register API routes
app.include_router(auth_router, prefix="/api/v1")
app.include_router(accounts_router, prefix="/api/v1")
app.include_router(transactions_router, prefix="/api/v1")
app.include_router(budgets_router, prefix="/api/v1")
app.include_router(organizations_router, prefix="/api/v1")
app.include_router(expenses_router, prefix="/api/v1")
app.include_router(integrations_router, prefix="/api/v1")
# Debug router — shard info and ring stats for interviews and local development.
# In production: only mount if DEBUG env var is set.
if os.environ.get("DEBUG", "true").lower() == "true":
    app.include_router(debug_router, prefix="/api/v1")

@app.get("/metrics", include_in_schema=False)
async def get_metrics():
    """
    Prometheus metrics endpoint — scraped every 15s by the Prometheus server.

    include_in_schema=False hides this from the Swagger UI docs.
    In production, restrict access to internal network via nginx allow/deny.

    Returns plain text in Prometheus exposition format:
      # HELP http_requests_total Total number of HTTP requests
      # TYPE http_requests_total counter
      http_requests_total{endpoint="...",method="GET",status_code="200"} 142.0
      ...
    """
    return metrics_endpoint()


@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify API is running.

    Health check endpoint for Docker/Kubernetes.

    Used by:
      - Docker HEALTHCHECK
      - Kubernetes liveness probe
      - Load balancers
    """
    return {"status": "ok"}
