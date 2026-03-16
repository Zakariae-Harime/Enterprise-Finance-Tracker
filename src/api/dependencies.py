"""
  Dependency Injection Functions

  FastAPI's Depends() system allows to :
    1. Share resources (db pool, kafka) across routes
    2. Keep route handlers clean and focused
    3. Make testing easier (can mock dependencies)
"""
from fastapi import Request
from src.domain.events_store import EventStore, OutboxRelay
from src.infrastructure.cache import CacheClient

def get_db_pool(request: Request):
    """
    Primary database pool — use for ALL writes.

    Inject with Depends(get_db_pool) in POST/PUT/DELETE route handlers.
    Also used by EventStore (always primary — replica lag would break
    optimistic concurrency version checks).

    Rule: if your query contains INSERT, UPDATE, or DELETE → use this.
    """
    return request.app.state.db_pool


def get_read_db_pool(request: Request):
    """
    Read replica pool — use for SELECT-only queries.

    Inject with Depends(get_read_db_pool) in GET route handlers.
    Points to the PostgreSQL streaming replica (read-only).

    In dev/CI: DATABASE_REPLICA_URL defaults to DATABASE_URL (same DB).
    In production: set DATABASE_REPLICA_URL to the actual replica host.

    Rule: if your query is a pure SELECT with no side effects → use this.

    The replica is always slightly behind the primary (replication lag, ~10ms).
    This is acceptable for list/detail reads. It is NOT acceptable for:
      - Read-after-write (user just created something and immediately reads it)
        → solved by our Redis cache (always fresh from the consumer)
      - EventStore version checks (must be linearizable)
        → EventStore always uses get_db_pool (primary)
    """
    return request.app.state.db_read_pool
def get_kafka_producer(request: Request):
    """
    Dependency to get the Kafka producer.
    Usage:
      - Inject into route handlers with Depends(get_kafka_producer)
      - Provides access to the shared Kafka producer created on startup (lifespan)
    """
    return request.app.state.kafka_producer
def get_event_store(request: Request) -> EventStore:
    """
      Create EventStore instance with shared pool.

      Why create new EventStore per request?
        - EventStore is lightweight (just holds pool reference)
        - Pool is shared (connections reused)
        - Could add request-specific context later (user, tenant)
      """
    pool = request.app.state.db_pool
    return EventStore(pool)
def get_outbox_relay(request: Request) -> OutboxRelay:
    pool=request.app.state.db_pool
    kafka_producer=request.app.state.kafka_producer
    return OutboxRelay(pool, kafka_producer)

def get_cache(request: Request) -> CacheClient:
    """Redis CacheClient — shared instance initialized at startup."""
    return request.app.state.cache

def get_categorizer(request: Request):
    """
    Dependency to get the shared TransactionCategorizer instance.

    The categorizer is initialized once at startup (loads 170MB ONNX model).
    All requests share the same instance — no per-request model loading.
    Returns None if categorizer failed to initialize (app still works without it).
    """
    return getattr(request.app.state, "categorizer", None)
