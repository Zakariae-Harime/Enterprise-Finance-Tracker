"""
  Dependency Injection Functions

  FastAPI's Depends() system allows to :
    1. Share resources (db pool, kafka) across routes
    2. Keep route handlers clean and focused
    3. Make testing easier (can mock dependencies)
"""
from fastapi import Request
from src.domain.events_store import EventStore, OutboxRelay

def get_db_pool(request: Request):
    """
    Dependency to get the database connection pool.

    Usage:
      - Inject into route handlers with Depends(get_db_pool)
      - Provides access to the shared db pool created on startup (lifespan)
        - Stored on app.state for global access
        - Every request can access via request.app
      """
    
    return request.app.state.db_pool
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

def get_categorizer(request: Request):
    """
    Dependency to get the shared TransactionCategorizer instance.

    The categorizer is initialized once at startup (loads 170MB ONNX model).
    All requests share the same instance — no per-request model loading.
    Returns None if categorizer failed to initialize (app still works without it).
    """
    return getattr(request.app.state, "categorizer", None)
