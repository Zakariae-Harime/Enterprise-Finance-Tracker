"""
Kafka fixtures for integration tests.

Placed in tests/integration/ so they only activate for integration tests.
The global tests/conftest.py provides db_pool — we build on top of that here.

Requires: docker-compose up -d kafka zookeeper
"""
import pytest_asyncio
from uuid import uuid4
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer

KAFKA_BOOTSTRAP = "localhost:9092"


@pytest_asyncio.fixture(scope="session")
async def kafka_producer():
    """
    Session-scoped AIOKafkaProducer.
    Created once for the entire test session, shared across all Kafka tests.

    Why session scope?
      Starting/stopping a Kafka producer takes ~200ms.
      Session scope avoids that overhead per test.
    """
    producer = AIOKafkaProducer(bootstrap_servers=KAFKA_BOOTSTRAP)
    await producer.start()
    yield producer
    await producer.stop()


@pytest_asyncio.fixture
async def kafka_consumer_factory():
    """
    Function-scoped factory that creates AIOKafkaConsumer instances on demand.

    Why a factory (not a direct fixture)?
      Different tests subscribe to different topics.
      A factory lets each test specify which topic to read from.

    Why unique group_id per consumer?
      Kafka tracks which messages each consumer group has read (committed offset).
      If tests share a group_id, the second test would skip messages the first
      test already consumed. A unique group_id ensures each consumer starts
      from offset 0 (earliest) — reads ALL messages on the topic.

    consumer_timeout_ms=5000:
      Stop iterating after 5 seconds with no new messages.
      Prevents tests hanging forever if a message never arrives.
    """
    consumers = []

    async def _make(topic: str) -> AIOKafkaConsumer:
        consumer = AIOKafkaConsumer(
            topic,
            bootstrap_servers=KAFKA_BOOTSTRAP,
            group_id=f"test-{uuid4().hex[:8]}",   # unique = always reads from earliest
            auto_offset_reset="earliest",
            consumer_timeout_ms=5000,
        )
        await consumer.start()
        consumers.append(consumer)
        return consumer

    yield _make

    # Cleanup: stop all consumers created in this test
    for c in consumers:
        await c.stop()
