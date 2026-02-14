"""
  Integration tests - test real database operations.
  REQUIRES: docker-compose up -d db
"""
import pytest
import json
from uuid import uuid4

# Mark ALL tests in this file as async
pytestmark = pytest.mark.asyncio

class TestDatabaseConnection:
      """Test basic database connectivity and table existence."""

      async def test_can_connect(self, db_pool):
          """Verify we can connect to TimescaleDB."""
          async with db_pool.acquire() as conn:
              result = await conn.fetchval("SELECT 1")
              assert result == 1

      async def test_tables_exist(self, db_pool):
          """Verify all critical tables were created by init-db.sql."""
          expected_tables = [
              "events", "outbox", "dlq_messages",
              "account_projections", "processed_events"
          ]
          async with db_pool.acquire() as conn:
              rows = await conn.fetch(
                  "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
              )
              actual_tables = [row["table_name"] for row in rows]

              for table in expected_tables:
                  assert table in actual_tables, f"Missing table: {table}"
class TestEventStoreDB:
      """Test event store operations against real TimescaleDB."""

      async def test_insert_event(self, db_conn, make_event):
          """Can we write and read an event?"""
          event = make_event(
              event_type="AccountCreated",
              event_data={"name": "Savings", "currency": "NOK"}
          )

          await db_conn.execute("""
              INSERT INTO events (event_id, aggregate_type, aggregate_id, event_type, event_data, version)
              VALUES ($1, $2, $3, $4, $5, $6)
          """,
              event["event_id"], event["aggregate_type"],
              event["aggregate_id"], event["event_type"],
              json.dumps(event["event_data"]), event["version"]
          )

          row = await db_conn.fetchrow(
              "SELECT * FROM events WHERE event_id = $1", event["event_id"]
          )
          assert row is not None
          assert row["event_type"] == "AccountCreated"
      async def test_version_conflict_rejected(self, db_conn, make_event):
          """Two events with same aggregate_id + version should fail."""
          aggregate_id = uuid4()

          await db_conn.execute("""
              INSERT INTO events (event_id, aggregate_type, aggregate_id, event_type, event_data, version)
              VALUES ($1, $2, $3, $4, $5, $6)
          """, uuid4(), "account", aggregate_id, "AccountCreated",
              json.dumps({"test": "first"}), 1)
          with pytest.raises(Exception):
              await db_conn.execute("""
                  INSERT INTO events (event_id, aggregate_type, aggregate_id, event_type, event_data, version)
                  VALUES ($1, $2, $3, $4, $5, $6)
              """, uuid4(), "account", aggregate_id, "AccountUpdated",
                  json.dumps({"test": "second"}), 1)
class TestDLQDatabase:
      """Test DLQ table operations."""

      async def test_insert_dlq_message(self, db_conn, make_dlq_message):
          """DLQ message is stored with correct defaults."""
          dlq = make_dlq_message(error_message="Connection timeout")

          await db_conn.execute("""
              INSERT INTO dlq_messages (event_id, consumer_name, error_message, error_category, original_event,
  original_topic)
              VALUES ($1, $2, $3, $4, $5, $6)
          """,
              dlq["event_id"], dlq["consumer_name"],
              dlq["error_message"], dlq["error_category"],
              dlq["original_event"], dlq["original_topic"]
          )

          row = await db_conn.fetchrow(
              "SELECT * FROM dlq_messages WHERE event_id = $1", dlq["event_id"]
          )
          assert row is not None
          assert row["status"] == "pending"
          assert row["error_category"] == "transient"