"""
Unit tests for transaction event replay logic.

Tests the replay_transaction_events() function — the core of event sourcing.
Each test builds a list of event dicts (simulating what EventStore.load_events returns)
and verifies the replayed state is correct.

No database, no async, no mocking — pure function in, dict out.
"""
import pytest
from datetime import datetime, timezone
from uuid import uuid4

from src.api.routes.transactions import replay_transaction_events


def _make_event(event_type: str, event_data: dict, version: int) -> dict:
    """
    Factory helper — builds an event dict matching EventStore.load_events() output.

    Why a helper instead of raw dicts in each test?
      - DRY: every test needs event_type, event_data, version, created_at
      - If load_events() format changes, we fix ONE place
      - Tests stay focused on WHAT they're testing, not boilerplate
    """
    return {
        "event_type": event_type,
        "event_data": event_data,
        "version": version,
        "created_at": datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc),
    }


# A reusable transaction ID — same across events in a test
# to simulate events belonging to the same aggregate
TX_ID = str(uuid4())


def _created_event(version: int = 1) -> dict:
    """Builds a standard TransactionCreated event for reuse across tests."""
    return _make_event(
        event_type="TransactionCreated",
        event_data={
            "aggregate_id": TX_ID,
            "amount": "450.00",
            "currency": "NOK",
            "transaction_type": "debit",
            "merchant_name": "REMA 1000 Grünerløkka",
            "description": "Weekly groceries",
            "category": "meals",
        },
        version=version,
    )


class TestReplayCreatedEvent:
    """Tests for the TransactionCreated event — the first event in every aggregate."""

    def test_sets_all_fields_from_event_data(self):
        """Arrange-Act-Assert: single created event produces correct state."""
        # Arrange
        events = [_created_event()]

        # Act
        state = replay_transaction_events(events)

        # Assert
        assert state["transaction_id"] == TX_ID
        assert state["amount"] == "450.00"
        assert state["currency"] == "NOK"
        assert state["transaction_type"] == "debit"
        assert state["merchant_name"] == "REMA 1000 Grünerløkka"
        assert state["description"] == "Weekly groceries"
        assert state["category"] == "meals"
        assert state["version"] == 1

    def test_defaults_when_fields_missing(self):
        """If event_data has missing fields, replay uses safe defaults."""
        # Arrange — minimal event with only aggregate_id
        events = [_make_event(
            event_type="TransactionCreated",
            event_data={"aggregate_id": TX_ID},
            version=1,
        )]

        # Act
        state = replay_transaction_events(events)

        # Assert — defaults from the initial state dict
        assert state["transaction_id"] == TX_ID
        assert state["amount"] == "0"
        assert state["currency"] == "NOK"
        assert state["merchant_name"] == "Unknown"
        assert state["description"] is None

    def test_is_not_disputed_by_default(self):
        """New transactions should never start as disputed."""
        events = [_created_event()]
        state = replay_transaction_events(events)

        assert state["is_disputed"] is False
        assert state["dispute_reason"] is None


class TestReplayCategorized:
    """Tests for TransactionCategorized — ML or user recategorization."""

    def test_categorized_overwrites_initial_category(self):
        """Created with 'meals', then recategorized to 'supplies'."""
        # Arrange
        events = [
            _created_event(version=1),
            _make_event(
                event_type="TransactionCategorized",
                event_data={"category": "supplies"},
                version=2,
            ),
        ]

        # Act
        state = replay_transaction_events(events)

        # Assert — category changed, everything else unchanged
        assert state["category"] == "supplies"
        assert state["amount"] == "450.00"       # untouched
        assert state["merchant_name"] == "REMA 1000 Grünerløkka"  # untouched
        assert state["version"] == 2

    def test_double_categorization_keeps_latest(self):
        """If categorized twice, the LAST event wins (left-fold property)."""
        events = [
            _created_event(version=1),
            _make_event(
                event_type="TransactionCategorized",
                event_data={"category": "supplies"},
                version=2,
            ),
            _make_event(
                event_type="TransactionCategorized",
                event_data={"category": "transportation"},
                version=3,
            ),
        ]

        state = replay_transaction_events(events)

        assert state["category"] == "transportation"
        assert state["version"] == 3


class TestReplayDisputed:
    """Tests for TransactionDisputed and TransactionDisputeResolved."""

    def test_disputed_sets_flag_and_reason(self):
        """Disputing a transaction flips is_disputed and stores reason."""
        events = [
            _created_event(version=1),
            _make_event(
                event_type="TransactionDisputed",
                event_data={"reason": "Unauthorized charge on my account"},
                version=2,
            ),
        ]

        state = replay_transaction_events(events)

        assert state["is_disputed"] is True
        assert state["dispute_reason"] == "Unauthorized charge on my account"
        assert state["version"] == 2

    def test_dispute_resolved_clears_flag(self):
        """After resolution, is_disputed goes back to False."""
        events = [
            _created_event(version=1),
            _make_event(
                event_type="TransactionDisputed",
                event_data={"reason": "Unauthorized charge"},
                version=2,
            ),
            _make_event(
                event_type="TransactionDisputeResolved",
                event_data={"resolution": "chargeback", "resolution_amount": "450.00"},
                version=3,
            ),
        ]

        state = replay_transaction_events(events)

        assert state["is_disputed"] is False
        assert state["dispute_reason"] is None
        assert state["version"] == 3


class TestReplayFullLifecycle:
    """Integration-style test: full transaction lifecycle through all event types."""

    def test_create_categorize_dispute_resolve(self):
        """
        Simulates a real transaction lifecycle:
          1. Created at REMA 1000 for 450 NOK
          2. ML categorizes as 'meals'
          3. User disputes it
          4. Bank resolves via chargeback

        After replay, the transaction should be:
          - NOT disputed (resolved)
          - Category = 'meals' (categorization survives dispute)
          - Version = 4
        """
        events = [
            _created_event(version=1),
            _make_event("TransactionCategorized", {"category": "meals"}, version=2),
            _make_event("TransactionDisputed", {"reason": "Not my purchase"}, version=3),
            _make_event("TransactionDisputeResolved", {"resolution": "chargeback"}, version=4),
        ]

        state = replay_transaction_events(events)

        assert state["transaction_id"] == TX_ID
        assert state["amount"] == "450.00"
        assert state["category"] == "meals"      # survives dispute
        assert state["is_disputed"] is False      # resolved
        assert state["dispute_reason"] is None    # cleared
        assert state["version"] == 4


class TestReplayUnknownEvents:
    """Tests for forward-compatibility — unknown event types must not crash."""

    def test_unknown_event_is_skipped(self):
        """
        If a new event type is added to the domain but the API hasn't been
        redeployed yet, replay should skip it gracefully.
        """
        events = [
            _created_event(version=1),
            _make_event(
                event_type="TransactionFlaggedForReview",  # doesn't exist in handlers
                event_data={"flagged_by": "compliance_bot"},
                version=2,
            ),
        ]

        state = replay_transaction_events(events)

        # State from Created is preserved, unknown event didn't crash
        assert state["transaction_id"] == TX_ID
        assert state["amount"] == "450.00"
        assert state["version"] == 2  # version still advances

    def test_unknown_event_between_known_events(self):
        """Unknown event sandwiched between known events — both known events apply."""
        events = [
            _created_event(version=1),
            _make_event("SomeNewEventType", {"data": "irrelevant"}, version=2),
            _make_event("TransactionCategorized", {"category": "rent"}, version=3),
        ]

        state = replay_transaction_events(events)

        assert state["category"] == "rent"  # categorization still applied
        assert state["version"] == 3
