---
name: personal-finance-tracker
description: "Comprehensive guide for building a production-grade Personal Finance Tracker portfolio project optimized for Norwegian job market. Covers event-driven architecture, ML-powered categorization, PSD2 Open Banking, GDPR compliance, and free-tier deployment. Use this skill when helping Zakariae build, debug, or extend any component of the finance tracker project."
author: Zakariae Elharime
target_market: Norway (DNB, Vipps, Equinor, Cognite, SpareBank1)
timeline: 4 weeks
---

# Personal Finance Tracker - Project Skill Guide

## Quick Reference

| Aspect | Decision |
|--------|----------|
| Backend | FastAPI (Python 3.11+) with async/await |
| Primary DB | TimescaleDB (PostgreSQL extension) |
| Event Streaming | Apache Kafka (Upstash free tier) |
| Orchestration | Dagster (asset-based pipelines) |
| ML Framework | PyTorch + Transformers (BERT) |
| Frontend | React 18 + TypeScript + Tailwind |
| Infrastructure | Terraform + Docker Compose |
| CI/CD | GitHub Actions |
| Monitoring | Grafana Cloud + Prometheus |
| Deployment | Oracle Cloud Free Tier (4 OCPU, 24GB RAM) |

---

## Part 1: Project Context and Goals

### 1.1 Why This Project Matters for Norwegian Employers

Norwegian fintech companies face a talent shortage of 16,000+ IT workers. This project demonstrates exact skills they seek:

**DNB (Norway's largest bank)** uses event sourcing for transaction systems, Kotlin/Spring Boot, and graph databases for fraud detection. Your event-sourced transaction system directly mirrors their architecture.

**Vipps (Norway's dominant payment app)** builds with TypeScript, REST APIs, and microservices. Your API design and frontend choices align with their stack.

**SpareBank1** practices trunk-based development with 5-minute deployment cycles. Your CI/CD pipeline and testing strategy demonstrate this DevOps maturity.

**Equinor and Cognite** run on Azure with Kubernetes. Your cloud-native deployment shows immediate platform familiarity.

### 1.2 Learning Objectives by Week

**Week 1 - Foundation**: Event sourcing fundamentals, CQRS pattern, PostgreSQL/TimescaleDB modeling, FastAPI async patterns, Docker multi-stage builds.

**Week 2 - Integration**: Kafka producers/consumers, GoCardless PSD2 API integration, dbt transformations, data pipeline design patterns.

**Week 3 - Intelligence**: BERT fine-tuning for transaction categorization, NER for merchant extraction, anomaly detection, real-time alerting.

**Week 4 - Production**: Terraform IaC, security hardening, GDPR implementation, comprehensive documentation, performance optimization.

### 1.3 Architecture Philosophy

This project implements **three architectural patterns** that signal senior-level thinking:

**Event Sourcing**: Every financial transaction is stored as an immutable event, not just current state. This provides complete audit trails required for financial compliance and enables powerful features like point-in-time balance queries.

**CQRS (Command Query Responsibility Segregation)**: Write operations go through the event store with strong consistency; read operations use denormalized views optimized for dashboard queries. This separation allows independent scaling and optimization.

**Outbox Pattern**: Solves the dual-write problem when you need to both save to database and publish to Kafka within a single logical transaction.

---

## Part 2: System Design Specifications

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT LAYER                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  React Frontend (TypeScript)  │  Mobile PWA  │  Grafana Dashboards          │
└───────────────┬───────────────┴──────┬───────┴──────────────┬───────────────┘
                │                      │                      │
                ▼                      ▼                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              API GATEWAY                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│  FastAPI (async)  │  JWT Auth  │  Rate Limiting  │  Request Validation      │
└───────────────┬───────────────┴──────────────────┴──────────────────────────┘
                │
        ┌───────┴───────┐
        ▼               ▼
┌───────────────┐ ┌─────────────────┐
│ COMMAND SIDE  │ │   QUERY SIDE    │
├───────────────┤ ├─────────────────┤
│ Event Store   │ │ Read Models     │
│ (TimescaleDB) │ │ (Materialized   │
│               │ │  Views)         │
│ - Transaction │ │                 │
│   Events      │ │ - Daily Agg     │
│ - Account     │ │ - Monthly Agg   │
│   Events      │ │ - Category Agg  │
│ - Outbox      │ │ - Budget Status │
└───────┬───────┘ └─────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           EVENT STREAMING                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  Apache Kafka (Upstash)                                                      │
│  Topics: transactions.created │ budgets.exceeded │ anomalies.detected       │
└───────────────┬───────────────┴─────────────┬───────────────────────────────┘
                │                             │
        ┌───────┴───────┐             ┌───────┴───────┐
        ▼               ▼             ▼               ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│ ML Categorizer│ │ Projection    │ │ Alert Engine  │ │ Analytics     │
│ Consumer      │ │ Consumer      │ │ Consumer      │ │ Consumer      │
├───────────────┤ ├───────────────┤ ├───────────────┤ ├───────────────┤
│ BERT Model    │ │ Update Read   │ │ Budget Rules  │ │ Dagster Jobs  │
│ NER Extraction│ │ Models        │ │ Anomaly ML    │ │ dbt Models    │
│ Confidence    │ │ Aggregations  │ │ Notifications │ │ Batch Agg     │
└───────────────┘ └───────────────┘ └───────────────┘ └───────────────┘
```

### 2.2 Database Schema (TimescaleDB)

```sql
-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- =============================================================================
-- EVENT STORE (Write Side)
-- =============================================================================

-- Core event store table
CREATE TABLE events (
    event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    aggregate_type VARCHAR(50) NOT NULL,  -- 'account', 'transaction', 'budget'
    aggregate_id UUID NOT NULL,
    event_type VARCHAR(100) NOT NULL,
    event_data JSONB NOT NULL,
    metadata JSONB DEFAULT '{}',
    version INTEGER NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Optimistic concurrency control
    UNIQUE (aggregate_id, version)
);

-- Convert to hypertable for time-series optimization
SELECT create_hypertable('events', 'created_at');

-- Compression policy (90% storage reduction)
ALTER TABLE events SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'aggregate_type, aggregate_id'
);
SELECT add_compression_policy('events', INTERVAL '7 days');

-- Transaction events specifically (for faster queries)
CREATE TABLE transaction_events (
    event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id UUID NOT NULL,
    transaction_id UUID NOT NULL,
    event_type VARCHAR(50) NOT NULL,  -- 'created', 'categorized', 'updated', 'deleted'
    amount DECIMAL(15, 2),
    currency VARCHAR(3) DEFAULT 'NOK',
    description TEXT,
    merchant_name VARCHAR(255),
    category VARCHAR(100),
    category_confidence DECIMAL(3, 2),
    occurred_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    raw_data JSONB  -- Original bank API response
);

SELECT create_hypertable('transaction_events', 'occurred_at');

-- =============================================================================
-- OUTBOX PATTERN (Guaranteed Kafka delivery)
-- =============================================================================

CREATE TABLE outbox (
    id BIGSERIAL PRIMARY KEY,
    aggregate_type VARCHAR(50) NOT NULL,
    aggregate_id UUID NOT NULL,
    event_type VARCHAR(100) NOT NULL,
    payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    published_at TIMESTAMPTZ,
    
    -- Index for polling unpublished events
    INDEX idx_outbox_unpublished (published_at) WHERE published_at IS NULL
);

-- =============================================================================
-- READ MODELS (Query Side - Denormalized for Performance)
-- =============================================================================

-- Current account state (projected from events)
CREATE TABLE account_projections (
    account_id UUID PRIMARY KEY,
    user_id UUID NOT NULL,
    bank_name VARCHAR(100),
    account_type VARCHAR(50),
    currency VARCHAR(3) DEFAULT 'NOK',
    current_balance DECIMAL(15, 2),
    last_synced_at TIMESTAMPTZ,
    last_event_version INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Daily spending aggregates (for dashboard)
CREATE TABLE daily_aggregates (
    id BIGSERIAL,
    user_id UUID NOT NULL,
    date DATE NOT NULL,
    category VARCHAR(100) NOT NULL,
    total_amount DECIMAL(15, 2) NOT NULL,
    transaction_count INTEGER NOT NULL,
    avg_amount DECIMAL(15, 2),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    PRIMARY KEY (id, date),
    UNIQUE (user_id, date, category)
);

SELECT create_hypertable('daily_aggregates', 'date');

-- Monthly budget tracking
CREATE TABLE budget_status (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID NOT NULL,
    category VARCHAR(100) NOT NULL,
    month DATE NOT NULL,  -- First day of month
    budget_amount DECIMAL(15, 2) NOT NULL,
    spent_amount DECIMAL(15, 2) NOT NULL DEFAULT 0,
    remaining_amount DECIMAL(15, 2) GENERATED ALWAYS AS (budget_amount - spent_amount) STORED,
    percentage_used DECIMAL(5, 2) GENERATED ALWAYS AS (
        CASE WHEN budget_amount > 0 THEN (spent_amount / budget_amount * 100) ELSE 0 END
    ) STORED,
    alert_threshold_reached BOOLEAN DEFAULT FALSE,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    UNIQUE (user_id, category, month)
);

-- =============================================================================
-- GDPR COMPLIANCE
-- =============================================================================

-- Consent records (required by GDPR)
CREATE TABLE user_consents (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID NOT NULL,
    consent_type VARCHAR(50) NOT NULL,  -- 'data_processing', 'marketing', 'analytics'
    granted BOOLEAN NOT NULL,
    granted_at TIMESTAMPTZ,
    revoked_at TIMESTAMPTZ,
    policy_version VARCHAR(20) NOT NULL,
    ip_address INET,
    user_agent TEXT
);

-- Data access audit log
CREATE TABLE audit_log (
    id BIGSERIAL,
    user_id UUID,
    action VARCHAR(50) NOT NULL,  -- 'view', 'export', 'delete', 'modify'
    resource_type VARCHAR(50) NOT NULL,
    resource_id UUID,
    details JSONB,
    ip_address INET,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable('audit_log', 'created_at');

-- =============================================================================
-- INDEXES FOR COMMON QUERIES
-- =============================================================================

CREATE INDEX idx_events_aggregate ON events (aggregate_type, aggregate_id, version);
CREATE INDEX idx_transaction_events_account ON transaction_events (account_id, occurred_at DESC);
CREATE INDEX idx_transaction_events_category ON transaction_events (category, occurred_at DESC);
CREATE INDEX idx_daily_agg_user_date ON daily_aggregates (user_id, date DESC);
CREATE INDEX idx_budget_user_month ON budget_status (user_id, month DESC);
```

### 2.3 API Design (OpenAPI-First)

```yaml
# openapi.yaml - Define contracts before implementation
openapi: 3.1.0
info:
  title: Personal Finance Tracker API
  version: 1.0.0
  description: |
    Event-sourced personal finance API with ML-powered categorization.
    Compliant with PSD2 Open Banking and GDPR regulations.

servers:
  - url: https://api.finance-tracker.example.com/v1
    description: Production
  - url: http://localhost:8000/v1
    description: Development

paths:
  /accounts:
    get:
      summary: List connected bank accounts
      tags: [Accounts]
      security:
        - bearerAuth: []
      responses:
        '200':
          description: List of accounts
          content:
            application/json:
              schema:
                type: array
                items:
                  $ref: '#/components/schemas/Account'
    
    post:
      summary: Connect a new bank account via PSD2
      tags: [Accounts]
      security:
        - bearerAuth: []
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/ConnectAccountRequest'
      responses:
        '202':
          description: Bank authorization URL returned
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/BankAuthResponse'

  /transactions:
    get:
      summary: List transactions with filtering
      tags: [Transactions]
      security:
        - bearerAuth: []
      parameters:
        - name: account_id
          in: query
          schema:
            type: string
            format: uuid
        - name: category
          in: query
          schema:
            type: string
        - name: from_date
          in: query
          schema:
            type: string
            format: date
        - name: to_date
          in: query
          schema:
            type: string
            format: date
        - name: min_amount
          in: query
          schema:
            type: number
        - name: max_amount
          in: query
          schema:
            type: number
        - name: page
          in: query
          schema:
            type: integer
            default: 1
        - name: limit
          in: query
          schema:
            type: integer
            default: 50
            maximum: 100
      responses:
        '200':
          description: Paginated transaction list
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/TransactionList'

    post:
      summary: Create a manual transaction
      tags: [Transactions]
      security:
        - bearerAuth: []
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/CreateTransactionRequest'
      responses:
        '201':
          description: Transaction created with ML categorization
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Transaction'

  /transactions/{transaction_id}/category:
    patch:
      summary: Override ML-assigned category
      tags: [Transactions]
      security:
        - bearerAuth: []
      parameters:
        - name: transaction_id
          in: path
          required: true
          schema:
            type: string
            format: uuid
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                category:
                  type: string
                  example: "groceries"
      responses:
        '200':
          description: Category updated

  /budgets:
    get:
      summary: List budgets for current month
      tags: [Budgets]
      security:
        - bearerAuth: []
      responses:
        '200':
          description: Budget status list
          content:
            application/json:
              schema:
                type: array
                items:
                  $ref: '#/components/schemas/BudgetStatus'
    
    post:
      summary: Create or update a budget
      tags: [Budgets]
      security:
        - bearerAuth: []
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/CreateBudgetRequest'
      responses:
        '201':
          description: Budget created

  /analytics/spending:
    get:
      summary: Get spending analytics
      tags: [Analytics]
      security:
        - bearerAuth: []
      parameters:
        - name: period
          in: query
          schema:
            type: string
            enum: [week, month, quarter, year]
            default: month
        - name: group_by
          in: query
          schema:
            type: string
            enum: [category, merchant, day, week]
            default: category
      responses:
        '200':
          description: Spending breakdown
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/SpendingAnalytics'

  /user/export:
    post:
      summary: Request GDPR data export
      tags: [GDPR]
      security:
        - bearerAuth: []
      responses:
        '202':
          description: Export job started
          content:
            application/json:
              schema:
                type: object
                properties:
                  export_id:
                    type: string
                    format: uuid
                  estimated_completion:
                    type: string
                    format: date-time

  /user/delete:
    post:
      summary: Request account deletion (GDPR right to erasure)
      tags: [GDPR]
      security:
        - bearerAuth: []
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                confirmation:
                  type: string
                  description: Must be "DELETE MY ACCOUNT"
      responses:
        '202':
          description: Deletion scheduled

components:
  securitySchemes:
    bearerAuth:
      type: http
      scheme: bearer
      bearerFormat: JWT

  schemas:
    Account:
      type: object
      properties:
        id:
          type: string
          format: uuid
        bank_name:
          type: string
          example: "DNB"
        account_type:
          type: string
          enum: [checking, savings, credit]
        currency:
          type: string
          example: "NOK"
        current_balance:
          type: number
          example: 15000.50
        last_synced_at:
          type: string
          format: date-time

    Transaction:
      type: object
      properties:
        id:
          type: string
          format: uuid
        account_id:
          type: string
          format: uuid
        amount:
          type: number
          example: -150.00
        currency:
          type: string
          example: "NOK"
        description:
          type: string
          example: "REMA 1000 OSLO"
        merchant_name:
          type: string
          example: "REMA 1000"
        category:
          type: string
          example: "groceries"
        category_confidence:
          type: number
          minimum: 0
          maximum: 1
          example: 0.94
        category_source:
          type: string
          enum: [ml, rule, user]
        occurred_at:
          type: string
          format: date-time
        created_at:
          type: string
          format: date-time

    BudgetStatus:
      type: object
      properties:
        category:
          type: string
        budget_amount:
          type: number
        spent_amount:
          type: number
        remaining_amount:
          type: number
        percentage_used:
          type: number
        alert_threshold_reached:
          type: boolean
        month:
          type: string
          format: date
```

---

## Part 3: Implementation Guidelines

### 3.1 Event Sourcing Implementation

**Learning Objective**: Understand why event sourcing matters for financial systems and implement it correctly.

```python
# src/domain/events.py
"""
Event definitions following Domain-Driven Design principles.
Each event represents something that HAS happened (past tense naming).
Events are immutable - once created, they never change.
"""
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Optional
from uuid import UUID, uuid4

@dataclass(frozen=True)  # Immutable
class DomainEvent:
    """Base class for all domain events."""
    event_id: UUID = field(default_factory=uuid4)
    occurred_at: datetime = field(default_factory=datetime.utcnow)
    version: int = 1

@dataclass(frozen=True)
class TransactionCreated(DomainEvent):
    """
    Fired when a new transaction enters the system.
    Could be from bank sync or manual entry.
    """
    transaction_id: UUID
    account_id: UUID
    amount: Decimal
    currency: str
    description: str
    occurred_at: datetime
    source: str  # 'bank_sync', 'manual', 'import'
    raw_data: Optional[dict] = None  # Original bank API response

@dataclass(frozen=True)
class TransactionCategorized(DomainEvent):
    """
    Fired when ML or rules engine assigns a category.
    Separate from TransactionCreated to maintain single responsibility.
    """
    transaction_id: UUID
    category: str
    confidence: Decimal
    categorization_source: str  # 'ml', 'rule', 'user'
    merchant_name: Optional[str] = None  # Extracted via NER

@dataclass(frozen=True)
class TransactionCategoryOverridden(DomainEvent):
    """
    Fired when user manually overrides ML category.
    This feeds back into ML training data.
    """
    transaction_id: UUID
    old_category: str
    new_category: str
    user_id: UUID

@dataclass(frozen=True)
class BudgetExceeded(DomainEvent):
    """
    Fired when spending in a category exceeds budget threshold.
    Triggers alert notifications.
    """
    user_id: UUID
    category: str
    budget_amount: Decimal
    current_spent: Decimal
    threshold_percentage: int  # 80, 90, 100

@dataclass(frozen=True)
class AnomalyDetected(DomainEvent):
    """
    Fired when ML detects unusual spending pattern.
    Could be fraud, duplicate charge, or just unusual behavior.
    """
    transaction_id: UUID
    anomaly_type: str  # 'unusual_amount', 'unusual_merchant', 'duplicate'
    anomaly_score: Decimal
    explanation: str
```

```python
# src/domain/aggregates.py
"""
Aggregates are the consistency boundaries in event sourcing.
All state changes go through aggregates, which emit events.
"""
from dataclasses import dataclass, field
from decimal import Decimal
from typing import List
from uuid import UUID

from .events import DomainEvent, TransactionCreated, TransactionCategorized

@dataclass
class Account:
    """
    Account aggregate - maintains consistency for all account operations.
    
    Key principle: The aggregate's state is DERIVED from its events.
    We never store current state directly - we rebuild it from events.
    """
    id: UUID
    user_id: UUID
    bank_name: str
    currency: str = "NOK"
    
    # Internal state (rebuilt from events)
    _balance: Decimal = Decimal("0")
    _transaction_count: int = 0
    _pending_events: List[DomainEvent] = field(default_factory=list)
    _version: int = 0
    
    def record_transaction(
        self,
        transaction_id: UUID,
        amount: Decimal,
        description: str,
        occurred_at: datetime,
        source: str = "bank_sync",
        raw_data: dict = None
    ) -> TransactionCreated:
        """
        Record a new transaction on this account.
        
        This method:
        1. Validates business rules
        2. Creates and stores the event
        3. Updates internal state
        4. Returns the event for persistence
        """
        # Business rule: Validate transaction
        if amount == Decimal("0"):
            raise ValueError("Transaction amount cannot be zero")
        
        # Create the event
        event = TransactionCreated(
            transaction_id=transaction_id,
            account_id=self.id,
            amount=amount,
            currency=self.currency,
            description=description,
            occurred_at=occurred_at,
            source=source,
            raw_data=raw_data,
            version=self._version + 1
        )
        
        # Apply to internal state
        self._apply(event)
        
        # Store for later persistence
        self._pending_events.append(event)
        
        return event
    
    def _apply(self, event: DomainEvent) -> None:
        """Apply an event to update internal state."""
        if isinstance(event, TransactionCreated):
            self._balance += event.amount
            self._transaction_count += 1
            self._version = event.version
    
    def get_pending_events(self) -> List[DomainEvent]:
        """Get events that need to be persisted."""
        events = self._pending_events.copy()
        self._pending_events.clear()
        return events
    
    @classmethod
    def rebuild_from_events(cls, events: List[DomainEvent]) -> "Account":
        """
        Rebuild aggregate state from event history.
        This is how we load an aggregate - not from a database row,
        but by replaying all its events.
        """
        if not events:
            raise ValueError("Cannot rebuild aggregate from empty event list")
        
        # Get initial state from first event
        first_event = events[0]
        account = cls(
            id=first_event.aggregate_id,
            user_id=first_event.user_id,
            # ... other fields
        )
        
        # Replay all events
        for event in events:
            account._apply(event)
        
        return account
```

```python
# src/infrastructure/event_store.py
"""
Event store implementation using TimescaleDB.
This is the persistence layer for event sourcing.
"""
import json
from dataclasses import asdict
from decimal import Decimal
from typing import List, Type, TypeVar
from uuid import UUID

import asyncpg
from asyncpg import Connection

from src.domain.events import DomainEvent

T = TypeVar('T', bound=DomainEvent)

class DecimalEncoder(json.JSONEncoder):
    """JSON encoder that handles Decimal types."""
    def default(self, obj):
        if isinstance(obj, Decimal):
            return str(obj)
        if isinstance(obj, UUID):
            return str(obj)
        return super().default(obj)

class EventStore:
    """
    Append-only event store with optimistic concurrency control.
    
    Key guarantees:
    1. Events are immutable once stored
    2. Events are ordered by version within an aggregate
    3. Concurrent writes to same aggregate are detected and rejected
    """
    
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool
    
    async def append(
        self,
        aggregate_type: str,
        aggregate_id: UUID,
        events: List[DomainEvent],
        expected_version: int
    ) -> None:
        """
        Append events to the store with optimistic concurrency control.
        
        Args:
            aggregate_type: Type of aggregate (e.g., 'account', 'transaction')
            aggregate_id: ID of the aggregate
            events: List of events to append
            expected_version: Expected current version (for concurrency check)
        
        Raises:
            ConcurrencyError: If expected_version doesn't match current version
        """
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                # Check current version
                current_version = await conn.fetchval(
                    """
                    SELECT COALESCE(MAX(version), 0)
                    FROM events
                    WHERE aggregate_id = $1
                    """,
                    aggregate_id
                )
                
                if current_version != expected_version:
                    raise ConcurrencyError(
                        f"Expected version {expected_version}, "
                        f"but current version is {current_version}"
                    )
                
                # Append all events
                for i, event in enumerate(events):
                    version = expected_version + i + 1
                    await conn.execute(
                        """
                        INSERT INTO events (
                            event_id, aggregate_type, aggregate_id,
                            event_type, event_data, version, created_at
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                        """,
                        event.event_id,
                        aggregate_type,
                        aggregate_id,
                        type(event).__name__,
                        json.dumps(asdict(event), cls=DecimalEncoder),
                        version,
                        event.occurred_at
                    )
                    
                    # Also insert into outbox for Kafka publishing
                    await conn.execute(
                        """
                        INSERT INTO outbox (
                            aggregate_type, aggregate_id, event_type, payload
                        ) VALUES ($1, $2, $3, $4)
                        """,
                        aggregate_type,
                        aggregate_id,
                        type(event).__name__,
                        json.dumps(asdict(event), cls=DecimalEncoder)
                    )
    
    async def get_events(
        self,
        aggregate_id: UUID,
        from_version: int = 0
    ) -> List[dict]:
        """
        Retrieve events for an aggregate, optionally from a specific version.
        Used for rebuilding aggregate state.
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT event_type, event_data, version, created_at
                FROM events
                WHERE aggregate_id = $1 AND version > $2
                ORDER BY version ASC
                """,
                aggregate_id,
                from_version
            )
            return [
                {
                    "event_type": row["event_type"],
                    "data": json.loads(row["event_data"]),
                    "version": row["version"],
                    "created_at": row["created_at"]
                }
                for row in rows
            ]

class ConcurrencyError(Exception):
    """Raised when optimistic concurrency check fails."""
    pass
```

### 3.2 ML Categorization Pipeline

**Learning Objective**: Implement production ML with confidence thresholds and human-in-the-loop feedback.

```python
# src/ml/categorizer.py
"""
Hybrid transaction categorization combining rules and ML.
Norwegian employers value practical ML that handles edge cases gracefully.
"""
import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional, Tuple, Dict, List

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

@dataclass
class CategorizationResult:
    """Result of transaction categorization."""
    category: str
    confidence: Decimal
    source: str  # 'rule', 'ml', 'fallback'
    merchant_name: Optional[str] = None
    alternatives: Optional[List[Tuple[str, Decimal]]] = None

class TransactionCategorizer:
    """
    Production-grade categorizer with:
    1. Fast rule-based matching for known merchants
    2. BERT-based ML for ambiguous cases
    3. Confidence thresholds for human review
    4. Continuous learning from user corrections
    """
    
    # Norwegian merchant rules (expand based on common merchants)
    MERCHANT_RULES: Dict[str, str] = {
        # Groceries
        "rema 1000": "groceries",
        "kiwi": "groceries",
        "meny": "groceries",
        "coop": "groceries",
        "extra": "groceries",
        "joker": "groceries",
        "bunnpris": "groceries",
        
        # Transport
        "ruter": "transport",
        "vy ": "transport",
        "nsb": "transport",
        "flytoget": "transport",
        "circle k": "transport",
        "esso": "transport",
        "shell": "transport",
        
        # Subscriptions
        "spotify": "subscriptions",
        "netflix": "subscriptions",
        "hbo": "subscriptions",
        "viaplay": "subscriptions",
        "youtube": "subscriptions",
        
        # Utilities
        "telenor": "utilities",
        "telia": "utilities",
        "ice.no": "utilities",
        "fjordkraft": "utilities",
        "tibber": "utilities",
        
        # Dining
        "peppes pizza": "dining",
        "burger king": "dining",
        "mcdonalds": "dining",
        "starbucks": "dining",
        "espresso house": "dining",
    }
    
    # Categories supported by the model
    CATEGORIES = [
        "groceries", "dining", "transport", "utilities", 
        "subscriptions", "shopping", "entertainment", "health",
        "education", "travel", "income", "transfer", "other"
    ]
    
    # Confidence thresholds
    HIGH_CONFIDENCE_THRESHOLD = Decimal("0.85")
    LOW_CONFIDENCE_THRESHOLD = Decimal("0.60")
    
    def __init__(self, model_path: str = "models/transaction-bert"):
        """
        Initialize categorizer with pre-trained BERT model.
        
        Args:
            model_path: Path to fine-tuned model or HuggingFace model ID
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()  # Set to evaluation mode
        
        # Load user corrections for continuous learning
        self._user_corrections: Dict[str, str] = {}
    
    def categorize(self, description: str, amount: Decimal) -> CategorizationResult:
        """
        Categorize a transaction using hybrid approach.
        
        Strategy:
        1. Check user corrections first (learned from feedback)
        2. Try rule-based matching (fast, deterministic)
        3. Fall back to ML model
        4. Return low-confidence result for human review if uncertain
        """
        description_lower = description.lower().strip()
        
        # Step 1: Check if user has corrected this exact description before
        if description_lower in self._user_corrections:
            return CategorizationResult(
                category=self._user_corrections[description_lower],
                confidence=Decimal("1.0"),
                source="user_correction",
                merchant_name=self._extract_merchant(description)
            )
        
        # Step 2: Rule-based matching
        for pattern, category in self.MERCHANT_RULES.items():
            if pattern in description_lower:
                return CategorizationResult(
                    category=category,
                    confidence=Decimal("0.99"),
                    source="rule",
                    merchant_name=pattern.title()
                )
        
        # Step 3: Income detection (positive amounts)
        if amount > 0:
            # Check common income patterns
            income_patterns = ["lønn", "salary", "refund", "tilbakebetaling"]
            for pattern in income_patterns:
                if pattern in description_lower:
                    return CategorizationResult(
                        category="income",
                        confidence=Decimal("0.95"),
                        source="rule",
                        merchant_name=None
                    )
        
        # Step 4: ML-based categorization
        return self._ml_categorize(description, amount)
    
    def _ml_categorize(self, description: str, amount: Decimal) -> CategorizationResult:
        """
        Use BERT model for categorization.
        Returns multiple alternatives if confidence is low.
        """
        # Tokenize input
        inputs = self.tokenizer(
            description,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True
        )
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)[0]
        
        # Get top predictions
        top_indices = torch.argsort(probabilities, descending=True)[:3]
        top_probs = probabilities[top_indices]
        
        primary_idx = top_indices[0].item()
        primary_category = self.CATEGORIES[primary_idx]
        primary_confidence = Decimal(str(round(top_probs[0].item(), 4)))
        
        # Build alternatives list
        alternatives = [
            (self.CATEGORIES[idx.item()], Decimal(str(round(prob.item(), 4))))
            for idx, prob in zip(top_indices[1:], top_probs[1:])
        ]
        
        # Extract merchant name using simple heuristics
        merchant_name = self._extract_merchant(description)
        
        # Determine if this needs human review
        if primary_confidence < self.LOW_CONFIDENCE_THRESHOLD:
            # Very uncertain - might want to flag for review
            return CategorizationResult(
                category="uncategorized",
                confidence=primary_confidence,
                source="ml_low_confidence",
                merchant_name=merchant_name,
                alternatives=[(primary_category, primary_confidence)] + alternatives
            )
        
        return CategorizationResult(
            category=primary_category,
            confidence=primary_confidence,
            source="ml",
            merchant_name=merchant_name,
            alternatives=alternatives if primary_confidence < self.HIGH_CONFIDENCE_THRESHOLD else None
        )
    
    def _extract_merchant(self, description: str) -> Optional[str]:
        """
        Extract merchant name from transaction description.
        Norwegian bank descriptions often follow patterns like:
        - "VISA VARE 1234 REMA 1000 OSLO"
        - "BANKAXEPT 12.01 KIWI GRØNLAND"
        """
        # Remove common prefixes
        cleaned = re.sub(
            r'^(visa|bankaxept|vipps|avtalegiro|nettgiro)\s*(vare|kode)?\s*\d*\s*',
            '',
            description.lower()
        ).strip()
        
        # Remove trailing location/date info
        cleaned = re.sub(r'\s+\d{2}\.\d{2}.*$', '', cleaned)
        cleaned = re.sub(r'\s+(oslo|bergen|trondheim|stavanger|drammen).*$', '', cleaned, flags=re.IGNORECASE)
        
        if cleaned:
            return cleaned.title()
        return None
    
    def record_correction(self, description: str, correct_category: str) -> None:
        """
        Record user correction for continuous learning.
        This creates training data for model fine-tuning.
        """
        self._user_corrections[description.lower().strip()] = correct_category
        
        # In production, also store this in database for batch retraining
        # self._store_correction_for_training(description, correct_category)
    
    def get_correction_stats(self) -> Dict[str, int]:
        """Get statistics about user corrections by category."""
        stats: Dict[str, int] = {}
        for category in self._user_corrections.values():
            stats[category] = stats.get(category, 0) + 1
        return stats
```

```python
# src/ml/train_categorizer.py
"""
Training script for transaction categorization model.
Run this periodically to incorporate user corrections.
"""
import json
from pathlib import Path
from typing import List, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

CATEGORIES = [
    "groceries", "dining", "transport", "utilities", 
    "subscriptions", "shopping", "entertainment", "health",
    "education", "travel", "income", "transfer", "other"
]

class TransactionDataset(Dataset):
    """Dataset for transaction categorization training."""
    
    def __init__(
        self,
        descriptions: List[str],
        labels: List[int],
        tokenizer,
        max_length: int = 128
    ):
        self.descriptions = descriptions
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.descriptions)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.descriptions[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx])
        }

def load_training_data(data_path: str) -> tuple:
    """
    Load training data from JSON file.
    Expected format: [{"description": "...", "category": "..."}, ...]
    """
    with open(data_path, "r") as f:
        data = json.load(f)
    
    descriptions = [item["description"] for item in data]
    labels = [CATEGORIES.index(item["category"]) for item in data]
    
    return descriptions, labels

def train_model(
    data_path: str,
    output_path: str,
    base_model: str = "bert-base-multilingual-cased",
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5
):
    """
    Fine-tune BERT for transaction categorization.
    
    Uses multilingual BERT to handle Norwegian text mixed with
    English merchant names and abbreviations.
    """
    # Load data
    descriptions, labels = load_training_data(data_path)
    
    # Split data
    train_desc, val_desc, train_labels, val_labels = train_test_split(
        descriptions, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=len(CATEGORIES),
        problem_type="single_label_classification"
    )
    
    # Create datasets
    train_dataset = TransactionDataset(train_desc, train_labels, tokenizer)
    val_dataset = TransactionDataset(val_desc, val_labels, tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_path,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        push_to_hub=False,
        logging_dir=f"{output_path}/logs",
        logging_steps=100,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    # Train
    trainer.train()
    
    # Save model and tokenizer
    trainer.save_model(output_path)
    tokenizer.save_pretrained(output_path)
    
    # Evaluate and print report
    predictions = trainer.predict(val_dataset)
    pred_labels = predictions.predictions.argmax(axis=-1)
    print(classification_report(
        val_labels,
        pred_labels,
        target_names=CATEGORIES
    ))

if __name__ == "__main__":
    train_model(
        data_path="data/training_transactions.json",
        output_path="models/transaction-bert"
    )
```

### 3.3 Kafka Integration with Outbox Pattern

**Learning Objective**: Implement reliable event publishing that guarantees exactly-once semantics.

```python
# src/infrastructure/kafka_producer.py
"""
Kafka producer with outbox pattern for guaranteed delivery.
"""
import asyncio
import json
from datetime import datetime
from typing import Optional

import asyncpg
from aiokafka import AIOKafkaProducer

class OutboxPublisher:
    """
    Polls the outbox table and publishes to Kafka.
    
    This pattern ensures:
    1. Events are stored in DB within the same transaction as domain changes
    2. Events are eventually published to Kafka (at-least-once)
    3. Consumers can deduplicate using event_id (exactly-once semantics)
    """
    
    def __init__(
        self,
        db_pool: asyncpg.Pool,
        kafka_bootstrap_servers: str,
        poll_interval: float = 1.0,
        batch_size: int = 100
    ):
        self.db_pool = db_pool
        self.kafka_bootstrap_servers = kafka_bootstrap_servers
        self.poll_interval = poll_interval
        self.batch_size = batch_size
        self.producer: Optional[AIOKafkaProducer] = None
        self._running = False
    
    async def start(self):
        """Start the outbox publisher."""
        self.producer = AIOKafkaProducer(
            bootstrap_servers=self.kafka_bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8') if k else None,
            acks='all',  # Wait for all replicas
            enable_idempotence=True  # Exactly-once semantics
        )
        await self.producer.start()
        self._running = True
        asyncio.create_task(self._poll_loop())
    
    async def stop(self):
        """Stop the outbox publisher."""
        self._running = False
        if self.producer:
            await self.producer.stop()
    
    async def _poll_loop(self):
        """Main polling loop."""
        while self._running:
            try:
                await self._process_outbox()
            except Exception as e:
                # Log error but don't crash - we'll retry next iteration
                print(f"Error processing outbox: {e}")
            
            await asyncio.sleep(self.poll_interval)
    
    async def _process_outbox(self):
        """Process unpublished outbox entries."""
        async with self.db_pool.acquire() as conn:
            # Fetch unpublished events
            rows = await conn.fetch(
                """
                SELECT id, aggregate_type, aggregate_id, event_type, payload
                FROM outbox
                WHERE published_at IS NULL
                ORDER BY id ASC
                LIMIT $1
                FOR UPDATE SKIP LOCKED
                """,
                self.batch_size
            )
            
            if not rows:
                return
            
            published_ids = []
            
            for row in rows:
                # Determine topic from aggregate type
                topic = self._get_topic(row["aggregate_type"], row["event_type"])
                
                try:
                    # Publish to Kafka
                    await self.producer.send_and_wait(
                        topic=topic,
                        key=str(row["aggregate_id"]),
                        value=json.loads(row["payload"])
                    )
                    published_ids.append(row["id"])
                except Exception as e:
                    print(f"Failed to publish event {row['id']}: {e}")
                    # Don't mark as published - will retry next iteration
            
            # Mark events as published
            if published_ids:
                await conn.execute(
                    """
                    UPDATE outbox
                    SET published_at = $1
                    WHERE id = ANY($2)
                    """,
                    datetime.utcnow(),
                    published_ids
                )
    
    def _get_topic(self, aggregate_type: str, event_type: str) -> str:
        """Map aggregate/event types to Kafka topics."""
        topic_map = {
            ("transaction", "TransactionCreated"): "transactions.created",
            ("transaction", "TransactionCategorized"): "transactions.categorized",
            ("budget", "BudgetExceeded"): "budgets.exceeded",
            ("anomaly", "AnomalyDetected"): "anomalies.detected",
        }
        return topic_map.get((aggregate_type, event_type), "events.default")
```

```python
# src/infrastructure/kafka_consumer.py
"""
Kafka consumers for different event types.
"""
import asyncio
import json
from abc import ABC, abstractmethod
from typing import Dict, Set

from aiokafka import AIOKafkaConsumer

class EventConsumer(ABC):
    """Base class for Kafka event consumers."""
    
    def __init__(
        self,
        bootstrap_servers: str,
        topic: str,
        group_id: str
    ):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.group_id = group_id
        self.consumer: AIOKafkaConsumer = None
        self._processed_ids: Set[str] = set()  # For deduplication
        self._running = False
    
    async def start(self):
        """Start consuming events."""
        self.consumer = AIOKafkaConsumer(
            self.topic,
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.group_id,
            value_deserializer=lambda v: json.loads(v.decode('utf-8')),
            auto_offset_reset='earliest',
            enable_auto_commit=False  # Manual commit for reliability
        )
        await self.consumer.start()
        self._running = True
        await self._consume_loop()
    
    async def stop(self):
        """Stop consuming events."""
        self._running = False
        if self.consumer:
            await self.consumer.stop()
    
    async def _consume_loop(self):
        """Main consumption loop."""
        async for message in self.consumer:
            if not self._running:
                break
            
            event_id = message.value.get("event_id")
            
            # Deduplication check
            if event_id in self._processed_ids:
                await self.consumer.commit()
                continue
            
            try:
                await self.handle_event(message.value)
                self._processed_ids.add(event_id)
                
                # Limit memory usage for dedup set
                if len(self._processed_ids) > 10000:
                    # Remove oldest entries (simple approach)
                    self._processed_ids = set(list(self._processed_ids)[-5000:])
                
                await self.consumer.commit()
            except Exception as e:
                print(f"Error processing event {event_id}: {e}")
                # Don't commit - will retry on next poll
    
    @abstractmethod
    async def handle_event(self, event: Dict) -> None:
        """Handle a single event. Override in subclasses."""
        pass

class CategorizationConsumer(EventConsumer):
    """
    Consumes TransactionCreated events and applies ML categorization.
    """
    
    def __init__(
        self,
        bootstrap_servers: str,
        categorizer,  # TransactionCategorizer instance
        event_store  # EventStore instance
    ):
        super().__init__(
            bootstrap_servers=bootstrap_servers,
            topic="transactions.created",
            group_id="categorization-service"
        )
        self.categorizer = categorizer
        self.event_store = event_store
    
    async def handle_event(self, event: Dict) -> None:
        """Categorize transaction and emit categorization event."""
        from decimal import Decimal
        from uuid import UUID
        from src.domain.events import TransactionCategorized
        
        # Run ML categorization
        result = self.categorizer.categorize(
            description=event["description"],
            amount=Decimal(event["amount"])
        )
        
        # Create categorization event
        categorization_event = TransactionCategorized(
            transaction_id=UUID(event["transaction_id"]),
            category=result.category,
            confidence=result.confidence,
            categorization_source=result.source,
            merchant_name=result.merchant_name
        )
        
        # Store event (which also adds to outbox for other consumers)
        await self.event_store.append(
            aggregate_type="transaction",
            aggregate_id=UUID(event["transaction_id"]),
            events=[categorization_event],
            expected_version=1  # After TransactionCreated
        )

class AlertConsumer(EventConsumer):
    """
    Consumes categorized transactions and checks budget thresholds.
    """
    
    def __init__(
        self,
        bootstrap_servers: str,
        db_pool,
        notification_service
    ):
        super().__init__(
            bootstrap_servers=bootstrap_servers,
            topic="transactions.categorized",
            group_id="alert-service"
        )
        self.db_pool = db_pool
        self.notification_service = notification_service
    
    async def handle_event(self, event: Dict) -> None:
        """Check if transaction triggers any budget alerts."""
        from decimal import Decimal
        
        category = event["category"]
        
        async with self.db_pool.acquire() as conn:
            # Get current budget status
            budget = await conn.fetchrow(
                """
                SELECT budget_amount, spent_amount, percentage_used,
                       alert_threshold_reached
                FROM budget_status
                WHERE user_id = $1 AND category = $2
                  AND month = date_trunc('month', CURRENT_DATE)
                """,
                event["user_id"],
                category
            )
            
            if not budget:
                return
            
            # Check thresholds
            thresholds = [80, 90, 100]
            for threshold in thresholds:
                if (budget["percentage_used"] >= threshold and 
                    not budget["alert_threshold_reached"]):
                    # Send alert
                    await self.notification_service.send_budget_alert(
                        user_id=event["user_id"],
                        category=category,
                        threshold=threshold,
                        budget_amount=budget["budget_amount"],
                        spent_amount=budget["spent_amount"]
                    )
                    
                    # Mark threshold as reached
                    await conn.execute(
                        """
                        UPDATE budget_status
                        SET alert_threshold_reached = TRUE
                        WHERE user_id = $1 AND category = $2
                          AND month = date_trunc('month', CURRENT_DATE)
                        """,
                        event["user_id"],
                        category
                    )
```

### 3.4 Dagster Orchestration

**Learning Objective**: Build observable, testable data pipelines with modern orchestration.

```python
# src/pipelines/dagster_assets.py
"""
Dagster asset definitions for data pipeline orchestration.
Assets represent data artifacts that can be materialized on demand.
"""
from datetime import datetime, timedelta
from typing import Dict, Any

from dagster import (
    asset,
    AssetExecutionContext,
    DailyPartitionsDefinition,
    MetadataValue,
    Output,
    AssetIn
)
import pandas as pd

# Daily partitioning for time-series data
daily_partitions = DailyPartitionsDefinition(start_date="2024-01-01")

@asset(
    partitions_def=daily_partitions,
    group_name="raw_data",
    description="Sync transactions from connected bank accounts"
)
async def bank_transactions(context: AssetExecutionContext) -> Output[pd.DataFrame]:
    """
    Fetch transactions from GoCardless API for the partition date.
    
    This asset:
    1. Connects to PSD2 Open Banking API
    2. Fetches transactions for the specified date
    3. Validates and normalizes the data
    4. Returns a DataFrame for downstream processing
    """
    partition_date = context.partition_key
    
    # Get database connection from resources
    db = context.resources.database
    gocardless = context.resources.gocardless_client
    
    # Fetch all connected accounts
    accounts = await db.fetch_all(
        "SELECT id, requisition_id FROM accounts WHERE user_id IS NOT NULL"
    )
    
    all_transactions = []
    
    for account in accounts:
        try:
            # Fetch from GoCardless
            transactions = await gocardless.get_transactions(
                account_id=account["requisition_id"],
                date_from=partition_date,
                date_to=partition_date
            )
            
            for txn in transactions:
                all_transactions.append({
                    "account_id": account["id"],
                    "external_id": txn["transactionId"],
                    "amount": float(txn["transactionAmount"]["amount"]),
                    "currency": txn["transactionAmount"]["currency"],
                    "description": txn.get("remittanceInformationUnstructured", ""),
                    "booking_date": txn["bookingDate"],
                    "value_date": txn.get("valueDate"),
                    "raw_data": txn
                })
        except Exception as e:
            context.log.warning(f"Failed to fetch account {account['id']}: {e}")
    
    df = pd.DataFrame(all_transactions)
    
    return Output(
        df,
        metadata={
            "row_count": MetadataValue.int(len(df)),
            "partition_date": MetadataValue.text(partition_date),
            "accounts_processed": MetadataValue.int(len(accounts))
        }
    )

@asset(
    ins={"bank_transactions": AssetIn()},
    partitions_def=daily_partitions,
    group_name="enriched_data",
    description="Categorize transactions using ML model"
)
async def categorized_transactions(
    context: AssetExecutionContext,
    bank_transactions: pd.DataFrame
) -> Output[pd.DataFrame]:
    """
    Apply ML categorization to raw transactions.
    
    This asset:
    1. Loads the trained categorization model
    2. Applies categorization to each transaction
    3. Records confidence scores and alternatives
    4. Flags low-confidence transactions for review
    """
    categorizer = context.resources.categorizer
    
    results = []
    low_confidence_count = 0
    
    for _, row in bank_transactions.iterrows():
        result = categorizer.categorize(
            description=row["description"],
            amount=row["amount"]
        )
        
        results.append({
            **row.to_dict(),
            "category": result.category,
            "category_confidence": float(result.confidence),
            "category_source": result.source,
            "merchant_name": result.merchant_name,
            "needs_review": result.confidence < 0.7
        })
        
        if result.confidence < 0.7:
            low_confidence_count += 1
    
    df = pd.DataFrame(results)
    
    return Output(
        df,
        metadata={
            "row_count": MetadataValue.int(len(df)),
            "low_confidence_count": MetadataValue.int(low_confidence_count),
            "category_distribution": MetadataValue.json(
                df["category"].value_counts().to_dict()
            )
        }
    )

@asset(
    ins={"categorized_transactions": AssetIn()},
    partitions_def=daily_partitions,
    group_name="aggregations",
    description="Compute daily spending aggregates by category"
)
async def daily_spending_aggregates(
    context: AssetExecutionContext,
    categorized_transactions: pd.DataFrame
) -> Output[pd.DataFrame]:
    """
    Aggregate transactions into daily spending by category.
    
    This powers the dashboard's spending breakdown visualizations.
    """
    if categorized_transactions.empty:
        return Output(
            pd.DataFrame(),
            metadata={"row_count": MetadataValue.int(0)}
        )
    
    # Filter to expenses only (negative amounts)
    expenses = categorized_transactions[categorized_transactions["amount"] < 0].copy()
    expenses["amount"] = expenses["amount"].abs()
    
    # Aggregate by user, date, category
    aggregates = expenses.groupby(
        ["user_id", "booking_date", "category"]
    ).agg({
        "amount": ["sum", "count", "mean"],
        "transaction_id": "count"
    }).reset_index()
    
    # Flatten column names
    aggregates.columns = [
        "user_id", "date", "category",
        "total_amount", "transaction_count", "avg_amount", "_count"
    ]
    aggregates = aggregates.drop("_count", axis=1)
    
    # Write to database
    db = context.resources.database
    await db.execute_many(
        """
        INSERT INTO daily_aggregates (
            user_id, date, category, total_amount, 
            transaction_count, avg_amount
        ) VALUES ($1, $2, $3, $4, $5, $6)
        ON CONFLICT (user_id, date, category)
        DO UPDATE SET
            total_amount = EXCLUDED.total_amount,
            transaction_count = EXCLUDED.transaction_count,
            avg_amount = EXCLUDED.avg_amount
        """,
        aggregates.values.tolist()
    )
    
    return Output(
        aggregates,
        metadata={
            "row_count": MetadataValue.int(len(aggregates)),
            "total_spending": MetadataValue.float(aggregates["total_amount"].sum()),
            "categories": MetadataValue.json(aggregates["category"].unique().tolist())
        }
    )

@asset(
    ins={"daily_spending_aggregates": AssetIn()},
    partitions_def=daily_partitions,
    group_name="alerts",
    description="Check budget thresholds and generate alerts"
)
async def budget_alerts(
    context: AssetExecutionContext,
    daily_spending_aggregates: pd.DataFrame
) -> Output[Dict[str, Any]]:
    """
    Check if any budgets have been exceeded and trigger alerts.
    
    This asset:
    1. Loads current budget definitions
    2. Calculates month-to-date spending
    3. Identifies budgets that crossed thresholds
    4. Publishes alert events
    """
    db = context.resources.database
    notification_service = context.resources.notifications
    
    alerts_triggered = []
    
    # Get unique users from aggregates
    user_ids = daily_spending_aggregates["user_id"].unique()
    
    for user_id in user_ids:
        # Get user's budgets
        budgets = await db.fetch_all(
            """
            SELECT category, budget_amount, spent_amount, percentage_used
            FROM budget_status
            WHERE user_id = $1 
              AND month = date_trunc('month', CURRENT_DATE)
            """,
            user_id
        )
        
        for budget in budgets:
            percentage = budget["percentage_used"]
            
            # Check each threshold
            for threshold in [80, 90, 100]:
                if percentage >= threshold:
                    alert = {
                        "user_id": str(user_id),
                        "category": budget["category"],
                        "threshold": threshold,
                        "percentage_used": float(percentage),
                        "budget_amount": float(budget["budget_amount"]),
                        "spent_amount": float(budget["spent_amount"])
                    }
                    alerts_triggered.append(alert)
                    
                    # Send notification
                    await notification_service.send_budget_alert(**alert)
                    
                    break  # Only send highest threshold alert
    
    return Output(
        {"alerts": alerts_triggered},
        metadata={
            "alerts_count": MetadataValue.int(len(alerts_triggered)),
            "affected_users": MetadataValue.int(len(set(a["user_id"] for a in alerts_triggered)))
        }
    )
```

---

## Part 4: Infrastructure and Deployment

### 4.1 Docker Configuration

```dockerfile
# Dockerfile
# Multi-stage build for minimal production image

# ============================================
# Stage 1: Build dependencies
# ============================================
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# ============================================
# Stage 2: Production image
# ============================================
FROM python:3.11-slim as production

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    libpq5 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home appuser

# Copy Python packages from builder
COPY --from=builder /root/.local /home/appuser/.local
ENV PATH=/home/appuser/.local/bin:$PATH

# Copy application code
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser alembic/ ./alembic/
COPY --chown=appuser:appuser alembic.ini .

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Run with Uvicorn
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
# Local development environment

version: '3.8'

services:
  # ===========================================
  # API Server
  # ===========================================
  api:
    build:
      context: .
      target: production
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:postgres@db:5432/finance_tracker
      - KAFKA_BOOTSTRAP_SERVERS=kafka:9092
      - REDIS_URL=redis://redis:6379
      - GOCARDLESS_SECRET_ID=${GOCARDLESS_SECRET_ID}
      - GOCARDLESS_SECRET_KEY=${GOCARDLESS_SECRET_KEY}
      - JWT_SECRET=${JWT_SECRET:-development-secret-change-in-production}
    depends_on:
      db:
        condition: service_healthy
      kafka:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # ===========================================
  # TimescaleDB (PostgreSQL with time-series)
  # ===========================================
  db:
    image: timescale/timescaledb:latest-pg15
    ports:
      - "5432:5432"
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=postgres
      - POSTGRES_DB=finance_tracker
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init-db.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5

  # ===========================================
  # Kafka (Redpanda for local dev - Kafka-compatible)
  # ===========================================
  kafka:
    image: redpandadata/redpanda:latest
    command:
      - redpanda
      - start
      - --smp 1
      - --memory 512M
      - --reserve-memory 0M
      - --overprovisioned
      - --node-id 0
      - --kafka-addr PLAINTEXT://0.0.0.0:9092
      - --advertise-kafka-addr PLAINTEXT://kafka:9092
    ports:
      - "9092:9092"
      - "8081:8081"  # Schema Registry
    healthcheck:
      test: ["CMD", "rpk", "cluster", "health"]
      interval: 10s
      timeout: 5s
      retries: 5

  # ===========================================
  # Redis (caching and real-time features)
  # ===========================================
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  # ===========================================
  # Dagster (pipeline orchestration)
  # ===========================================
  dagster:
    build:
      context: .
      dockerfile: Dockerfile.dagster
    ports:
      - "3000:3000"
    environment:
      - DAGSTER_HOME=/opt/dagster/dagster_home
      - DATABASE_URL=postgresql://postgres:postgres@db:5432/finance_tracker
    volumes:
      - dagster_home:/opt/dagster/dagster_home
    depends_on:
      - db

  # ===========================================
  # Grafana (monitoring dashboards)
  # ===========================================
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3001:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning

  # ===========================================
  # Prometheus (metrics collection)
  # ===========================================
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus

volumes:
  postgres_data:
  dagster_home:
  grafana_data:
  prometheus_data:
```

### 4.2 Terraform Infrastructure (Oracle Cloud Free Tier)

```hcl
# terraform/main.tf
# Oracle Cloud Infrastructure - Free Tier Deployment

terraform {
  required_providers {
    oci = {
      source  = "oracle/oci"
      version = "~> 5.0"
    }
  }
}

provider "oci" {
  tenancy_ocid     = var.tenancy_ocid
  user_ocid        = var.user_ocid
  fingerprint      = var.fingerprint
  private_key_path = var.private_key_path
  region           = var.region
}

# ===========================================
# Variables
# ===========================================
variable "tenancy_ocid" {}
variable "user_ocid" {}
variable "fingerprint" {}
variable "private_key_path" {}
variable "region" { default = "eu-amsterdam-1" }  # Closest to Norway
variable "compartment_ocid" {}

# ===========================================
# Network Configuration
# ===========================================
resource "oci_core_vcn" "finance_tracker_vcn" {
  compartment_id = var.compartment_ocid
  cidr_blocks    = ["10.0.0.0/16"]
  display_name   = "finance-tracker-vcn"
  dns_label      = "financetracker"
}

resource "oci_core_subnet" "public_subnet" {
  compartment_id    = var.compartment_ocid
  vcn_id            = oci_core_vcn.finance_tracker_vcn.id
  cidr_block        = "10.0.1.0/24"
  display_name      = "public-subnet"
  dns_label         = "public"
  security_list_ids = [oci_core_security_list.public_security_list.id]
  route_table_id    = oci_core_route_table.public_route_table.id
}

resource "oci_core_internet_gateway" "internet_gateway" {
  compartment_id = var.compartment_ocid
  vcn_id         = oci_core_vcn.finance_tracker_vcn.id
  display_name   = "internet-gateway"
}

resource "oci_core_route_table" "public_route_table" {
  compartment_id = var.compartment_ocid
  vcn_id         = oci_core_vcn.finance_tracker_vcn.id
  display_name   = "public-route-table"

  route_rules {
    destination       = "0.0.0.0/0"
    destination_type  = "CIDR_BLOCK"
    network_entity_id = oci_core_internet_gateway.internet_gateway.id
  }
}

resource "oci_core_security_list" "public_security_list" {
  compartment_id = var.compartment_ocid
  vcn_id         = oci_core_vcn.finance_tracker_vcn.id
  display_name   = "public-security-list"

  # Allow SSH
  ingress_security_rules {
    protocol = "6"  # TCP
    source   = "0.0.0.0/0"
    tcp_options {
      min = 22
      max = 22
    }
  }

  # Allow HTTP
  ingress_security_rules {
    protocol = "6"
    source   = "0.0.0.0/0"
    tcp_options {
      min = 80
      max = 80
    }
  }

  # Allow HTTPS
  ingress_security_rules {
    protocol = "6"
    source   = "0.0.0.0/0"
    tcp_options {
      min = 443
      max = 443
    }
  }

  # Allow all outbound
  egress_security_rules {
    protocol    = "all"
    destination = "0.0.0.0/0"
  }
}

# ===========================================
# Compute Instance (ARM - Free Tier)
# ===========================================
data "oci_core_images" "ubuntu_arm" {
  compartment_id           = var.compartment_ocid
  operating_system         = "Canonical Ubuntu"
  operating_system_version = "22.04"
  shape                    = "VM.Standard.A1.Flex"
  sort_by                  = "TIMECREATED"
  sort_order               = "DESC"
}

resource "oci_core_instance" "finance_tracker_instance" {
  compartment_id      = var.compartment_ocid
  availability_domain = data.oci_identity_availability_domains.ads.availability_domains[0].name
  display_name        = "finance-tracker-server"
  
  # ARM shape - Free tier allows up to 4 OCPUs and 24GB RAM
  shape = "VM.Standard.A1.Flex"
  shape_config {
    ocpus         = 4
    memory_in_gbs = 24
  }

  source_details {
    source_type = "image"
    source_id   = data.oci_core_images.ubuntu_arm.images[0].id
    boot_volume_size_in_gbs = 100  # Free tier includes 200GB total
  }

  create_vnic_details {
    subnet_id        = oci_core_subnet.public_subnet.id
    assign_public_ip = true
  }

  metadata = {
    ssh_authorized_keys = file("~/.ssh/id_rsa.pub")
    user_data = base64encode(file("${path.module}/cloud-init.yaml"))
  }

  # Prevent accidental deletion
  lifecycle {
    prevent_destroy = true
  }
}

data "oci_identity_availability_domains" "ads" {
  compartment_id = var.tenancy_ocid
}

# ===========================================
# Outputs
# ===========================================
output "instance_public_ip" {
  value = oci_core_instance.finance_tracker_instance.public_ip
}

output "ssh_command" {
  value = "ssh ubuntu@${oci_core_instance.finance_tracker_instance.public_ip}"
}
```

```yaml
# terraform/cloud-init.yaml
# Cloud-init configuration for instance setup

#cloud-config
package_update: true
package_upgrade: true

packages:
  - docker.io
  - docker-compose
  - nginx
  - certbot
  - python3-certbot-nginx
  - fail2ban
  - ufw

runcmd:
  # Enable Docker
  - systemctl enable docker
  - systemctl start docker
  - usermod -aG docker ubuntu
  
  # Configure firewall
  - ufw default deny incoming
  - ufw default allow outgoing
  - ufw allow ssh
  - ufw allow http
  - ufw allow https
  - ufw --force enable
  
  # Enable fail2ban
  - systemctl enable fail2ban
  - systemctl start fail2ban
  
  # Create app directory
  - mkdir -p /opt/finance-tracker
  - chown ubuntu:ubuntu /opt/finance-tracker
  
  # Install Docker Compose v2
  - curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
  - chmod +x /usr/local/bin/docker-compose
  
  # Create keepalive cron job (prevents Oracle from reclaiming idle instances)
  - echo "*/5 * * * * root /usr/bin/curl -s http://localhost:8000/health > /dev/null" >> /etc/crontab
```

### 4.3 GitHub Actions CI/CD

```yaml
# .github/workflows/ci.yml
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  PYTHON_VERSION: "3.11"
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  # ===========================================
  # Code Quality
  # ===========================================
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      
      - name: Install dependencies
        run: |
          pip install ruff mypy
          pip install -r requirements.txt
      
      - name: Lint with Ruff
        run: ruff check src/ tests/
      
      - name: Type check with MyPy
        run: mypy src/ --ignore-missing-imports

  # ===========================================
  # Unit Tests
  # ===========================================
  test-unit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      
      - name: Install dependencies
        run: pip install -r requirements.txt -r requirements-dev.txt
      
      - name: Run unit tests
        run: |
          pytest tests/unit \
            --cov=src \
            --cov-report=xml \
            --cov-report=html \
            --junitxml=test-results.xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          files: coverage.xml
          fail_ci_if_error: true

  # ===========================================
  # Integration Tests
  # ===========================================
  test-integration:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: timescale/timescaledb:latest-pg15
        env:
          POSTGRES_USER: test
          POSTGRES_PASSWORD: test
          POSTGRES_DB: test
        ports:
          - 5432:5432
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
      
      redis:
        image: redis:7-alpine
        ports:
          - 6379:6379
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      
      - name: Install dependencies
        run: pip install -r requirements.txt -r requirements-dev.txt
      
      - name: Run integration tests
        env:
          DATABASE_URL: postgresql://test:test@localhost:5432/test
          REDIS_URL: redis://localhost:6379
        run: pytest tests/integration -v

  # ===========================================
  # Security Scanning
  # ===========================================
  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Run Snyk to check for vulnerabilities
        uses: snyk/actions/python@master
        continue-on-error: true  # Don't fail on vulnerabilities, just report
        env:
          SNYK_TOKEN: ${{ secrets.SNYK_TOKEN }}
        with:
          args: --severity-threshold=high
      
      - name: Run CodeQL Analysis
        uses: github/codeql-action/init@v3
        with:
          languages: python
      
      - name: Perform CodeQL Analysis
        uses: github/codeql-action/analyze@v3

  # ===========================================
  # Build Docker Image
  # ===========================================
  build:
    needs: [lint, test-unit, test-integration]
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
      
      - name: Login to GitHub Container Registry
        if: github.event_name != 'pull_request'
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      
      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=sha,prefix=
            type=ref,event=branch
            type=semver,pattern={{version}}
      
      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          push: ${{ github.event_name != 'pull_request' }}
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
      
      - name: Scan image for vulnerabilities
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
          format: 'sarif'
          output: 'trivy-results.sarif'

  # ===========================================
  # Deploy to Production
  # ===========================================
  deploy:
    needs: [build, security]
    if: github.ref == 'refs/heads/main' && github.event_name == 'push'
    runs-on: ubuntu-latest
    environment: production
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Deploy to Oracle Cloud
        env:
          SSH_PRIVATE_KEY: ${{ secrets.ORACLE_SSH_KEY }}
          HOST: ${{ secrets.ORACLE_HOST }}
        run: |
          # Setup SSH
          mkdir -p ~/.ssh
          echo "$SSH_PRIVATE_KEY" > ~/.ssh/id_rsa
          chmod 600 ~/.ssh/id_rsa
          ssh-keyscan -H $HOST >> ~/.ssh/known_hosts
          
          # Deploy
          ssh ubuntu@$HOST << 'ENDSSH'
            cd /opt/finance-tracker
            docker-compose pull
            docker-compose up -d
            docker system prune -f
          ENDSSH
      
      - name: Verify deployment
        run: |
          sleep 30
          curl -f https://api.finance-tracker.example.com/health || exit 1
```

---

## Part 5: Testing Strategy

### 5.1 Test Pyramid

```
        /\
       /  \      E2E Tests (10%)
      /----\     - Full user flows
     /      \    - Critical paths only
    /--------\
   /          \  Integration Tests (20%)
  /            \ - API endpoints
 /--------------\ - Database operations
/                \ - Kafka consumers
/------------------\
        Unit Tests (70%)
        - Domain logic
        - ML categorization
        - Event handling
```

### 5.2 Example Tests

```python
# tests/unit/test_categorizer.py
"""
Unit tests for transaction categorization.
"""
from decimal import Decimal

import pytest

from src.ml.categorizer import TransactionCategorizer, CategorizationResult

class TestTransactionCategorizer:
    """Tests for the hybrid categorization pipeline."""
    
    @pytest.fixture
    def categorizer(self):
        """Create categorizer with mock ML model for unit tests."""
        # In unit tests, we mock the ML model
        return TransactionCategorizer(model_path="tests/fixtures/mock-model")
    
    def test_rule_based_groceries(self, categorizer):
        """Norwegian grocery stores should be categorized by rules."""
        result = categorizer.categorize(
            description="VISA VARE 1234 REMA 1000 OSLO",
            amount=Decimal("-150.00")
        )
        
        assert result.category == "groceries"
        assert result.confidence == Decimal("0.99")
        assert result.source == "rule"
        assert result.merchant_name == "Rema 1000"
    
    def test_rule_based_transport(self, categorizer):
        """Transport merchants should be detected."""
        result = categorizer.categorize(
            description="RUTER AS BILLETT",
            amount=Decimal("-39.00")
        )
        
        assert result.category == "transport"
        assert result.source == "rule"
    
    def test_income_detection(self, categorizer):
        """Positive amounts with salary keywords should be income."""
        result = categorizer.categorize(
            description="LØNN DESEMBER 2024",
            amount=Decimal("45000.00")
        )
        
        assert result.category == "income"
        assert result.confidence >= Decimal("0.95")
    
    def test_ml_fallback_for_unknown(self, categorizer):
        """Unknown merchants should use ML categorization."""
        result = categorizer.categorize(
            description="RANDOM MERCHANT XYZ",
            amount=Decimal("-99.00")
        )
        
        # Should fall back to ML
        assert result.source in ["ml", "ml_low_confidence"]
    
    def test_user_correction_takes_priority(self, categorizer):
        """User corrections should override ML."""
        # Record a correction
        categorizer.record_correction(
            description="RANDOM MERCHANT XYZ",
            correct_category="entertainment"
        )
        
        # Now categorize the same description
        result = categorizer.categorize(
            description="RANDOM MERCHANT XYZ",
            amount=Decimal("-99.00")
        )
        
        assert result.category == "entertainment"
        assert result.confidence == Decimal("1.0")
        assert result.source == "user_correction"
    
    def test_low_confidence_flagged(self, categorizer):
        """Low confidence predictions should include alternatives."""
        # Use a very ambiguous description
        result = categorizer.categorize(
            description="PAYMENT",
            amount=Decimal("-50.00")
        )
        
        if result.confidence < Decimal("0.70"):
            assert result.alternatives is not None
            assert len(result.alternatives) > 0

class TestMerchantExtraction:
    """Tests for merchant name extraction from descriptions."""
    
    @pytest.fixture
    def categorizer(self):
        return TransactionCategorizer(model_path="tests/fixtures/mock-model")
    
    @pytest.mark.parametrize("description,expected_merchant", [
        ("VISA VARE 1234 REMA 1000 OSLO", "Rema 1000"),
        ("BANKAXEPT 12.01 KIWI GRØNLAND", "Kiwi"),
        ("VIPPS *SPOTIFY", "Spotify"),
        ("NETTGIRO FRA DNB", None),  # Not a merchant
    ])
    def test_merchant_extraction(self, categorizer, description, expected_merchant):
        """Merchant names should be extracted from various formats."""
        result = categorizer.categorize(description, Decimal("-100"))
        assert result.merchant_name == expected_merchant
```

```python
# tests/integration/test_event_store.py
"""
Integration tests for event sourcing infrastructure.
"""
import asyncio
from decimal import Decimal
from uuid import uuid4

import pytest
import asyncpg

from src.infrastructure.event_store import EventStore, ConcurrencyError
from src.domain.events import TransactionCreated

@pytest.fixture
async def db_pool():
    """Create database connection pool for tests."""
    pool = await asyncpg.create_pool(
        "postgresql://test:test@localhost:5432/test"
    )
    
    # Setup schema
    async with pool.acquire() as conn:
        await conn.execute(open("schema.sql").read())
    
    yield pool
    
    # Cleanup
    async with pool.acquire() as conn:
        await conn.execute("TRUNCATE events, outbox CASCADE")
    
    await pool.close()

@pytest.fixture
def event_store(db_pool):
    return EventStore(db_pool)

class TestEventStore:
    """Integration tests for event persistence."""
    
    @pytest.mark.asyncio
    async def test_append_single_event(self, event_store):
        """Should persist a single event."""
        aggregate_id = uuid4()
        event = TransactionCreated(
            transaction_id=uuid4(),
            account_id=aggregate_id,
            amount=Decimal("-150.00"),
            currency="NOK",
            description="Test transaction",
            occurred_at=datetime.utcnow(),
            source="test"
        )
        
        await event_store.append(
            aggregate_type="transaction",
            aggregate_id=aggregate_id,
            events=[event],
            expected_version=0
        )
        
        # Verify event was stored
        events = await event_store.get_events(aggregate_id)
        assert len(events) == 1
        assert events[0]["event_type"] == "TransactionCreated"
    
    @pytest.mark.asyncio
    async def test_optimistic_concurrency(self, event_store):
        """Should reject concurrent writes with wrong version."""
        aggregate_id = uuid4()
        event = TransactionCreated(
            transaction_id=uuid4(),
            account_id=aggregate_id,
            amount=Decimal("-100.00"),
            currency="NOK",
            description="First",
            occurred_at=datetime.utcnow(),
            source="test"
        )
        
        # First write succeeds
        await event_store.append(
            aggregate_type="transaction",
            aggregate_id=aggregate_id,
            events=[event],
            expected_version=0
        )
        
        # Second write with wrong version fails
        with pytest.raises(ConcurrencyError):
            await event_store.append(
                aggregate_type="transaction",
                aggregate_id=aggregate_id,
                events=[event],
                expected_version=0  # Should be 1
            )
    
    @pytest.mark.asyncio
    async def test_outbox_populated(self, event_store, db_pool):
        """Events should be added to outbox for Kafka publishing."""
        aggregate_id = uuid4()
        event = TransactionCreated(
            transaction_id=uuid4(),
            account_id=aggregate_id,
            amount=Decimal("-50.00"),
            currency="NOK",
            description="Outbox test",
            occurred_at=datetime.utcnow(),
            source="test"
        )
        
        await event_store.append(
            aggregate_type="transaction",
            aggregate_id=aggregate_id,
            events=[event],
            expected_version=0
        )
        
        # Check outbox
        async with db_pool.acquire() as conn:
            outbox_count = await conn.fetchval(
                "SELECT COUNT(*) FROM outbox WHERE aggregate_id = $1",
                aggregate_id
            )
        
        assert outbox_count == 1
```

---

## Part 6: Documentation Standards

### 6.1 README Template

Your README should follow this structure to impress Norwegian recruiters:

1. **Badges** (CI status, coverage, license)
2. **One-line description** with key differentiators
3. **Live demo link** (critical for portfolio impact)
4. **Architecture diagram** (visual immediately shows system thinking)
5. **Tech stack table** with "why" column
6. **Quick start** (under 5 commands)
7. **API examples** (show real requests/responses)
8. **Testing section** (coverage and how to run)
9. **Design decisions** (link to ADRs)
10. **Deployment guide**

### 6.2 Architecture Decision Records (ADRs)

Create `docs/adr/` directory with decisions documented:

```markdown
# ADR-001: Event Sourcing for Financial Transactions

## Status
Accepted

## Context
We need to track all financial transactions with complete audit history. 
Norwegian financial regulations require maintaining transaction records for 5+ years.
Users may dispute transactions or need to reconstruct historical balances.

## Decision
Implement event sourcing where every transaction state change is stored as an immutable event.

## Consequences
### Positive
- Complete audit trail by design
- Can reconstruct state at any point in time
- Natural fit for CQRS pattern
- Events can be replayed for debugging

### Negative
- More complex than CRUD
- Requires careful event schema versioning
- Read models must be kept in sync

### Risks
- Event store could grow large over time
- Mitigation: TimescaleDB compression (90% reduction)
```

---

## Part 7: Claude Assistance Guidelines

When helping with this project, Claude should:

### 7.1 Code Review Checklist

- [ ] Event names are past tense (e.g., `TransactionCreated`, not `CreateTransaction`)
- [ ] Events are immutable (using `@dataclass(frozen=True)`)
- [ ] Outbox pattern used for all Kafka publishing
- [ ] Async/await used consistently (no blocking calls)
- [ ] Type hints on all function signatures
- [ ] Docstrings explain "why" not just "what"
- [ ] Error handling with specific exception types
- [ ] Logging at appropriate levels (DEBUG, INFO, WARNING, ERROR)
- [ ] Tests follow Arrange-Act-Assert pattern
- [ ] Norwegian merchant names in categorization rules

### 7.2 Common Pitfalls to Catch

1. **Blocking calls in async code**: Never use `time.sleep()` in async, use `asyncio.sleep()`
2. **Missing outbox entries**: Every event store append must also add to outbox
3. **Version conflicts**: Always check expected_version in event store
4. **Decimal precision**: Use `Decimal` for all money amounts, never `float`
5. **Timezone handling**: Store everything in UTC, convert on display
6. **SQL injection**: Always use parameterized queries
7. **Missing indexes**: Check query plans for full table scans

### 7.3 Learning Prompts to Ask

When Zakariae is implementing a component, prompt with:

- "What problem does this pattern solve?"
- "How would a Norwegian bank like DNB implement this?"
- "What happens if this component fails?"
- "How would you test this in isolation?"
- "What metrics would you monitor for this?"

### 7.4 Norwegian Market Talking Points

Help Zakariae articulate these in interviews:

- **Event sourcing**: "DNB uses this pattern for transaction systems because financial regulations require complete audit trails. I implemented the same architecture."

- **PSD2 integration**: "I integrated with GoCardless to demonstrate understanding of Open Banking APIs that Norwegian fintechs like Vipps must comply with."

- **GDPR compliance**: "The system implements data export, consent management, and right to erasure - critical requirements for any European fintech."

- **TimescaleDB choice**: "PostgreSQL is the dominant database in Norwegian fintech. TimescaleDB adds time-series optimization while keeping full PostgreSQL compatibility."

---

## Quick Command Reference

```bash
# Start local development
docker-compose up -d

# Run all tests
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_categorizer.py -v

# Apply database migrations
alembic upgrade head

# Generate new migration
alembic revision --autogenerate -m "description"

# Start Dagster UI
dagster dev

# Build production image
docker build -t finance-tracker:latest .

# Deploy to Oracle Cloud
terraform apply -auto-approve

# View logs
docker-compose logs -f api

# Access database
docker-compose exec db psql -U postgres -d finance_tracker
```

---

*This skill document should be referenced whenever working on the Personal Finance Tracker project. Update it as the project evolves and new patterns emerge.*
