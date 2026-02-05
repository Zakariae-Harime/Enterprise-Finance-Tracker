# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Personal finance application with event-driven architecture, ML-powered transaction categorization, and PSD2 Open Banking integration. Built with FastAPI, TimescaleDB, Kafka, and PyTorch.

## Development Commands

```bash
# Start all services
docker-compose up -d

# Run database migrations
./scripts/migrate.sh

# Run tests
pytest --cov=src --cov-report=html    # Full suite with coverage
pytest tests/unit                      # Unit tests only
pytest tests/integration               # Integration tests
pytest tests/e2e                       # End-to-end tests
```

## Architecture

- **Event Sourcing + CQRS**: Transactions are event-sourced with separated read/write models
- **ML Categorization**: BERT-based classifier for automatic transaction categorization
- **Real-time Processing**: Kafka-powered event streaming for alerts and notifications
- **GDPR Compliance**: Built-in data export, consent management, and audit logging

## Tech Stack

- Backend: FastAPI + Python 3.11
- Database: TimescaleDB (time-series optimized PostgreSQL)
- Streaming: Apache Kafka
- ML: PyTorch + Transformers (BERT)
- Orchestration: Airflow
- Monitoring: Grafana + Prometheus
- Infrastructure: Terraform + Docker
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
