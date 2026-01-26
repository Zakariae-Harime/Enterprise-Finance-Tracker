# Personal Finance Tracker 🇳🇴

[![CI/CD](https://github.com/username/finance-tracker/actions/workflows/ci.yml/badge.svg)](link)
[![Coverage](https://codecov.io/gh/username/finance-tracker/badge.svg)](link)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](link)

**[Live Demo](https://finance-tracker.example.com)** | **[API Docs](https://api.finance-tracker.example.com/docs)**

A production-grade personal finance application demonstrating event-driven architecture,
ML-powered transaction categorization, and PSD2 Open Banking integration—built with 
technologies used by DNB, Vipps, and Equinor.

## Architecture Overview

![Architecture Diagram](docs/architecture.png)

- **Event Sourcing**: Complete audit trail using PostgreSQL/TimescaleDB
- **CQRS**: Separated read/write models for optimal performance
- **ML Categorization**: BERT-based classifier achieving 94% accuracy
- **Real-time Alerts**: Kafka-powered budget and anomaly notifications
- **GDPR Compliant**: Full data export, consent management, audit logging

## Tech Stack

| Layer | Technology | Why This Choice |
|-------|------------|-----------------|
| Backend | FastAPI + Python 3.11 | Type-safe, async-native, OpenAPI auto-generation |
| Database | TimescaleDB | Time-series optimized PostgreSQL (used in Nordic fintech) |
| Streaming | Apache Kafka | Industry standard for financial event streaming |
| ML | PyTorch + Transformers | BERT fine-tuned on 100K+ transaction descriptions |
| Orchestration | Dagster | Asset-based pipelines with built-in data quality |
| Monitoring | Grafana + Prometheus | Industry-standard observability stack |
| Infrastructure | Terraform + Docker | IaC with multi-stage container builds |

## Quick Start
```bash
# Clone and start all services
git clone https://github.com/username/finance-tracker
cd finance-tracker
docker-compose up -d

# Run database migrations
./scripts/migrate.sh

# Access the application
open http://localhost:3000
```

## API Examples
```bash
# Create a transaction (event sourced)
curl -X POST http://localhost:8000/api/v1/transactions \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"amount": -150.00, "description": "REMA 1000 OSLO", "date": "2025-01-25"}'

# Response includes ML-predicted category
{
  "id": "txn_abc123",
  "amount": -150.00,
  "category": "groceries",
  "category_confidence": 0.97,
  "merchant": "REMA 1000"
}
```

## Testing
```bash
# Run full test suite (80%+ coverage)
pytest --cov=src --cov-report=html

# Run specific test types
pytest tests/unit          # 2s
pytest tests/integration   # 15s
pytest tests/e2e          # 60s
```

## Design Decisions

See [Architecture Decision Records](docs/adr/) for detailed reasoning:
- [ADR-001: Event Sourcing for Transactions](docs/adr/001-event-sourcing.md)
- [ADR-002: TimescaleDB over ClickHouse](docs/adr/002-database-choice.md)
- [ADR-003: BERT vs Rule-Based Categorization](docs/adr/003-ml-approach.md)

## Deployment

The application runs on free-tier cloud services:
- **Compute**: Oracle Cloud ARM (4 OCPU, 24GB RAM)
- **Database**: Supabase PostgreSQL + self-hosted TimescaleDB
- **Streaming**: Upstash Kafka
- **Monitoring**: Grafana Cloud

See [Deployment Guide](docs/deployment.md) for step-by-step instructions.