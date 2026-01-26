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
- Orchestration: Dagster
- Monitoring: Grafana + Prometheus
- Infrastructure: Terraform + Docker
