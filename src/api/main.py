"""
  FastAPI Application Entry Point
    1. Creates the FastAPI app instance
    2. Sets up database connection pool on startup
    3. Registers API routes
    4. Handles graceful shutdown
"""
from contextlib import asynccontextmanager
from os import sync
from fastapi import FastAPI
import asyncpg
from aiokafka import AIOKafkaProducer
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes.accounts import router as accounts_router
from src.api.routes.transactions import router as transactions_router
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages application lifecycle.

    Startup: Create database pool, Kafka producer
    Shutdown: Close connections gracefully

    lifespan vs @app.on_event?
      - @app.on_event is deprecated in FastAPI
      - lifespan is the modern, recommended approach
    """
    # STARTUP 
    # Create database connection pool
    app.state.db_pool = await asyncpg.create_pool(
        dsn="postgresql://user:password@localhost:5432/finance_db",
        min_size=5,
        max_size=20  # Maximum 20 concurrent connections
    )
    # Create Kafka producer
    app.state.kafka_producer = AIOKafkaProducer(
        bootstrap_servers='localhost:9092'
    )
    await app.state.kafka_producer.start()
    print("Database pool and Kafka producer initialized")
    yield
    # SHUTDOWN
    await app.state.db_pool.close()
    await app.state.kafka_producer.stop()
    print("Database pool and Kafka producer closed gracefully")

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
#Register API routes
app.include_router(accounts_router, prefix="/api/v1")
app.include_router(transactions_router, prefix="/api/v1")
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
