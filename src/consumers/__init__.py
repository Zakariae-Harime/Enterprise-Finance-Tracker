"""
Kafka Consumers

Each consumer:
    - Inherits from IdempotentConsumer
    - Implements process_event() method
    - Handles specific event types

Consumers:
    - EmailConsumer: Sends notification emails
    - DataLakeConsumer: Uploads events to Azure Data Lake Bronze layer
    - DLQProcessor: Handles failed messages from Dead Letter Queue
    - AnalyticsConsumer: Updates dashboards (future)
    - FraudConsumer: Detects suspicious activity (future)
"""
from src.consumers.email_consumer import EmailConsumer
from src.consumers.data_lake_consumer import DataLakeConsumer, start_data_lake_consumer
from src.consumers.dlq_processor import DLQProcessor, start_dlq_processor

__all__ = [
    "EmailConsumer",
    "DataLakeConsumer",
    "start_data_lake_consumer",
    "DLQProcessor",
    "start_dlq_processor",
]