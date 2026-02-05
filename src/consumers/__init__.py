"""  Kafka Consumers

  Each consumer:
    - Inherits from IdempotentConsumer
    - Implements process_event() method
    - Handles specific event types

  Consumers:
    - EmailConsumer: Sends notification emails
    - AnalyticsConsumer: Updates dashboards (future)
    - FraudConsumer: Detects suspicious activity (future)
  """