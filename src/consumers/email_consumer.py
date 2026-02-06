"""
  Email Notification Consumer

  Listens to Kafka events and sends appropriate emails:
    - AccountCreated → Welcome email
    - TransactionCreated → Transaction confirmation
    - BudgetExceeded → Budget alert

  Uses IdempotentConsumer to ensure emails are sent exactly once.
 """
import asyncpg
from src.infrastructure.kafka_consumer import IdempotentConsumer
class EmailConsumer (IdempotentConsumer):
    """
    Kafka consumer for sending notification emails.

    Inherits from IdempotentConsumer to ensure exactly-once processing.
    """
    def __init__(self, db_pool: asyncpg.Pool, consumer_name: str, email_client):
        """
          Initialize email consumer.

          Args:
              db_pool: Database connection pool
              consumer_name: Name of the consumer
              email_client: Service for sending emails (could be SendGrid, AWS SES, etc.)
        """
        # Call parent constructor with consumer name
        super().__init__(db_pool, consumer_name="email_service")
        self._email_client = email_client  # Store email client instance (e.g., SendGrid, AWS SES)
    async def process_event(self, event_type: str, event_data: dict)-> None:
        """
        Process an incoming event and send email notification.
                  This method is called by handle_event() ONLY if the event
          hasn't been processed before (deduplication handled by parent).

        Args:
            event_type: Type of the event (e.g., 'AccountCreated')
            event_data: Payload of the event as dict
        """
        if event_type == "AccountCreated":
            await self._handle_account_created(event_data)
        elif event_type == "TransactionCreated":
            await self._transaction_created(event_data)
        elif event_type == "BudgetExceeded":
            await self._handle_budget_exceeded(event_data)
        else:
            print(f"[email_service] Unhandled event type: {event_type}")
    async def _handle_account_created(self, event_data: dict) -> None:
        """
        Handle AccountCreated event by sending welcome email.

        Args:
            event_data: Payload of the AccountCreated event
        """
        user_email = event_data.get("user_email")
        account_name = event_data.get("account_name", "your account")
        if user_email:
            subject = "Welcome to Finance Tracker!"
            body = f"Hello,\n\nYour account '{account_name}' has been successfully created.\n\nBest regards,\nFinance Tracker Team"
            await self._email_client.send_email(to=user_email, subject=subject, body=body)
            print(f"[email_service] Sent AccountCreated email to {user_email}")
        if not user_email:
            return
        #send welcome email
        await self._email_client.send_email(
            to=user_email,
            subject="Welcome to Finance Tracker!",
            body=f"Hello,\n\nYour account '{account_name}' has been successfully created.\n\nBest regards,\nFinance Tracker Team"
        )
        print(f"[email_service] Welcome email has been sent to {user_email}")
    async def _transaction_created(self, event_data: dict) -> None:
        """
        Handle TransactionCreated event by sending transaction confirmation email.

        Args:
            event_data: Payload of the TransactionCreated event
        """
        user_email = event_data.get("user_email")
        amount = event_data.get("amount", "0.00")
        currency = event_data.get("currency", "NOK")
        merchant = event_data.get("merchant_name", "unknown merchant")
        if not user_email:
            return
        await self._email_client.send_email(
            to=user_email,
            subject="Transaction Confirmation",
            body=f"Hello,\n\nA transaction of {amount} {currency} has been recorded in your account. Merchant: {merchant}\n\nIf you didn't make this transaction, please contact support.\n\nBest regards,\nFinance Tracker Team"
        )
    async def _handle_budget_exceeded(self, event_data: dict) -> None:
        """
        Handle BudgetExceeded event by sending budget alert email.

        Args:
            event_data: Payload of the BudgetExceeded event
        """
        user_email = event_data.get("user_email")
        budget_name = event_data.get("budget_name", "your budget")
        spent_amount = event_data.get("spent_amount", "0.00")
        budget_limit = event_data.get("budget_limit", "0.00")
        if not user_email:
            return
        await self._email_client.send_email(
            to=user_email,
            subject="Budget Exceeded Alert",
            body=f"Hello,\n\nYou have exceeded your budget '{budget_name}'. You have spent {spent_amount}, which is over your limit of {budget_limit}.\n\nPlease review your expenses.\n\nBest regards,\nFinance Tracker Team"
        )
        print (f"[email_service] BudgetExceeded email sent to {user_email}")