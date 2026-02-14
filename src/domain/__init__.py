from dataclasses import dataclass, field
from decimal import Decimal
from datetime import datetime, timezone
from enum import Enum
from uuid import UUID
from uuid6 import uuid7
from typing import Optional


# =============================================================================
# HELPER FUNCTION
# =============================================================================

def generate_event_id() -> UUID:
    """
    Generate a UUID7 (time-ordered) for event IDs.

    UUID7 benefits:
    - IDs sort chronologically (great for event ordering)
    - Database B-tree indexes stay balanced
    - You can extract creation time from the UUID
    """
    return uuid7()


# =============================================================================
# EVENT METADATA
# =============================================================================

@dataclass(frozen=True)
class EventMetadata:
    """
    Metadata attached to every event for tracing and compliance.

    Example workflow:
    ┌─────────────────────────────────────────────────────────────┐
    │ User uploads invoice                                        │
    │   correlation_id: abc-123                                   │
    │   causation_id: None (user initiated)                       │
    ├─────────────────────────────────────────────────────────────┤
    │ OCR processes invoice                                       │
    │   correlation_id: abc-123  ← Same! Links to original        │
    │   causation_id: upload-event-id                             │
    ├─────────────────────────────────────────────────────────────┤
    │ Invoice approved                                            │
    │   correlation_id: abc-123  ← Still same                     │
    │   causation_id: ocr-event-id                                │
    └─────────────────────────────────────────────────────────────┘
    """
    # Fields WITH defaults (using field(default_factory=...))
    event_id: UUID = field(default_factory=generate_event_id)
    correlation_id: UUID = field(default_factory=generate_event_id)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    # Fields WITH simple defaults
    causation_id: Optional[UUID] = None
    user_id: Optional[UUID] = None
    schema_version: int = 1


# =============================================================================
# BASE EVENT CLASS
# =============================================================================

@dataclass(frozen=True)
class DomainEvent:
    """
    Base class for all domain events.

    Every event must have:
    - aggregate_id: The entity this event belongs to (e.g., account_id)
    - metadata: Tracing information for debugging and compliance

    frozen=True makes the dataclass immutable - you cannot change fields
    after creation. This is critical because events represent historical
    facts that CANNOT change.

    Example:
        event = TransactionCreated(aggregate_id=uuid7(), ...)
        event.amount = 100  # ERROR! frozen=True prevents this
    """
    aggregate_id: UUID
    metadata: EventMetadata = field(default_factory=EventMetadata, kw_only=True)

    @property
    def event_type(self) -> str:
        """Returns the class name as the event type for serialization."""
        return self.__class__.__name__

    @property
    def occurred_at(self) -> datetime:
        """Returns the event timestamp from metadata."""
        return self.metadata.timestamp

    @property
    def event_id(self) -> UUID:
        """Returns the unique event ID from metadata."""
        return self.metadata.event_id


# =============================================================================
# ENUMS - Business Domain Constants
# =============================================================================

class TransactionType(str, Enum):
    """Types of financial transactions."""
    CREDIT = "credit"    # Money coming IN
    DEBIT = "debit"      # Money going OUT
    TRANSFER = "transfer"  # Between accounts


class Currency(str, Enum):
    """Supported currencies (ISO 4217 codes)."""
    NOK = "NOK"  # Norwegian Krone
    DKK = "DKK"  # Danish Krone
    MAD = "MAD"  # Moroccan Dirham
    SEK = "SEK"  # Swedish Krona
    USD = "USD"  # US Dollar
    EUR = "EUR"  # Euro
    GBP = "GBP"  # British Pound
    JPY = "JPY"  # Japanese Yen
    AUD = "AUD"  # Australian Dollar
    CAD = "CAD"  # Canadian Dollar
    CHF = "CHF"  # Swiss Franc


class AccountType(str, Enum):
    """Types of financial accounts."""
    CHECKING = "checking"
    SAVINGS = "savings"
    CREDIT_CARD = "credit_card"
    LOAN = "loan"
    INVESTMENT = "investment"
    CASH = "cash"
    OTHER = "other"


class ApprovalStatus(str, Enum):
    """Status of approval workflow."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


class ExpenseCategory(str, Enum):
    """Enterprise expense categories (aligned with NS 4102)."""
    MEALS = "meals"
    SUPPLIES = "supplies"
    RENT = "rent"
    UTILITIES = "utilities"
    TRANSPORTATION = "transportation"
    SOFTWARE = "software"
    HEALTHCARE = "healthcare"
    MARKETING = "marketing"
    CONSULTING = "consulting"
    TRAVEL = "travel"
    OTHER = "other"


# =============================================================================
# ACCOUNT EVENTS - Lifecycle of a bank account
# =============================================================================

@dataclass(frozen=True)
class AccountCreated(DomainEvent):
    """
    Fired when a new account is created.

    This is the first event in an account's lifecycle.
    Contains all initial state needed to create the account projection.

    Note: Balance starts at initial_balance. All balance changes
    come from TransactionCreated events.
    """
    account_name: str
    account_type: AccountType
    currency: Currency
    initial_balance: Decimal = Decimal("0.00")
    # Enterprise: multi-tenant fields
    organization_id: Optional[UUID] = None
    department_id: Optional[UUID] = None


@dataclass(frozen=True)
class AccountRenamed(DomainEvent):
    """
    Fired when an account is renamed.

    Why store old_name? For audit - regulators may ask
    "what was this account called before?"
    """
    old_name: str
    new_name: str


@dataclass(frozen=True)
class AccountClosed(DomainEvent):
    """
    Fired when an account is closed.

    We NEVER delete accounts - we mark them as closed.
    Norwegian law requires 5+ years of financial records.
    """
    reason: str
    final_balance: Decimal

# TRANSACTION EVENTS - Financial movements (CORE)
@dataclass(frozen=True)
class TransactionCreated(DomainEvent):
    """
    Fired when a financial transaction is created.

    This event drives balance changes on accounts.
    """
    amount: Decimal
    currency: Currency
    transaction_type: TransactionType
    merchant_name: str
    description: Optional[str] = None
    category: Optional[ExpenseCategory] = None
    transaction_date: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    #enrichment fields
    merchant_category_code: Optional[str] = None  # MCC code (e.g., "5411" = grocery)
    external_reference: Optional[str] = None  # Bank's transaction ID
    raw_description: Optional[str] = None  # Original bank description
    # Enterprise fields
    organization_id: Optional[UUID] = None
    department_id: Optional[UUID] = None
    project_id: Optional[UUID] = None
    cost_center_id: Optional[UUID] = None

@dataclass(frozen=True)
class TransactionCategorized(DomainEvent):
    """
    Fired when a transaction is categorized by ML or user.
categorized_by options:
"ml_model": BERT classifier prediction
"user": Manual categorization
"rule": Rule-based (e.g., "REMA" → groceries)
    """
    category:str
    subcategory:Optional[str] = None
    confidence_score:Optional[float] = None  # 0.0 to 1.0
    categorized_by:str="user"
    previous_category:Optional[str] = None
@dataclass(frozen=True)
class TransactionTagged(DomainEvent):
    """
    Fired when a transaction is tagged with custom labels.
    """
    tags: tuple[str, ...]  # Variable-length tuple of tag strings
@dataclass(frozen=True)
class TransactionNoteAdded(DomainEvent):
    """
    Fired when a note is added to a transaction.
    """
    note: str

@dataclass(frozen=True)
class TransactionDisputed(DomainEvent):
    """
    Fired when a transaction is disputed by the user.
    """
    reason: str

@dataclass(frozen=True)
class TransactionDisputeResolved(DomainEvent):
    """
    Fired when a disputed transaction is resolved.

    """
    resolution: str  # "chargeback", "upheld", "denied"
    resolution_amount: Optional[Decimal] = None

# BUDGET EVENTS - Spending limits and tracking
@dataclass(frozen=True)
class BudgetCreated(DomainEvent):
    """
    Fired when a new budget is created.
    """
    # Fields WITHOUT defaults first
    budget_name: str
    amount: Decimal
    currency: Currency
    period: str  # "monthly", "quarterly", "yearly"
    start_date: datetime
    end_date: datetime
    # Fields WITH defaults last
    alert_threshold: float = 0.8
    category: Optional[ExpenseCategory] = None
    organization_id: Optional[UUID] = None
    department_id: Optional[UUID] = None
@dataclass(frozen=True)
class BudgetUpdated(DomainEvent):
    old_amount_limit: Decimal
    new_amount_limit: Decimal
@dataclass(frozen=True)
class BudgetThresholdExceeded(DomainEvent):
    """
    Fired when a budget's alert threshold is exceeded.
    Kafka consumers listen for this to send notifications:
    Push notification to mobile
    Email alert
    Slack message (enterprise)
    """
    category : str
    budget_limit : Decimal
    current_spending :Decimal
    percentage_used : float 
    
    
@dataclass(frozen=True)
class BudgetExceeded(DomainEvent):
    """
    Fired when a budget is exceeded.
    """
    budget_name: str
    amount: Decimal
    current_spending: Decimal
    budget_limit: Decimal
    exceeded_by: Decimal
    currency: Currency
    category: Optional[ExpenseCategory] = None  
    organization_id: Optional[UUID] = None
    department_id: Optional[UUID] = None

#ORGANIZATION EVENTS - Enterprise multi-tenant
@dataclass(frozen=True)
class OrganizationCreated(DomainEvent):
    """
    Fired when a new organization is created.

    """
    organization_name: str
    org_number: Optional[str] = None  # Business registration number
    country: str = "NO"  # Country of registration


@dataclass(frozen=True)
class DepartmentCreated(DomainEvent):
    """
    Fired when a new department is created within an organization.

    """
    department_name: str
    organization_id: UUID
    parent_department_id: Optional[UUID] = None  # For nested departments
    cost_center_code: Optional[str] = None  # Linked cost center

@dataclass(frozen=True)
class MemberInvitedToOrganization(DomainEvent):
    """
    Fired when a member is invited to join an organization.

    """
    organization_id: UUID
    email: str  # Email of the invited member
    role: str  # e.g., "admin", "viewer"

@dataclass(frozen=True)
class MemberJoinedOrganization(DomainEvent):
    """
    Fired when a member joins an organization.
    """
    organization_id: UUID
    user_id: UUID  # User ID of the member
    role: str  # e.g., "admin", "viewer"
@dataclass(frozen=True)
class MemberRoleUpdated(DomainEvent):
    """
    Fired when a member's role is updated within an organization.

    """
    organization_id: UUID
    user_id: UUID  # User ID of the member whose role changed
    old_role: str
    new_role: str


@dataclass(frozen=True)
class MemberRemovedFromOrganization(DomainEvent):
    """
    Fired when a member is removed from an organization.

    """
    organization_id: UUID
    user_id: UUID  # User ID of the removed member
    reason: Optional[str] = None  # Optional reason for removal

# EXPENSE EVENTS - Enterprise expense management
@dataclass(frozen=True)
class ExpenseSubmitted(DomainEvent):
    """
    Fired when an employee submits an expense for approval.

    Use metadata.user_id for who submitted.
    Use metadata.timestamp for submission time.
    """
    amount: Decimal
    currency: Currency
    description: str
    category: ExpenseCategory
    receipt_urls: tuple[str, ...] = field(default_factory=tuple)
    organization_id: Optional[UUID] = None
    department_id: Optional[UUID] = None
    project_id: Optional[UUID] = None


@dataclass(frozen=True)
class ExpenseApprovalRequested(DomainEvent):
    """
    Fired when an expense approval is requested.

    Approval rules determine who approves:
    - < 1000 NOK: Direct manager
    - 1000-10000 NOK: Department head
    - > 10000 NOK: Finance director
    """
    approver_id: UUID
    due_date: datetime
    approval_level: int
@dataclass(frozen=True)
class ExpenseApproved(DomainEvent):
    """
    Fired when an expense is approved by a manager.
    """
    notes: Optional[str] = None  # Optional approval notes

@dataclass(frozen=True)
class ExpenseRejected(DomainEvent):
    """
    Fired when an expense is rejected by a manager.
    """
    rejection_reason: str
@dataclass(frozen=True)
class ExpenseReimbursed(DomainEvent):
    """
    Fired when an approved expense is reimbursed to the employee.
    """
    reimbursement_amount: Decimal  
    payment_reference: Optional[str] = None         
# INVOICE EVENTS - Invoice processing (OCR integration)
@dataclass(frozen=True)
class InvoiceUploaded(DomainEvent):
    """
    Fired when an invoice is uploaded.
    """
    file_url: str  # URL to the uploaded invoice file
    file_name: str
    organization_id: Optional[UUID] = None
    mime_type: Optional[str] = None # e.g., "application/pdf"
@dataclass(frozen=True)
class InvoiceProcessed(DomainEvent):
    """
    Fired when an invoice has been processed by OCR.
    """
    invoice_number: str
    vendor_name: str
    amount_due: Decimal
    invoice_date: datetime
    currency: Currency
    due_date: datetime
    tax_amount: Optional[Decimal] = None    
    confidence_score: float = 0.0 # Confidence score from OCR
    kid_number: Optional[str] = None  # Norwegian payment reference
@dataclass(frozen=True)
class InvoiceProcessingFailed(DomainEvent):
    """
    Fired when invoice processing fails.
    """
    error_message: str
    error_code: Optional[str] = None
@dataclass(frozen=True)
class InvoiceApproved(DomainEvent):
    """
    Fired when an invoice has been approved for payment.
    """
    payment_date: Optional[datetime] = None  # Scheduled payment date
@dataclass(frozen=True)
class InvoicePaid(DomainEvent):
    """
    Fired when an invoice has been marked as paid.
    """
    payment_amount: Decimal
    payment_reference: Optional[str] = None  # Payment transaction reference

#SYNC EVENTS - ERP/Accounting integration
@dataclass(frozen=True)
class SyncStarted(DomainEvent):
    """
    Fired when a synchronization with an external system starts.
    """
    organization_id: UUID
    system_name: str  # e.g., "SAP", "Oracle"
    sync_type: str  # e.g., "full", "incremental"
@dataclass(frozen=True)
class SyncCompleted(DomainEvent):
    """
    Fired when a synchronization with an external system completes.
    """
    organization_id: UUID
    system_name: str  # e.g., "SAP", "Oracle"
    records_synced: int

@dataclass(frozen=True)
class SyncFailed(DomainEvent):
    """
    Fired when a synchronization with an external system fails.
    """
    organization_id: UUID
    system_name: str  # e.g., "SAP", "Oracle"
    error_message: str
    will_retry: bool = True
# GDPR EVENTS - Privacy compliance (MANDATORY in EU/Norway)
@dataclass(frozen=True)
class DataRequested(DomainEvent):
    """
    Fired when a user requests their personal data (GDPR).
    """
    request_type: str  # "export" or "deletion"
    export_format: str = "json"
@dataclass(frozen=True)
class DataExportCompleted(DomainEvent):
    """
    Fired when a user's personal data export is completed.
    """
    export_url: str  # URL to download the exported data
    consent_type: str  # "analytics", "marketing", "third_party_sharing"
@dataclass(frozen=True)
class DataDeletionRequested(DomainEvent):
    """
    Fired when a user requests deletion of their personal data (GDPR).
     
      
        "Right to be forgotten" - BUT financial records must be kept
        for regulatory compliance (5 years in Norway per Bokføringsloven).
      
         We delete personal data but retain anonymized financial records.
    """
    reason: Optional[str] = None  # Optional reason for deletion
@dataclass(frozen=True)
class DataDeletionCompleted(DomainEvent):
    """
    Fired when a user's personal data deletion is completed.
     GDPR requires EXPLICIT consent. We track:
     "analytics": Usage tracking
     "marketing": Email campaigns
     "third_party_sharing": Share with partners
    """
    consent_type: str  # "analytics", "marketing", "third_party_sharing"
@dataclass(frozen=True)
class UserConsentGranted(DomainEvent):
    """
    Fired when a user grants consent for data processing.
    """
    consent_type: str  # "analytics", "marketing", "third_party_sharing"
@dataclass(frozen=True)
class UserConsentRevoked(DomainEvent):
    """
    Fired when a user revokes consent for data processing.
    """
    consent_type: str  # "analytics", "marketing", "third_party_sharing"

# EVENT REGISTRY - Maps event names to classes for deserialization
EVENT_REGISTRY : dict[str, type[DomainEvent]] = {
    "AccountCreated": AccountCreated,
    "AccountRenamed": AccountRenamed,
    "AccountClosed": AccountClosed,
    "TransactionCreated": TransactionCreated,
    "TransactionCategorized": TransactionCategorized,
    "TransactionTagged": TransactionTagged,
    "TransactionDisputed": TransactionDisputed,
    "TransactionDisputeResolved": TransactionDisputeResolved,
    "BudgetCreated": BudgetCreated,
    "BudgetExceeded": BudgetExceeded,
    "OrganizationCreated": OrganizationCreated,
    "DepartmentCreated": DepartmentCreated,
    "MemberJoinedOrganization": MemberJoinedOrganization,
    "MemberRoleUpdated": MemberRoleUpdated,
    "ExpenseSubmitted": ExpenseSubmitted,
    "ExpenseApproved": ExpenseApproved,
    "ExpenseRejected": ExpenseRejected,
    "InvoiceUploaded": InvoiceUploaded,
    "InvoiceProcessed": InvoiceProcessed,
    "InvoicePaid": InvoicePaid,
    "SyncStarted": SyncStarted,
    "SyncCompleted": SyncCompleted,
    "SyncFailed": SyncFailed,
    "DataRequested": DataRequested,
    "DataExportCompleted": DataExportCompleted,
    "UserConsentGranted": UserConsentGranted,
    "UserConsentRevoked": UserConsentRevoked,
    "TransactionNoteAdded": TransactionNoteAdded,
    "BudgetUpdated": BudgetUpdated,
    "BudgetThresholdExceeded": BudgetThresholdExceeded,
    "MemberInvitedToOrganization": MemberInvitedToOrganization,
    "MemberRemovedFromOrganization": MemberRemovedFromOrganization,
    "ExpenseApprovalRequested": ExpenseApprovalRequested,
    "ExpenseReimbursed": ExpenseReimbursed,
    "InvoiceProcessingFailed": InvoiceProcessingFailed,
    "InvoiceApproved": InvoiceApproved,
    "DataDeletionRequested": DataDeletionRequested,
    "DataDeletionCompleted": DataDeletionCompleted,
}