from pydantic import BaseModel, ConfigDict
from typing import Optional
from uuid import UUID
from datetime import datetime


class ConnectIntegrationRequest(BaseModel):
    provider: str                  # "tripletex" | "visma" | "xero" | "quickbooks" | "sap" | "dynamics"
    credentials: dict              # provider-specific — encrypted before storage


class IntegrationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    provider: str
    status: str
    created_at: datetime
    last_synced_at: Optional[datetime] = None


class TriggerSyncResponse(BaseModel):
    integration_id: UUID
    records_synced: int
    success: bool
    errors: list[str]


class SyncJobResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    integration_id: UUID
    status: str
    records_processed: Optional[int] = None
    error_details: Optional[str] = None
    started_at: datetime
    finished_at: Optional[datetime] = None
