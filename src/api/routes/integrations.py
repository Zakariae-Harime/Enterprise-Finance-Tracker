"""
ERP Integration Routes — thin HTTP layer only.

Endpoints:
  POST   /integrations/              — connect a new ERP provider
  GET    /integrations/              — list all integrations for this org
  POST   /integrations/{id}/sync     — trigger a full pull sync
  GET    /integrations/{id}/sync-jobs — list sync history
  DELETE /integrations/{id}          — disconnect (soft delete)
"""
import uuid
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.dependencies import get_db_pool
from src.api.schemas.integration import (
    ConnectIntegrationRequest,
    IntegrationResponse,
    SyncJobResponse,
    TriggerSyncResponse,
)
from src.auth.dependencies import UserContext, get_current_user, require_role
from src.erp import ADAPTERS, get_adapter
from src.services.credentials import encrypt_credentials
from src.services.sync_service import SyncService

router = APIRouter(prefix="/integrations", tags=["integrations"])


@router.post("/", response_model=IntegrationResponse, status_code=status.HTTP_201_CREATED)
async def connect_integration(
    request: ConnectIntegrationRequest,
    current_user: UserContext = Depends(require_role("owner", "admin")),
    db_pool=Depends(get_db_pool),
):
    """
    Connect a new ERP provider.
    Validates the provider name, encrypts credentials, inserts into DB.
    """
    if request.provider not in ADAPTERS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Unknown provider '{request.provider}'. Supported: {list(ADAPTERS)}",
        )

    encrypted = encrypt_credentials(request.credentials)
    integration_id = uuid.uuid4()

    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO integrations (id, organization_id, provider, credentials_encrypted, status)
            VALUES ($1, $2, $3, $4, 'active')
            RETURNING id, provider, status, created_at, last_sync_at
            """,
            integration_id,
            current_user.organization_id,
            request.provider,
            encrypted,
        )

    return IntegrationResponse(**dict(row))


@router.get("/", response_model=List[IntegrationResponse])
async def list_integrations(
    current_user: UserContext = Depends(get_current_user),
    db_pool=Depends(get_db_pool),
):
    """List all active ERP integrations for this organisation."""
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, provider, status, created_at, last_sync_at
            FROM integrations
            WHERE organization_id = $1 AND status != 'disconnected'
            ORDER BY created_at DESC
            """,
            current_user.organization_id,
        )
    return [IntegrationResponse(**dict(r)) for r in rows]


@router.post("/{integration_id}/sync", response_model=TriggerSyncResponse)
async def trigger_sync(
    integration_id: uuid.UUID,
    current_user: UserContext = Depends(require_role("owner", "admin", "finance")),
    db_pool=Depends(get_db_pool),
):
    """
    Pull invoices from the ERP and store results.
    Updates last_synced_at on success.
    """
    service = SyncService(db_pool)
    result = await service.run_sync(integration_id, current_user.organization_id)

    if result.success:
        async with db_pool.acquire() as conn:
            await conn.execute(
                "UPDATE integrations SET last_sync_at = NOW() WHERE id = $1",
                integration_id,
            )

    return TriggerSyncResponse(
        integration_id=integration_id,
        records_synced=result.records_synced,
        success=result.success,
        errors=result.errors,
    )


@router.get("/{integration_id}/sync-jobs", response_model=List[SyncJobResponse])
async def list_sync_jobs(
    integration_id: uuid.UUID,
    current_user: UserContext = Depends(get_current_user),
    db_pool=Depends(get_db_pool),
):
    """Return sync history for a given integration, newest first."""
    async with db_pool.acquire() as conn:
        # Verify integration belongs to this org before exposing its jobs
        integration = await conn.fetchrow(
            "SELECT id FROM integrations WHERE id = $1 AND organization_id = $2",
            integration_id,
            current_user.organization_id,
        )
        if not integration:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Integration not found")

        rows = await conn.fetch(
            """
            SELECT id, integration_id, status, records_processed,
                   error_log, started_at, completed_at
            FROM sync_jobs
            WHERE integration_id = $1
            ORDER BY started_at DESC
            LIMIT 50
            """,
            integration_id,
        )

    return [SyncJobResponse(**dict(r)) for r in rows]


@router.delete("/{integration_id}", status_code=status.HTTP_204_NO_CONTENT)
async def disconnect_integration(
    integration_id: uuid.UUID,
    current_user: UserContext = Depends(require_role("owner", "admin")),
    db_pool=Depends(get_db_pool),
):
    """Soft-delete: marks integration as 'disconnected'. Credentials stay encrypted in DB for audit."""
    async with db_pool.acquire() as conn:
        result = await conn.execute(
            "UPDATE integrations SET status = 'disconnected' "
            "WHERE id = $1 AND organization_id = $2 AND status = 'active'",
            integration_id,
            current_user.organization_id,
        )
    if result == "UPDATE 0":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Integration not found")
