import json
import uuid
from datetime import date, timedelta
from typing import Optional
from uuid import UUID

from src.erp import get_adapter
from src.services.credentials import decrypt_credentials


class SyncResult:
    def __init__(self, integration_id: UUID, records_synced: int, errors: list[str]):
        self.integration_id = integration_id
        self.records_synced = records_synced
        self.errors = errors
        self.success = len(errors) == 0


class SyncService:

    def __init__(self, db_pool):
        self._db_pool = db_pool

    async def run_sync(self, integration_id: UUID, org_id: UUID) -> SyncResult:
        async with self._db_pool.acquire() as conn:
            integration = await self._load_integration(conn, integration_id, org_id)
            if not integration:
                return SyncResult(integration_id, 0, [f"Integration {integration_id} not found"])

            sync_job_id = await self._create_sync_job(conn, integration_id)

            try:
                records, errors = await self._do_sync(integration)
                await self._complete_sync_job(conn, sync_job_id, len(records), errors)
                return SyncResult(integration_id, len(records), errors)

            except Exception as exc:
                await self._fail_sync_job(conn, sync_job_id, str(exc))
                return SyncResult(integration_id, 0, [str(exc)])

    async def _load_integration(self, conn, integration_id: UUID, org_id: UUID) -> Optional[dict]:
        row = await conn.fetchrow(
            "SELECT id, provider, encrypted_credentials, last_synced_at "
            "FROM integrations WHERE id = $1 AND organization_id = $2 AND status = 'active'",
            integration_id,
            org_id,
        )
        if not row:
            return None
        return dict(row)

    async def _create_sync_job(self, conn, integration_id: UUID) -> UUID:
        job_id = uuid.uuid4()
        await conn.execute(
            "INSERT INTO sync_jobs (id, integration_id, status, started_at) "
            "VALUES ($1, $2, 'running', NOW())",
            job_id,
            integration_id,
        )
        return job_id

    async def _do_sync(self, integration: dict) -> tuple[list, list[str]]:
        provider = integration["provider"]
        credentials = decrypt_credentials(integration["encrypted_credentials"])

        since_date = (
            integration["last_synced_at"].date().isoformat()
            if integration["last_synced_at"]
            else (date.today() - timedelta(days=30)).isoformat()
        )

        adapter_class = get_adapter(provider)
        adapter = adapter_class(**credentials)

        invoices = await adapter.pull_invoices(since_date)

        errors = []
        synced = []
        for invoice in invoices:
            try:
                synced.append(invoice)
            except Exception as exc:
                errors.append(f"Invoice {invoice.external_id}: {exc}")

        return synced, errors

    async def _complete_sync_job(self, conn, job_id: UUID, records: int, errors: list[str]):
        status = "completed" if not errors else "completed_with_errors"
        await conn.execute(
            "UPDATE sync_jobs SET status = $1, records_processed = $2, "
            "error_details = $3, finished_at = NOW() WHERE id = $4",
            status,
            records,
            json.dumps(errors) if errors else None,
            job_id,
        )

    async def _fail_sync_job(self, conn, job_id: UUID, error: str):
        await conn.execute(
            "UPDATE sync_jobs SET status = 'failed', error_details = $1, finished_at = NOW() "
            "WHERE id = $2",
            json.dumps([error]),
            job_id,
        )
