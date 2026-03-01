"""
E2E tests for the Data Lake pipeline (Medallion Architecture).

Tests the DataLakeClient directly against Azurite — the local Azure Storage emulator.
Azurite is to Azure what LocalStack is to AWS: a Docker container that implements
the Azure Storage APIs locally so tests never hit the real cloud.

Medallion layers tested:
  Bronze → raw Parquet upload/read
  Parquet → schema validation, compression, row count

Requires: docker-compose up -d azurite
  (Azurite runs on localhost:10000)

Note: DataLakeClient uses the synchronous Azure SDK (not async).
These tests use regular 'def' functions — no async, no await, no pytest.mark.asyncio.
This is intentional: we test each layer as it actually runs.
"""
import json
import pytest
from datetime import datetime, timezone
from uuid import uuid4
from io import BytesIO

import pyarrow.parquet as pq
from azure.storage.filedatalake import DataLakeServiceClient
from azure.core.exceptions import ResourceExistsError

from src.infrastructure.data_lake_client import DataLakeClient

# Azurite uses a fixed well-known account name and key (same for everyone).
# In production this would be a real Azure Storage connection string or
# a DefaultAzureCredential (Managed Identity on Azure VMs).
AZURITE_CONNECTION_STRING = (
    "DefaultEndpointsProtocol=http;"
    "AccountName=devstoreaccount1;"
    "AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;"
    "BlobEndpoint=http://localhost:10000/devstoreaccount1;"
    "DfsEndpoint=http://localhost:10000/devstoreaccount1;"  # Required for azure-storage-file-datalake
)

# Enterprise transaction events — shape matches TransactionCreated event_data
# These represent software licensing and consulting expenses (B2B finance context)
_SAMPLE_EVENTS = [
    {
        "event_id": str(uuid4()),
        "event_type": "TransactionCreated",
        "aggregate_id": str(uuid4()),
        "aggregate_type": "Transaction",
        "event_data": json.dumps({
            "amount": "15000.00",
            "currency": "NOK",
            "merchant": "SAP AG",
            "category": "software",
        }),
        "created_at": datetime(2026, 2, 1, 12, 0, 0, tzinfo=timezone.utc),
        "tenant_id": "00000000-0000-0000-0000-000000000001",
    },
    {
        "event_id": str(uuid4()),
        "event_type": "TransactionCreated",
        "aggregate_id": str(uuid4()),
        "aggregate_type": "Transaction",
        "event_data": json.dumps({
            "amount": "87500.00",
            "currency": "NOK",
            "merchant": "Accenture Norge AS",
            "category": "consulting",
        }),
        "created_at": datetime(2026, 2, 1, 14, 30, 0, tzinfo=timezone.utc),
        "tenant_id": "00000000-0000-0000-0000-000000000001",
    },
]


@pytest.fixture(scope="module")
def data_lake_client():
    """
    Module-scoped DataLakeClient pointed at Azurite.

    scope="module" means this fixture is created ONCE for the entire test file,
    not once per test. Creating/connecting to Azurite takes ~100ms — module scope
    avoids that overhead for every test.

    Creates the bronze/silver/gold file systems (containers) before tests run.
    In production, these would be created by Terraform during infrastructure setup.
    """
    client = DataLakeClient(
        account_name="devstoreaccount1",
        connection_string=AZURITE_CONNECTION_STRING,
        api_version="2020-02-10",  # Pin to version Azurite v3 supports
    )

    # Create file systems (Azure "containers") — bronze, silver, gold
    # Using DataLakeServiceClient directly because DataLakeClient
    # doesn't expose create_file_system() — it only manages files
    service_client = DataLakeServiceClient.from_connection_string(
        AZURITE_CONNECTION_STRING,
        api_version="2020-02-10",
    )
    for fs_name in ["bronze", "silver", "gold"]:
        try:
            service_client.create_file_system(fs_name)
        except ResourceExistsError:
            pass  # Already exists from a previous test run — that's fine

    return client


class TestParquetConversion:
    """
    In-memory Parquet conversion — no Azurite needed.

    These tests call _convert_events_to_parquet() which:
      1. Builds a PyArrow table from the event list
      2. Writes it to an in-memory BytesIO buffer with Snappy compression
      3. Returns the raw bytes

    Why test in-memory first?
      If Parquet conversion is broken, no upload test can pass.
      Testing conversion in isolation pinpoints the failure faster.
    """

    def test_returns_valid_parquet_bytes(self, data_lake_client):
        """
        Every valid Parquet file starts with the magic bytes b'PAR1' and
        ends with b'PAR1'. This is part of the Parquet file format specification.
        Checking the header is the fastest way to verify output is real Parquet.
        """
        result = data_lake_client._convert_events_to_parquet(_SAMPLE_EVENTS)

        assert isinstance(result, bytes)
        assert result[:4] == b"PAR1"  # Parquet magic header (format spec)

    def test_row_count_matches_input(self, data_lake_client):
        """
        All events must be present in the Parquet output — no silent data loss.
        pq.read_table() reads back from in-memory bytes, no file system needed.
        """
        parquet_bytes = data_lake_client._convert_events_to_parquet(_SAMPLE_EVENTS)

        # BytesIO wraps raw bytes as a file-like object — pq.read_table accepts it
        table = pq.read_table(BytesIO(parquet_bytes))

        assert table.num_rows == len(_SAMPLE_EVENTS)

    def test_schema_contains_enterprise_metadata_columns(self, data_lake_client):
        """
        The Parquet schema must include all columns needed by downstream analytics.

        In the Medallion Architecture:
          - Spark/Databricks reads Bronze Parquet files by column name
          - Missing a column = Spark job crashes silently or produces wrong results
          - tenant_id is required for multi-tenant query isolation
          - created_at enables time-series partitioning queries
        """
        parquet_bytes = data_lake_client._convert_events_to_parquet(_SAMPLE_EVENTS)
        table = pq.read_table(BytesIO(parquet_bytes))

        column_names = table.schema.names
        assert "event_id" in column_names
        assert "event_type" in column_names
        assert "aggregate_id" in column_names
        assert "aggregate_type" in column_names
        assert "tenant_id" in column_names
        assert "event_data" in column_names   # stored as JSON string in Bronze
        assert "created_at" in column_names

    def test_event_data_stored_as_json_string(self, data_lake_client):
        """
        In Bronze, event_data is stored as a raw JSON string (not parsed).

        Why not parse it in Bronze?
          Bronze = raw, immutable. Never interpret data at this layer.
          The Silver transform parses and flattens it into typed columns.
          If parsing logic changes, you can reprocess from Bronze without re-reading Kafka.
        """
        parquet_bytes = data_lake_client._convert_events_to_parquet(_SAMPLE_EVENTS)
        table = pq.read_table(BytesIO(parquet_bytes))

        first_event_data = table.column("event_data")[0].as_py()

        # Must be a JSON string, not a dict — Bronze never interprets data
        assert isinstance(first_event_data, str)
        parsed = json.loads(first_event_data)   # must parse without error
        assert "amount" in parsed or "merchant" in parsed


class TestBronzeLayer:
    """
    Tests Bronze upload logic using mocks — no live Azurite calls.

    WHY MOCKS HERE?
    Azurite's ADLS Gen2 create_directory() requires Hierarchical Namespace
    (HNS) enabled at the account level — only configurable in real Azure,
    not via connection string. This is a known Azurite limitation.

    The strategy: mock the Azure SDK boundary.
      - We test OUR code (path construction, call sequence, data passed)
      - We trust the Azure SDK to handle actual HTTP calls correctly
      - Microsoft tests the SDK — we test our usage of it

    This is how DNB, Equinor, and Aker BP test Azure integrations in CI.
    Real Azure calls only happen in staging pipelines.
    """

    def test_upload_constructs_hive_partition_path(self, data_lake_client):
        """
        upload_event_to_bronze() must construct a Hive-style partition path:
          year=YYYY/month=MM/day=DD/{timestamp}.parquet

        Why this matters: Spark/Databricks uses this exact format for
        partition elimination — filtering by date skips unrelated partitions.
        A wrong format (e.g., YYYY/MM/DD/) = full table scan = slow analytics.
        """
        from unittest.mock import MagicMock

        mock_fs = MagicMock()
        mock_dir = MagicMock()
        mock_file = MagicMock()
        mock_fs.get_directory_client.return_value = mock_dir
        mock_dir.get_file_client.return_value = mock_file

        # Patch the service client so no real HTTP call is made
        data_lake_client._service_client.get_file_system_client = lambda name: mock_fs

        path = data_lake_client.upload_event_to_bronze(
            event_id=uuid4(),
            event_type="TransactionCreated",
            events=_SAMPLE_EVENTS,
            partition_date=datetime(2026, 2, 5, tzinfo=timezone.utc),
        )

        assert "year=2026" in path    # Hive year partition
        assert "month=02" in path     # zero-padded month
        assert "day=05" in path       # zero-padded day
        assert path.endswith(".parquet")

    def test_upload_sends_parquet_bytes_to_azure(self, data_lake_client):
        """
        upload_event_to_bronze() must call Azure SDK in this order:
          1. get_directory_client(partition_path) — locate folder
          2. create_directory()                   — ensure it exists
          3. get_file_client(filename)            — target the file
          4. upload_data(parquet_bytes)           — write data

        Also verifies the bytes passed to upload_data() are valid Parquet
        (start with magic header b'PAR1'), not empty or uncompressed JSON.
        """
        from unittest.mock import MagicMock

        mock_fs = MagicMock()
        mock_dir = MagicMock()
        mock_file = MagicMock()
        mock_fs.get_directory_client.return_value = mock_dir
        mock_dir.get_file_client.return_value = mock_file

        data_lake_client._service_client.get_file_system_client = lambda name: mock_fs

        data_lake_client.upload_event_to_bronze(
            event_id=uuid4(),
            event_type="TransactionCreated",
            events=_SAMPLE_EVENTS,
            partition_date=datetime(2026, 2, 5, tzinfo=timezone.utc),
        )

        # Verify the Azure SDK call sequence ran
        mock_dir.create_directory.assert_called_once()
        mock_file.upload_data.assert_called_once()

        # Verify actual Parquet bytes were passed — not None, not JSON string
        uploaded_bytes = mock_file.upload_data.call_args[0][0]
        assert uploaded_bytes[:4] == b"PAR1"  # valid Parquet magic header
