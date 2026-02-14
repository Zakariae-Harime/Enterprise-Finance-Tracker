"""
Integration tests for DataLakeClient against Azurite (local Azure emulator).
REQUIRES: docker-compose up -d azurite

Strategy:
  - Production code uses DataLakeServiceClient (DFS API) — the real Azure Data Lake SDK
  - Tests use BlobServiceClient (Blob API) — fully supported by Azurite
  - Both APIs access the SAME underlying storage in Azure, so this is valid
  - We test Parquet conversion + path logic using the Blob SDK
"""
import pytest
import json
import pyarrow.parquet as pq
from io import BytesIO
from datetime import datetime, timezone
from uuid import uuid4

from azure.storage.blob import BlobServiceClient
from src.infrastructure.data_lake_client import DataLakeClient

# Azurite's default connection string (same for everyone, not a secret)
AZURITE_CONNECTION_STRING = (
    "DefaultEndpointsProtocol=http;"
    "AccountName=devstoreaccount1;"
    "AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;"
    "BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1;"
)


@pytest.fixture
def blob_client():
    """BlobServiceClient for direct Azurite access (test infrastructure)."""
    return BlobServiceClient.from_connection_string(
        AZURITE_CONNECTION_STRING,
        api_version="2024-11-04"
    )


@pytest.fixture
def bronze_container(blob_client):
    """Create Bronze container in Azurite, clean up after test."""
    container = blob_client.create_container("bronze")
    yield container
    blob_client.delete_container("bronze")


@pytest.fixture
def lake_client():
    """Create DataLakeClient for testing Parquet conversion logic."""
    return DataLakeClient(
        account_name="devstoreaccount1",
        connection_string=AZURITE_CONNECTION_STRING,
        api_version="2024-11-04"
    )


@pytest.fixture
def sample_events():
    """Sample Norwegian finance events for testing."""
    return [
        {
            "event_id": str(uuid4()),
            "event_type": "AccountCreated",
            "aggregate_id": str(uuid4()),
            "aggregate_type": "account",
            "event_data": {"account_name": "Sparekonto", "currency": "NOK"},
            "created_at": datetime.now(timezone.utc),
            "tenant_id": "tenant-001"
        },
        {
            "event_id": str(uuid4()),
            "event_type": "AccountCreated",
            "aggregate_id": str(uuid4()),
            "aggregate_type": "account",
            "event_data": {"account_name": "Brukskonto", "currency": "NOK"},
            "created_at": datetime.now(timezone.utc),
            "tenant_id": "tenant-001"
        }
    ]


class TestAzuriteConnection:
    """Test basic Azurite connectivity using Blob SDK."""

    def test_can_create_container(self, blob_client):
        """Verify we can create and delete a container in Azurite."""
        container = blob_client.create_container("test-connection")
        assert container is not None
        blob_client.delete_container("test-connection")

    def test_can_upload_and_download_blob(self, blob_client, bronze_container):
        """Verify basic upload/download roundtrip works."""
        bronze_container.upload_blob("test.txt", b"hello azurite")
        download = bronze_container.download_blob("test.txt")
        content = download.readall()
        assert content == b"hello azurite"


class TestParquetConversion:
    """Test the Parquet conversion logic (core of DataLakeClient)."""

    def test_convert_events_to_parquet(self, lake_client, sample_events):
        """Verify events are correctly converted to Parquet bytes."""
        # ACT - call the internal conversion method
        parquet_bytes = lake_client._convert_events_to_parquet(sample_events)

        # ASSERT - bytes start with PAR1 magic number (Parquet file signature)
        assert parquet_bytes[:4] == b"PAR1"

        # ASSERT - can read back as a PyArrow table
        table = pq.read_table(BytesIO(parquet_bytes))
        assert table.num_rows == 2
        assert "event_id" in table.column_names
        assert "event_type" in table.column_names
        assert "event_data" in table.column_names

    def test_parquet_preserves_event_data_as_json(self, lake_client, sample_events):
        """event_data dict should be serialized as JSON string in Parquet."""
        parquet_bytes = lake_client._convert_events_to_parquet(sample_events)
        table = pq.read_table(BytesIO(parquet_bytes))

        # event_data column should contain JSON strings, not dicts
        for row in table.to_pylist():
            event_data = json.loads(row["event_data"])
            assert "currency" in event_data
            assert event_data["currency"] == "NOK"

    def test_parquet_schema_has_all_columns(self, lake_client, sample_events):
        """Verify the Parquet schema includes all expected columns."""
        parquet_bytes = lake_client._convert_events_to_parquet(sample_events)
        table = pq.read_table(BytesIO(parquet_bytes))

        expected_columns = [
            "event_id", "event_type", "aggregate_id",
            "aggregate_type", "event_data", "created_at", "tenant_id"
        ]
        for col in expected_columns:
            assert col in table.column_names, f"Missing column: {col}"


class TestBronzeLayerRoundtrip:
    """Test full upload → download roundtrip via Blob SDK + Parquet."""

    def test_upload_parquet_and_read_back(self, lake_client, blob_client, bronze_container, sample_events):
        """Upload Parquet to Azurite and read it back — full data roundtrip."""
        # ARRANGE - convert events to Parquet (using DataLakeClient logic)
        fixed_date = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        parquet_bytes = lake_client._convert_events_to_parquet(sample_events)

        # Build the same path structure that upload_event_to_bronze would create
        blob_path = "year=2024/month=06/day=15/20240615_120000.parquet"

        # ACT - upload via Blob SDK (what Azurite supports)
        bronze_container.upload_blob(name=blob_path, data=parquet_bytes)

        # ACT - download and parse
        download = bronze_container.download_blob(blob_path)
        content = download.readall()
        table = pq.read_table(BytesIO(content))
        retrieved = table.to_pylist()

        # ASSERT - data survived the full roundtrip
        assert len(retrieved) == 2
        assert all(r["event_type"] == "AccountCreated" for r in retrieved)

    def test_list_blobs_by_partition(self, blob_client, bronze_container, lake_client, sample_events):
        """Verify Hive-style partition paths work with blob prefix listing."""
        parquet_bytes = lake_client._convert_events_to_parquet(sample_events)

        # Upload to multiple partitions (different days)
        bronze_container.upload_blob("year=2024/month=03/day=01/batch1.parquet", parquet_bytes)
        bronze_container.upload_blob("year=2024/month=03/day=02/batch2.parquet", parquet_bytes)
        bronze_container.upload_blob("year=2024/month=04/day=01/batch3.parquet", parquet_bytes)

        # ACT - list only March 2024 partition
        march_blobs = list(bronze_container.list_blobs(
            name_starts_with="year=2024/month=03"
        ))

        # ASSERT - only March files returned, not April
        assert len(march_blobs) == 2
        assert all("month=03" in b.name for b in march_blobs)

    def test_event_data_roundtrip_preserves_norwegian_data(self, lake_client, blob_client, bronze_container):
        """Norwegian merchant/account names survive Parquet serialization."""
        # ARRANGE - events with Norwegian characters
        norwegian_events = [
            {
                "event_id": str(uuid4()),
                "event_type": "TransactionCreated",
                "aggregate_id": str(uuid4()),
                "aggregate_type": "transaction",
                "event_data": {
                    "merchant": "Rema 1000 Grønland",
                    "category": "dagligvarer",
                    "amount": 459.90,
                    "currency": "NOK"
                },
                "created_at": datetime.now(timezone.utc),
                "tenant_id": "tenant-001"
            }
        ]

        # ACT - convert → upload → download → parse
        parquet_bytes = lake_client._convert_events_to_parquet(norwegian_events)
        bronze_container.upload_blob("norwegian_test.parquet", parquet_bytes)
        content = bronze_container.download_blob("norwegian_test.parquet").readall()
        table = pq.read_table(BytesIO(content))
        retrieved = table.to_pylist()

        # ASSERT - Norwegian characters preserved
        event_data = json.loads(retrieved[0]["event_data"])
        assert event_data["merchant"] == "Rema 1000 Grønland"
        assert event_data["category"] == "dagligvarer"
        assert event_data["amount"] == 459.90
