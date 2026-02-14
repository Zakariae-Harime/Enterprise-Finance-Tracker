"""
  Azure Data Lake Storage Gen2 Client

  Provides methods to:
    - Upload events to Bronze layer (raw data)
    - Read/write Parquet files (columnar, compressed)
    - Organize data by date partitions

  SDK: azure-storage-file-datalake
  Docs: https://learn.microsoft.com/en-us/python/api/overview/azure/storage-file-datalake-readme

  Authentication options:
    1. Connection string (simple, for development)
    2. DefaultAzureCredential (recommended for production)
    3. Service Principal (client_id, client_secret, tenant_id)
"""
import io
import json
from typing import Optional, List
from datetime import datetime, timezone
from uuid import UUID
import pyarrow as pa # Create/read Parquet files (columnar format)
import pyarrow.parquet as pq
from io import BytesIO # In-memory file buffer (avoid disk I/O)
# Azure SDK imports
from azure.identity import DefaultAzureCredential # Auto-detects authentication method 
from azure.storage.filedatalake import( DataLakeServiceClient, #Top-level client for the storage account
FileSystemClient, #Client for a container (bronze, silver, gold)
DataLakeFileClient,DataLakeDirectoryClient)
class DataLakeClient:
    """
    Client for Azure Data Lake Storage Gen2.

    Organizes data using the Medallion Architecture:
      - Bronze: Raw events (as received from Kafka)
      - Silver: Cleaned, validated, deduplicated
      - Gold: Aggregated, ready for analytics

    File format: Parquet (columnar, compressed, fast queries)
    Partitioning: By date (year/month/day) for efficient queries
    """

    def __init__(
        self,
        account_name: str,
        credential: Optional[DefaultAzureCredential] = None,
        connection_string: Optional[str] = None,
        api_version: Optional[str] = None
    ):
        """
        Initialize Data Lake client.

        Args:
            account_name: Azure storage account name (e.g., 'financetrackersa')
            credential: Azure credential (DefaultAzureCredential for production)
            connection_string: Connection string (for development only!)
            api_version: Pin API version (needed for Azurite compatibility)

        Authentication priority:
            1. Connection string (if provided)
            2. DefaultAzureCredential (tries multiple methods automatically)
        """
        self.account_name = account_name
        if connection_string:
            # Development: Use connection string
            # api_version lets us pin to a version Azurite supports
            self._service_client = DataLakeServiceClient.from_connection_string(
                connection_string,
                **({"api_version": api_version} if api_version else {})
            )
        else:
            # Production: Use DefaultAzureCredential
            # This tries (in order):
            #   1. Environment variables (AZURE_CLIENT_ID, etc.)
            #   2. Managed Identity (when running in Azure)
            #   3. Azure CLI credentials (az login)
            #   4. Visual Studio Code credentials
            account_url = f"https://{account_name}.dfs.core.windows.net"
            self._service_client = DataLakeServiceClient(
                account_url=account_url,
                credential=credential or DefaultAzureCredential()
            )

    def get_filesystem_client(self, filesystem_name: str) -> FileSystemClient:
        """
        Get client for a specific container (filesystem).

          Args:
              filesystem_name: Container name ('bronze', 'silver', 'gold')

          Returns:
              FileSystemClient for the container

          Note: Container must exist. Create in Azure Portal or with:
                self._service_client.create_file_system(filesystem_name)
          """
        return self._service_client.get_file_system_client(filesystem_name)
    def upload_event_to_bronze(
        self,
        event_id: UUID,
        event_type: str,
        events: List[dict],
        partition_date: Optional[datetime] = None) -> str:
        """
        Upload raw events to Bronze layer as Parquet file.

        Path structure:
          bronze/events/{event_type}/year=2024/month=01/day=15/{timestamp}.parquet

        Why this structure?
          - Partitioned by date: Fast queries for specific time ranges
          - Hive-compatible: Works with Spark, Databricks, etc.
          - Event type separation: Easy to process specific events

        Args:
            events: List of event dictionaries
            event_type: Type like 'AccountCreated', 'TransactionCreated'
            partition_date: Date for partitioning (default: now)

        Returns:
            Path where file was uploaded
        """
        # use current datetime if not provided
        partition_date = partition_date or datetime.now(timezone.utc)
        # Build partition path (Hive-style partitioning)
        partition_path = f"year={partition_date.year}/month={partition_date.month:02d}/day={partition_date.day:02d}"
        # Generate unique filename with timestamp
        timestamp = partition_date.strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}.parquet"
        full_path = f"{partition_path}/{filename}"
        # Convert events to Parquet format
        parquet_bytes = self._convert_events_to_parquet(events)
        # Upload to Azure Data Lake
        bronze_fs = self.get_filesystem_client("bronze")
        # Create directory if it doesn't exist
        directory_client = bronze_fs.get_directory_client(partition_path)
        directory_client.create_directory()
        # Upload file
        file_client = directory_client.get_file_client(filename)
        file_client.upload_data(parquet_bytes, overwrite=True)
        print(f"Uploaded {len(events)} events to bronze/{full_path}")
        return full_path
    def _convert_events_to_parquet(self, events: List[dict]) -> bytes:
        """
        Convert list of events to Parquet format.

        Why Parquet?
          - Columnar: Read only columns you need (fast!)
          - Compressed: ~10x smaller than JSON
          - Typed: Schema enforced, no surprises
          - Industry standard: Works everywhere

        Args:
            events: List of event dictionaries

        Returns:
            Parquet file as bytes (ready for upload)
        """
        # Define schema (explicit types for safety)
        schema = pa.schema([
            ("event_id", pa.string()),
            ("event_type", pa.string()),
            ("aggregate_id", pa.string()),
            ("aggregate_type", pa.string()),
            ("event_data", pa.string()), # Store event_data as JSON string
            ("created_at", pa.timestamp("ms")),
            ("tenant_id", pa.string())
        ])
        # Convert events to PyArrow table
        arrays = {
            "event_id": [str(e.get("event_id")) for e in events],
            "event_type": [e.get("event_type") for e in events],
            "aggregate_id": [str(e.get("aggregate_id")) for e in events],
            "aggregate_type": [e.get("aggregate_type") for e in events],
            "event_data": [json.dumps(e.get("event_data")) for e in events],
            "created_at": [e.get("created_at") for e in events],
            "tenant_id": [str(e.get("tenant_id", "")) for e in events]
        }
        table = pa.table(arrays, schema=schema)
        # write to in-memory buffer
        # Step 1: Create empty in-memory buffer
        buffer = BytesIO() # This is like an in-memory file (no disk I/O)
        # Step 2: Write Parquet data to buffer with compression
        pq.write_table(
            table, # PyArrow table with our data
            buffer,
            compression='snappy'  # Fast compression, good ratio
        )
        return buffer.getvalue()
      # Returns the raw bytes: b'PAR1\x00\x00...'
      # These bytes can be uploaded directly to Azure
    def read_events_from_bronze(self, event_type: str, year: int, month: int, day: Optional[int] = None) -> List[dict]:
        """
        Read events from Bronze layer.

        Args:
            event_type: Type of events to read
            year: Year partition
            month: Month partition
            day: Day partition (optional - reads whole month if None)

        Returns:
            List of event dictionaries
        """
        # Build partition path
        bronze_fs = self.get_filesystem_client("bronze")
        if day:
            partition_path = f"year={year}/month={month:02d}/day={day:02d}"
        else:
            partition_path = f"year={year}/month={month:02d}"
        paths = bronze_fs.get_paths(path=partition_path, recursive=True)
        all_events = []
        for path_item in paths:
            if path_item.name.endswith(".parquet"):
              # Download and read parquet file
             file_client = bronze_fs.get_file_client(path_item.name)
             download = file_client.download_file()
             content=download.readall()
             #read parquet from in-memory bytes
             table = pq.read_table(BytesIO(content))
             # Convert to list of dicts
             for row in table.to_pylist():
                 all_events.append(row)
        return all_events
 # buffer is automatically garbage collected! No need to close or delete it.
    def transform_bronze_to_silver(self, event_type: str, year: int, month: int, day: Optional[int] = None) -> str:
            """
             Transform raw Bronze data to cleaned Silver data.

          Transformations:
            1. Deduplicate by event_id
            2. Parse event_data JSON into columns
            3. Validate required fields
            4. Add processing metadata

          Args:
              event_type: Type of events to process
              year, month, day: Partition to process

          Returns:
              Path where Silver data was written
          """
            # Step 1: Read from Bronze
            bronze_fs = self.get_filesystem_client("bronze")
            bronze_path= f"events/{event_type}/year={year}/month={month:02d}/day={day:02d}"
            #collect all events from bronze
            all_events = []
            paths = bronze_fs.get_paths(path=bronze_path, recursive=True)
            for path_item in paths:
                if path_item.name.endswith(".parquet"):
                    file_client = bronze_fs.get_file_client(path_item.name)
                    download = file_client.download_file()
                    content=download.readall()
                    table = pq.read_table(BytesIO(content))
                    for row in table.to_pylist():
                        all_events.append(row)
            if not all_events:
                print(f"No events found in {bronze_path}")
                return ""
            # transforme deduplicate events by event_id
            seen_ids= set()
            unique_events = []
            for event in all_events:
                if event["event_id"] not in seen_ids:
                    seen_ids.add(event["event_id"])
                    unique_events.append(event)
            duplicates_removed = len(all_events) - len(unique_events) 
            print(f"Deduplicated {duplicates_removed} events")
            # Tranforme : Parse event_data JSON
            enriched_events = []
            for event in unique_events:
                # Parse the JSON string into dict
                event_data = json.loads(event.get('event_data', '{}'))
                # Flatten into columns
                enriched = {
                    'event_id': event['event_id'],
                    'event_type': event['event_type'],
                    'aggregate_id': event.get('aggregate_id'),
                    'tenant_id': event.get('tenant_id'),
                    'created_at': event.get('created_at'),
                    # Flattened from event_data
                    'account_name': event_data.get('account_name'),
                    'amount': event_data.get('amount'),
                    'currency': event_data.get('currency'),
                    'category': event_data.get('category'),
                    # Processing metadata
                    'processed_at': datetime.now(timezone.utc).isoformat(),
                    'source_file': bronze_path,
                }
                enriched_events.append(enriched)
                # Write to Silver layer
                silver_fs = self.get_filesystem_client("silver")
                silver_path = f"events/{event_type}/year={year}/month={month:02d}/day={day:02d}/silver_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
                filename = f"silver_{year}{month:02d}{day:02d}_{datetime.now().strftime('%H%M%S')}.parquet"
                # Create Silver schema (enriched columns)
                schema = pa.schema([
                    ("event_id", pa.string()),
                    ("event_type", pa.string()),
                    ("aggregate_id", pa.string()),
                    ("tenant_id", pa.string()),
                    ("created_at", pa.timestamp("ms")),
                    # Flattened fields from event_data
                    ("account_name", pa.string()),
                    ("amount", pa.float64()),
                    ("currency", pa.string()),
                    ("category", pa.string()),
                    # Processing metadata
                    ("processed_at", pa.timestamp("ms")),
                    ("source_file", pa.string())
                ])
                # Convert to Parquet
                arrays = {col.name: [e.get(col.name) for e in enriched_events] for col in schema}
                table = pa.table(arrays, schema=schema)

                buffer = BytesIO()
                pq.write_table(table, buffer, compression='snappy')
                 # Upload to Silver
                directory_client = silver_fs.get_directory_client(silver_path)
                try:
                    directory_client.create_directory()
                except:
                    pass

                file_client = directory_client.get_file_client(filename)
                file_client.upload_data(buffer.getvalue(), overwrite=True)

                full_path = f"{silver_path}/{filename}"
                print(f"Silver data written: {len(enriched_events)} events to silver/{full_path}")
                return full_path

    #Gold layer transformations would go here (aggregations, etc.)
    def transform_silver_to_gold(self, event_type: str, year: int, month: int) -> str:
        """
          Aggregate Silver data into Gold analytics tables.

          Creates:
            - Monthly spending by category
            - Transaction counts
            - Average transaction amounts

          Args:
              year, month: Month to aggregate

          Returns:
              Path where Gold data was written
        """
        import pandas as pd  # Use pandas for aggregations

          # Read all Silver data for the month
        silver_fs = self.get_filesystem_client("silver")
        silver_path = f"TransactionCreated/year={year}/month={month:02d}"

        all_data = []
        try:
              paths = silver_fs.get_paths(path=silver_path, recursive=True)
              for path_item in paths:
                  if path_item.name.endswith('.parquet'):
                      file_client = silver_fs.get_file_client(path_item.name)
                      content = file_client.download_file().readall()
                      table = pq.read_table(BytesIO(content))
                      all_data.extend(table.to_pylist())
        except Exception as e:
            print(f"No silver data found: {e}")
            return ""

        if not all_data:
            return ""
        # Convert to pandas DataFrame for easy aggregation
        df = pd.DataFrame(all_data)
        # Ensure amount is numeric
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)
        # Monthly spending by category aggregation
        summary = df.groupby('category').agg({
        'amount': ['sum', 'mean', 'count'],
        'event_id': 'nunique'  # Unique transactions
         }).reset_index()
        # Flatten column names 
        summary.columns = ['category', 'total_amount', 'avg_amount', 'transaction_count', 'unique_events']
        # Add metadata
        summary['year'] = year
        summary['month'] = month
        summary['generated_at'] = datetime.now(timezone.utc).isoformat()

        # Write to Gold layer
        gold_fs = self.get_filesystem_client("gold")
        gold_path = f"spending_by_category/year={year}/month={month:02d}"
        filename = f"monthly_summary.parquet"

        # Convert to PyArrow and write
        table = pa.Table.from_pandas(summary)
        buffer = BytesIO()
        pq.write_table(table, buffer, compression='snappy')

        directory_client = gold_fs.get_directory_client(gold_path)
        try:
            directory_client.create_directory()
        except:
              pass

        file_client = directory_client.get_file_client(filename)
        file_client.upload_data(buffer.getvalue(), overwrite=True)

        full_path = f"{gold_path}/{filename}"
        print(f"🥇 Gold data written: {len(summary)} categories to gold/{full_path}")
        return full_path    