"""
Azure ML job submitter — run this from your laptop.

This script submits finetune_bert.py to run on Azure ML GPU compute.
YOUR LAPTOP DOES NOT DO THE TRAINING.
You just send the job spec (~2KB) to Azure, then watch logs in browser.

Cost: Standard_NC4as_T4_v3 (T4 GPU) = $0.52/hour × ~2 hours = ~$1.04
      Well within your $10 budget. You can run this 9 times and still be under budget.

SETUP REQUIRED (one time, ~10 minutes):
────────────────────────────────────────
1. Create Azure ML Workspace in Azure portal:
     portal.azure.com → Create Resource → Machine Learning
     Name: "finance-tracker-ml"
     Resource Group: "finance-tracker-rg"
     Region: "norwayeast" (lowest latency from Norway)

2. Install Azure CLI + ML extension:
     winget install Microsoft.AzureCLI
     az extension add -n ml
     az login

3. Set your config in environment variables (or .env file):
     AZURE_SUBSCRIPTION_ID=your-subscription-id
     AZURE_RESOURCE_GROUP=finance-tracker-rg
     AZURE_ML_WORKSPACE=finance-tracker-ml

4. Run dataset preparation first:
     python -m src.ml.data.prepare_dataset --n-per-category 500

5. Run this script:
     python -m src.ml.train.azure_ml_job

6. Watch training in browser:
     https://ml.azure.com → Experiments → finance-tracker-categorization

CONCEPTS EXPLAINED:
────────────────────

COMPUTE CLUSTER:
  A pool of Azure VMs (virtual machines) that Azure manages for you.
  When idle: 0 VMs running → $0/hour (cluster "scales to zero")
  When job submitted: Azure provisions 1 VM → training runs → VM deallocated
  You pay ONLY for the time the VM is running your job.
  Like hiring a freelancer for 2 hours vs keeping an employee full-time.

ENVIRONMENT:
  A Docker container image with all Python dependencies pre-installed.
  We specify pip packages (transformers, torch, etc.) and Azure ML
  builds a Docker image. The training job runs inside this container
  on the GPU VM — completely isolated from your laptop's environment.

DATA ASSET:
  Our Parquet files (train/val/test) registered in Azure ML as a named dataset.
  Instead of uploading files every time, we reference the registered asset.
  Version-controlled: "data version 1" → first run, "data version 2" → after adding GoCardless data.
  This is how you trace "which data produced which model" (data lineage).

COMMAND JOB:
  The actual job definition. Specifies:
    - What script to run (finetune_bert.py)
    - What arguments to pass
    - What compute to use (T4 GPU cluster)
    - What environment (Docker image with dependencies)
    - What data to mount (our Parquet files)
"""
import os
from pathlib import Path

from azure.ai.ml import Input, MLClient, command
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import (
    AmlCompute,           # GPU compute cluster definition
    BuildContext,         # How to build the Docker environment
    Environment,          # Python/CUDA environment for training
)
from azure.identity import DefaultAzureCredential


# ─── Azure Configuration ───────────────────────────────────────────────────────
# Read from environment variables — never hardcode credentials in code!
# Set these before running: export AZURE_SUBSCRIPTION_ID=...
# Or add to .env and load with python-dotenv

SUBSCRIPTION_ID  = os.environ.get("AZURE_SUBSCRIPTION_ID", "")
RESOURCE_GROUP   = os.environ.get("AZURE_RESOURCE_GROUP", "finance-tracker-rg")
WORKSPACE_NAME   = os.environ.get("AZURE_ML_WORKSPACE", "finance-tracker-ml")

# GPU compute cluster settings
COMPUTE_NAME     = "gpu-cluster-t4"
COMPUTE_SIZE     = "Standard_NC4as_T4_v3"  # T4 GPU, 4 vCPUs, 28GB RAM, $0.52/hour
MIN_NODES        = 0    # Scale to 0 when idle → $0 cost when not training
MAX_NODES        = 1    # Only need 1 node for our dataset size

# Project root for locating training script
PROJECT_ROOT = Path(__file__).parents[3]


# ─── Azure ML Client ──────────────────────────────────────────────────────────

def get_ml_client() -> MLClient:
    """
    Create authenticated Azure ML client.

    DefaultAzureCredential tries multiple auth methods in order:
      1. Environment variables (AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, AZURE_TENANT_ID)
      2. Workload Identity (for Azure-hosted apps)
      3. Managed Identity
      4. Azure CLI credentials (az login) ← this is what we use locally
      5. Visual Studio Code credentials
      6. Azure PowerShell credentials

    Since you ran "az login" during setup, method #4 will succeed automatically.
    """
    if not SUBSCRIPTION_ID:
        raise ValueError(
            "AZURE_SUBSCRIPTION_ID environment variable not set.\n"
            "Set it with: export AZURE_SUBSCRIPTION_ID=your-subscription-id\n"
            "Find it at: portal.azure.com → Subscriptions"
        )

    credential = DefaultAzureCredential()
    client = MLClient(
        credential=credential,
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME,
    )
    print(f"[azure_ml] Connected to workspace: {WORKSPACE_NAME}")
    return client


# ─── Compute Cluster ──────────────────────────────────────────────────────────

def get_or_create_compute(client: MLClient) -> str:
    """
    Get existing compute cluster or create a new T4 GPU cluster.

    Cluster creation takes ~3 minutes (first time only).
    Subsequent calls reuse the existing cluster immediately.

    T4 GPU specs:
      - 16GB GDDR6 VRAM (plenty for BERT fine-tuning with batch_size=16)
      - ~65 TFLOPS FP16 (much faster than V100 for small models)
      - Turing architecture (2018) — mature, stable, well-supported

    Returns:
        Compute cluster name (same COMPUTE_NAME constant)
    """
    try:
        # Check if cluster already exists
        cluster = client.compute.get(COMPUTE_NAME)
        print(f"[azure_ml] Using existing compute: {COMPUTE_NAME} ({cluster.size})")
        return COMPUTE_NAME
    except Exception:
        pass  # Cluster doesn't exist yet → create it

    print(f"[azure_ml] Creating compute cluster: {COMPUTE_NAME} ({COMPUTE_SIZE})")
    print("[azure_ml] This takes ~3 minutes on first run...")

    cluster = AmlCompute(
        name=COMPUTE_NAME,
        type="amlcompute",
        size=COMPUTE_SIZE,
        min_instances=MIN_NODES,   # 0 = scales to zero when idle → $0/hour at rest
        max_instances=MAX_NODES,   # 1 = only spin up 1 GPU node per job
        idle_time_before_scale_down=120,  # Deallocate after 2min idle → saves money
    )

    poller = client.compute.begin_create_or_update(cluster)
    poller.result()  # Wait for creation to complete

    print(f"[azure_ml] Compute cluster ready: {COMPUTE_NAME}")
    return COMPUTE_NAME


# ─── Environment (Docker image) ────────────────────────────────────────────────

def get_or_create_environment(client: MLClient) -> str:
    """
    Create a Docker environment with all training dependencies.

    WHAT IS A DOCKER ENVIRONMENT?
    ──────────────────────────────
    Docker = a lightweight virtual machine that packages code + dependencies.
    When Azure runs finetune_bert.py, it runs inside this Docker container.
    The container is built from our specification (pip packages list).

    Base image: PyTorch 2.1 + CUDA 11.8 (GPU driver) pre-installed by Microsoft.
    We add our pip packages on top (transformers, scikit-learn, etc.).

    Azure ML caches the built image — you only rebuild when pip packages change.
    First build takes ~10 minutes. Subsequent runs reuse the cached image.

    Returns:
        Environment name:version string
    """
    env_name = "finance-tracker-training-env"
    env_version = "1.0.0"

    try:
        # Check if environment already exists
        env = client.environments.get(env_name, version=env_version)
        print(f"[azure_ml] Using existing environment: {env_name}:{env_version}")
        return f"{env_name}:{env_version}"
    except Exception:
        pass  # Environment doesn't exist yet → create it

    print(f"[azure_ml] Creating training environment: {env_name}:{env_version}")
    print("[azure_ml] Building Docker image (first time only, ~10 min)...")

    env = Environment(
        name=env_name,
        version=env_version,
        description="NB-BERT fine-tuning environment for Norwegian transaction categorization",

        # Base image: PyTorch 2.1 with CUDA 11.8 support (from Azure ML curated images)
        # This is maintained by Microsoft — security patches, CUDA updates included.
        image="mcr.microsoft.com/azureml/curated/acpt-pytorch-2.1-cuda11.8:latest",

        # Additional pip packages on top of the base image
        # These are NOT in the base image — we install them on first container build.
        conda_file={
            "name": "finance-tracker-env",
            "channels": ["conda-forge", "defaults"],
            "dependencies": [
                "python=3.11",
                "pip",
                {
                    "pip": [
                        "transformers>=4.36.0",
                        "datasets>=2.16.0",
                        "scikit-learn>=1.4.0",
                        "pyarrow>=15.0.0",
                        "onnx>=1.15.0",
                        "onnxruntime>=1.17.0",
                        "numpy>=1.24.0",
                        "pandas>=2.1.0",
                    ]
                },
            ],
        },
    )

    client.environments.create_or_update(env)
    print(f"[azure_ml] Environment created: {env_name}:{env_version}")
    return f"{env_name}:{env_version}"


# ─── Data Asset ────────────────────────────────────────────────────────────────

def register_data_asset(client: MLClient) -> Input:
    """
    Register local Parquet files as an Azure ML Data Asset.

    DATA ASSET = a versioned reference to data in Azure ML.
    Instead of: copy files to Azure every run (slow, duplicates)
    We do: register once → reference by name + version in all jobs

    This also creates an "data lineage" audit trail:
      Job run #1 → used data version 1 → produced model version 1
      Job run #2 → used data version 2 (added GoCardless) → produced model version 2

    The data gets uploaded to your Azure ML default datastore (ADLS Gen2).
    This is the SAME storage account as your Bronze/Silver/Gold layers.

    Returns:
        Input object pointing to the registered data asset in Azure
    """
    data_dir = PROJECT_ROOT / "data" / "ml"

    if not (data_dir / "train.parquet").exists():
        raise FileNotFoundError(
            f"Training data not found at {data_dir}\n"
            "Run first: python -m src.ml.data.prepare_dataset --n-per-category 500"
        )

    from azure.ai.ml.entities import Data
    from azure.ai.ml.constants import AssetTypes as AT

    data_asset = Data(
        name="transaction-categorization-dataset",
        version="1.0",
        description="Norwegian transaction categorization dataset — train/val/test Parquet splits",
        path=str(data_dir),             # Local path — Azure ML uploads this directory
        type=AT.URI_FOLDER,             # Type: a folder (not a single file)
    )

    # create_or_update: upload files if they don't exist, update metadata if they do
    registered_data = client.data.create_or_update(data_asset)
    print(f"[azure_ml] Data registered: {registered_data.name} v{registered_data.version}")
    print(f"[azure_ml] Uploaded to: {registered_data.path}")

    # Return an Input object — this is how the job references the data
    return Input(
        type=AssetTypes.URI_FOLDER,
        path=f"azureml:{registered_data.name}:{registered_data.version}",
    )


# ─── Submit Training Job ───────────────────────────────────────────────────────

def submit_training_job(client: MLClient, compute_name: str, env_name: str, data_input: Input) -> None:
    """
    Submit the BERT fine-tuning job to Azure ML.

    WHAT IS A COMMAND JOB?
    ──────────────────────
    A Command Job = "run this shell command on this compute with this environment."
    Like SSH-ing into a GPU machine and running: python finetune_bert.py --data-dir ...
    Except Azure manages the VM lifecycle and you never SSH manually.

    The job:
      1. Provisions a T4 GPU VM (Standard_NC4as_T4_v3)
      2. Pulls our Docker environment image
      3. Mounts our data asset at /data
      4. Runs: python finetune_bert.py --data-dir /data --output-dir /outputs
      5. Uploads /outputs/ (trained model files) to Azure ML storage
      6. Deallocates the VM

    You can monitor in Azure ML portal:
      https://ml.azure.com → Jobs → finance-tracker-categorization
      Live metrics, logs, GPU utilization — all visible in browser.
    """
    training_script = PROJECT_ROOT / "src" / "ml" / "train" / "finetune_bert.py"

    job = command(
        name="finance-tracker-bert-finetuning",
        display_name="NB-BERT Fine-tuning — Norwegian Transaction Categorization",
        description=(
            "Fine-tune NbAiLab/nb-bert-base on Norwegian transaction data. "
            "Target: weighted F1 > 0.95 across 7 expense categories."
        ),

        # The command to run (inside the Docker container, on the GPU VM)
        # ${{inputs.data}} = Azure ML substitutes the actual data path
        # ${{outputs.model}} = Azure ML substitutes the output path
        command=(
            "python src/ml/train/finetune_bert.py "
            "--data-dir ${{inputs.data}} "
            "--output-dir ${{outputs.model}}"
        ),

        # Code directory — Azure ML uploads this to the job
        # It uploads the entire src/ directory so finetune_bert.py can import our modules
        code=str(PROJECT_ROOT),

        # Inputs: our registered Parquet dataset
        inputs={"data": data_input},

        # Outputs: where trained model files go (auto-uploaded to Azure ML)
        outputs={
            "model": {
                "type": "uri_folder",
                "mode": "rw_mount",   # Read-write mount (training can write here)
            }
        },

        # Where to run
        compute=compute_name,

        # What Docker environment to use
        environment=env_name,

        # Azure ML experiment name (groups all runs together in the portal)
        experiment_name="finance-tracker-categorization",
    )

    # Submit job to Azure ML — this call returns immediately
    # Training continues on Azure VM even if you close your laptop
    submitted_job = client.jobs.create_or_update(job)

    print("\n" + "=" * 60)
    print("[azure_ml] Job submitted successfully!")
    print(f"[azure_ml] Job name: {submitted_job.name}")
    print(f"[azure_ml] Status: {submitted_job.status}")
    print(f"[azure_ml] Monitor at: {submitted_job.studio_url}")
    print("=" * 60)
    print("\n[azure_ml] Training on T4 GPU (~2 hours, ~$1.04)")
    print("[azure_ml] You can close your laptop — job runs in Azure.")
    print("[azure_ml] You'll get email when it completes.")
    print("\n[azure_ml] When done, download the model:")
    print(f"  az ml job download --name {submitted_job.name} --output-name model")
    print("[azure_ml] Then run: python -m src.ml.train.export_onnx")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Full pipeline: connect → create compute → register data → submit job.
    """
    print("Finance Tracker ML — Azure ML Job Submitter")
    print("=" * 60)

    # Step 1: Connect to Azure ML
    client = get_ml_client()

    # Step 2: Get or create T4 GPU compute cluster
    compute_name = get_or_create_compute(client)

    # Step 3: Get or create training Docker environment
    env_name = get_or_create_environment(client)

    # Step 4: Register training data as Azure ML Data Asset
    data_input = register_data_asset(client)

    # Step 5: Submit training job (returns immediately — job runs in background)
    submit_training_job(client, compute_name, env_name, data_input)


if __name__ == "__main__":
    main()
