"""
Download ML inference models from ADLS Gen2 before Docker build.
Called by GitHub Actions ci-cd.yml — reads CONN_STR from environment.
"""
import os
import pathlib
from azure.storage.blob import BlobServiceClient

conn_str = os.environ["CONN_STR"]
client = BlobServiceClient.from_connection_string(conn_str)

pathlib.Path("src/ml/models").mkdir(parents=True, exist_ok=True)

models = [
    "tfidf_vectorizer.joblib",
    "tfidf_classifier.joblib",
    "categorizer_int8.onnx",
]

for blob_name in models:
    blob_client = client.get_blob_client(container="models", blob=blob_name)
    data = blob_client.download_blob().readall()
    dest = f"src/ml/models/{blob_name}"
    with open(dest, "wb") as f:
        f.write(data)
    print(f"Downloaded {blob_name}: {len(data):,} bytes")
