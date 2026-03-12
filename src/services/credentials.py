import json
import os

from cryptography.fernet import Fernet


def _fernet() -> Fernet:
    key = os.environ.get("ENCRYPTION_KEY")
    if not key:
        raise RuntimeError("ENCRYPTION_KEY environment variable is not set")
    return Fernet(key.encode() if isinstance(key, str) else key)


def encrypt_credentials(credentials: dict) -> str:
    raw = json.dumps(credentials).encode()
    return _fernet().encrypt(raw).decode()


def decrypt_credentials(encrypted: str) -> dict:
    raw = _fernet().decrypt(encrypted.encode() if isinstance(encrypted, str) else encrypted)
    return json.loads(raw)
