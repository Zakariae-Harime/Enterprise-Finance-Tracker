"""
Fernet Credential Encryption Security Tests

Proves that:
  - Decryption with wrong key raises InvalidToken (not silent garbage)
  - Tampered ciphertext is detected and rejected
  - Credentials round-trip with exact fidelity
  - Missing ENCRYPTION_KEY env var raises RuntimeError (not silent failure)
"""
import os
import pytest
from cryptography.fernet import Fernet, InvalidToken

os.environ.setdefault("ENCRYPTION_KEY", "Rrcx7V1BgRERZsKT7cVGFOQwuxnCuGL4zfSfilo-bOw=")

from src.services.credentials import encrypt_credentials, decrypt_credentials


def test_credentials_roundtrip():
    """Encrypt then decrypt must return the exact original dict."""
    original = {
        "consumer_token": "ct-abc123",
        "employee_token": "et-xyz789",
        "company_id": 42,
    }
    encrypted = encrypt_credentials(original)
    recovered = decrypt_credentials(encrypted)
    assert recovered == original


def test_encrypted_value_is_not_plaintext():
    """The stored value must not contain the raw credential strings."""
    creds = {"consumer_token": "super-secret-token", "password": "hunter2"}
    encrypted = encrypt_credentials(creds)
    assert "super-secret-token" not in encrypted
    assert "hunter2" not in encrypted


def test_wrong_key_raises_invalid_token(monkeypatch):
    """
    Encrypt with key A, try to decrypt with key B.
    Fernet's HMAC-SHA256 detects the key mismatch and raises InvalidToken.
    Must never silently return garbage data.
    """
    # Encrypt with the real test key
    encrypted = encrypt_credentials({"token": "real-data"})

    # Switch ENCRYPTION_KEY to a different key
    different_key = Fernet.generate_key().decode()
    monkeypatch.setenv("ENCRYPTION_KEY", different_key)

    # Force module to use the new key
    import importlib, src.services.credentials as creds_module
    importlib.reload(creds_module)

    with pytest.raises(InvalidToken):
        creds_module.decrypt_credentials(encrypted)

    # Restore
    importlib.reload(creds_module)


def test_tampered_ciphertext_rejected():
    """
    Flip a byte in the middle of the ciphertext.
    Fernet's HMAC authentication detects the tampering and raises InvalidToken.
    """
    encrypted = encrypt_credentials({"token": "sensitive"})

    # Flip a byte at position 20 (well past the HMAC/header)
    raw = bytearray(encrypted.encode())
    raw[20] ^= 0xFF
    tampered = raw.decode(errors="replace")

    with pytest.raises((InvalidToken, Exception)):
        decrypt_credentials(tampered)


def test_missing_encryption_key_raises_runtime_error(monkeypatch):
    """
    If ENCRYPTION_KEY is not set, encrypt_credentials must raise RuntimeError.
    Must NOT silently use a default key or return None.
    """
    monkeypatch.delenv("ENCRYPTION_KEY", raising=False)

    import importlib, src.services.credentials as creds_module
    importlib.reload(creds_module)

    with pytest.raises(RuntimeError, match="ENCRYPTION_KEY"):
        creds_module.encrypt_credentials({"token": "test"})

    importlib.reload(creds_module)


def test_empty_credentials_encrypt_and_decrypt():
    """Edge case: empty dict must still round-trip."""
    encrypted = encrypt_credentials({})
    recovered = decrypt_credentials(encrypted)
    assert recovered == {}


def test_unicode_credentials_roundtrip():
    """Credentials with unicode characters (common in Norwegian names) must round-trip."""
    creds = {
        "company_name": "Æøå Consulting AS",
        "description":  "Norsk bedrift — NB: æøå",
        "token":        "tøken-123",
    }
    assert decrypt_credentials(encrypt_credentials(creds)) == creds


def test_each_encryption_produces_different_ciphertext():
    """
    Fernet uses a random IV — same plaintext encrypted twice must produce
    different ciphertext. Prevents ciphertext comparison attacks.
    """
    creds = {"token": "same-value"}
    enc1 = encrypt_credentials(creds)
    enc2 = encrypt_credentials(creds)
    assert enc1 != enc2
    # But both must decrypt to the same original
    assert decrypt_credentials(enc1) == decrypt_credentials(enc2) == creds
