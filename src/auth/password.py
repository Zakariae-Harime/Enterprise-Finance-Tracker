"""
Password hashing and verification using bcrypt + SHA-256 pre-hashing.

Why pre-hash with SHA-256?
  - bcrypt silently truncates passwords longer than 72 bytes
  - SHA-256 converts any length input to exactly 32 bytes
  - bcrypt then hashes those 32 bytes safely
  - Pattern used by: Auth0, Django (with bcrypt backend)
"""
import hashlib
import base64
import bcrypt

def _pre_hash(plain: str) -> bytes:
    """SHA-256 the password → always 32 bytes → safe for bcrypt's 72-byte limit."""
    digest = hashlib.sha256(plain.encode("utf-8")).digest()
    return base64.b64encode(digest)  # 44 bytes, well within limit


def hash_password(plain: str) -> str:
    """Hash password with bcrypt directly — bypasses passlib version issues."""
    return bcrypt.hashpw(_pre_hash(plain), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """Verify password using bcrypt directly."""
    return bcrypt.checkpw(_pre_hash(plain), hashed.encode("utf-8"))
