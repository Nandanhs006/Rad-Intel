"""
Key material and encryption at rest.

Two different problems are solved here, and they need different tools.

Secrets nobody ever needs to read back -- passwords, one-time codes, session
tokens -- are *hashed*. There is no decryption and no key to steal: a copy of
the database tells an attacker nothing it can replay.

Contact details the administrator genuinely has to read back -- a phone
number, so a verification code can be sent to it -- cannot be hashed. Those
are *encrypted* with AES-256-GCM, which is authenticated, so a tampered
ciphertext fails to decrypt rather than returning something plausible.

The key comes from RAD_INTEL_SECRET_KEY when it is set. When it is not, a
random key is written once to `.radintel_key` with owner-only permissions and
reused, because the alternative -- a fresh key per process -- would lock every
stored phone number out of the system on the next restart.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import os
import secrets
import stat

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from rad_intel.config import BASE_DIR

logger = logging.getLogger(__name__)

_KEY_FILE = BASE_DIR / ".radintel_key"


def _load_master_secret() -> bytes:
    """The root secret everything else is derived from."""
    env = os.environ.get("RAD_INTEL_SECRET_KEY", "").strip()
    if env:
        return env.encode("utf-8")

    if _KEY_FILE.exists():
        return _KEY_FILE.read_bytes().strip()

    generated = secrets.token_hex(32).encode("ascii")
    _KEY_FILE.write_bytes(generated)
    # Owner read/write only. A key file other local accounts can read is not
    # a key file.
    os.chmod(_KEY_FILE, stat.S_IRUSR | stat.S_IWUSR)
    logger.warning(
        "RAD_INTEL_SECRET_KEY is not set; generated %s instead. Keep it out of "
        "version control and back it up: losing it makes stored phone numbers "
        "unreadable.", _KEY_FILE.name,
    )
    return generated


_MASTER = _load_master_secret()


def derive_key(purpose: str, length: int = 32) -> bytes:
    """
    A separate key per purpose, so one cannot be used against another.

    HMAC-based extract-and-expand over the master secret. `purpose` labels the
    use ("field-encryption", "session-tokens") and changing it changes the key.
    """
    prk = hmac.new(b"rad-intel-hkdf-salt", _MASTER, hashlib.sha256).digest()
    out, block, counter = b"", b"", 1
    while len(out) < length:
        block = hmac.new(prk, block + purpose.encode("utf-8") + bytes([counter]),
                         hashlib.sha256).digest()
        out += block
        counter += 1
    return out[:length]


_FIELD_KEY = derive_key("field-encryption")


# --- reversible field encryption -------------------------------------------

def encrypt_field(plaintext: str | None) -> str | None:
    """
    Encrypt a value that has to be read back later, e.g. a phone number.

    Returns base64 of nonce || ciphertext || tag. A fresh 96-bit nonce per
    call means the same phone number stored twice produces different
    ciphertext, so the store cannot be mined for duplicates.
    """
    if plaintext is None or plaintext == "":
        return None
    nonce = secrets.token_bytes(12)
    sealed = AESGCM(_FIELD_KEY).encrypt(nonce, plaintext.encode("utf-8"), None)
    return base64.b64encode(nonce + sealed).decode("ascii")


def decrypt_field(blob: str | None) -> str | None:
    """Reverse encrypt_field. Returns None when the value is absent or unreadable."""
    if not blob:
        return None
    try:
        raw = base64.b64decode(blob)
        return AESGCM(_FIELD_KEY).decrypt(raw[:12], raw[12:], None).decode("utf-8")
    except Exception:
        # A wrong key or a tampered value. Returning None is correct: the
        # caller must not receive a guess at what the number might have been.
        logger.warning("A stored encrypted field could not be decrypted.")
        return None


# --- one-way hashing -------------------------------------------------------

_SCRYPT_N = 2 ** 15
_SCRYPT_R = 8
_SCRYPT_P = 1
_KEYLEN = 64
# OpenSSL caps scrypt memory at 32 MB by default, which 128 * N * r reaches
# exactly, so raise the ceiling or hashing raises "memory limit exceeded".
_SCRYPT_MAXMEM = 96 * 1024 * 1024


def hash_secret(secret: str, salt: str | None = None) -> tuple[str, str]:
    """
    Return (salt, hash) as hex for a password or a Rad-Intel ID.

    scrypt's memory cost is what matters: it makes an offline guessing run
    expensive per candidate rather than merely slow.
    """
    salt = salt or secrets.token_hex(16)
    digest = hashlib.scrypt(
        secret.encode("utf-8"),
        salt=bytes.fromhex(salt),
        n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P, dklen=_KEYLEN,
        maxmem=_SCRYPT_MAXMEM,
    )
    return salt, digest.hex()


def verify_secret(secret: str, salt: str, expected_hash: str) -> bool:
    """Constant-time comparison, so a wrong guess leaks nothing by timing."""
    if not salt or not expected_hash:
        return False
    try:
        _, candidate = hash_secret(secret, salt)
    except ValueError:
        return False
    return hmac.compare_digest(candidate, expected_hash)


def fast_hash(value: str) -> str:
    """
    A keyed digest for short-lived, high-entropy values.

    Used for one-time codes and session tokens. These already carry ~70 bits
    of entropy and expire in minutes, so the work factor scrypt exists to
    provide buys nothing, while a cheap keyed hash keeps them unreadable in
    the database and keeps verification fast enough to rate-limit honestly.
    """
    return hmac.new(derive_key("short-lived-tokens"), value.encode("utf-8"),
                    hashlib.sha256).hexdigest()
