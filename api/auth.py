"""Auth for the hosted deployment: SQLite users, PBKDF2 password hashes,
and hand-rolled HS256 JWTs (stdlib-only — no extra dependency).

Enforcement is opt-in via ULTRABULL_AUTH=1 (set in production; local dev runs
open so the pipeline can be used without creating users). User management goes
through manage.py at the repo root — the same functions below.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUTH_DB = os.path.join(ROOT, "data_cache", "auth.db")
SECRET_FILE = os.path.join(ROOT, "data_cache", "jwt_secret")
TOKEN_TTL_SECONDS = 7 * 24 * 3600  # re-login weekly
PBKDF2_ITERATIONS = 200_000


# ---------- user store ----------

def _db() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(AUTH_DB), exist_ok=True)
    conn = sqlite3.connect(AUTH_DB, timeout=10.0)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS users ("
        "  username TEXT PRIMARY KEY,"
        "  pw_hash TEXT NOT NULL,"
        "  salt TEXT NOT NULL,"
        "  created_at TEXT NOT NULL)"
    )
    return conn


def _hash_password(password: str, salt: bytes) -> str:
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, PBKDF2_ITERATIONS)
    return dk.hex()


def create_user(username: str, password: str) -> None:
    username = username.strip().lower()
    if not username or not password:
        raise ValueError("username and password are required")
    salt = secrets.token_bytes(16)
    conn = _db()
    try:
        with conn:
            conn.execute(
                "INSERT INTO users (username, pw_hash, salt, created_at) VALUES (?, ?, ?, ?)",
                (username, _hash_password(password, salt), salt.hex(),
                 time.strftime("%Y-%m-%dT%H:%M:%S")),
            )
    except sqlite3.IntegrityError:
        raise ValueError(f"user '{username}' already exists (use passwd to reset)")
    finally:
        conn.close()


def set_password(username: str, password: str) -> None:
    username = username.strip().lower()
    salt = secrets.token_bytes(16)
    conn = _db()
    try:
        with conn:
            cur = conn.execute(
                "UPDATE users SET pw_hash = ?, salt = ? WHERE username = ?",
                (_hash_password(password, salt), salt.hex(), username),
            )
        if cur.rowcount == 0:
            raise ValueError(f"no such user '{username}'")
    finally:
        conn.close()


def delete_user(username: str) -> None:
    conn = _db()
    try:
        with conn:
            cur = conn.execute("DELETE FROM users WHERE username = ?", (username.strip().lower(),))
        if cur.rowcount == 0:
            raise ValueError(f"no such user '{username}'")
    finally:
        conn.close()


def list_users() -> list[tuple[str, str]]:
    conn = _db()
    try:
        return conn.execute("SELECT username, created_at FROM users ORDER BY username").fetchall()
    finally:
        conn.close()


def verify_user(username: str, password: str) -> bool:
    conn = _db()
    try:
        row = conn.execute(
            "SELECT pw_hash, salt FROM users WHERE username = ?",
            (username.strip().lower(),),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        # burn comparable time so missing users aren't distinguishable
        _hash_password(password, b"x" * 16)
        return False
    pw_hash, salt = row
    return hmac.compare_digest(pw_hash, _hash_password(password, bytes.fromhex(salt)))


# ---------- JWT (HS256, stdlib) ----------

def _secret() -> bytes:
    env = os.getenv("ULTRABULL_JWT_SECRET")
    if env:
        return env.encode()
    os.makedirs(os.path.dirname(SECRET_FILE), exist_ok=True)
    if not os.path.exists(SECRET_FILE):
        with open(SECRET_FILE, "w") as f:
            f.write(secrets.token_hex(32))
        os.chmod(SECRET_FILE, 0o600)
    with open(SECRET_FILE) as f:
        return f.read().strip().encode()


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def make_token(username: str) -> tuple[str, int]:
    """Return (jwt, exp_unix)."""
    exp = int(time.time()) + TOKEN_TTL_SECONDS
    header = _b64(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    payload = _b64(json.dumps({"sub": username, "exp": exp, "iat": int(time.time())}).encode())
    signing_input = f"{header}.{payload}".encode()
    sig = _b64(hmac.new(_secret(), signing_input, hashlib.sha256).digest())
    return f"{header}.{payload}.{sig}", exp


def verify_token(token: str) -> str | None:
    """Return the username, or None for anything invalid or expired."""
    try:
        header, payload, sig = token.split(".")
        signing_input = f"{header}.{payload}".encode()
        expected = _b64(hmac.new(_secret(), signing_input, hashlib.sha256).digest())
        if not hmac.compare_digest(sig, expected):
            return None
        claims = json.loads(_unb64(payload))
        if int(claims.get("exp", 0)) < time.time():
            return None
        sub = claims.get("sub")
        return str(sub) if sub else None
    except Exception:
        return None
