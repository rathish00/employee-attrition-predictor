"""Authentication helpers for the app's login gate.

Kept separate from app.py so credential-checking logic can be unit
tested directly, without needing a running Streamlit session.

This is intentionally simple (single shared username/password, not
per-user accounts) — appropriate for gating a demo/portfolio app, not a
substitute for real auth (OAuth, per-user accounts, rate limiting) on
anything handling actual employee data.
"""
from __future__ import annotations

import hashlib
import hmac


def hash_password(password: str) -> str:
    """SHA-256 hex digest of a password string."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


DEFAULT_USERNAME = "admin"
# Demo-only default password: "attrition2026". This is intentionally not a
# secret — it's a documented placeholder (see README) meant to be
# overridden via Streamlit secrets before sharing the app publicly.
DEFAULT_PASSWORD_HASH = hash_password("attrition2026")


def resolve_credentials(secrets: dict | None) -> tuple[str, str]:
    """Return (username, password_hash) to check logins against.

    Args:
        secrets: a dict-like object (e.g. ``st.secrets.get("credentials")``)
            expected to have "username" and "password_hash" keys if
            configured. Pass None or {} to fall back to the demo default.

    Returns:
        (username, password_hash) — either from secrets, or the documented
        demo default.
    """
    if secrets:
        username = secrets.get("username")
        password_hash = secrets.get("password_hash")
        if username and password_hash:
            return username, password_hash
    return DEFAULT_USERNAME, DEFAULT_PASSWORD_HASH


def verify_credentials(
    entered_username: str,
    entered_password: str,
    valid_username: str,
    valid_password_hash: str,
) -> bool:
    """Check entered credentials against the resolved valid ones.

    Uses a constant-time comparison for the password hash to avoid
    leaking timing information.
    """
    if not entered_username or not entered_password:
        return False
    return entered_username == valid_username and hmac.compare_digest(
        hash_password(entered_password), valid_password_hash
    )
