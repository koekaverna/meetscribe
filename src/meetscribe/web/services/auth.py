"""Authentication service: registration, login, session management."""

import hashlib
import hmac
import logging
import os
import secrets
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from meetscribe.config import WebConfig, get_config
from meetscribe.database import (
    create_auth_session,
    create_user,
    delete_auth_session,
    delete_expired_sessions,
    get_auth_session,
    get_team,
    get_user_by_username,
)

logger = logging.getLogger(__name__)

COOKIE_NAME = "meetscribe_session"
PBKDF2_ITERATIONS = 600_000

_web_cfg: WebConfig | None = None


def _get_web_config() -> WebConfig:
    """Get cached web config."""
    global _web_cfg
    if _web_cfg is None:
        _web_cfg = get_config().web
    return _web_cfg


def get_session_ttl_days() -> int:
    """Get session TTL from config."""
    return _get_web_config().session_ttl_days


def get_secure_cookies() -> bool:
    """Get secure cookies flag from config."""
    return _get_web_config().secure_cookies


@dataclass
class AuthUser:
    """Authenticated user context."""

    id: int
    username: str
    team_id: int
    team_name: str
    is_admin: bool = False


def hash_password(password: str) -> str:
    """Hash password using PBKDF2-SHA256."""
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, PBKDF2_ITERATIONS)
    return f"pbkdf2:sha256:{PBKDF2_ITERATIONS}${salt.hex()}${dk.hex()}"


def verify_password(password: str, password_hash: str) -> bool:
    """Verify password against stored hash."""
    try:
        parts = password_hash.split("$")
        if len(parts) != 3:
            return False
        header, salt_hex, hash_hex = parts
        _, _, iterations_str = header.split(":")
        iterations = int(iterations_str)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(hash_hex)
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
        return hmac.compare_digest(dk, expected)
    except (ValueError, KeyError):
        return False


class AuthService:
    """Handles user registration, login, and session verification."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def register(self, username: str, password: str, team_name: str) -> tuple[AuthUser, str]:
        """Register a new user. Returns (AuthUser, session_token).

        Raises ValueError if username taken or team not found.
        """
        team = get_team(self.conn, team_name)
        if not team:
            raise ValueError(f"Team '{team_name}' not found")

        existing = get_user_by_username(self.conn, username)
        if existing:
            raise ValueError("Username already taken")

        pw_hash = hash_password(password)
        user_id = create_user(self.conn, username, pw_hash, team["id"])

        token = secrets.token_hex(32)
        expires = datetime.now(UTC) + timedelta(days=get_session_ttl_days())
        create_auth_session(self.conn, user_id, token, expires.isoformat())

        user = AuthUser(
            id=user_id,
            username=username,
            team_id=team["id"],
            team_name=team_name,
            is_admin=False,
        )
        return user, token

    def login(self, username: str, password: str) -> tuple[AuthUser, str]:
        """Login a user. Returns (AuthUser, session_token).

        Raises ValueError if credentials are invalid.
        """
        row = get_user_by_username(self.conn, username)
        if not row or not verify_password(password, row["password_hash"]):
            logger.warning("Failed login attempt")
            raise ValueError("Invalid username or password")

        # Clean up expired sessions
        delete_expired_sessions(self.conn)

        token = secrets.token_hex(32)
        expires = datetime.now(UTC) + timedelta(days=get_session_ttl_days())
        create_auth_session(self.conn, row["id"], token, expires.isoformat())

        user = AuthUser(
            id=row["id"],
            username=row["username"],
            team_id=row["team_id"],
            team_name=row["team_name"],
            is_admin=bool(row["is_admin"]),
        )
        return user, token

    def verify_session(self, token: str) -> AuthUser | None:
        """Verify a session token. Returns AuthUser or None."""
        row = get_auth_session(self.conn, token)
        if not row:
            return None
        return AuthUser(
            id=row["user_id"],
            username=row["username"],
            team_id=row["team_id"],
            team_name=row["team_name"],
            is_admin=bool(row["is_admin"]),
        )

    def logout(self, token: str) -> None:
        """Delete auth session."""
        delete_auth_session(self.conn, token)


# Singleton
_auth_service: AuthService | None = None


def init_auth_service(conn: sqlite3.Connection) -> AuthService:
    """Initialize the auth service singleton with a shared connection."""
    global _auth_service
    _auth_service = AuthService(conn)
    return _auth_service


def get_auth_service() -> AuthService:
    """Get the auth service singleton."""
    if _auth_service is None:
        raise RuntimeError("AuthService not initialized — call init_auth_service(conn) first")
    return _auth_service
