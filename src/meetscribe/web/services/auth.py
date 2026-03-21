"""Authentication service: registration, login, session management."""

import hashlib
import hmac
import logging
import os
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from meetscribe import config
from meetscribe.database import (
    create_auth_session,
    create_user,
    delete_auth_session,
    delete_expired_sessions,
    get_auth_session,
    get_db,
    get_team,
    get_user_by_username,
)

logger = logging.getLogger(__name__)

SESSION_TTL_DAYS = 7
COOKIE_NAME = "meetscribe_session"
PBKDF2_ITERATIONS = 600_000


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
    except Exception:
        return False


class AuthService:
    """Handles user registration, login, and session verification."""

    def register(self, username: str, password: str, team_name: str) -> tuple[AuthUser, str]:
        """Register a new user. Returns (AuthUser, session_token).

        Raises ValueError if username taken or team not found.
        """
        conn = get_db(config.DB_PATH)
        try:
            team = get_team(conn, team_name)
            if not team:
                raise ValueError(f"Team '{team_name}' not found")

            existing = get_user_by_username(conn, username)
            if existing:
                raise ValueError("Username already taken")

            pw_hash = hash_password(password)
            user_id = create_user(conn, username, pw_hash, team["id"])

            token = secrets.token_hex(32)
            expires = datetime.now(timezone.utc) + timedelta(days=SESSION_TTL_DAYS)
            create_auth_session(conn, user_id, token, expires.isoformat())

            user = AuthUser(
                id=user_id,
                username=username,
                team_id=team["id"],
                team_name=team_name,
                is_admin=False,
            )
            return user, token
        finally:
            conn.close()

    def login(self, username: str, password: str) -> tuple[AuthUser, str]:
        """Login a user. Returns (AuthUser, session_token).

        Raises ValueError if credentials are invalid.
        """
        conn = get_db(config.DB_PATH)
        try:
            row = get_user_by_username(conn, username)
            if not row or not verify_password(password, row["password_hash"]):
                logger.warning("Failed login attempt for user: %s", username)
                raise ValueError("Invalid username or password")

            # Clean up expired sessions
            delete_expired_sessions(conn)

            token = secrets.token_hex(32)
            expires = datetime.now(timezone.utc) + timedelta(days=SESSION_TTL_DAYS)
            create_auth_session(conn, row["id"], token, expires.isoformat())

            user = AuthUser(
                id=row["id"],
                username=row["username"],
                team_id=row["team_id"],
                team_name=row["team_name"],
                is_admin=bool(row["is_admin"]),
            )
            return user, token
        finally:
            conn.close()

    def verify_session(self, token: str) -> AuthUser | None:
        """Verify a session token. Returns AuthUser or None."""
        conn = get_db(config.DB_PATH)
        try:
            row = get_auth_session(conn, token)
            if not row:
                return None
            return AuthUser(
                id=row["user_id"],
                username=row["username"],
                team_id=row["team_id"],
                team_name=row["team_name"],
                is_admin=bool(row["is_admin"]),
            )
        finally:
            conn.close()

    def logout(self, token: str) -> None:
        """Delete auth session."""
        conn = get_db(config.DB_PATH)
        try:
            delete_auth_session(conn, token)
        finally:
            conn.close()


# Singleton
_auth_service: AuthService | None = None


def get_auth_service() -> AuthService:
    """Get the auth service singleton."""
    global _auth_service
    if _auth_service is None:
        _auth_service = AuthService()
    return _auth_service
