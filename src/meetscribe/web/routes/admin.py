"""Admin panel routes: user/team management, server status, disk usage, error log."""

import logging
import shutil
import sqlite3
import time
from pathlib import Path

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query

from meetscribe import config
from meetscribe.config import get_config
from meetscribe.database import (
    count_admins,
    count_superadmins,
    create_team,
    create_user,
    delete_auth_sessions_for_user,
    delete_team,
    delete_user,
    get_db,
    get_team,
    get_user_by_username,
    list_teams_with_counts,
    list_users,
)

from ..deps import get_admin_user, get_superadmin_user
from ..models import (
    AdminPasswordReset,
    AdminTeam,
    AdminTeamCreate,
    AdminUser,
    AdminUserCreate,
    AdminUserPatch,
    DiskUsage,
    ErrorLogTail,
    ServerStatus,
)
from ..services.auth import AuthUser, hash_password

logger = logging.getLogger(__name__)

router = APIRouter()

STATUS_TIMEOUT_S = 3.0


# --- Users ---


def _to_admin_user(row: sqlite3.Row) -> AdminUser:
    return AdminUser(
        id=row["id"],
        username=row["username"],
        team_name=row["team_name"],
        is_admin=bool(row["is_admin"]),
        is_superadmin=bool(row["is_superadmin"]),
        created_at=row["created_at"],
    )


def _get_target_user(conn: sqlite3.Connection, username: str, admin: AuthUser) -> sqlite3.Row:
    """Fetch a user within the requester's scope. 404 hides other teams' users."""
    target = get_user_by_username(conn, username)
    if not target or (not admin.is_superadmin and target["team_id"] != admin.team_id):
        raise HTTPException(status_code=404, detail="User not found")
    return target


def _check_password(password: str) -> None:
    if len(password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")


@router.get("/users", response_model=list[AdminUser])
def get_users(admin: AuthUser = Depends(get_admin_user)) -> list[AdminUser]:
    """List users: all for superadmins, own team only for team admins."""
    team_id = None if admin.is_superadmin else admin.team_id
    return [_to_admin_user(row) for row in list_users(get_db(), team_id)]


@router.post("/users", response_model=AdminUser)
def post_user(req: AdminUserCreate, admin: AuthUser = Depends(get_admin_user)) -> AdminUser:
    """Create a user (optionally admin). Team admins only within their own team."""
    _check_password(req.password)
    team_name = req.team_name or admin.team_name
    if not admin.is_superadmin and team_name != admin.team_name:
        raise HTTPException(
            status_code=403, detail="Team admins can only create users in their own team"
        )

    conn = get_db()
    try:
        conn.execute("BEGIN IMMEDIATE")
        team = get_team(conn, team_name)
        if not team:
            raise HTTPException(status_code=400, detail=f"Team '{team_name}' not found")
        if get_user_by_username(conn, req.username):
            raise HTTPException(status_code=409, detail="Username already taken")
        create_user(conn, req.username, hash_password(req.password), team["id"], req.is_admin)
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    logger.info("User created via admin panel", extra={"username": req.username})
    row = get_user_by_username(conn, req.username)
    assert row is not None
    return _to_admin_user(row)


@router.delete("/users/{username}")
def remove_user(username: str, admin: AuthUser = Depends(get_admin_user)) -> dict[str, str]:
    """Delete a user. Refuses self-deletion and deleting the last (super)admin."""
    conn = get_db()
    target = _get_target_user(conn, username, admin)
    if target["is_superadmin"] and not admin.is_superadmin:
        raise HTTPException(status_code=403, detail="Cannot delete a superadmin")
    # Last-role checks first: they are only reachable via self-deletion (the requester
    # holds the same role), so the self check would otherwise shadow them.
    if target["is_superadmin"] and count_superadmins(conn) == 1:
        raise HTTPException(status_code=400, detail="Cannot delete the last superadmin")
    if target["is_admin"] and count_admins(conn) == 1:
        raise HTTPException(status_code=400, detail="Cannot delete the last admin")
    if username == admin.username:
        raise HTTPException(status_code=400, detail="Cannot delete your own account")
    delete_user(conn, username)
    conn.commit()
    logger.info("User deleted via admin panel", extra={"username": username})
    return {"status": "deleted"}


@router.patch("/users/{username}", response_model=AdminUser)
def patch_user(
    username: str, req: AdminUserPatch, admin: AuthUser = Depends(get_admin_user)
) -> AdminUser:
    """Grant or revoke a user's admin role."""
    conn = get_db()
    target = _get_target_user(conn, username, admin)
    if target["is_superadmin"]:
        raise HTTPException(status_code=403, detail="Superadmin role is managed via CLI")
    if username == admin.username:
        raise HTTPException(status_code=400, detail="Cannot change your own role")
    conn.execute("UPDATE users SET is_admin = ? WHERE id = ?", (int(req.is_admin), target["id"]))
    conn.commit()
    logger.info(
        "User role changed via admin panel",
        extra={"username": username, "is_admin": req.is_admin},
    )
    row = get_user_by_username(conn, username)
    assert row is not None
    return _to_admin_user(row)


@router.post("/users/{username}/password")
def reset_password(
    username: str, req: AdminPasswordReset, admin: AuthUser = Depends(get_admin_user)
) -> dict[str, str]:
    """Set a user's new password and invalidate all their sessions."""
    _check_password(req.password)
    conn = get_db()
    target = _get_target_user(conn, username, admin)
    if target["is_superadmin"] and not admin.is_superadmin:
        raise HTTPException(status_code=403, detail="Cannot reset a superadmin's password")
    conn.execute(
        "UPDATE users SET password_hash = ? WHERE id = ?",
        (hash_password(req.password), target["id"]),
    )
    delete_auth_sessions_for_user(conn, target["id"])
    conn.commit()
    logger.info("Password reset via admin panel", extra={"username": username})
    return {"status": "reset"}


# --- Teams (superadmin only) ---


@router.get("/teams", response_model=list[AdminTeam], dependencies=[Depends(get_superadmin_user)])
def get_teams() -> list[AdminTeam]:
    """List all teams with user/session/voiceprint counts."""
    return [AdminTeam(**dict(row)) for row in list_teams_with_counts(get_db())]


@router.post("/teams", response_model=AdminTeam, dependencies=[Depends(get_superadmin_user)])
def post_team(req: AdminTeamCreate) -> AdminTeam:
    """Create a team and its sample directories."""
    conn = get_db()
    try:
        create_team(conn, req.name, req.description)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except sqlite3.IntegrityError:
        conn.rollback()
        raise HTTPException(status_code=409, detail=f"Team '{req.name}' already exists")
    config.ensure_team_dirs(req.name)
    logger.info("Team created via admin panel", extra={"team": req.name})
    row = next(r for r in list_teams_with_counts(conn) if r["name"] == req.name)
    return AdminTeam(**dict(row))


@router.delete("/teams/{name}", dependencies=[Depends(get_superadmin_user)])
def remove_team(name: str) -> dict[str, str]:
    """Delete a team. Voiceprints cascade; users/sessions block deletion (FK NO ACTION)."""
    conn = get_db()
    try:
        deleted = delete_team(conn, name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except sqlite3.IntegrityError:
        conn.rollback()
        raise HTTPException(
            status_code=409,
            detail="Team still has users or sessions. Delete them first.",
        )
    if not deleted:
        raise HTTPException(status_code=404, detail="Team not found")
    # Safe: the name matched a DB row, and team names are validated on creation
    # (alphanumeric/hyphen/underscore only), so no path traversal is possible.
    team_dir = config.TEAMS_DIR / name
    if team_dir.exists():
        shutil.rmtree(team_dir)
    logger.info("Team deleted via admin panel", extra={"team": name})
    return {"status": "deleted"}


# --- Speaches server status (superadmin only) ---


@router.get(
    "/status", response_model=list[ServerStatus], dependencies=[Depends(get_superadmin_user)]
)
def get_status() -> list[ServerStatus]:
    """Ping each configured Speaches server's /health endpoint."""
    statuses = []
    for server in get_config().servers:
        start = time.perf_counter()
        try:
            httpx.get(f"{server.url}/health", timeout=STATUS_TIMEOUT_S)
            statuses.append(
                ServerStatus(
                    name=server.name,
                    url=server.url,
                    reachable=True,
                    latency_ms=int((time.perf_counter() - start) * 1000),
                )
            )
        except httpx.HTTPError as e:
            statuses.append(
                ServerStatus(name=server.name, url=server.url, reachable=False, error=str(e))
            )
    return statuses


# --- Disk usage ---


def _dir_size(path: Path) -> int:
    """Sum of file sizes under a directory (0 if it doesn't exist)."""
    if not path.is_dir():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


@router.get("/disk", response_model=DiskUsage, dependencies=[Depends(get_superadmin_user)])
def get_disk() -> DiskUsage:
    """Disk usage of the data directory: total, sessions, team samples."""
    return DiskUsage(
        total_bytes=_dir_size(config.DATA_DIR),
        sessions_bytes=_dir_size(config.DATA_DIR / "sessions"),
        samples_bytes=_dir_size(config.TEAMS_DIR),
    )


# --- Error log ---


@router.get("/errors", response_model=ErrorLogTail, dependencies=[Depends(get_superadmin_user)])
def get_errors(limit: int = Query(50, ge=1, le=500)) -> ErrorLogTail:
    """Last ERROR-level lines from the newest log file."""
    log_files = sorted(config.LOGS_DIR.glob("*.log"), key=lambda p: p.stat().st_mtime)
    if not log_files:
        return ErrorLogTail()
    newest = log_files[-1]
    lines = newest.read_text(encoding="utf-8", errors="replace").splitlines()
    error_lines = [line for line in lines if "[ERROR]" in line]
    return ErrorLogTail(file=newest.name, lines=error_lines[-limit:])
