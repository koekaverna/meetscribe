"""Enrolled speakers routes (team-scoped via authenticated user)."""

from fastapi import APIRouter, Depends, HTTPException

from ..deps import get_current_user
from ..models import GlobalSpeaker
from ..services.auth import AuthUser
from ..services.pipeline import list_team_speakers, remove_team_speaker

router = APIRouter()


@router.get("", response_model=list[GlobalSpeaker])
async def list_speakers(user: AuthUser = Depends(get_current_user)):
    """List all enrolled speakers for the user's team."""
    names = list_team_speakers(user.team_name)
    return [GlobalSpeaker(name=name) for name in names]


@router.delete("/{name}")
async def delete_speaker(name: str, user: AuthUser = Depends(get_current_user)):
    """Remove a speaker from the user's team."""
    if not remove_team_speaker(name, user.team_name):
        raise HTTPException(status_code=404, detail="Speaker not found")
    return {"status": "deleted"}
