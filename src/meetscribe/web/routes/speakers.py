"""Global speakers routes."""

from fastapi import APIRouter, HTTPException

from ..models import GlobalSpeaker
from ..services.pipeline import list_global_speakers, remove_global_speaker

router = APIRouter()


@router.get("", response_model=list[GlobalSpeaker])
async def list_speakers():
    """List all enrolled global speakers."""
    names = list_global_speakers()
    return [GlobalSpeaker(name=name) for name in names]


@router.delete("/{name}")
async def delete_speaker(name: str):
    """Remove a global speaker."""
    if not remove_global_speaker(name):
        raise HTTPException(status_code=404, detail="Speaker not found")
    return {"status": "deleted"}
