"""Enrolled speakers routes (team-scoped).

The name list is available to any team member (the workflow needs it for
suggestions and open-space assignment); mutations and sample access are
admin-only."""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from meetscribe.errors import SpeachesAPIError

from ..deps import get_admin_user, get_current_user
from ..models import GlobalSpeaker, SpeakerRename, SpeakerSample
from ..services.auth import AuthUser
from ..services.pipeline import (
    delete_speaker_sample,
    get_speaker_sample_path,
    list_speaker_samples,
    list_team_speakers,
    remove_team_speaker,
    rename_team_speaker,
)

router = APIRouter()


@router.get("", response_model=list[GlobalSpeaker])
def list_speakers(user: AuthUser = Depends(get_current_user)) -> list[GlobalSpeaker]:
    """List all enrolled speakers for the user's team."""
    return list_team_speakers(user.team_name)


@router.patch("/{name}", dependencies=[Depends(get_admin_user)])
def rename_speaker(
    name: str, data: SpeakerRename, user: AuthUser = Depends(get_current_user)
) -> dict[str, str]:
    """Rename an enrolled speaker (voiceprint + samples dir)."""
    try:
        rename_team_speaker(name, data.name, user.team_name)
    except LookupError:
        raise HTTPException(status_code=404, detail="Speaker not found")
    except FileExistsError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "renamed"}


@router.delete("/{name}", dependencies=[Depends(get_admin_user)])
def delete_speaker(name: str, user: AuthUser = Depends(get_current_user)) -> dict[str, str]:
    """Remove a speaker from the user's team."""
    if not remove_team_speaker(name, user.team_name):
        raise HTTPException(status_code=404, detail="Speaker not found")
    return {"status": "deleted"}


@router.get(
    "/{name}/samples", response_model=list[SpeakerSample], dependencies=[Depends(get_admin_user)]
)
def list_samples(name: str, user: AuthUser = Depends(get_current_user)) -> list[SpeakerSample]:
    """List enrolled samples of a speaker."""
    try:
        return list_speaker_samples(name, user.team_name)
    except LookupError:
        raise HTTPException(status_code=404, detail="Speaker not found")


@router.get("/{name}/samples/{filename}/audio", dependencies=[Depends(get_admin_user)])
def get_sample_audio(
    name: str, filename: str, user: AuthUser = Depends(get_current_user)
) -> FileResponse:
    """Stream an enrolled sample."""
    try:
        path = get_speaker_sample_path(name, filename, user.team_name)
    except LookupError:
        raise HTTPException(status_code=404, detail="Sample not found")
    return FileResponse(path, media_type="audio/wav")


@router.delete("/{name}/samples/{filename}", dependencies=[Depends(get_admin_user)])
def delete_sample(
    name: str, filename: str, user: AuthUser = Depends(get_current_user)
) -> dict[str, str]:
    """Delete an enrolled sample and recompute the voiceprint from the rest."""
    try:
        delete_speaker_sample(name, filename, user.team_name)
    except LookupError:
        raise HTTPException(status_code=404, detail="Sample not found")
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except SpeachesAPIError as e:
        raise HTTPException(status_code=502, detail=str(e))
    return {"status": "deleted"}
