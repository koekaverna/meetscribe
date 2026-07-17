"""Transcript segment editing routes."""

from fastapi import APIRouter, Depends, HTTPException

from ..deps import get_current_user, get_session_for_user
from ..models import SegmentPatch, SegmentSplit, SessionStatus
from ..services.auth import AuthUser
from ..services.session import get_session_service

router = APIRouter()


def _require_transcribed(session_id: str, user: AuthUser) -> None:
    """Check access and that the session is editable (transcribed)."""
    state = get_session_for_user(session_id, user)
    if state.status != SessionStatus.TRANSCRIBED:
        raise HTTPException(status_code=409, detail="Session is not transcribed")


@router.patch("/{session_id}/segments/{segment_id}")
def update_segment(
    session_id: str,
    segment_id: int,
    data: SegmentPatch,
    user: AuthUser = Depends(get_current_user),
) -> dict[str, str]:
    """Update a segment's text and/or speaker."""
    _require_transcribed(session_id, user)
    if data.text is None and data.speaker is None:
        raise HTTPException(status_code=400, detail="Nothing to update")
    if not get_session_service().update_segment(session_id, segment_id, data.text, data.speaker):
        raise HTTPException(status_code=404, detail="Segment not found")
    return {"status": "updated"}


@router.delete("/{session_id}/segments/{segment_id}")
def delete_segment(
    session_id: str, segment_id: int, user: AuthUser = Depends(get_current_user)
) -> dict[str, str]:
    """Delete a segment."""
    _require_transcribed(session_id, user)
    if not get_session_service().delete_segment(session_id, segment_id):
        raise HTTPException(status_code=404, detail="Segment not found")
    return {"status": "deleted"}


@router.post("/{session_id}/segments/{segment_id}/merge-next")
def merge_segment_with_next(
    session_id: str, segment_id: int, user: AuthUser = Depends(get_current_user)
) -> dict[str, str]:
    """Merge a segment with the next adjacent one."""
    _require_transcribed(session_id, user)
    try:
        merged = get_session_service().merge_segment_with_next(session_id, segment_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not merged:
        raise HTTPException(status_code=404, detail="Segment not found")
    return {"status": "merged"}


@router.post("/{session_id}/segments/{segment_id}/split")
def split_segment(
    session_id: str,
    segment_id: int,
    data: SegmentSplit,
    user: AuthUser = Depends(get_current_user),
) -> dict[str, str]:
    """Split a segment at a character offset (time divided proportionally)."""
    _require_transcribed(session_id, user)
    try:
        split = get_session_service().split_segment(session_id, segment_id, data.offset)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not split:
        raise HTTPException(status_code=404, detail="Segment not found")
    return {"status": "split"}
