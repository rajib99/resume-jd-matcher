from fastapi import APIRouter, Depends, HTTPException, status

from app.dependencies import get_matcher_service
from app.models.request import MatchRequest
from app.models.response import MatchResponse
from app.services.matcher import MatcherService

router = APIRouter(prefix="/match", tags=["match"])


@router.post(
    "/",
    response_model=MatchResponse,
    summary="Match a resume against a job description",
    status_code=status.HTTP_200_OK,
)
async def match_resume(
    payload: MatchRequest,
    service: MatcherService = Depends(get_matcher_service),
) -> MatchResponse:
    try:
        return service.match(
            resume_text=payload.resume_text,
            job_description=payload.job_description,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Agent error: {exc}",
        ) from exc
