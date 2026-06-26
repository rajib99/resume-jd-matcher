import asyncio
import logging
import time

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.dependencies import get_matcher_service
from app.models.request import MatchRequest
from app.models.response import MatchResponse
from app.services.matcher import MatcherService
from app.services.parser import extract_text_from_file

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/match", tags=["match"])

_MAX_FILE_BYTES = 5 * 1024 * 1024  # 5 MB
_ALLOWED_CONTENT_TYPES = frozenset(
    {
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        # Some browsers send this for .docx
        "application/octet-stream",
    }
)
_ALLOWED_EXTENSIONS = frozenset({"pdf", "docx"})


# ── POST /match/text ────────────────────────────────────────────────────────


@router.post(
    "/text",
    response_model=MatchResponse,
    summary="Match resume and job description supplied as plain text",
    status_code=status.HTTP_200_OK,
)
async def match_text(
    payload: MatchRequest,
    service: MatcherService = Depends(get_matcher_service),
) -> MatchResponse:
    start = time.monotonic()
    logger.info("POST /match/text received — calling Groq")
    try:
        result = await asyncio.to_thread(
            service.match,
            resume_text=payload.resume,
            job_description=payload.job_description,
        )
    except Exception as exc:
        logger.error("match/text agent error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Agent error: {exc}",
        ) from exc

    elapsed = int((time.monotonic() - start) * 1000)
    logger.info("POST /match/text completed in %d ms", elapsed)
    return result


# ── POST /match/upload ──────────────────────────────────────────────────────


@router.post(
    "/upload",
    response_model=MatchResponse,
    summary="Match resume and job description uploaded as PDF or DOCX files",
    status_code=status.HTTP_200_OK,
)
async def match_upload(
    resume_file: UploadFile = File(..., description="Resume as PDF or DOCX (max 5 MB)"),
    jd_file: UploadFile = File(..., description="Job description as PDF or DOCX (max 5 MB)"),
    service: MatcherService = Depends(get_matcher_service),
) -> MatchResponse:
    start = time.monotonic()

    resume_content = await _read_upload(resume_file, "resume_file")
    jd_content = await _read_upload(jd_file, "jd_file")

    try:
        resume_text = extract_text_from_file(
            resume_content,
            resume_file.filename or "",
            resume_file.content_type or "",
        )
        jd_text = extract_text_from_file(
            jd_content,
            jd_file.filename or "",
            jd_file.content_type or "",
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    _validate_min_length(resume_text, "resume_file", min_chars=50)
    _validate_min_length(jd_text, "jd_file", min_chars=50)

    try:
        result = await asyncio.to_thread(
            service.match,
            resume_text=resume_text,
            job_description=jd_text,
        )
    except Exception as exc:
        logger.error("match/upload agent error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Agent error: {exc}",
        ) from exc

    elapsed = int((time.monotonic() - start) * 1000)
    logger.info(
        "POST /match/upload completed in %d ms (resume=%s, jd=%s)",
        elapsed,
        resume_file.filename,
        jd_file.filename,
    )
    return result


# ── helpers ─────────────────────────────────────────────────────────────────


async def _read_upload(file: UploadFile, field_name: str) -> bytes:
    """Read upload, enforcing the 5 MB size limit and allowed types."""
    _validate_file_type(file, field_name)

    # Read one extra byte so we can detect files that are exactly at the limit
    content = await file.read(_MAX_FILE_BYTES + 1)
    if len(content) > _MAX_FILE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"'{field_name}' exceeds the 5 MB size limit.",
        )
    return content


def _validate_file_type(file: UploadFile, field_name: str) -> None:
    ext = ""
    if file.filename and "." in file.filename:
        ext = file.filename.rsplit(".", 1)[-1].lower()

    content_type = file.content_type or ""

    # Allow octet-stream only when the extension unambiguously identifies the format
    if content_type == "application/octet-stream" and ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"'{field_name}' has an unrecognised type. "
                "Upload a PDF (.pdf) or Word document (.docx)."
            ),
        )

    if content_type not in _ALLOWED_CONTENT_TYPES and ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"'{field_name}' must be a PDF or DOCX file. "
                f"Got content-type '{content_type}'."
            ),
        )


def _validate_min_length(text: str, field_name: str, min_chars: int) -> None:
    if len(text.strip()) < min_chars:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Extracted text from '{field_name}' is too short "
                f"(minimum {min_chars} characters)."
            ),
        )
