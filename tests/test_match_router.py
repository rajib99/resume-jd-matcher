import io
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.dependencies import get_matcher_service
from app.main import app
from app.models.response import MatchReport, MatchResponse

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SAMPLE_REPORT = MatchReport(
    overall_match_score=82,
    matched_skills=["Python", "FastAPI"],
    missing_skills=["Kubernetes"],
    strengths=["Strong Python background", "FastAPI expertise", "REST API design"],
    gaps=["No Kubernetes experience", "No CI/CD exposure"],
    recommendation="Strong Match",
    summary="Strong match. Candidate has most required skills. Minor gaps in infrastructure tooling.",
)

SAMPLE_RESPONSE = MatchResponse(
    report=SAMPLE_REPORT,
    model_used="claude-sonnet-4-6",
    processing_time_ms=1200,
)

_LONG_RESUME = "John Doe\n" + "Python developer with 5 years of FastAPI experience. " * 5
_LONG_JD = "We need a Python engineer with FastAPI skills. " * 5

TEXT_PAYLOAD = {"resume": _LONG_RESUME, "job_description": _LONG_JD}


@pytest.fixture()
def client():
    mock_service = MagicMock()
    mock_service.match.return_value = SAMPLE_RESPONSE
    app.dependency_overrides[get_matcher_service] = lambda: mock_service
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["version"] == "1.0.0"


# ---------------------------------------------------------------------------
# POST /match/text
# ---------------------------------------------------------------------------


def test_text_endpoint_success(client):
    resp = client.post("/api/v1/match/text", json=TEXT_PAYLOAD)
    assert resp.status_code == 200
    data = resp.json()
    assert data["report"]["overall_match_score"] == 82
    assert data["report"]["recommendation"] == "Strong Match"
    assert data["model_used"] == "claude-sonnet-4-6"
    assert data["processing_time_ms"] == 1200


def test_text_endpoint_response_shape(client):
    resp = client.post("/api/v1/match/text", json=TEXT_PAYLOAD)
    report = resp.json()["report"]
    for key in ("overall_match_score", "matched_skills", "missing_skills",
                "strengths", "gaps", "recommendation", "summary"):
        assert key in report, f"Missing key: {key}"


def test_text_short_resume_rejected(client):
    resp = client.post(
        "/api/v1/match/text",
        json={"resume": "too short", "job_description": _LONG_JD},
    )
    assert resp.status_code == 422


def test_text_short_jd_rejected(client):
    resp = client.post(
        "/api/v1/match/text",
        json={"resume": _LONG_RESUME, "job_description": "too short"},
    )
    assert resp.status_code == 422


def test_text_missing_fields_rejected(client):
    resp = client.post("/api/v1/match/text", json={"resume": _LONG_RESUME})
    assert resp.status_code == 422


def test_text_agent_error_returns_502(client):
    mock_service = MagicMock()
    mock_service.match.side_effect = RuntimeError("API down")
    app.dependency_overrides[get_matcher_service] = lambda: mock_service
    resp = client.post("/api/v1/match/text", json=TEXT_PAYLOAD)
    assert resp.status_code == 502
    assert "Agent error" in resp.json()["detail"]
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# POST /match/upload — helpers
# ---------------------------------------------------------------------------

# Minimal but structurally valid PDF (hand-crafted, no external dependency)
_MINIMAL_PDF = (
    b"%PDF-1.4\n"
    b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
    b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
    b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R"
    b"/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj\n"
    b"4 0 obj<</Length 44>>stream\n"
    b"BT /F1 12 Tf 100 700 Td (Hello World) Tj ET\n"
    b"endstream\nendobj\n"
    b"5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n"
    b"xref\n0 6\n0000000000 65535 f \n"
    b"trailer<</Size 6/Root 1 0 R>>\nstartxref\n9\n%%EOF"
)


def _make_docx_bytes(text: str) -> bytes:
    """Create a real minimal DOCX in memory."""
    import docx

    doc = docx.Document()
    doc.add_paragraph(text)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def _upload_files(client, resume_bytes, resume_name, jd_bytes, jd_name,
                  resume_ct="application/pdf", jd_ct="application/pdf"):
    return client.post(
        "/api/v1/match/upload",
        files={
            "resume_file": (resume_name, io.BytesIO(resume_bytes), resume_ct),
            "jd_file": (jd_name, io.BytesIO(jd_bytes), jd_ct),
        },
    )


# ---------------------------------------------------------------------------
# POST /match/upload — success paths
# ---------------------------------------------------------------------------


def test_upload_pdf_success(client):
    with patch("app.routers.match.extract_text_from_file", return_value=_LONG_RESUME) as mock_extract:
        resp = _upload_files(client, _MINIMAL_PDF, "resume.pdf", _MINIMAL_PDF, "jd.pdf")
    assert resp.status_code == 200
    assert mock_extract.call_count == 2
    data = resp.json()
    assert data["report"]["overall_match_score"] == 82


def test_upload_docx_success(client):
    docx_bytes = _make_docx_bytes(_LONG_RESUME)
    with patch("app.routers.match.extract_text_from_file", return_value=_LONG_RESUME):
        resp = _upload_files(
            client,
            docx_bytes, "resume.docx",
            docx_bytes, "jd.docx",
            resume_ct="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            jd_ct="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
    assert resp.status_code == 200


def test_upload_octet_stream_with_pdf_extension(client):
    with patch("app.routers.match.extract_text_from_file", return_value=_LONG_RESUME):
        resp = _upload_files(
            client,
            _MINIMAL_PDF, "resume.pdf",
            _MINIMAL_PDF, "jd.pdf",
            resume_ct="application/octet-stream",
            jd_ct="application/octet-stream",
        )
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# POST /match/upload — validation failures
# ---------------------------------------------------------------------------


def test_upload_unsupported_type_returns_415(client):
    txt = b"plain text content that is long enough " * 5
    resp = _upload_files(
        client, txt, "resume.txt", txt, "jd.txt",
        resume_ct="text/plain", jd_ct="text/plain",
    )
    assert resp.status_code == 415


def test_upload_file_too_large_returns_413(client):
    big = b"x" * (5 * 1024 * 1024 + 1)
    resp = _upload_files(client, big, "resume.pdf", _MINIMAL_PDF, "jd.pdf")
    assert resp.status_code == 413


def test_upload_extracted_text_too_short_returns_422(client):
    with patch("app.routers.match.extract_text_from_file", return_value="short"):
        resp = _upload_files(client, _MINIMAL_PDF, "resume.pdf", _MINIMAL_PDF, "jd.pdf")
    assert resp.status_code == 422


def test_upload_extraction_error_returns_422(client):
    with patch(
        "app.routers.match.extract_text_from_file",
        side_effect=ValueError("PDF contains no extractable text"),
    ):
        resp = _upload_files(client, _MINIMAL_PDF, "resume.pdf", _MINIMAL_PDF, "jd.pdf")
    assert resp.status_code == 422
    assert "no extractable text" in resp.json()["detail"]


def test_upload_agent_error_returns_502(client):
    mock_service = MagicMock()
    mock_service.match.side_effect = RuntimeError("model unavailable")
    app.dependency_overrides[get_matcher_service] = lambda: mock_service
    with patch("app.routers.match.extract_text_from_file", return_value=_LONG_RESUME):
        resp = _upload_files(client, _MINIMAL_PDF, "resume.pdf", _MINIMAL_PDF, "jd.pdf")
    assert resp.status_code == 502
    assert "Agent error" in resp.json()["detail"]
    app.dependency_overrides.clear()


def test_upload_missing_jd_file_returns_422(client):
    resp = client.post(
        "/api/v1/match/upload",
        files={"resume_file": ("resume.pdf", io.BytesIO(_MINIMAL_PDF), "application/pdf")},
    )
    assert resp.status_code == 422
