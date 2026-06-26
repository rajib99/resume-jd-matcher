"""
Shared fixtures for the resume-jd-matcher test suite.

Fixture scopes:
  session  – expensive objects built once (DOCX bytes, constants)
  function – stateful objects reset per test (mock service, test client)
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.dependencies import get_matcher_service
from app.main import app
from app.models.response import MatchReport, MatchResponse

# ── Canonical sample data ────────────────────────────────────────────────────

RESUME_TEXT = """\
Jane Smith  |  jane@example.com  |  github.com/janesmith

EXPERIENCE
Senior Software Engineer – Acme Corp (2019–present)
  • Built REST APIs with Python, FastAPI, and PostgreSQL serving 10 M requests/day
  • Led migration to Docker and Kubernetes on AWS EKS; cut deploy time by 60 %
  • Mentored four junior engineers and conducted 50+ technical interviews

Software Engineer – Beta Inc (2016–2019)
  • Developed microservices in Python and Go
  • Implemented CI/CD pipelines with GitHub Actions and Jenkins

SKILLS
Languages:     Python, Go, TypeScript
Frameworks:    FastAPI, Flask, React
Infrastructure: Docker, Kubernetes, AWS, Terraform
Databases:     PostgreSQL, Redis, MongoDB

EDUCATION
B.S. Computer Science – State University (2016)
"""

JD_TEXT = """\
Senior Software Engineer – CloudTech Inc

We are looking for a Senior Software Engineer to join our platform team.

Requirements:
  • 5+ years of Python development
  • Strong knowledge of FastAPI or Django REST Framework
  • Experience with Docker and Kubernetes
  • Proficiency with PostgreSQL
  • Cloud platforms: AWS, GCP, or Azure
  • CI/CD pipeline experience

Nice to have: Go, TypeScript, Terraform, startup background.

Responsibilities:
  • Design and implement scalable REST APIs
  • Collaborate with frontend and data-engineering teams
  • Participate in on-call rotation and code reviews
"""

VALID_REPORT_DATA: dict = {
    "overall_match_score": 82,
    "matched_skills": ["Python", "FastAPI", "Docker", "Kubernetes", "PostgreSQL"],
    "missing_skills": ["AWS certification"],
    "strengths": [
        "Strong Python and FastAPI expertise",
        "Docker and Kubernetes experience directly matches requirements",
        "PostgreSQL proficiency aligns with the database requirement",
    ],
    "gaps": [
        "No cloud certification mentioned",
        "No formal on-call experience listed",
    ],
    "recommendation": "Strong Match",
    "summary": (
        "Jane is a strong match for this role. "
        "Her Python, FastAPI, Docker, and Kubernetes experience directly aligns with requirements. "
        "Minor gaps in cloud certifications should not be disqualifying."
    ),
}

# ── Session-scoped fixtures (built once per test run) ────────────────────────


@pytest.fixture(scope="session")
def sample_resume() -> str:
    return RESUME_TEXT


@pytest.fixture(scope="session")
def sample_jd() -> str:
    return JD_TEXT


@pytest.fixture(scope="session")
def valid_report_data() -> dict:
    return VALID_REPORT_DATA


@pytest.fixture(scope="session")
def sample_report() -> MatchReport:
    return MatchReport(**VALID_REPORT_DATA)


@pytest.fixture(scope="session")
def sample_response(sample_report: MatchReport) -> MatchResponse:
    return MatchResponse(
        report=sample_report,
        model_used="gemini-2.0-flash",
        processing_time_ms=950,
    )


@pytest.fixture(scope="session")
def minimal_pdf_bytes() -> bytes:
    """Hand-crafted minimal PDF that satisfies the PDF header check."""
    return (
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


@pytest.fixture(scope="session")
def sample_docx_bytes() -> bytes:
    """Real minimal DOCX file created with python-docx."""
    import docx

    doc = docx.Document()
    doc.add_paragraph(RESUME_TEXT)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


# ── Function-scoped fixtures (reset per test) ────────────────────────────────


@pytest.fixture()
def make_api_message():
    """Factory: return a fake Gemini API response wrapping *text*."""

    def _factory(text: str) -> MagicMock:
        resp = MagicMock()
        resp.text = text
        return resp

    return _factory


@pytest.fixture()
def mock_service(sample_response: MatchResponse) -> MagicMock:
    """Pre-configured mock MatcherService that returns VALID_REPORT_DATA."""
    svc = MagicMock()
    svc.match.return_value = sample_response
    return svc


@pytest.fixture()
def test_client(mock_service: MagicMock) -> TestClient:
    """TestClient with get_matcher_service dependency overridden."""
    app.dependency_overrides[get_matcher_service] = lambda: mock_service
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()
