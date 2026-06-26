from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.dependencies import get_matcher_service
from app.main import app
from app.models.response import MatchReport, MatchResponse

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

VALID_PAYLOAD = {
    "resume_text": "John Doe\n" + "Python developer with 5 years of FastAPI experience. " * 5,
    "job_description": "We need a Python engineer with FastAPI skills. " * 5,
}


@pytest.fixture()
def client():
    mock_service = MagicMock()
    mock_service.match.return_value = SAMPLE_RESPONSE
    app.dependency_overrides[get_matcher_service] = lambda: mock_service
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_match_success(client):
    resp = client.post("/api/v1/match/", json=VALID_PAYLOAD)
    assert resp.status_code == 200
    data = resp.json()
    assert data["report"]["overall_match_score"] == 82
    assert data["report"]["recommendation"] == "Strong Match"
    assert data["model_used"] == "claude-sonnet-4-6"
    assert data["processing_time_ms"] == 1200


def test_match_response_shape(client):
    resp = client.post("/api/v1/match/", json=VALID_PAYLOAD)
    report = resp.json()["report"]
    assert "overall_match_score" in report
    assert "matched_skills" in report
    assert "missing_skills" in report
    assert "strengths" in report
    assert "gaps" in report
    assert "recommendation" in report
    assert "summary" in report


def test_match_short_resume_rejected(client):
    resp = client.post(
        "/api/v1/match/",
        json={"resume_text": "short", "job_description": VALID_PAYLOAD["job_description"]},
    )
    assert resp.status_code == 422


def test_match_short_jd_rejected(client):
    resp = client.post(
        "/api/v1/match/",
        json={"resume_text": VALID_PAYLOAD["resume_text"], "job_description": "short"},
    )
    assert resp.status_code == 422


def test_match_agent_error_returns_502(client):
    mock_service = MagicMock()
    mock_service.match.side_effect = RuntimeError("API down")
    app.dependency_overrides[get_matcher_service] = lambda: mock_service
    resp = client.post("/api/v1/match/", json=VALID_PAYLOAD)
    assert resp.status_code == 502
    assert "Agent error" in resp.json()["detail"]
    app.dependency_overrides.clear()
