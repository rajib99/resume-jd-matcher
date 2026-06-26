import json
from unittest.mock import MagicMock, patch

import pytest
from google.genai import errors as genai_errors

from app.agents.resume_agent import AgentError, ResumeMatchAgent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_DATA = {
    "overall_match_score": 75,
    "matched_skills": ["Python", "FastAPI"],
    "missing_skills": ["Docker"],
    "strengths": ["Strong Python background", "FastAPI expertise", "REST API design"],
    "gaps": ["No Docker experience", "No CI/CD exposure"],
    "recommendation": "Strong Match",
    "summary": "Good match overall. Candidate covers most requirements.",
}


def _make_agent() -> ResumeMatchAgent:
    return ResumeMatchAgent(client=MagicMock(), model="gemini-2.0-flash")


def _mock_response(text: str) -> MagicMock:
    resp = MagicMock()
    resp.text = text
    return resp


def _client_err(code: int, msg: str = "error") -> genai_errors.ClientError:
    return genai_errors.ClientError(code, {"error": {"message": msg}})


def _server_err(code: int, msg: str = "error") -> genai_errors.ServerError:
    return genai_errors.ServerError(code, {"error": {"message": msg}})


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_run_parses_plain_json():
    agent = _make_agent()
    agent._client.models.generate_content.return_value = _mock_response(json.dumps(VALID_DATA))

    report = agent.run("resume text here", "job description here")

    assert report.overall_match_score == 75
    assert report.matched_skills == ["Python", "FastAPI"]
    assert report.missing_skills == ["Docker"]
    assert report.recommendation == "Strong Match"
    assert len(report.strengths) == 3
    assert len(report.gaps) == 2


def test_run_parses_markdown_fenced_json():
    agent = _make_agent()
    fenced = f"```json\n{json.dumps(VALID_DATA)}\n```"
    agent._client.models.generate_content.return_value = _mock_response(fenced)

    report = agent.run("resume text here", "job description here")
    assert report.overall_match_score == 75


def test_run_parses_unfenced_markdown_block():
    agent = _make_agent()
    fenced = f"```\n{json.dumps(VALID_DATA)}\n```"
    agent._client.models.generate_content.return_value = _mock_response(fenced)

    report = agent.run("resume text here", "job description here")
    assert report.recommendation == "Strong Match"


# ---------------------------------------------------------------------------
# Error handling — non-retryable
# ---------------------------------------------------------------------------


def test_invalid_json_raises_agent_error_immediately():
    agent = _make_agent()
    agent._client.models.generate_content.return_value = _mock_response("not json at all")

    with pytest.raises(AgentError, match="Failed to parse model response"):
        agent.run("resume", "jd")

    agent._client.models.generate_content.assert_called_once()


def test_non_retryable_http_error_raises_immediately():
    agent = _make_agent()
    agent._client.models.generate_content.side_effect = _client_err(401, "Unauthorized")

    with pytest.raises(AgentError, match="401"):
        agent.run("resume", "jd")

    agent._client.models.generate_content.assert_called_once()


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------


@patch("app.agents.resume_agent.time.sleep")
def test_retries_on_rate_limit_then_succeeds(mock_sleep):
    agent = _make_agent()
    rate_limit_err = _client_err(429, "Too Many Requests")
    agent._client.models.generate_content.side_effect = [
        rate_limit_err,
        rate_limit_err,
        _mock_response(json.dumps(VALID_DATA)),
    ]

    report = agent.run("resume", "jd")

    assert report.overall_match_score == 75
    assert agent._client.models.generate_content.call_count == 3
    assert mock_sleep.call_count == 2
    # Exponential backoff: 1.0 s then 2.0 s
    mock_sleep.assert_any_call(1.0)
    mock_sleep.assert_any_call(2.0)


@patch("app.agents.resume_agent.time.sleep")
def test_retries_on_server_error(mock_sleep):
    agent = _make_agent()
    server_err = _server_err(503, "Service Unavailable")
    agent._client.models.generate_content.side_effect = [
        server_err,
        _mock_response(json.dumps(VALID_DATA)),
    ]

    report = agent.run("resume", "jd")
    assert report.overall_match_score == 75
    mock_sleep.assert_called_once_with(1.0)


@patch("app.agents.resume_agent.time.sleep")
def test_exhausted_retries_raises_agent_error(mock_sleep):
    agent = _make_agent()
    err = _server_err(503, "Service Unavailable")
    agent._client.models.generate_content.side_effect = err

    with pytest.raises(AgentError, match="failed after 3 attempts"):
        agent.run("resume", "jd")

    assert agent._client.models.generate_content.call_count == 3
    assert mock_sleep.call_count == 2


@patch("app.agents.resume_agent.time.sleep")
def test_retries_on_500_error(mock_sleep):
    agent = _make_agent()
    server_err = _server_err(500, "Internal Server Error")
    agent._client.models.generate_content.side_effect = [
        server_err,
        _mock_response(json.dumps(VALID_DATA)),
    ]

    report = agent.run("resume", "jd")
    assert report.recommendation == "Strong Match"


# ---------------------------------------------------------------------------
# Pydantic model validation
# ---------------------------------------------------------------------------


def test_recommendation_literal_is_validated():
    agent = _make_agent()
    bad_data = {**VALID_DATA, "recommendation": "Maybe"}
    agent._client.models.generate_content.return_value = _mock_response(json.dumps(bad_data))

    with pytest.raises(AgentError, match="Failed to parse model response"):
        agent.run("resume", "jd")


def test_score_out_of_range_is_rejected():
    agent = _make_agent()
    bad_data = {**VALID_DATA, "overall_match_score": 150}
    agent._client.models.generate_content.return_value = _mock_response(json.dumps(bad_data))

    with pytest.raises(AgentError, match="Failed to parse model response"):
        agent.run("resume", "jd")
