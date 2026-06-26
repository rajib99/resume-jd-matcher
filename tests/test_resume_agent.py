import json
from unittest.mock import MagicMock, call, patch

import anthropic
import pytest

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
    return ResumeMatchAgent(client=MagicMock(), model="claude-sonnet-4-6")


def _mock_message(text: str) -> MagicMock:
    msg = MagicMock()
    msg.content = [MagicMock(text=text)]
    return msg


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_run_parses_plain_json():
    agent = _make_agent()
    agent._client.messages.create.return_value = _mock_message(json.dumps(VALID_DATA))

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
    agent._client.messages.create.return_value = _mock_message(fenced)

    report = agent.run("resume text here", "job description here")
    assert report.overall_match_score == 75


def test_run_parses_unfenced_markdown_block():
    agent = _make_agent()
    fenced = f"```\n{json.dumps(VALID_DATA)}\n```"
    agent._client.messages.create.return_value = _mock_message(fenced)

    report = agent.run("resume text here", "job description here")
    assert report.recommendation == "Strong Match"


# ---------------------------------------------------------------------------
# Error handling — non-retryable
# ---------------------------------------------------------------------------


def test_invalid_json_raises_agent_error_immediately():
    agent = _make_agent()
    agent._client.messages.create.return_value = _mock_message("not json at all")

    with pytest.raises(AgentError, match="Failed to parse model response"):
        agent.run("resume", "jd")

    # Must NOT retry on bad model output
    agent._client.messages.create.assert_called_once()


def test_non_retryable_http_error_raises_immediately():
    agent = _make_agent()
    agent._client.messages.create.side_effect = anthropic.APIStatusError(
        "Unauthorized",
        response=MagicMock(status_code=401),
        body={},
    )

    with pytest.raises(AgentError, match="401"):
        agent.run("resume", "jd")

    agent._client.messages.create.assert_called_once()


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------


@patch("app.agents.resume_agent.time.sleep")
def test_retries_on_rate_limit_then_succeeds(mock_sleep):
    agent = _make_agent()
    rate_limit_err = anthropic.APIStatusError(
        "Too Many Requests",
        response=MagicMock(status_code=429),
        body={},
    )
    agent._client.messages.create.side_effect = [
        rate_limit_err,
        rate_limit_err,
        _mock_message(json.dumps(VALID_DATA)),
    ]

    report = agent.run("resume", "jd")

    assert report.overall_match_score == 75
    assert agent._client.messages.create.call_count == 3
    assert mock_sleep.call_count == 2
    # Exponential backoff: 1.0 s then 2.0 s
    mock_sleep.assert_any_call(1.0)
    mock_sleep.assert_any_call(2.0)


@patch("app.agents.resume_agent.time.sleep")
def test_retries_on_connection_error(mock_sleep):
    agent = _make_agent()
    conn_err = anthropic.APIConnectionError(request=MagicMock())
    agent._client.messages.create.side_effect = [
        conn_err,
        _mock_message(json.dumps(VALID_DATA)),
    ]

    report = agent.run("resume", "jd")
    assert report.overall_match_score == 75
    mock_sleep.assert_called_once_with(1.0)


@patch("app.agents.resume_agent.time.sleep")
def test_exhausted_retries_raises_agent_error(mock_sleep):
    agent = _make_agent()
    err = anthropic.APIStatusError(
        "Service Unavailable",
        response=MagicMock(status_code=503),
        body={},
    )
    agent._client.messages.create.side_effect = err

    with pytest.raises(AgentError, match="failed after 3 attempts"):
        agent.run("resume", "jd")

    assert agent._client.messages.create.call_count == 3
    assert mock_sleep.call_count == 2


@patch("app.agents.resume_agent.time.sleep")
def test_timeout_error_is_retried(mock_sleep):
    agent = _make_agent()
    timeout_err = anthropic.APITimeoutError(request=MagicMock())
    agent._client.messages.create.side_effect = [
        timeout_err,
        _mock_message(json.dumps(VALID_DATA)),
    ]

    report = agent.run("resume", "jd")
    assert report.recommendation == "Strong Match"


# ---------------------------------------------------------------------------
# Pydantic model validation
# ---------------------------------------------------------------------------


def test_recommendation_literal_is_validated():
    agent = _make_agent()
    bad_data = {**VALID_DATA, "recommendation": "Maybe"}
    agent._client.messages.create.return_value = _mock_message(json.dumps(bad_data))

    with pytest.raises(AgentError, match="Failed to parse model response"):
        agent.run("resume", "jd")


def test_score_out_of_range_is_rejected():
    agent = _make_agent()
    bad_data = {**VALID_DATA, "overall_match_score": 150}
    agent._client.messages.create.return_value = _mock_message(json.dumps(bad_data))

    with pytest.raises(AgentError, match="Failed to parse model response"):
        agent.run("resume", "jd")
