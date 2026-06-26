"""
Unit tests for app/agents/resume_agent.py and app/services/matcher.py.

Coverage targets
----------------
_extract_json            – standalone helper, all fence/whitespace variants
ResumeMatchAgent.__init__ – stores client and model
ResumeMatchAgent._call_api – passes correct kwargs to the OpenAI/Groq SDK
ResumeMatchAgent._parse_report – all recommendation values, score boundaries
ResumeMatchAgent.run     – happy paths, all retryable error codes,
                           all non-retryable codes, exhausted retries,
                           malformed / missing-key responses
MatcherService.match     – normalises text, delegates to agent, wraps response
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, call, patch

import openai
import pytest

from app.agents.resume_agent import (
    AgentError,
    ResumeMatchAgent,
    _extract_json,
)
from app.services.matcher import MatcherService

# ── Helpers ──────────────────────────────────────────────────────────────────


def _status_err(code: int, msg: str = "error") -> openai.APIStatusError:
    return openai.APIStatusError(
        message=msg,
        response=MagicMock(status_code=code),
        body={},
    )


def _conn_err() -> openai.APIConnectionError:
    return openai.APIConnectionError(request=MagicMock())


def _agent(client: MagicMock | None = None) -> ResumeMatchAgent:
    return ResumeMatchAgent(client=client or MagicMock(), model="llama-3.3-70b-versatile")


# ── _extract_json ─────────────────────────────────────────────────────────────


class TestExtractJson:
    def test_plain_json_object(self, valid_report_data):
        assert _extract_json(json.dumps(valid_report_data)) == valid_report_data

    def test_strips_leading_trailing_whitespace(self, valid_report_data):
        padded = f"\n\n  {json.dumps(valid_report_data)}  \n"
        assert _extract_json(padded) == valid_report_data

    def test_strips_json_fence(self, valid_report_data):
        fenced = f"```json\n{json.dumps(valid_report_data)}\n```"
        assert _extract_json(fenced) == valid_report_data

    def test_strips_plain_fence(self, valid_report_data):
        fenced = f"```\n{json.dumps(valid_report_data)}\n```"
        assert _extract_json(fenced) == valid_report_data

    def test_raises_on_invalid_json(self):
        with pytest.raises(json.JSONDecodeError):
            _extract_json("not json {{{")

    def test_raises_on_empty_string(self):
        with pytest.raises((json.JSONDecodeError, ValueError)):
            _extract_json("")


# ── ResumeMatchAgent constructor ──────────────────────────────────────────────


class TestResumeMatchAgentInit:
    def test_stores_client_and_model(self):
        client = MagicMock()
        agent = ResumeMatchAgent(client=client, model="llama-3.1-8b-instant")
        assert agent._client is client
        assert agent._model == "llama-3.1-8b-instant"


# ── _call_api arguments ───────────────────────────────────────────────────────


class TestCallApiArguments:
    def test_passes_model_and_max_tokens(self, valid_report_data, make_api_message):
        client = MagicMock()
        client.chat.completions.create.return_value = make_api_message(
            json.dumps(valid_report_data)
        )
        agent = ResumeMatchAgent(client=client, model="llama-3.3-70b-versatile")

        agent.run("resume text here and more", "jd text here and more")

        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["model"] == "llama-3.3-70b-versatile"
        assert kwargs["max_tokens"] == 2048

    def test_embeds_resume_and_jd_in_user_message(self, valid_report_data, make_api_message):
        client = MagicMock()
        client.chat.completions.create.return_value = make_api_message(
            json.dumps(valid_report_data)
        )
        agent = _agent(client)

        agent.run("MY RESUME CONTENT", "MY JD CONTENT")

        messages = client.chat.completions.create.call_args.kwargs["messages"]
        user_content = messages[1]["content"]
        assert "MY RESUME CONTENT" in user_content
        assert "MY JD CONTENT" in user_content

    def test_passes_system_prompt(self, valid_report_data, make_api_message):
        client = MagicMock()
        client.chat.completions.create.return_value = make_api_message(
            json.dumps(valid_report_data)
        )
        agent = _agent(client)

        agent.run("resume", "jd")

        messages = client.chat.completions.create.call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert len(messages[0]["content"]) > 0


# ── _parse_report / happy-path parsing ───────────────────────────────────────


class TestParseReport:
    @pytest.mark.parametrize("recommendation", [
        "Strong Match",
        "Moderate Match",
        "Weak Match",
    ])
    def test_all_recommendation_values_accepted(
        self, recommendation, valid_report_data, make_api_message
    ):
        data = {**valid_report_data, "recommendation": recommendation}
        agent = _agent()
        agent._client.chat.completions.create.return_value = make_api_message(json.dumps(data))

        report = agent.run("resume", "jd")
        assert report.recommendation == recommendation

    @pytest.mark.parametrize("score", [0, 1, 49, 50, 74, 75, 99, 100])
    def test_score_boundary_values_accepted(self, score, valid_report_data, make_api_message):
        data = {**valid_report_data, "overall_match_score": score}
        agent = _agent()
        agent._client.chat.completions.create.return_value = make_api_message(json.dumps(data))

        report = agent.run("resume", "jd")
        assert report.overall_match_score == score

    def test_empty_skill_lists_accepted(self, valid_report_data, make_api_message):
        data = {**valid_report_data, "matched_skills": [], "missing_skills": []}
        agent = _agent()
        agent._client.chat.completions.create.return_value = make_api_message(json.dumps(data))

        report = agent.run("resume", "jd")
        assert report.matched_skills == []
        assert report.missing_skills == []

    def test_full_report_fields_populated(self, valid_report_data, make_api_message):
        agent = _agent()
        agent._client.chat.completions.create.return_value = make_api_message(
            json.dumps(valid_report_data)
        )

        report = agent.run("resume", "jd")
        assert report.overall_match_score == valid_report_data["overall_match_score"]
        assert report.matched_skills == valid_report_data["matched_skills"]
        assert report.missing_skills == valid_report_data["missing_skills"]
        assert report.strengths == valid_report_data["strengths"]
        assert report.gaps == valid_report_data["gaps"]
        assert report.summary == valid_report_data["summary"]


# ── Non-retryable API errors ──────────────────────────────────────────────────


class TestNonRetryableErrors:
    @pytest.mark.parametrize("status_code", [400, 401, 403, 404])
    def test_non_retryable_status_raises_immediately(self, status_code):
        agent = _agent()
        agent._client.chat.completions.create.side_effect = _status_err(status_code)

        with pytest.raises(AgentError, match=str(status_code)):
            agent.run("resume", "jd")

        agent._client.chat.completions.create.assert_called_once()

    def test_non_retryable_error_message_includes_status(self):
        agent = _agent()
        agent._client.chat.completions.create.side_effect = _status_err(403, "Forbidden")

        with pytest.raises(AgentError) as exc_info:
            agent.run("resume", "jd")

        assert "403" in str(exc_info.value)


# ── Retryable API errors ──────────────────────────────────────────────────────


class TestRetryableErrors:
    @pytest.mark.parametrize("status_code", [429, 500, 502, 503])
    @patch("app.agents.resume_agent.time.sleep")
    def test_all_retryable_codes_trigger_retry(
        self, mock_sleep, status_code, valid_report_data, make_api_message
    ):
        agent = _agent()
        agent._client.chat.completions.create.side_effect = [
            _status_err(status_code),
            make_api_message(json.dumps(valid_report_data)),
        ]

        report = agent.run("resume", "jd")

        assert report.overall_match_score == valid_report_data["overall_match_score"]
        assert agent._client.chat.completions.create.call_count == 2
        mock_sleep.assert_called_once_with(1.0)

    @patch("app.agents.resume_agent.time.sleep")
    def test_connection_error_triggers_retry(
        self, mock_sleep, valid_report_data, make_api_message
    ):
        agent = _agent()
        agent._client.chat.completions.create.side_effect = [
            _conn_err(),
            make_api_message(json.dumps(valid_report_data)),
        ]

        report = agent.run("resume", "jd")
        assert report.recommendation == valid_report_data["recommendation"]
        mock_sleep.assert_called_once_with(1.0)

    @patch("app.agents.resume_agent.time.sleep")
    def test_timeout_error_triggers_retry(
        self, mock_sleep, valid_report_data, make_api_message
    ):
        agent = _agent()
        agent._client.chat.completions.create.side_effect = [
            openai.APITimeoutError(request=MagicMock()),
            make_api_message(json.dumps(valid_report_data)),
        ]

        report = agent.run("resume", "jd")
        assert report.overall_match_score == valid_report_data["overall_match_score"]

    @patch("app.agents.resume_agent.time.sleep")
    def test_backoff_doubles_on_each_attempt(
        self, mock_sleep, valid_report_data, make_api_message
    ):
        agent = _agent()
        err = _status_err(429, "rate limit")
        agent._client.chat.completions.create.side_effect = [
            err,
            err,
            make_api_message(json.dumps(valid_report_data)),
        ]

        agent.run("resume", "jd")

        assert mock_sleep.call_count == 2
        assert mock_sleep.call_args_list == [call(1.0), call(2.0)]

    @patch("app.agents.resume_agent.time.sleep")
    def test_exhausted_retries_raise_agent_error(self, mock_sleep):
        agent = _agent()
        agent._client.chat.completions.create.side_effect = _status_err(503, "unavailable")

        with pytest.raises(AgentError, match="failed after 3 attempts"):
            agent.run("resume", "jd")

        assert agent._client.chat.completions.create.call_count == 3
        assert mock_sleep.call_count == 2

    @patch("app.agents.resume_agent.time.sleep")
    def test_no_sleep_after_final_attempt(self, mock_sleep):
        agent = _agent()
        agent._client.chat.completions.create.side_effect = _conn_err()

        with pytest.raises(AgentError):
            agent.run("resume", "jd")

        assert mock_sleep.call_count == 2


# ── Malformed model responses ─────────────────────────────────────────────────


class TestMalformedResponses:
    def test_invalid_json_raises_immediately_without_retry(self, make_api_message):
        agent = _agent()
        agent._client.chat.completions.create.return_value = make_api_message("not json {{{")

        with pytest.raises(AgentError, match="Failed to parse model response"):
            agent.run("resume", "jd")

        agent._client.chat.completions.create.assert_called_once()

    def test_missing_required_key_raises_agent_error(
        self, valid_report_data, make_api_message
    ):
        data = {k: v for k, v in valid_report_data.items() if k != "recommendation"}
        agent = _agent()
        agent._client.chat.completions.create.return_value = make_api_message(json.dumps(data))

        with pytest.raises(AgentError, match="Failed to parse model response"):
            agent.run("resume", "jd")

    def test_invalid_recommendation_literal_raises_agent_error(
        self, valid_report_data, make_api_message
    ):
        data = {**valid_report_data, "recommendation": "Maybe Hire"}
        agent = _agent()
        agent._client.chat.completions.create.return_value = make_api_message(json.dumps(data))

        with pytest.raises(AgentError, match="Failed to parse model response"):
            agent.run("resume", "jd")

    def test_score_above_100_raises_agent_error(self, valid_report_data, make_api_message):
        data = {**valid_report_data, "overall_match_score": 101}
        agent = _agent()
        agent._client.chat.completions.create.return_value = make_api_message(json.dumps(data))

        with pytest.raises(AgentError, match="Failed to parse model response"):
            agent.run("resume", "jd")

    def test_score_below_0_raises_agent_error(self, valid_report_data, make_api_message):
        data = {**valid_report_data, "overall_match_score": -1}
        agent = _agent()
        agent._client.chat.completions.create.return_value = make_api_message(json.dumps(data))

        with pytest.raises(AgentError, match="Failed to parse model response"):
            agent.run("resume", "jd")

    def test_malformed_response_does_not_retry(self, make_api_message):
        agent = _agent()
        agent._client.chat.completions.create.return_value = make_api_message("oops")

        with pytest.raises(AgentError):
            agent.run("resume", "jd")

        agent._client.chat.completions.create.assert_called_once()


# ── MatcherService integration ────────────────────────────────────────────────


class TestMatcherService:
    def test_match_wraps_agent_output_in_response(self, sample_report):
        mock_agent = MagicMock()
        mock_agent.run.return_value = sample_report

        with patch("app.services.matcher.openai.OpenAI"), \
             patch("app.services.matcher.ResumeMatchAgent", return_value=mock_agent):
            svc = MatcherService(groq_api_key="test-key", model="llama-3.3-70b-versatile")
            result = svc.match(resume_text="resume text here", job_description="jd text here")

        assert result.report is sample_report
        assert result.model_used == "llama-3.3-70b-versatile"
        assert result.processing_time_ms >= 0

    def test_match_normalises_whitespace_before_agent(self, sample_report):
        mock_agent = MagicMock()
        mock_agent.run.return_value = sample_report

        with patch("app.services.matcher.openai.OpenAI"), \
             patch("app.services.matcher.ResumeMatchAgent", return_value=mock_agent):
            svc = MatcherService(groq_api_key="key", model="llama-3.3-70b-versatile")
            svc.match(
                resume_text="  Alice\n\n\n\nPython developer  ",
                job_description="\n\nSeeking Python engineer\n\n\n",
            )

        kwargs = mock_agent.run.call_args.kwargs
        assert kwargs["resume_text"] == "Alice\n\nPython developer"
        assert kwargs["job_description"] == "Seeking Python engineer"

    def test_match_records_non_negative_processing_time(self, sample_report):
        mock_agent = MagicMock()
        mock_agent.run.return_value = sample_report

        with patch("app.services.matcher.openai.OpenAI"), \
             patch("app.services.matcher.ResumeMatchAgent", return_value=mock_agent):
            svc = MatcherService(groq_api_key="key", model="llama-3.3-70b-versatile")
            result = svc.match("resume", "jd")

        assert result.processing_time_ms >= 0

    def test_match_propagates_agent_error(self, sample_report):
        mock_agent = MagicMock()
        mock_agent.run.side_effect = AgentError("Groq unreachable")

        with patch("app.services.matcher.openai.OpenAI"), \
             patch("app.services.matcher.ResumeMatchAgent", return_value=mock_agent):
            svc = MatcherService(groq_api_key="key", model="llama-3.3-70b-versatile")

            with pytest.raises(AgentError, match="Groq unreachable"):
                svc.match("resume", "jd")
