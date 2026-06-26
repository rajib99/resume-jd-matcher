import json
import logging
import re
import time

import openai

from app.models.response import MatchReport

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_BACKOFF_BASE = 1.0  # seconds; delay = base * 2^attempt

# HTTP status codes that are safe to retry
_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503})

_SYSTEM_PROMPT = """\
You are a senior technical recruiter with 15+ years of experience evaluating \
software engineering candidates. Your assessments are precise, evidence-based, \
and actionable.

Given a resume and a job description, produce a structured match report.

Rules:
- Score honestly: 0–49 = weak, 50–74 = moderate, 75–100 = strong.
- matched_skills: only skills explicitly present in BOTH documents.
- missing_skills: skills the JD requires that are absent or undemonstrated in the resume.
- strengths: 3 to 5 bullet points that highlight the candidate's strongest alignment.
- gaps: 2 to 4 bullet points that describe the most significant misalignments.
- recommendation: exactly one of "Strong Match", "Moderate Match", or "Weak Match".
  Use "Strong Match" for score ≥ 75, "Moderate Match" for 50–74, "Weak Match" for < 50.
- summary: 2–3 plain-English sentences a hiring manager can read in 10 seconds.

Respond ONLY with a single valid JSON object — no prose, no markdown fences:
{
  "overall_match_score": <integer 0-100>,
  "matched_skills": ["<skill>"],
  "missing_skills": ["<skill>"],
  "strengths": ["<point>", "<point>", "<point>"],
  "gaps": ["<point>", "<point>"],
  "recommendation": "<Strong Match|Moderate Match|Weak Match>",
  "summary": "<2-3 sentences>"
}\
"""

_USER_TEMPLATE = """\
## Resume
{resume}

## Job Description
{jd}

Analyze the candidate's fit and return the JSON match report.\
"""


class AgentError(Exception):
    """Raised when the agent cannot produce a valid report after all retries."""


class ResumeMatchAgent:
    def __init__(self, client: openai.OpenAI, model: str) -> None:
        self._client = client
        self._model = model

    def run(self, resume_text: str, job_description: str) -> MatchReport:
        """Call Groq and return a validated MatchReport.

        Retries up to _MAX_RETRIES times on transient API errors with
        exponential backoff (1 s, 2 s).
        """
        last_exc: Exception | None = None

        for attempt in range(_MAX_RETRIES):
            try:
                raw = self._call_api(resume_text, job_description)
                return self._parse_report(raw)

            except openai.APIStatusError as exc:
                if exc.status_code not in _RETRYABLE_STATUS_CODES:
                    raise AgentError(
                        f"Groq API error {exc.status_code}: {exc.message}"
                    ) from exc
                last_exc = exc
                logger.warning(
                    "Groq API returned %s (attempt %d/%d) — retrying",
                    exc.status_code,
                    attempt + 1,
                    _MAX_RETRIES,
                )

            except openai.APIConnectionError as exc:
                # Covers APITimeoutError too (it's a subclass)
                last_exc = exc
                logger.warning(
                    "Transient Groq error: %s (attempt %d/%d) — retrying",
                    exc,
                    attempt + 1,
                    _MAX_RETRIES,
                )

            except AgentError:
                raise

            except Exception as exc:
                # Bad model output — not retryable, fail fast
                raise AgentError(f"Failed to parse model response: {exc}") from exc

            if attempt < _MAX_RETRIES - 1:
                delay = _BACKOFF_BASE * (2**attempt)
                logger.info("Backing off %.1f s before next attempt", delay)
                time.sleep(delay)

        raise AgentError(
            f"Groq API failed after {_MAX_RETRIES} attempts: {last_exc}"
        ) from last_exc

    # ── private helpers ────────────────────────────────────────────────────────

    def _call_api(self, resume_text: str, job_description: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": _USER_TEMPLATE.format(
                        resume=resume_text,
                        jd=job_description,
                    ),
                },
            ],
            max_tokens=2048,
        )
        content = response.choices[0].message.content
        if content is None:
            raise AgentError("Model returned no content")
        return content

    def _parse_report(self, text: str) -> MatchReport:
        data = _extract_json(text)
        return MatchReport(
            overall_match_score=data["overall_match_score"],
            matched_skills=data.get("matched_skills", []),
            missing_skills=data.get("missing_skills", []),
            strengths=data["strengths"],
            gaps=data["gaps"],
            recommendation=data["recommendation"],
            summary=data["summary"],
        )


def _extract_json(text: str) -> dict:
    """Return the first JSON object found in text, stripping any markdown fences."""
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1)
    return json.loads(text.strip())
