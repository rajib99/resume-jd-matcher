import time

import openai

from app.agents.resume_agent import ResumeMatchAgent
from app.models.response import MatchResponse
from app.services.parser import parse_job_description, parse_resume

_GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class MatcherService:
    def __init__(self, groq_api_key: str, model: str) -> None:
        client = openai.OpenAI(
            api_key=groq_api_key,
            base_url=_GROQ_BASE_URL,
            timeout=30.0,
        )
        self._agent = ResumeMatchAgent(client=client, model=model)
        self._model = model

    def match(self, resume_text: str, job_description: str) -> MatchResponse:
        resume = parse_resume(resume_text)
        jd = parse_job_description(job_description)

        start = time.monotonic()
        report = self._agent.run(resume_text=resume, job_description=jd)
        elapsed_ms = int((time.monotonic() - start) * 1000)

        return MatchResponse(
            report=report,
            model_used=self._model,
            processing_time_ms=elapsed_ms,
        )
