from pydantic import BaseModel, Field


class MatchRequest(BaseModel):
    resume: str = Field(..., min_length=50, description="Full text content of the resume")
    job_description: str = Field(..., min_length=50, description="Full text of the job description")
