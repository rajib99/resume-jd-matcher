from typing import Literal

from pydantic import BaseModel, Field


class MatchReport(BaseModel):
    overall_match_score: int = Field(..., ge=0, le=100, description="Overall match score 0–100")
    matched_skills: list[str] = Field(default_factory=list, description="Skills present in both resume and JD")
    missing_skills: list[str] = Field(default_factory=list, description="JD skills absent from resume")
    strengths: list[str] = Field(default_factory=list, min_length=3, max_length=5)
    gaps: list[str] = Field(default_factory=list, min_length=2, max_length=4)
    recommendation: Literal["Strong Match", "Moderate Match", "Weak Match"]
    summary: str = Field(..., description="2–3 sentence plain English summary")


class MatchResponse(BaseModel):
    report: MatchReport
    model_used: str
    processing_time_ms: int
