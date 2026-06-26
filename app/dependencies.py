from functools import lru_cache

from app.config import Settings
from app.services.matcher import MatcherService


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def get_matcher_service() -> MatcherService:
    settings = get_settings()
    return MatcherService(
        gemini_api_key=settings.gemini_api_key,
        model=settings.gemini_model,
    )
