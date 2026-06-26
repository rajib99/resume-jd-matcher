import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import Settings
from app.routers import match_router

settings = Settings()

logging.basicConfig(
    level=settings.log_level.upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

logger = logging.getLogger(__name__)

@asynccontextmanager
async def _lifespan(app: FastAPI):
    from app.dependencies import get_matcher_service, get_settings
    s = get_settings()
    if not s.groq_api_key:
        logger.warning(
            "GROQ_API_KEY is not set. Get a free key at https://console.groq.com/"
        )
    get_matcher_service()  # pre-warm — surfaces init errors at startup
    logger.info("MatcherService ready (model=%s)", s.groq_model)
    yield


app = FastAPI(
    title="Resume JD Matcher",
    description="AI agent that matches a resume against a job description and returns a structured match report.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=_lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

_STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

app.include_router(match_router, prefix="/api/v1")


@app.get("/", include_in_schema=False)
async def frontend() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")


@app.get("/health", tags=["health"])
async def health() -> dict[str, str]:
    return {"status": "ok", "version": app.version}
