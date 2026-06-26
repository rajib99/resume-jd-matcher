# resume-jd-matcher

An AI agent that matches a resume against a job description and returns a structured match report powered by Claude.

## Local development

### Prerequisites

- Python 3.12+
- An [Anthropic API key](https://console.anthropic.com/)

### Setup

```bash
# 1. Clone and enter the project
git clone <repo-url>
cd resume-jd-matcher

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements-dev.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY

# 5. Run the dev server (auto-reload)
uvicorn app.main:app --reload
```

The API is now available at `http://localhost:8000`.

### Run tests

```bash
pytest tests/ -v
```

## Docker usage

### Build and run with Docker Compose

The compose file attaches the container to an **external** Docker network called `web` (no host port is exposed). Create the network once if it does not exist:

```bash
docker network create web
```

Then start the service:

```bash
cp .env.example .env   # fill in ANTHROPIC_API_KEY
docker compose up -d --build
```

The container listens on port `8000` inside the `web` network. Pair it with a reverse proxy (e.g. Traefik, Nginx) on the same network to expose it externally.

### Build image only

```bash
docker build -t resume-jd-matcher:latest .
```

## API reference

Interactive docs are served by the running application:

| URL | Description |
|-----|-------------|
| `/docs` | Swagger UI |
| `/redoc` | ReDoc |
| `/health` | Health check |

### POST `/api/v1/match/`

**Request body**

```json
{
  "resume_text": "Full text of the candidate's resume...",
  "job_description": "Full text of the job posting..."
}
```

**Response**

```json
{
  "report": {
    "overall_score": 82.5,
    "summary": "Strong match. Candidate has most required skills.",
    "matched_skills": [
      { "skill": "Python", "found_in_resume": true, "context": "5 years Python" }
    ],
    "missing_skills": ["Kubernetes"],
    "strengths": ["FastAPI expertise"],
    "gaps": ["No K8s experience"],
    "recommendations": ["Take a Kubernetes certification course"]
  },
  "model_used": "claude-sonnet-4-6",
  "processing_time_ms": 1340
}
```

## Environment variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | Yes | — | Anthropic API key |
| `CLAUDE_MODEL` | No | `claude-sonnet-4-6` | Claude model ID |
| `LOG_LEVEL` | No | `INFO` | Uvicorn log level |
| `ENVIRONMENT` | No | `production` | Runtime label |
