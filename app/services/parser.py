import re


def normalize_whitespace(text: str) -> str:
    """Collapse multiple blank lines and strip leading/trailing whitespace."""
    text = text.strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def parse_resume(raw: str) -> str:
    return normalize_whitespace(raw)


def parse_job_description(raw: str) -> str:
    return normalize_whitespace(raw)
