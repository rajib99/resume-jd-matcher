from app.services.parser import parse_job_description, parse_resume


def test_parse_resume_strips_whitespace():
    raw = "  \n\nJohn Doe\n\nPython developer  \n\n"
    result = parse_resume(raw)
    assert result.startswith("John Doe")
    assert result.endswith("Python developer")


def test_parse_resume_collapses_blank_lines():
    raw = "Section A\n\n\n\nSection B"
    result = parse_resume(raw)
    assert "\n\n\n" not in result
    assert "Section A\n\nSection B" == result


def test_parse_jd_strips_whitespace():
    raw = "   Software Engineer role\n\n\n"
    result = parse_job_description(raw)
    assert result == "Software Engineer role"


def test_parse_resume_preserves_content():
    content = "Alice\nSkills: Python, FastAPI"
    assert parse_resume(content) == content
