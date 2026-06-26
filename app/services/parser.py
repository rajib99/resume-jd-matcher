import io
import re


def normalize_whitespace(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def parse_resume(raw: str) -> str:
    return normalize_whitespace(raw)


def parse_job_description(raw: str) -> str:
    return normalize_whitespace(raw)


def extract_text_from_pdf(content: bytes) -> str:
    import pdfplumber

    with pdfplumber.open(io.BytesIO(content)) as pdf:
        pages = [page.extract_text() or "" for page in pdf.pages]
    text = "\n\n".join(p for p in pages if p.strip())
    if not text.strip():
        raise ValueError("PDF contains no extractable text (may be scanned/image-only)")
    return normalize_whitespace(text)


def extract_text_from_docx(content: bytes) -> str:
    import docx

    doc = docx.Document(io.BytesIO(content))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    text = "\n\n".join(paragraphs)
    if not text.strip():
        raise ValueError("DOCX contains no extractable text")
    return normalize_whitespace(text)


def extract_text_from_file(content: bytes, filename: str, content_type: str) -> str:
    """Dispatch to the correct extractor based on content-type or file extension."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    is_pdf = content_type == "application/pdf" or ext == "pdf"
    is_docx = (
        content_type
        == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        or ext == "docx"
    )

    if is_pdf:
        return extract_text_from_pdf(content)
    if is_docx:
        return extract_text_from_docx(content)

    raise ValueError(
        f"Unsupported file type '{content_type}' (extension: .{ext}). "
        "Upload a PDF or DOCX file."
    )
