"""Resume text extraction.

Supports the same file types already handled elsewhere in this repo
(pdf via ``pypdf``, docx via ``docx2txt``, plain text) but processes the
upload in-memory / via a short-lived temp file for a single request — no
uploaded resume is persisted to disk, since a user's resume may belong to
someone other than the site owner.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from werkzeug.datastructures import FileStorage

SUPPORTED_EXTENSIONS = frozenset({".pdf", ".docx", ".txt"})


class ResumeTextError(ValueError):
    """Raised when a resume file cannot be read or is of an unsupported type."""


def _extract_pdf(file_like) -> str:
    from pypdf import PdfReader

    reader = PdfReader(file_like)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages).strip()


def _extract_docx(raw_bytes: bytes) -> str:
    import docx2txt

    fd, tmp_path = tempfile.mkstemp(suffix=".docx")
    try:
        with os.fdopen(fd, "wb") as tmp_file:
            tmp_file.write(raw_bytes)
        return (docx2txt.process(tmp_path) or "").strip()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def extract_text_from_upload(file_storage: FileStorage, *, max_bytes: int) -> str:
    """Extract plain text from an uploaded resume file. Never persisted to disk."""
    if not file_storage or not file_storage.filename:
        raise ResumeTextError("No resume file was uploaded.")

    suffix = Path(file_storage.filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ResumeTextError(f"Unsupported resume file type '{suffix or '(none)'}'. Allowed: {supported}")

    raw_bytes = file_storage.read(max_bytes + 1)
    if len(raw_bytes) > max_bytes:
        raise ResumeTextError(f"Resume file is too large (max {max_bytes // 1_000_000} MB).")

    if suffix == ".pdf":
        from io import BytesIO

        text = _extract_pdf(BytesIO(raw_bytes))
    elif suffix == ".docx":
        text = _extract_docx(raw_bytes)
    else:
        text = raw_bytes.decode("utf-8", errors="ignore")

    if not text.strip():
        raise ResumeTextError("Could not extract any text from the uploaded resume.")
    return text


def extract_text_from_default_resume(resume_path: Path) -> str:
    """Extract text from the site owner's default resume (Profile/*.pdf)."""
    if not resume_path.exists():
        raise ResumeTextError(f"Default resume not found at {resume_path}.")
    with resume_path.open("rb") as f:
        text = _extract_pdf(f)
    if not text.strip():
        raise ResumeTextError("Could not extract any text from the default resume.")
    return text
