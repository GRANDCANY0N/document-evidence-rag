from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


class PDFPreflightError(RuntimeError):
    pass


@dataclass(frozen=True)
class PDFPreflight:
    path: Path
    sha256: str
    page_count: int
    encrypted: bool
    file_size: int
    rotations: tuple[int, ...]
    page_sizes: tuple[tuple[float, float], ...]


def inspect_pdf(path: Path) -> PDFPreflight:
    path = path.resolve()
    if not path.is_file() or path.suffix.lower() != ".pdf":
        raise PDFPreflightError(f"Not a readable PDF: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        raise PDFPreflightError(f"PDF cannot be opened: {type(exc).__name__}") from exc
    if reader.is_encrypted:
        try:
            unlocked = reader.decrypt("")
        except Exception:
            unlocked = 0
        if not unlocked:
            raise PDFPreflightError("PDF is encrypted and cannot be opened without a password")
    rotations: list[int] = []
    sizes: list[tuple[float, float]] = []
    for page in reader.pages:
        rotations.append(int(page.get("/Rotate", 0) or 0) % 360)
        sizes.append((float(page.mediabox.width), float(page.mediabox.height)))
    return PDFPreflight(
        path=path,
        sha256=digest.hexdigest(),
        page_count=len(reader.pages),
        encrypted=reader.is_encrypted,
        file_size=path.stat().st_size,
        rotations=tuple(rotations),
        page_sizes=tuple(sizes),
    )

