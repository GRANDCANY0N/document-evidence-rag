from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4


def create_fresh_submission_copy(pdf_path: Path, output_dir: Path) -> tuple[Path, str]:
    """Create a visually identical, byte-distinct PDF for a fresh MinerU run."""
    output_dir.mkdir(parents=True, exist_ok=True)
    nonce = uuid4().hex
    target = output_dir / f"{pdf_path.stem}__fresh_{nonce[:12]}.pdf"
    with pdf_path.open("rb") as source, target.open("wb") as destination:
        shutil.copyfileobj(source, destination, length=1024 * 1024)
        destination.write(f"\n% MinerU fresh-run nonce: {nonce}\n".encode("ascii"))
    return target, nonce
