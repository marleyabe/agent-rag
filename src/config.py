from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    root_dir: Path = Path(__file__).resolve().parents[1]
    storage_dir: Path = root_dir / "storage"
    files_dir: Path = storage_dir / "files"
    chroma_dir: Path = storage_dir / "chroma"
    db_path: Path = storage_dir / "app.db"
    top_k: int = 5

