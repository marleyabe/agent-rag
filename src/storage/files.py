from __future__ import annotations

import shutil
from pathlib import Path


class FileStorage:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_upload(self, source_path: Path, target_name: str) -> Path:
        target = self.base_dir / target_name
        shutil.copy2(source_path, target)
        return target

