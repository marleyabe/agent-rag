from __future__ import annotations

from config import AppConfig


def test_app_config_paths_are_composed() -> None:
    cfg = AppConfig()
    assert cfg.files_dir == cfg.storage_dir / "files"
    assert cfg.chroma_dir == cfg.storage_dir / "chroma"
    assert cfg.db_path == cfg.storage_dir / "app.db"
    assert cfg.top_k == 5

