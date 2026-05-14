from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from models import Answer, Citation, DocumentRecord


class AppDatabase:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    file_id TEXT PRIMARY KEY,
                    file_name TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    file_path TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS citations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id INTEGER NOT NULL,
                    file_name TEXT NOT NULL,
                    page_number INTEGER,
                    paragraph_number INTEGER,
                    start_line INTEGER,
                    end_line INTEGER,
                    source_text TEXT NOT NULL,
                    url TEXT NOT NULL,
                    FOREIGN KEY(chat_id) REFERENCES chat_history(id)
                )
                """
            )

    def upsert_document(self, record: DocumentRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO documents(file_id, file_name, file_type, file_path)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(file_id) DO UPDATE SET
                    file_name=excluded.file_name,
                    file_type=excluded.file_type,
                    file_path=excluded.file_path
                """,
                (record.file_id, record.file_name, record.file_type, record.file_path),
            )

    def get_document(self, file_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT file_id, file_name, file_type, file_path FROM documents WHERE file_id = ?",
                (file_id,),
            ).fetchone()
        if not row:
            return None
        return {
            "file_id": row[0],
            "file_name": row[1],
            "file_type": row[2],
            "file_path": row[3],
        }

    def save_answer(self, question: str, answer: Answer) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO chat_history(question, answer) VALUES (?, ?)",
                (question, answer.answer),
            )
            chat_id = int(cursor.lastrowid)
            conn.executemany(
                """
                INSERT INTO citations(
                    chat_id, file_name, page_number, paragraph_number,
                    start_line, end_line, source_text, url
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [self._citation_row(chat_id, c) for c in answer.citations],
            )
        return chat_id

    def get_citation_by_url(self, url: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT file_name, page_number, paragraph_number, start_line, end_line, source_text, url
                FROM citations
                WHERE url = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (url,),
            ).fetchone()
        if not row:
            return None
        return {
            "file_name": row[0],
            "page_number": row[1],
            "paragraph_number": row[2],
            "start_line": row[3],
            "end_line": row[4],
            "source_text": row[5],
            "url": row[6],
        }

    @staticmethod
    def _citation_row(chat_id: int, c: Citation) -> tuple[Any, ...]:
        return (
            chat_id,
            c.file_name,
            c.page_number,
            c.paragraph_number,
            c.start_line,
            c.end_line,
            c.source_text,
            c.url,
        )
