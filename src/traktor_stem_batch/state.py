from __future__ import annotations

import sqlite3
import time
from pathlib import Path


SCHEMA = """
create table if not exists jobs (
  path text primary key,
  size integer not null,
  mtime_ns integer not null,
  status text not null,
  output_path text,
  error text,
  updated_at real not null
);
"""


class JobState:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.execute(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def get(self, audio_path: Path) -> dict[str, object] | None:
        row = self.conn.execute(
            "select path, size, mtime_ns, status, output_path, error, updated_at "
            "from jobs where path = ?",
            (str(audio_path),),
        ).fetchone()
        if row is None:
            return None
        keys = ("path", "size", "mtime_ns", "status", "output_path", "error", "updated_at")
        return dict(zip(keys, row))

    def set(
        self,
        audio_path: Path,
        status: str,
        output_path: Path | None = None,
        error: str | None = None,
    ) -> None:
        stat = audio_path.stat()
        self.conn.execute(
            """
            insert into jobs(path, size, mtime_ns, status, output_path, error, updated_at)
            values (?, ?, ?, ?, ?, ?, ?)
            on conflict(path) do update set
              size=excluded.size,
              mtime_ns=excluded.mtime_ns,
              status=excluded.status,
              output_path=excluded.output_path,
              error=excluded.error,
              updated_at=excluded.updated_at
            """,
            (
                str(audio_path),
                stat.st_size,
                stat.st_mtime_ns,
                status,
                str(output_path) if output_path else None,
                error,
                time.time(),
            ),
        )
        self.conn.commit()

    def is_done_current(self, audio_path: Path) -> bool:
        row = self.get(audio_path)
        if row is None or row["status"] != "done":
            return False
        stat = audio_path.stat()
        return row["size"] == stat.st_size and row["mtime_ns"] == stat.st_mtime_ns
