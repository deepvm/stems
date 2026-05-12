from __future__ import annotations

import json
import re
import shlex
import subprocess
from pathlib import Path
from typing import Iterable


def sanitize_filename(value: str, fallback: str = "untitled") -> str:
    value = value.strip()
    value = re.sub(r"[/:\\]+", " - ", value)
    value = re.sub(r"[\0\r\n\t]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip(" .")
    return value or fallback


def shell_join(args: Iterable[str | Path]) -> str:
    return " ".join(shlex.quote(str(arg)) for arg in args)


def run(args: list[str], dry_run: bool = False, cwd: Path | None = None) -> None:
    if dry_run:
        print(shell_join(args))
        return
    subprocess.run(args, cwd=cwd, check=True)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def human_bool(value: bool) -> str:
    return "yes" if value else "no"
