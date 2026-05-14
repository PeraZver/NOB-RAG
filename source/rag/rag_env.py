from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


def load_local_env(book_dir: Path | None = None) -> None:
    candidate_paths = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parent / ".env",
        Path(__file__).resolve().parent.parent / ".env",
    ]
    if book_dir is not None:
        candidate_paths.append(book_dir / ".env")

    seen: set[Path] = set()
    for path in candidate_paths:
        if path in seen:
            continue
        seen.add(path)
        if path.exists():
            load_dotenv(path, override=False)


def resolve_provider_model(provider: str, explicit_model: str | None = None) -> str:
    if explicit_model:
        return explicit_model
    if provider == "anthropic":
        return os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    return os.getenv("OPENAI_MODEL", "gpt-5.3-codex")
