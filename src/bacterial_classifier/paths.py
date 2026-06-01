from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(value: str | os.PathLike[str]) -> Path:
    """Resolve a user-supplied path without requiring it to exist."""
    return Path(os.path.expandvars(os.path.expanduser(str(value)))).resolve()


def project_path(*parts: str) -> Path:
    """Return a path rooted at the repository."""
    return PROJECT_ROOT.joinpath(*parts)
