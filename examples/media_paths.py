"""Shared paths for rendered media under repo-root ``Media/``."""
from __future__ import annotations

import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MEDIA_DIR = os.path.join(REPO_ROOT, "Media")


def media_path(*parts: str) -> str:
    return os.path.join(MEDIA_DIR, *parts)
