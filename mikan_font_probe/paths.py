"""Stable repository paths for package modules and command-line entrypoints."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "mikan_font_probe"


def resolve_repo_path(relative: str | Path) -> Path:
    """Resolve data paths and legacy source references after package migration."""
    relative = Path(relative)
    direct = ROOT / relative
    if direct.exists():
        return direct
    return PACKAGE_ROOT / relative
