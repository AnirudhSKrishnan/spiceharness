"""Workspace resolution and short follow-up commands for the ``spice`` shortcut.

Resolution order when a command is run without ``-w/--workspace``:

1. ``SPICEHARNESS_WORKSPACE`` (explicit per-shell override)
2. the current directory, when it contains ``.spiceharness/``
3. ``SPICEHARNESS_DEFAULT_WORKSPACE`` (baked into the ``spice`` wrapper by
   ``scripts/install_shortcuts.sh``)
4. the current directory (historical behaviour for plain ``uv run spiceharness``)

An explicit ``-w/--workspace`` always wins and is step 0.
"""

from __future__ import annotations

import os
import shlex
import shutil
from pathlib import Path

ENV_WORKSPACE = "SPICEHARNESS_WORKSPACE"
ENV_DEFAULT_WORKSPACE = "SPICEHARNESS_DEFAULT_WORKSPACE"
ENV_SHORTCUT = "SPICEHARNESS_SHORTCUT"  # "1"/"0" forces the hint style; otherwise detected
SHORTCUT_MARKER = "# spiceharness-shortcut"


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name, "").strip()
    return Path(value).expanduser() if value else None


def resolve_workspace(explicit: Path | str | None = None) -> Path:
    """Return the absolute workspace for ``explicit`` or the documented defaults."""
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    chosen = _env_path(ENV_WORKSPACE)
    if chosen is None:
        cwd = Path.cwd()
        if (cwd / ".spiceharness").is_dir():
            chosen = cwd
        else:
            chosen = _env_path(ENV_DEFAULT_WORKSPACE) or cwd
    return chosen.resolve()


def workspace_option(value: Path | None) -> Path:
    """Typer callback: keep an explicit ``-w`` (as given) or resolve the default."""
    return Path(value) if value is not None else resolve_workspace()


def shortcut_installed() -> bool:
    """True when the ``spice`` wrapper written by install_shortcuts.sh is available."""
    forced = os.environ.get(ENV_SHORTCUT)
    if forced in {"0", "1"}:
        return forced == "1"
    candidates = [shutil.which("spice"), str(Path.home() / ".local" / "bin" / "spice")]
    for name in candidates:
        if not name:
            continue
        try:
            path = Path(name)
            if path.is_file() and SHORTCUT_MARKER in path.read_text(errors="ignore")[:4096]:
                return True
        except OSError:
            continue
    return False


def command_hint(subcommand: str, workspace: Path | None) -> str:
    """A copy-pasteable command; ``-w`` is added only when it differs from the resolved default."""
    command = ("spice " if shortcut_installed() else "uv run spiceharness ") + subcommand
    if workspace is not None:
        resolved = Path(workspace).expanduser().resolve()
        if resolved != resolve_workspace():
            command += " -w " + shlex.quote(str(resolved))
    return command
