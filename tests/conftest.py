"""Keep CLI tests independent of the developer's installed `spice` shortcut and workspace settings."""
import pytest


@pytest.fixture(autouse=True)
def _isolated_shortcut_environment(monkeypatch):
    for name in ("SPICEHARNESS_WORKSPACE", "SPICEHARNESS_DEFAULT_WORKSPACE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SPICEHARNESS_SHORTCUT", "0")
