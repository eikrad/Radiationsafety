"""Tests that CI workflow configuration uses expected action versions."""

from pathlib import Path

CI_FILE = Path(__file__).parent.parent / ".github" / "workflows" / "ci.yml"


def _ci_text() -> str:
    return CI_FILE.read_text()


def test_setup_uv_uses_stable_version():
    """Confirm setup-uv is pinned to a known-stable major version."""
    text = _ci_text()
    assert (
        "astral-sh/setup-uv@v" in text
    ), "CI must pin astral-sh/setup-uv to a major version"
    # v8 was listed in PR #7 but is not yet released; v7 is the current stable.
    assert (
        "astral-sh/setup-uv@v7" in text
    ), "astral-sh/setup-uv@v8 does not yet exist; must stay on @v7"
