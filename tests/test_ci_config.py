"""Tests that CI workflow configuration uses expected action versions."""
import re
from pathlib import Path

CI_FILE = Path(__file__).parent.parent / ".github" / "workflows" / "ci.yml"


def _ci_text() -> str:
    return CI_FILE.read_text()


def test_setup_uv_is_v8():
    assert "astral-sh/setup-uv@v8" in _ci_text(), (
        "CI should use astral-sh/setup-uv@v8"
    )


def test_no_setup_uv_v7():
    assert "astral-sh/setup-uv@v7" not in _ci_text(), (
        "Old astral-sh/setup-uv@v7 reference should be removed"
    )
