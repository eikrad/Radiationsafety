"""Tests that pyproject.toml dependency constraints match intended minimums."""
import re
from pathlib import Path

PYPROJECT = Path(__file__).parent.parent / "pyproject.toml"


def _pyproject_text() -> str:
    return PYPROJECT.read_text()


def test_langchain_google_genai_requires_v4():
    text = _pyproject_text()
    match = re.search(r'langchain-google-genai([^,\n"\']+)', text)
    assert match, "langchain-google-genai not found in pyproject.toml"
    constraint = match.group(1).strip()
    # Must not allow versions below 4 (e.g. >=2.0.0 would allow 2.x/3.x)
    assert ">=4" in constraint or "^4" in constraint, (
        f"langchain-google-genai constraint '{constraint}' should pin to >=4.0.0"
    )


def test_eslint_requires_v10():
    text = _pyproject_text()
    # ESLint version is tracked in frontend/package.json, not pyproject.toml
    # This test checks the package.json instead
    import json
    pkg = json.loads(
        (Path(__file__).parent.parent / "frontend" / "package.json").read_text()
    )
    dev = pkg.get("devDependencies", {})
    eslint_version = dev.get("eslint", "")
    assert eslint_version.startswith("^10") or eslint_version.startswith("~10"), (
        f"eslint in package.json should be ^10.x, got '{eslint_version}'"
    )
    eslint_js_version = dev.get("@eslint/js", "")
    assert eslint_js_version.startswith("^10") or eslint_js_version.startswith("~10"), (
        f"@eslint/js in package.json should be ^10.x, got '{eslint_js_version}'"
    )
