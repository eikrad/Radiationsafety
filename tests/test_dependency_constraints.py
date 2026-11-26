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
    assert (
        ">=4" in constraint or "^4" in constraint
    ), f"langchain-google-genai constraint '{constraint}' should pin to >=4.0.0"


def _frontend_package() -> dict:
    import json

    return json.loads(
        (Path(__file__).parent.parent / "frontend" / "package.json").read_text()
    )


def test_frontend_lint_uses_pinned_oxlint():
    # Oxlint's React Compiler rules are experimental, so the version is pinned exactly.
    pkg = _frontend_package()
    dev = pkg.get("devDependencies", {})
    oxlint_version = dev.get("oxlint", "")
    assert re.fullmatch(
        r"\d+\.\d+\.\d+", oxlint_version
    ), f"oxlint should be pinned to an exact version, got '{oxlint_version}'"
    assert pkg["scripts"]["lint"].startswith("oxlint")


def test_frontend_has_no_eslint_toolchain():
    # ESLint and typescript-eslint were replaced by Oxlint so TypeScript 7 can be used.
    dev = _frontend_package().get("devDependencies", {})
    leftovers = [
        name
        for name in dev
        if name in {"eslint", "@eslint/js", "globals"}
        or name.startswith("@typescript-eslint/")
        or name.startswith("eslint-plugin-")
    ]
    assert not leftovers, f"ESLint packages left in devDependencies: {leftovers}"


def test_vite_requires_v8():
    import json

    pkg = json.loads(
        (Path(__file__).parent.parent / "frontend" / "package.json").read_text()
    )
    dev = pkg.get("devDependencies", {})
    vite_version = dev.get("vite", "")
    assert vite_version.startswith("^8") or vite_version.startswith(
        "~8"
    ), f"vite should be ^8.x, got '{vite_version}'"
    plugin_react = dev.get("@vitejs/plugin-react", "")
    assert plugin_react.startswith("^6") or plugin_react.startswith(
        "~6"
    ), f"@vitejs/plugin-react should be ^6.x, got '{plugin_react}'"
