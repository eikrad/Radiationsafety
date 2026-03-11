#!/usr/bin/env python3
"""Regenerate architecture.png from architecture.mmd using mermaid.ink (no local Mermaid/Node required)."""
import base64
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MMD = ROOT / "architecture.mmd"
PNG = ROOT / "architecture.png"


def main() -> None:
    content = MMD.read_text()
    b64 = base64.urlsafe_b64encode(content.encode()).decode().rstrip("=")
    url = f"https://mermaid.ink/img/{b64}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r:
        PNG.write_bytes(r.read())
    print(f"Written {PNG}")


if __name__ == "__main__":
    main()
