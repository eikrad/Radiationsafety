#!/usr/bin/env python3
"""Regenerate architecture diagram assets from architecture.mmd using mermaid.ink."""

import base64
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MMD = ROOT / "architecture.mmd"
PNG = ROOT / "architecture.png"
SVG = ROOT / "architecture.svg"


def main() -> None:
    content = MMD.read_text()
    b64 = base64.urlsafe_b64encode(content.encode()).decode().rstrip("=")
    user_agent = {"User-Agent": "Mozilla/5.0"}

    # SVG stays crisp at any zoom in README renderers.
    svg_url = f"https://mermaid.ink/svg/{b64}"
    svg_req = urllib.request.Request(svg_url, headers=user_agent)
    with urllib.request.urlopen(svg_req) as r:
        SVG.write_bytes(r.read())
    print(f"Written {SVG}")

    # Keep PNG as a convenience fallback for viewers that prefer raster files.
    png_url = f"https://mermaid.ink/img/{b64}?bgColor=white"
    png_req = urllib.request.Request(png_url, headers=user_agent)
    try:
        with urllib.request.urlopen(png_req) as r:
            PNG.write_bytes(r.read())
        print(f"Written {PNG}")
    except urllib.error.URLError as exc:
        print(f"Warning: PNG render skipped ({exc}). SVG was generated.")


if __name__ == "__main__":
    main()
