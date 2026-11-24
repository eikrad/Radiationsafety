"""Export the indexed corpus as Markdown files, one per document.

For a comparison with an agentic harness (GEUS, Scientific Friday Bar 27 Nov 2026)
that searches and reads text files instead of a vector index. The export holds the
same sources and the same text as the index:

- the five Danish executive orders from their saved Retsinformation XML
  (documents/Bekendtgørelse/*_current.xml), with chapters, paragraphs, Stk. and
  annexes as Markdown structure and annex tables as Markdown tables;
- the IAEA PDFs (documents/IAEA, documents/IAEA_other) and the two Danish guidance
  PDFs, converted by docling with the same default converter as ingestion.py.

The orders' PDFs (B2025*.pdf) are not exported: the index reads those orders
from XML.

Usage:
    uv run python scripts/export_markdown.py              # everything
    uv run python scripts/export_markdown.py --only dk    # XML orders only (seconds)
    uv run python scripts/export_markdown.py --only pdf   # PDFs only (slow)

PDFs already exported are skipped, so an interrupted run can be restarted.
Output: export/markdown/{dk,iaea}/*.md plus MANIFEST.md.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "documents"
DK_DIR = DOCS / "Bekendtgørelse"
PDF_DIRS = [DOCS / "IAEA", DOCS / "IAEA_other"]
GUIDANCE_PDFS = [
    DK_DIR / "Brug af aabne radioaktive kilder.pdf",
    DK_DIR / "Udarbejdelse af en sikkerhedsvurderingAAK.pdf",
]

# --- Danish orders: Retsinformation XML -> Markdown ---------------------------------

_NO_SPACE_BEFORE = (",", ".", ";", ":", ")")


def _render(elem) -> list[str]:
    """Text pieces in document order; super/subscripts glued with ^ and _
    (same rule as ingestion_dk._render, so 1·10^6 Bq stays readable)."""
    if elem.tag == "Meta":
        return []
    pieces: list[str] = []

    def add(text, glue=""):
        text = (text or "").strip()
        if not text:
            return
        if pieces and (glue or text.startswith(_NO_SPACE_BEFORE)):
            pieces[-1] += glue + text
        else:
            pieces.append(text)

    add(elem.text)
    for child in elem:
        sub = _render(child)
        if sub:
            form = child.get("formaChar", "") if child.tag == "Char" else ""
            glue = "^" if "Superscript" in form else "_" if "Subscript" in form else ""
            add(sub[0], glue)
            pieces.extend(sub[1:])
        add(child.tail)
    return pieces


def _text(elem) -> str:
    return re.sub(r"\s+", " ", " ".join(_render(elem))).strip()


def _heading(elem) -> str:
    parts = [elem.find(t) for t in ("Explicatus", "Rubrica")]
    return " ".join(_text(p) for p in parts if p is not None).strip()


def _cell(td) -> str:
    return _text(td).replace("|", "\\|")


def _table(table) -> list[str]:
    """A Retsinformation table as a Markdown table.

    rowspan/colspan are expanded into a grid (a spanned cell repeats its text), and
    leading header rows are merged column by column, so 'Nedre værdi ... /
    C [TBq]' becomes one header cell and every data row is one grep-able line.
    """
    grid: list[list[str]] = []
    pending: dict[tuple[int, int], str] = {}  # (row, col) -> text from a rowspan
    header_rows = 1
    for r, tr in enumerate(tr for tr in table if tr.tag == "Tr"):
        row: list[str] = []
        col = 0
        for td in (td for td in tr if td.tag == "Td"):
            while (r, col) in pending:
                row.append(pending.pop((r, col)))
                col += 1
            text = _cell(td)
            rs, cs = int(td.get("rowspan", 1)), int(td.get("colspan", 1))
            if r == 0:
                header_rows = max(header_rows, rs)
            for c in range(cs):
                row.append(text)
                for k in range(1, rs):
                    pending[(r + k, col + c)] = text
            col += cs
        while (r, col) in pending:
            row.append(pending.pop((r, col)))
            col += 1
        grid.append(row)
    grid = [row for row in grid if any(row)]
    if not grid:
        return []
    width = max(len(row) for row in grid)
    grid = [row + [""] * (width - len(row)) for row in grid]
    header_rows = min(header_rows, len(grid) - 1) or 1
    header = []
    for c in range(width):
        parts: list[str] = []
        for row in grid[:header_rows]:
            if row[c] and row[c] not in parts:
                parts.append(row[c])
        # joined by a space, as in the index ("Primært referenceniveau P_KA [Gy·cm²]")
        header.append(" ".join(parts))
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(row) + " |" for row in grid[header_rows:]]
    return ["\n".join(lines)]


def _has(elem, tags) -> bool:
    return any(e.tag in tags for e in elem.iter() if e is not elem)


class _Writer:
    def __init__(self):
        self.blocks: list[str] = []
        self.prefix = ""  # bold label waiting for the next paragraph ("**§ 80.**")

    def para(self, text: str):
        if not text:
            return
        if self.prefix:
            text = f"{self.prefix} {text}"
            self.prefix = ""
        self.blocks.append(text)

    def raw(self, block: str):
        if self.prefix:  # a label directly before a table stands on its own line
            self.blocks.append(self.prefix)
            self.prefix = ""
        self.blocks.append(block)

    def walk(self, elem, depth: int = 2):
        tag = elem.tag
        if tag in ("Meta",):
            return
        if tag == "Table":
            for block in _table(elem):
                self.raw(block)
            return
        if tag == "Image":
            self.para("*[Figur, ikke gengivet i teksten]*")
            return
        if tag in ("Kapitel", "Bilag", "Afsnit", "ParagrafGruppe"):
            title = _heading(elem)
            if title:  # an untitled part (Afsnit) adds no heading level
                self.raw("#" * min(depth, 6) + " " + title)
            for child in elem:
                if child.tag not in ("Explicatus", "Rubrica"):
                    self.walk(child, depth + 1 if title else depth)
            return
        if tag in ("Paragraf", "Stk", "Indentatio"):
            label = elem.find("Explicatus")
            label_text = _text(label) if label is not None else ""
            if tag == "Paragraf" and label_text:
                # own heading: an agent can grep '^#### § 80' and read on
                self.raw("#" * min(depth, 6) + " " + label_text.rstrip("."))
            elif label_text:
                self.prefix = f"**{label_text}**" if tag == "Stk" else f"- {label_text}"
            if not _has(elem, ("Table", "Image", "Indentatio", "Stk")):
                body = " ".join(
                    _text(c) for c in elem if c.tag not in ("Explicatus", "Rubrica")
                )
                self.para(body.strip())
                return
            for child in elem:
                if child.tag != "Explicatus":
                    self.walk(child, depth)
            return
        # paragraph-level containers (Exitus, TekstGruppe, Index, Titel, ...)
        if len(elem) == 0 or not _has(elem, ("Table", "Image", "Exitus", "Indentatio", "Stk", "Paragraf")):
            self.para(_text(elem))
            return
        if (elem.text or "").strip():
            self.para(elem.text.strip())
        for child in elem:
            self.walk(child, depth)


def xml_to_markdown(xml_path: Path) -> tuple[str, str]:
    root = ET.parse(str(xml_path)).getroot()
    meta_elem = root.find("Meta")
    meta = {c.tag: (c.text or "").strip() for c in (meta_elem if meta_elem is not None else [])}
    title = meta.get("DocumentTitle", xml_path.stem)
    number, signed = meta.get("Number", ""), meta.get("DiesSigni", "")
    date = ".".join(reversed(signed.split("-"))) if signed else ""
    version = f"BEK nr {number} af {date}" if number else ""
    w = _Writer()
    w.raw(f"# {title}" + (f" ({version})" if version else ""))
    w.raw(
        f"Kilde: Retsinformation (retsinformation.dk), {version}. "
        "Konverteret fra XML; tabeller som Markdown-tabeller."
    )
    for child in root:
        if child.tag == "TitelGruppe":
            # the title is the H1 above; keep what follows it (the EU footnote)
            full = _text(child)
            rest = full[len(title):].strip() if full.startswith(title) else full
            if rest:
                w.para(f"Fodnote: {rest}")
            continue
        w.walk(child, 2)
    text = "\n\n".join(b for b in w.blocks if b.strip())
    return re.sub(r"\n{3,}", "\n\n", text) + "\n", f"{title} ({version})"


# --- PDFs: docling -> Markdown ------------------------------------------------------


def _converter(cpu: bool = False):
    from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    if not cpu:
        return DocumentConverter()  # same as DoclingLoader(converter=None) in ingestion
    options = PdfPipelineOptions(
        accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CPU)
    )
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )


def pdf_to_markdown(pdf: Path) -> str:
    try:
        doc = _converter().convert(str(pdf)).document
    except Exception as e:  # GPU busy (local model loaded): retry on the CPU
        print(f"  default device failed ({e}); retrying on CPU", file=sys.stderr)
        doc = _converter(cpu=True).convert(str(pdf)).document
    return doc.export_to_markdown()


# --- main ---------------------------------------------------------------------------


def _slug(name: str) -> str:
    name = re.sub(r"\.(pdf|xml)$", "", name, flags=re.I).removesuffix("_current")
    name = re.sub(r"[^\w\-]+", "_", name, flags=re.UNICODE).strip("_")
    return re.sub(r"_+", "_", name)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", choices=["dk", "pdf", "all"], default="all")
    ap.add_argument("--out", type=Path, default=ROOT / "export" / "markdown")
    args = ap.parse_args()
    out_dk, out_pdf = args.out / "dk", args.out / "iaea"
    rows: list[str] = []

    if args.only in ("dk", "all"):
        out_dk.mkdir(parents=True, exist_ok=True)
        for xml in sorted(DK_DIR.glob("*_current.xml")):
            md, label = xml_to_markdown(xml)
            target = out_dk / f"{_slug(xml.name)}.md"
            target.write_text(md, encoding="utf-8")
            rows.append(f"| dk/{target.name} | {label} | {xml.name} | {_sha(xml)} |")
            print(f"dk  {target.name}: {len(md):,} chars")

    if args.only in ("pdf", "all"):
        pdfs = [p for d in PDF_DIRS for p in sorted(d.rglob("*.pdf"))]
        out_pdf.mkdir(parents=True, exist_ok=True)
        out_guid = out_dk
        out_guid.mkdir(parents=True, exist_ok=True)
        jobs = [(p, out_pdf) for p in pdfs] + [(p, out_guid) for p in GUIDANCE_PDFS]
        for i, (pdf, folder) in enumerate(jobs, 1):
            target = folder / f"{_slug(pdf.name)}.md"
            rel = f"{folder.name}/{target.name}"
            rows.append(f"| {rel} | {pdf.stem} | {pdf.name} | {_sha(pdf)} |")
            if target.exists() and target.stat().st_size > 0:
                print(f"[{i}/{len(jobs)}] {pdf.name}: already exported")
                continue
            t0 = time.time()
            print(f"[{i}/{len(jobs)}] {pdf.name} ...", flush=True)
            md = f"# {pdf.stem}\n\nSource file: {pdf.name}\n\n" + pdf_to_markdown(pdf)
            target.write_text(md, encoding="utf-8")
            print(f"  {len(md):,} chars in {time.time() - t0:.0f} s", flush=True)

    manifest = args.out / "MANIFEST.md"
    old = manifest.read_text(encoding="utf-8").splitlines() if manifest.exists() else []
    keep = [r for r in old if r.startswith("| ") and not r.startswith("| File")]
    new_files = {r.split(" | ")[0] for r in rows}
    merged = sorted({r for r in keep if r.split(" | ")[0] not in new_files} | set(rows))
    manifest.write_text(
        "# Corpus export (Radiation Safety RAG)\n\n"
        "One Markdown file per indexed document. `dk/`: the five Danish executive "
        "orders (from Retsinformation XML) and two Danish guidance documents (PDF, "
        "docling). `iaea/`: IAEA publications (PDF, docling).\n\n"
        "| File | Document | Source file | sha256 (12) |\n|---|---|---|---|\n"
        + "\n".join(merged)
        + "\n",
        encoding="utf-8",
    )
    print(f"manifest: {manifest}")


if __name__ == "__main__":
    main()
