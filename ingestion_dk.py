"""Danish law from Retsinformation XML: plain text and structure-aware chunks (#130).

A Retsinformation order is Afsnit > Kapitel > ParagrafGruppe > Paragraf > Stk, with
numbered items (Index > Indentatio) and annexes (Bilag) holding tables (Table > Tr).
Chunks follow that structure instead of a character count: short paragraphs of one
group share a chunk, a long one is split only between its Stk., items, lines or
table rows (a split table repeats its header row), and every chunk starts with the
law, chapter and paragraph it comes from, e.g.

    Bekendtgørelse om radioaktive stoffer (BEK nr 1385 af 18.11.2025)
    › Kapitel 5 Registrering › § 19
"""

import re
import xml.etree.ElementTree as ET
from pathlib import Path

from langchain_core.documents import Document

# Body characters per chunk; the header line comes on top. Most paragraphs are a
# few hundred characters, so several of one group fit; the long ones (definitions,
# annex tables) are split at their own item boundaries.
MAX_CHUNK_CHARS = 1500

HEADER_SEPARATOR = " › "

_NO_SPACE_BEFORE = (",", ".", ";", ":", ")")

# Elements split no further: a line, a table row and labels stay whole.
_ATOMIC = {"Linea", "Tr", "Char", "Explicatus", "Rubrica"}


def xml_meta(root) -> dict[str, str]:
    """The Retsinformation <Meta> fields (title, number, date, accession number)."""
    meta = root.find(".//Meta")
    if meta is None:
        return {}
    return {child.tag: (child.text or "").strip() for child in meta}


def _render(elem) -> list[str]:
    """Text pieces of an element in document order, Meta left out.

    Superscripts and subscripts are attached to the preceding text as ^ and _
    (10^6, CTDI_vol): flattening them turns 1·10^6 Bq into "1·10 6" here, and
    into "106" in the PDF text.
    """
    if elem.tag == "Meta":
        return []
    pieces: list[str] = []

    def add(text: str | None, glue: str = "") -> None:
        text = (text or "").strip()
        if not text:
            return
        if pieces and (glue or text.startswith(_NO_SPACE_BEFORE)):
            pieces[-1] += glue + text
        else:
            pieces.append(text)

    add(elem.text)
    for child in elem:
        child_pieces = _render(child)
        if child_pieces:
            add(child_pieces[0], _glue_of(child))
            pieces.extend(child_pieces[1:])
        add(child.tail)
    return pieces


def _glue_of(elem) -> str:
    """How a Char attaches to the text before it: ^ superscript, _ subscript."""
    form = elem.get("formaChar", "") if elem.tag == "Char" else ""
    return "^" if "Superscript" in form else "_" if "Subscript" in form else ""


def _text(elem) -> str:
    return re.sub(r"\s+", " ", " ".join(_render(elem))).strip()


def law_title_line(meta: dict[str, str]) -> str:
    """'Bekendtgørelse om radioaktive stoffer (BEK nr 1385 af 18.11.2025)'."""
    title = meta.get("DocumentTitle", "")
    number, signed = meta.get("Number", ""), meta.get("DiesSigni", "")
    if not title or not number:
        return title
    date = ".".join(reversed(signed.split("-"))) if signed else ""
    return f"{title} (BEK nr {number}{' af ' + date if date else ''})"


def _parse(xml_path: Path):
    try:
        return ET.parse(str(xml_path)).getroot()
    except (ET.ParseError, OSError):
        return None


def xml_to_text(xml_path: Path) -> str:
    """Plain text of a Retsinformation XML law, headed by its title and number.

    The <Meta> block (document type codes, signatures) is not law text and is
    left out; the title line names the version.
    """
    root = _parse(xml_path)
    if root is None:
        return ""
    body = " ".join(_render(root))
    title = law_title_line(xml_meta(root))
    text = f"{title} {body}" if title else body
    return re.sub(r"\s+", " ", text).strip()


# --- structure-aware chunks ----------------------------------------------------------


def _pack(pieces: list[str], budget: int) -> list[str]:
    """Join consecutive pieces while they fit the budget; a piece is never cut."""
    packed: list[str] = []
    for piece in pieces:
        if packed and len(packed[-1]) + 1 + len(piece) <= budget:
            packed[-1] += " " + piece
        else:
            packed.append(piece)
    return packed


def _split(elem, budget: int) -> list[str]:
    """The element's text in pieces of at most `budget` characters, cut only between
    child elements. A piece that cannot be cut (one long line or row) stays whole."""
    text = _text(elem)
    if len(text) <= budget or elem.tag in _ATOMIC or len(elem) == 0:
        return [text] if text else []
    if elem.tag == "Table":
        return _split_table(elem, budget)
    label = ""
    pieces: list[str] = []
    for child in elem:
        if child.tag == "Explicatus" and not pieces and not label:
            label = _text(child)
        elif child.tag != "Rubrica":
            pieces.extend(_split(child, budget))
    if label:
        pieces = [f"{label} {pieces[0]}", *pieces[1:]] if pieces else [label]
    return _pack(pieces, budget)


def _split_table(table, budget: int) -> list[str]:
    """Rows packed into pieces, each headed by the table's first row."""
    rows = [t for t in (_text(tr) for tr in table) if t]
    if len(rows) < 2:
        return rows
    header, body = rows[0], rows[1:]
    room = max(budget - len(header) - 1, 1)
    return [f"{header} {group}" for group in _pack(body, room)]


def _section_label(paragraf) -> str:
    label = paragraf.find("Explicatus")
    return _text(label).rstrip(".") if label is not None else ""


def _section_range(first: str, last: str) -> str:
    if first == last:
        return first
    return f"{first}–{last.removeprefix('§ ')}"


def _heading(elem) -> str:
    """'Kapitel 5 Registrering', 'Bilag 2 Dosisgrænser', a group's title, or ''."""
    parts = [elem.find(tag) for tag in ("Explicatus", "Rubrica")]
    return " ".join(_text(p) for p in parts if p is not None).strip()


def _units(root, budget: int):
    """(heading path, section label, text) in document order, before packing."""
    for elem in (root.find(".//Titel"), root.find(".//Indledning")):
        if elem is not None:
            for piece in _split(elem, budget):
                yield (), "Indledning", piece
    parent = {child: elem for elem in root.iter() for child in elem}
    for paragraf in root.iter("Paragraf"):
        path, node = [], parent.get(paragraf)
        while node is not None:
            if node.tag in ("Kapitel", "ParagrafGruppe") and _heading(node):
                path.insert(0, _heading(node))
            node = parent.get(node)
        label = _section_label(paragraf)
        for piece in _split(paragraf, budget):
            yield tuple(path), label, piece
    for bilag in root.iter("Bilag"):
        content = bilag.find("BilagIndhold")
        if content is None:
            continue
        number = bilag.find("Explicatus")
        label = _text(number) if number is not None else "Bilag"
        for piece in _split(content, budget):
            yield (_heading(bilag),), label, piece


def structure_chunks(
    xml_path: Path, source_label: str, max_chars: int = MAX_CHUNK_CHARS
) -> list[Document]:
    """Chunks of a Retsinformation XML law along its paragraphs and annexes.

    Paragraphs of one group (same chapter, same group title) are packed together
    up to max_chars; a new group, chapter or annex starts a new chunk.
    """
    root = _parse(xml_path)
    if root is None:
        return []
    meta = xml_meta(root)
    title = law_title_line(meta)
    base = {
        "source": source_label,
        "document_type": "Danish law",
        "law_title": meta.get("DocumentTitle", ""),
        "doc_id": meta.get("AccessionNumber", ""),
    }

    groups: list[list] = []  # [path, first label, last label, body]
    for path, label, text in _units(root, max_chars):
        last = groups[-1] if groups else None
        if (
            last is not None
            and last[0] == path
            and (label == last[2] or label[:1] == last[2][:1] == "§")
            and len(last[3]) + 1 + len(text) <= max_chars
        ):
            last[2], last[3] = label, f"{last[3]} {text}"
        else:
            groups.append([path, label, label, text])

    chunks = []
    for path, first, last, body in groups:
        section = _section_range(first, last)
        header = HEADER_SEPARATOR.join(p for p in (title, *path, section) if p)
        # An annex's number is already part of its heading.
        if path and path[0].startswith(section):
            header = HEADER_SEPARATOR.join(p for p in (title, *path) if p)
        chunks.append(
            Document(
                page_content=f"{header}\n{body}",
                metadata={
                    **base,
                    "section": section,
                    "chapter": path[0] if path else "",
                },
            )
        )
    return chunks
