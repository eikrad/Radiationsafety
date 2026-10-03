"""Danish law from Retsinformation XML, chunked along its own structure (#130)."""

import json
from collections import Counter
from pathlib import Path

import pytest

from eval.scoring import normalize
from ingestion_dk import structure_chunks, xml_to_text

LAW = """<?xml version="1.0" encoding="utf-8"?>
<Dokument>
  <Meta>
    <AccessionNumber>B20250138405</AccessionNumber>
    <DocumentTitle>Bekendtgørelse om ioniserende stråling og strålebeskyttelse</DocumentTitle>
    <DiesSigni>2025-11-18</DiesSigni>
    <Number>1384</Number>
  </Meta>
  <DokumentIndhold>
    <Kapitel>
      <Explicatus>Kapitel 1</Explicatus>
      <Rubrica><Linea><Char>Gyldighedsområde</Char></Linea></Rubrica>
      <ParagrafGruppe>
        <Paragraf><Explicatus>§ 1.</Explicatus>
          <Stk><Exitus><Linea><Char>Bekendtgørelsen gælder for brug af strålekilder.</Char></Linea></Exitus></Stk>
        </Paragraf>
        <Paragraf><Explicatus>§ 2.</Explicatus>
          <Stk><Exitus><Linea><Char>Bekendtgørelsen gælder ikke for radon.</Char></Linea></Exitus></Stk>
        </Paragraf>
      </ParagrafGruppe>
      <ParagrafGruppe>
        <Rubrica><Linea><Char>Definitioner</Char></Linea></Rubrica>
        <Paragraf><Explicatus>§ 3.</Explicatus>
          <Stk>
            <Exitus><Linea><Char>I denne bekendtgørelse forstås ved:</Char></Linea></Exitus>
            <Exitus><Index>
              <Indentatio><Explicatus>1)</Explicatus><Exitus><Linea><Char>Anlæg: DEFINITION_ONE</Char></Linea></Exitus></Indentatio>
              <Indentatio><Explicatus>2)</Explicatus><Exitus><Linea><Char>Kilde: DEFINITION_TWO</Char></Linea></Exitus></Indentatio>
              <Indentatio><Explicatus>3)</Explicatus><Exitus><Linea><Char>Sikkerhedsvurdering: DEFINITION_THREE</Char></Linea></Exitus></Indentatio>
            </Index></Exitus>
          </Stk>
        </Paragraf>
      </ParagrafGruppe>
    </Kapitel>
    <Kapitel>
      <Explicatus>Kapitel 2</Explicatus>
      <Rubrica><Linea><Char>Dosisgrænser</Char></Linea></Rubrica>
      <ParagrafGruppe>
        <Paragraf><Explicatus>§ 14.</Explicatus>
          <Stk><Exitus><Linea><Char>Dosisgrænserne fremgår af bilag 2.</Char></Linea></Exitus></Stk>
        </Paragraf>
      </ParagrafGruppe>
    </Kapitel>
  </DokumentIndhold>
  <Bilag>
    <Explicatus>Bilag 2</Explicatus>
    <Rubrica><Linea><Char>Dosisgrænser</Char></Linea></Rubrica>
    <BilagIndhold><TekstGruppe>
      <Exitus><Linea><Char>Tabel 2.1.</Char></Linea></Exitus>
      <Exitus><Table>
        <Tr><Td><Exitus><Linea><Char>Gruppe</Char></Linea></Exitus></Td><Td><Exitus><Linea><Char>Grænse</Char></Linea></Exitus></Td></Tr>
        <Tr><Td><Exitus><Linea><Char>Arbejdstager over 18</Char></Linea></Exitus></Td><Td><Exitus><Linea><Char>20</Char></Linea></Exitus></Td><Td><Exitus><Linea><Char>ROW_ONE</Char></Linea></Exitus></Td></Tr>
        <Tr><Td><Exitus><Linea><Char>Arbejdstager 16-18</Char></Linea></Exitus></Td><Td><Exitus><Linea><Char>6</Char></Linea></Exitus></Td><Td><Exitus><Linea><Char>ROW_TWO</Char></Linea></Exitus></Td></Tr>
        <Tr><Td><Exitus><Linea><Char>Befolkningen</Char></Linea></Exitus></Td><Td><Exitus><Linea><Char>1</Char></Linea></Exitus></Td><Td><Exitus><Linea><Char>ROW_THREE</Char></Linea></Exitus></Td></Tr>
      </Table></Exitus>
    </TekstGruppe></BilagIndhold>
  </Bilag>
</Dokument>
"""

TITLE = (
    "Bekendtgørelse om ioniserende stråling og strålebeskyttelse "
    "(BEK nr 1384 af 18.11.2025)"
)


@pytest.fixture
def law(tmp_path) -> Path:
    path = tmp_path / "law_current.xml"
    path.write_text(LAW, encoding="utf-8")
    return path


def _chunk_with(chunks, text):
    [chunk] = [c for c in chunks if text in c.page_content]
    return chunk


def test_each_chunk_starts_with_its_law_chapter_and_paragraph(law):
    chunks = structure_chunks(law, "BEK nr 1384")

    chunk = _chunk_with(chunks, "Dosisgrænserne fremgår af bilag 2.")
    assert chunk.page_content.startswith(f"{TITLE} › Kapitel 2 Dosisgrænser › § 14\n")
    assert chunk.metadata["section"] == "§ 14"
    assert chunk.metadata["law_title"].startswith("Bekendtgørelse om ioniserende")
    assert chunk.metadata["doc_id"] == "B20250138405"
    assert chunk.metadata["document_type"] == "Danish law"
    assert chunk.metadata["source"] == "BEK nr 1384"


def test_short_paragraphs_of_one_group_share_a_chunk(law):
    chunks = structure_chunks(law, "BEK nr 1384")

    chunk = _chunk_with(chunks, "gælder for brug af strålekilder")
    assert "gælder ikke for radon" in chunk.page_content
    assert chunk.metadata["section"] == "§ 1–2"


def test_a_new_group_or_chapter_starts_a_new_chunk(law):
    chunks = structure_chunks(law, "BEK nr 1384")

    assert "DEFINITION_ONE" not in _chunk_with(chunks, "radon").page_content
    assert (
        "Dosisgrænserne fremgår"
        not in _chunk_with(chunks, "DEFINITION_ONE").page_content
    )


def test_a_long_paragraph_is_split_between_items_never_inside_one(law):
    chunks = structure_chunks(law, "BEK nr 1384", max_chars=95)

    definitions = [c for c in chunks if c.metadata["section"] == "§ 3"]
    assert len(definitions) > 1
    for marker in ("DEFINITION_ONE", "DEFINITION_TWO", "DEFINITION_THREE"):
        assert "Definitioner" in _chunk_with(chunks, marker).page_content.split("\n")[0]
    assert all(c.page_content.split("\n", 1)[0].endswith("› § 3") for c in definitions)


def test_an_annex_is_its_own_chunk_with_its_title(law):
    chunks = structure_chunks(law, "BEK nr 1384")

    chunk = _chunk_with(chunks, "ROW_TWO")
    assert chunk.page_content.startswith(f"{TITLE} › Bilag 2 Dosisgrænser\n")
    assert chunk.metadata["section"] == "Bilag 2"


def test_a_split_table_repeats_its_header_row_and_keeps_rows_whole(law):
    chunks = structure_chunks(law, "BEK nr 1384", max_chars=70)

    for marker, row in (
        ("ROW_ONE", "Arbejdstager over 18 20 ROW_ONE"),
        ("ROW_THREE", "Befolkningen 1 ROW_THREE"),
    ):
        chunk = _chunk_with(chunks, marker)
        assert row in chunk.page_content
        assert "Gruppe Grænse" in chunk.page_content
    assert "ROW_ONE" not in _chunk_with(chunks, "ROW_THREE").page_content


# --- the real orders: re-chunking must not cut an evidence quote ---------------------

_DOCS = Path(__file__).resolve().parent.parent / "documents" / "Bekendtgørelse"
_GOLDEN = Path(__file__).resolve().parent.parent / "eval" / "data" / "golden.json"


@pytest.mark.parametrize(
    "xml", sorted(_DOCS.glob("*_current.xml")), ids=lambda p: p.stem
)
def test_every_golden_quote_in_a_law_lies_within_one_of_its_chunks(xml):
    quotes = {
        q
        for item in json.loads(_GOLDEN.read_text(encoding="utf-8"))
        for n in item.get("nuggets") or []
        for q in n["evidence"]
    }
    full_text = normalize(xml_to_text(xml))
    chunks = [normalize(c.page_content) for c in structure_chunks(xml, xml.stem)]

    in_law = [q for q in quotes if normalize(q) in full_text]
    cut = [q for q in in_law if not any(normalize(q) in c for c in chunks)]
    assert in_law and cut == []


@pytest.mark.parametrize(
    "xml", sorted(_DOCS.glob("*_current.xml")), ids=lambda p: p.stem
)
def test_no_word_of_a_law_is_lost_between_its_chunks(xml):
    words = Counter(normalize(xml_to_text(xml)).split())
    chunked = Counter(
        " ".join(
            normalize(c.page_content) for c in structure_chunks(xml, xml.stem)
        ).split()
    )

    assert words - chunked == Counter()
