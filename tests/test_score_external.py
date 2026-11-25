"""eval.score_external: context from an external agent's trace, evidence in tables."""

import json

from eval.score_external import (
    CorpusFiles,
    build_outputs,
    evidence_context,
    flatten_markdown,
    linearise_tables,
    load_answers,
    rebuild_context,
)
from eval.scoring import evidence_found

TABLE = """Tabel 8

| Transport index | Maximum dose rate at any point on external surface | Category. |
|---|---|---|
| 0 a | Not more than 0.005 mSv/h | I-WHITE |
| More than 0 but not more than 1 a | More than 0.005 mSv/h but not more than 0.5 mSv/h | II-YELLOW |
"""


def test_tables_are_linearised_like_the_index_chunks():
    lines = linearise_tables(TABLE).splitlines()
    assert lines[1] == (
        "More than 0 but not more than 1 a, Maximum dose rate at any point on "
        "external surface = More than 0.005 mSv/h but not more than 0.5 mSv/h. "
        "More than 0 but not more than 1 a, Category. = II-YELLOW"
    )


def test_evidence_quoted_from_a_linearised_chunk_is_found_in_a_markdown_table():
    quote = "More than 0 but not more than 1 a, Category. = II-YELLOW"
    assert not evidence_found([quote], [TABLE])
    assert evidence_found([quote], [evidence_context(TABLE)])


def test_danish_tables_and_lists_flatten_like_the_xml_chunks():
    text = "| Co-58 | 1 | 100 |\n|---|---|---|\n| Co-60 | 0,1 | 10 |\n\n- 5) Transport af fissilt\n\n**Stk. 2.** P_KA [Gy·cm^2]"
    flat = flatten_markdown(text)
    assert "Co-58 1 100 Co-60 0,1 10" in flat
    assert "5) Transport af fissilt" in flat
    assert evidence_found(["Stk. 2. P_KA [Gy·cm²]"], [flat])


def test_header_cells_joined_by_a_colon_in_the_first_export_still_match():
    text = "| Undersøgelse | Primært referenceniveau: P_KA [Gy·cm^2] |\n|---|---|\n| Thorax | 0,3 |"
    quote = "Undersøgelse Primært referenceniveau P_KA [Gy·cm²]"
    assert evidence_found([quote], [evidence_context(text)])


def test_the_judge_sees_the_context_unchanged():
    # the re-renderings are appended, never replacing what the agent read
    assert evidence_context(TABLE).startswith(TABLE)


def _corpus(tmp_path):
    (tmp_path / "dk").mkdir()
    (tmp_path / "dk" / "law.md").write_text(
        "\n".join(f"line {n}" for n in range(1, 101)), encoding="utf-8"
    )
    return CorpusFiles(tmp_path)


def test_read_steps_are_rebuilt_from_the_export(tmp_path):
    corpus = _corpus(tmp_path)
    record = {
        "trace": [
            {"tool": "grep", "output": "given output"},  # text in the trace is kept
            {"file": "dk/law.md", "offset": 9, "limit": 2},  # 0-based offset
            {"path": "export/markdown/dk/law.md", "lines": "50-51"},
            {"file": "law.md", "start": 99, "end": 100},  # file name alone
            {
                "tool": "read",
                "file": "dk/law.md",
                "offset": 50,
                "limit": 2,
                "lines": [50, 51],
            },
        ]
    }
    context, source, problems = rebuild_context(record, corpus)
    assert source == "trace" and not problems
    assert context.split("\n\n") == [
        "given output",
        "[dk/law.md, lines 10-11]\nline 10\nline 11",
        "[dk/law.md, lines 50-51]\nline 50\nline 51",  # read twice, shown once
        "[dk/law.md, lines 99-100]\nline 99\nline 100",
    ]


def test_grep_output_is_replayed_on_the_export(tmp_path):
    corpus = _corpus(tmp_path)
    record = {
        "trace": [
            {
                "tool": "grep",
                "pattern": "line 4[2-3]$",
                "path": "/home/x/corpus",
                "n_matches": 2,
            },
            {
                "tool": "grep",
                "pattern": "line 7$",
                "include": "dk/*.md",
                "n_matches": 1,
            },
            {
                "tool": "grep",
                "pattern": "line 9$",
                "include": "other*.md",
                "n_matches": 0,
            },
            {"tool": "glob", "pattern": "dk/*.md"},
        ]
    }
    context, _, problems = rebuild_context(record, corpus)
    assert not problems
    assert context.split("\n\n") == [
        "[dk/law.md, lines 7-7]\nline 7",
        "[dk/law.md, lines 42-43]\nline 42\nline 43",
    ]


def test_a_search_that_found_nothing_on_one_side_is_reported(tmp_path):
    record = {"trace": [{"tool": "grep", "pattern": "line", "n_matches": 0}]}
    _, _, problems = rebuild_context(record, _corpus(tmp_path))
    assert problems == ["grep replay found 100 matches, the trace says 0: line"]


def test_include_with_a_folder_is_matched_below_the_searched_folder(tmp_path):
    # ripgrep: include "dk/*.md" under path "dk" looks for dk/dk/*.md
    record = {
        "trace": [
            {
                "tool": "grep",
                "pattern": "line 7$",
                "path": "dk",
                "include": "dk/*.md",
                "n_matches": 0,
            }
        ]
    }
    context, _, problems = rebuild_context(record, _corpus(tmp_path))
    assert (context, problems) == ("", [])


def test_without_a_trace_the_quotes_stand_in(tmp_path):
    record = {
        "trace": [{"tool": "read", "file": "missing.md"}],
        "quotes": [{"file": "x", "quote": "Q"}],
    }
    context, source, problems = rebuild_context(record, _corpus(tmp_path))
    assert (context, source) == ("Q", "quotes")
    assert problems == ["trace file not in the export: missing.md"]


def test_missing_answers_and_changed_questions_are_reported(tmp_path):
    golden = [
        {"id": "a", "question": "Q a?", "expected_behavior": "answer", "nuggets": []},
        {"id": "b", "question": "Q b?", "expected_behavior": "answer", "nuggets": []},
    ]
    path = tmp_path / "answers.jsonl"
    path.write_text(
        json.dumps({"id": "a", "question": "Other?", "answer": "A"})
        + "\n"
        + json.dumps({"id": "z", "answer": "Z"})
        + "\n",
        encoding="utf-8",
    )
    outputs, problems = build_outputs(golden, load_answers(path), _corpus(tmp_path))
    assert outputs["a"]["generation"] == "A"
    assert outputs["b"]["error"]
    assert any("question text differs" in p for p in problems)
    assert any("no answer" in p for p in problems)
    assert any("not in the golden selection: z" in p for p in problems)


def test_main_loads_the_env_file_like_run_eval(monkeypatch):
    """The judge settings and API keys live in .env, as for eval.run_eval."""
    import dotenv
    import pytest

    from eval import score_external

    class Loaded(Exception):
        pass

    def fake_load_dotenv(*args, **kwargs):
        raise Loaded

    monkeypatch.setattr(dotenv, "load_dotenv", fake_load_dotenv)
    monkeypatch.setattr("sys.argv", ["score_external", "answers.jsonl"])
    with pytest.raises(Loaded):
        score_external.main()


def test_each_file_is_shown_with_its_title_like_the_index_chunks(tmp_path):
    # every index chunk starts with its law's title line ("... (BEK nr 1385 af
    # 18.11.2025)"), so the judge can check a cited order number; the export has
    # it only on line 1, which an agent that reads from an offset never sees
    (tmp_path / "dk").mkdir()
    (tmp_path / "dk" / "law.md").write_text(
        "# Bekendtgørelse om x (BEK nr 1385 af 18.11.2025)\n"
        + "\n".join(f"line {n}" for n in range(2, 101)),
        encoding="utf-8",
    )
    record = {
        "trace": [
            {"file": "dk/law.md", "offset": 9, "limit": 2},
            {"file": "dk/law.md", "offset": 49, "limit": 1},
        ]
    }
    context, _, problems = rebuild_context(record, CorpusFiles(tmp_path))
    assert not problems
    assert context.split("\n\n") == [
        "[dk/law.md, lines 1-1]\n# Bekendtgørelse om x (BEK nr 1385 af 18.11.2025)",
        "[dk/law.md, lines 10-11]\nline 10\nline 11",
        "[dk/law.md, lines 50-50]\nline 50",
    ]
