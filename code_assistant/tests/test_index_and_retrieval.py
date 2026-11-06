"""Outlines, symbol-aligned chunks, incremental indexing and hybrid search."""

from __future__ import annotations

from pathlib import Path

import pytest

from code_assistant.workspace.indexer import Indexer, chunk_source, file_tree, outline, outline_python
from code_assistant.workspace.retriever import Retriever, query_terms
from code_assistant.workspace.sandbox import Sandbox, Workspace, workspace_id_for

from .conftest import _toy_embedding


def _workspace(root: Path) -> Workspace:
    return Workspace(workspace_id_for(root), root, root.name)


def test_python_outline_and_chunks():
    src = "import os\n\n\n@dec\ndef a():\n    pass\n\n\nclass B:\n    def m(self):\n        return 1\n"
    symbols = outline_python(src)
    assert [(s.name, s.kind, s.start_line, s.end_line, s.parent) for s in symbols] == [
        ("a", "function", 4, 6, None), ("B", "class", 9, 11, None), ("m", "method", 10, 11, "B"),
    ]
    chunks = chunk_source("x.py", src, symbols)
    assert [(c.start_line, c.end_line, c.symbol) for c in chunks] == [(1, 3, None), (4, 6, "a"), (9, 11, "B")]  # blank lines 7-8 are skipped


def test_regex_outline_for_js():
    src = "export class Foo {\n  bar() {}\n}\n\nexport async function baz(x) {\n  return x;\n}\nconst qux = (a) => a;\n"
    names = {(s.name, s.kind) for s in outline("a.ts", src)}
    assert {("Foo", "class"), ("baz", "function"), ("qux", "function")} <= names


def test_large_symbol_split_into_windows():
    body = "\n".join(f"    x{i} = {i}" for i in range(300))
    src = f"def big():\n{body}\n"
    chunks = chunk_source("big.py", src, outline_python(src), max_lines=120)
    assert len(chunks) == 3 and chunks[0].start_line == 1 and chunks[-1].end_line == 301


def test_chunk_text_is_redacted():
    src = 'API_KEY = "sk-abcdefghijklmnopqrstuvwxyz012345"\n'
    assert "sk-abc" not in chunk_source("c.py", src, [])[0].text


def test_file_tree_excludes_secrets_and_ignored(workspace_dir):
    tree = file_tree(Sandbox(workspace_dir))
    assert "calc.py" in tree and ".env" not in tree and "node_modules" not in tree


def test_incremental_indexing_only_reembeds_changed_files(workspace_dir, tmp_path):
    embedded: list[str] = []

    def embed(texts):
        embedded.extend(texts)
        return [_toy_embedding(t) for t in texts]

    ws = _workspace(workspace_dir)
    indexer = Indexer(tmp_path / "idx", embed_fn=embed)
    index = indexer.build(ws)
    assert index.status == "ready" and "pkg/calc.py" in index.files and index.vectors
    first = len(embedded)

    indexer.build(ws)
    assert len(embedded) == first  # nothing changed

    (workspace_dir / "pkg" / "calc.py").write_text("def mul(a, b):\n    return a * b\n", encoding="utf-8")
    indexer.build(ws)
    assert len(embedded) == first + 1
    assert [s.name for s in index.symbols_for("pkg/calc.py")] == ["mul"]

    reloaded = Indexer(tmp_path / "idx", embed_fn=embed).get(ws)  # from disk
    assert reloaded.status == "ready" and len(reloaded.vectors) == len(index.vectors)


def test_index_without_embeddings_is_keyword_only(workspace_dir, tmp_path):
    index = Indexer(tmp_path / "idx", embed_fn=None).build(_workspace(workspace_dir))
    assert index.status == "ready" and not index.vectors and "keyword" in index.error


def test_query_terms_prefers_identifiers():
    assert query_terms("why does Calculator.total return wrong value in add_items?")[:2] == ["Calculator.total", "add_items"]


@pytest.mark.parametrize("rg", ["auto", None])
def test_keyword_search_excludes_secrets(workspace_dir, rg):
    retriever = Retriever(Sandbox(workspace_dir), rg_path=rg)
    paths = {p for p, _l, _t in retriever.keyword("OPENAI_API_KEY")}
    assert ".env" not in paths
    hits = retriever.keyword("return a - b")
    assert hits and hits[0][0] == "pkg/calc.py" and hits[0][1] == 5


def test_hybrid_search_returns_line_ranges(workspace_dir, tmp_path):
    ws = _workspace(workspace_dir)
    index = Indexer(tmp_path / "idx", embed_fn=lambda ts: [_toy_embedding(t) for t in ts]).build(ws)
    retriever = Retriever(Sandbox(workspace_dir), index, embed_query=_toy_embedding)
    hits = retriever.search("the add function returns the wrong result")
    assert hits[0].path == "pkg/calc.py"
    assert hits[0].symbol == "add" and hits[0].start_line == 4
    assert {"keyword", "semantic"} & hits[0].sources
    assert "pkg/calc.py:4-" in hits[0].render()
