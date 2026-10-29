"""Hybrid code search: exact/regex (ripgrep, Python fallback) + embeddings, merged and reranked.

- Keyword search finds exact identifiers with ripgrep when it is installed, or
  a bounded Python scan otherwise. Secret and ignored files are always excluded.
- Semantic search uses the workspace ``CodeIndex`` vectors (if embedded).
- Results are merged at chunk level with reciprocal-rank fusion, then reranked
  with a lexical bonus (query terms in the symbol name / path / text).
- Every hit carries a file path and a line range so the agent can read_file it.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Callable

from .indexer import CodeIndex
from .sandbox import Sandbox, SandboxError, redact

logger = logging.getLogger(__name__)

QueryEmbedFn = Callable[[str], list[float]]

_RRF_K = 60
_SNIPPET_LINES = 40
_STOPWORDS = frozenset("""
a an and are as at be by can code do does file files find fix for from how i in is it its me my of on or
please show that the this to used uses using what when where which who why with work works
function method class bug error should would could there their them into about all any not
""".split())


@dataclass
class SearchHit:
    """One search result: a file region plus where the score came from."""

    path: str
    start_line: int
    end_line: int
    snippet: str
    score: float = 0.0
    sources: set[str] = field(default_factory=set)
    symbol: str | None = None

    def render(self) -> str:
        label = f" ({self.symbol})" if self.symbol else ""
        return f"--- {self.path}:{self.start_line}-{self.end_line}{label} [{'+'.join(sorted(self.sources))}]\n{self.snippet}"


def query_terms(query: str) -> list[str]:
    """Distinctive identifiers from a natural-language query, most specific first."""
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_\.]{2,}", query)
    seen: dict[str, None] = {}
    for token in tokens:
        token = token.strip(".")
        if token.lower() not in _STOPWORDS and token not in seen:
            seen[token] = None
    terms = list(seen)
    # Identifiers that look like code (snake_case, CamelCase, dotted) first.
    terms.sort(key=lambda t: (not ("_" in t or "." in t or (t[0].islower() and any(c.isupper() for c in t)) or t[1:] != t[1:].lower()), -len(t)))
    return terms[:8]


class Retriever:
    """Search one workspace. ``index`` and ``embed_query`` are optional (keyword-only without them)."""

    def __init__(
        self,
        sandbox: Sandbox,
        index: CodeIndex | None = None,
        embed_query: QueryEmbedFn | None = None,
        *,
        rg_path: str | None = "auto",
        timeout: float = 20.0,
    ) -> None:
        self.sandbox = sandbox
        self.index = index
        self.embed_query = embed_query
        self.rg_path = shutil.which("rg") if rg_path == "auto" else rg_path
        self.timeout = timeout

    # --- keyword ------------------------------------------------------------------------------

    def keyword(self, pattern: str, *, regex: bool = False, ignore_case: bool = True, max_results: int = 60) -> list[tuple[str, int, str]]:
        """Line hits ``(path, line, text)`` for a literal string or a regex."""
        if not pattern:
            return []
        if regex:
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"invalid regex: {exc}") from exc
        hits = self._ripgrep(pattern, regex, ignore_case, max_results) if self.rg_path else None
        if hits is None:
            hits = self._python_grep(pattern, regex, ignore_case, max_results)
        return hits

    def _ripgrep(self, pattern: str, regex: bool, ignore_case: bool, max_results: int) -> list[tuple[str, int, str]] | None:
        args = [self.rg_path, "--json", "--line-number", "--max-count", "20", "--max-filesize", "1M",
                "--max-columns", "400", "--no-follow"]
        if ignore_case:
            args.append("-i")
        if not regex:
            args.append("-F")
        args += ["-e", pattern, "--", "."]
        try:
            proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
                args, cwd=self.sandbox.root, capture_output=True, timeout=self.timeout, shell=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            logger.warning("ripgrep failed (%s); using Python search", exc)
            return None
        if proc.returncode not in (0, 1):
            return None
        hits: list[tuple[str, int, str]] = []
        for raw in proc.stdout.decode("utf-8", errors="replace").splitlines():
            try:
                event = json.loads(raw)
            except ValueError:
                continue
            if event.get("type") != "match":
                continue
            data = event["data"]
            path = (data["path"].get("text") or "").replace("\\", "/")
            path = path[2:] if path.startswith("./") else path
            if not self.sandbox.is_readable_file(path):
                continue
            text = (data["lines"].get("text") or "").rstrip("\n")
            hits.append((path, int(data["line_number"]), redact(text)))
            if len(hits) >= max_results:
                break
        return hits

    def _python_grep(self, pattern: str, regex: bool, ignore_case: bool, max_results: int) -> list[tuple[str, int, str]]:
        flags = re.IGNORECASE if ignore_case else 0
        compiled = re.compile(pattern if regex else re.escape(pattern), flags)
        hits: list[tuple[str, int, str]] = []
        for rel in self.sandbox.iter_files(max_files=5000):
            try:
                text = self.sandbox.read_text(rel)
            except (SandboxError, OSError):
                continue
            per_file = 0
            for number, line in enumerate(text.splitlines(), start=1):
                if compiled.search(line):
                    hits.append((rel, number, redact(line[:400])))
                    per_file += 1
                    if len(hits) >= max_results:
                        return hits
                    if per_file >= 20:
                        break
        return hits

    # --- chunk-level helpers ------------------------------------------------------------------

    def _hit_for_line(self, path: str, line: int) -> SearchHit:
        chunk = self.index.chunk_at(path, line) if self.index else None
        if chunk is not None:
            lines = chunk.text.splitlines()
            start = chunk.start_line
            if len(lines) > _SNIPPET_LINES:  # centre long chunks on the match
                offset = max(0, min(line - start - _SNIPPET_LINES // 2, len(lines) - _SNIPPET_LINES))
                lines, start = lines[offset:offset + _SNIPPET_LINES], start + offset
            return SearchHit(path, start, start + len(lines) - 1, "\n".join(lines), symbol=chunk.symbol)
        try:
            text_lines = self.sandbox.read_text(path).splitlines()
        except (SandboxError, OSError):
            return SearchHit(path, line, line, "")
        start, end = max(1, line - 6), min(len(text_lines), line + 6)
        return SearchHit(path, start, end, redact("\n".join(text_lines[start - 1:end])))

    @staticmethod
    def _key(hit: SearchHit) -> tuple[str, int]:
        return hit.path, hit.start_line

    # --- hybrid -------------------------------------------------------------------------------

    def search(self, query: str, *, k: int = 8) -> list[SearchHit]:
        """Hybrid search for a natural-language or identifier query."""
        query = query.strip()
        if not query:
            return []
        ranked_lists: list[list[SearchHit]] = []

        terms = query_terms(query) or [query]
        keyword_hits: list[SearchHit] = []
        seen: set[tuple[str, int]] = set()
        pattern = "|".join(re.escape(t) for t in terms)
        try:
            lines = self.keyword(rf"\b(?:{pattern})\b", regex=True, max_results=120)
        except ValueError:
            lines = []
        counts: dict[tuple[str, int], set[str]] = {}
        for path, number, text in lines:
            hit = self._hit_for_line(path, number)
            key = self._key(hit)
            matched = {t for t in terms if t.lower() in text.lower()}
            counts.setdefault(key, set()).update(matched)
            if key not in seen:
                seen.add(key)
                hit.sources.add("keyword")
                keyword_hits.append(hit)
        keyword_hits.sort(key=lambda h: -len(counts.get(self._key(h), ())))
        ranked_lists.append(keyword_hits)

        if self.index is not None and self.embed_query is not None and self.index.vectors:
            try:
                vector = self.embed_query(query)
                semantic = []
                for chunk, score in self.index.search_vectors(vector, k * 3):
                    hit = SearchHit(chunk.path, chunk.start_line, chunk.end_line,
                                    "\n".join(chunk.text.splitlines()[:_SNIPPET_LINES]), score, {"semantic"}, chunk.symbol)
                    semantic.append(hit)
                ranked_lists.append(semantic)
            except Exception as exc:  # noqa: BLE001 - semantic search is best-effort
                logger.warning("semantic search failed: %s", exc)

        return self._fuse(ranked_lists, terms, k)

    def _fuse(self, ranked_lists: list[list[SearchHit]], terms: list[str], k: int) -> list[SearchHit]:
        merged: dict[tuple[str, int], SearchHit] = {}
        scores: dict[tuple[str, int], float] = {}
        for ranked in ranked_lists:
            for rank, hit in enumerate(ranked):
                key = self._key(hit)
                scores[key] = scores.get(key, 0.0) + 1.0 / (_RRF_K + rank + 1)
                if key in merged:
                    merged[key].sources |= hit.sources
                else:
                    merged[key] = hit
        lowered = [t.lower() for t in terms]
        for key, hit in merged.items():
            bonus = 0.0
            symbol = (hit.symbol or "").lower()
            path = hit.path.lower()
            text = hit.snippet.lower()
            for term in lowered:
                if term == symbol:
                    bonus += 0.02
                elif term in symbol:
                    bonus += 0.01
                if term in path:
                    bonus += 0.005
                if term in text:
                    bonus += 0.002
            if len(hit.sources) > 1:
                bonus += 0.01
            hit.score = round(scores[key] + bonus, 5)
        return sorted(merged.values(), key=lambda h: -h.score)[:k]
