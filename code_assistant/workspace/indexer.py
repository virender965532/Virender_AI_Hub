"""Codebase index: file tree, per-file outlines, symbol-aware chunks and embeddings.

- Outlines come from Python's ``ast`` for ``.py`` files and from per-language
  regexes otherwise.
- Chunks are split at top-level function/class boundaries (oversized symbols
  are split into line windows), never at fixed character offsets.
- Chunk text is redacted before it is embedded or stored.
- Vectors are stored per workspace under ``CA_DATA_DIR/index/<workspace_id>/``
  together with each file's content hash; re-indexing only re-embeds files
  whose hash changed. Search uses FAISS when available, NumPy otherwise.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .sandbox import Sandbox, SandboxError, Workspace, redact, sha256_text

try:
    import faiss
except ImportError:  # pragma: no cover - faiss-cpu is in requirements.txt
    faiss = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

EmbedFn = Callable[[list[str]], list[list[float]]]

INDEX_VERSION = 1
MAX_CHUNK_LINES = 120
MAX_INDEXED_FILES = 5000
TEXT_SUFFIXES = frozenset({
    ".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".java", ".kt", ".kts", ".go", ".rs", ".rb",
    ".php", ".cs", ".c", ".h", ".cpp", ".hpp", ".cc", ".swift", ".scala", ".sh", ".bash", ".ps1", ".sql",
    ".html", ".htm", ".css", ".scss", ".vue", ".svelte", ".md", ".rst", ".txt", ".json", ".yaml", ".yml",
    ".toml", ".ini", ".cfg", ".xml", ".gradle", ".dockerfile", ".tf", ".proto", ".graphql",
})
TEXT_NAMES = frozenset({"Dockerfile", "Makefile", "Procfile", "Jenkinsfile", ".gitignore", ".dockerignore"})


@dataclass(frozen=True)
class Symbol:
    """A class / function / method in a file outline (1-based inclusive lines)."""

    name: str
    kind: str
    start_line: int
    end_line: int
    parent: str | None = None


@dataclass(frozen=True)
class Chunk:
    """A retrievable slice of a file, aligned to symbol boundaries."""

    id: str
    path: str
    start_line: int
    end_line: int
    symbol: str | None
    text: str


@dataclass
class FileEntry:
    sha256: str
    symbols: list[Symbol] = field(default_factory=list)
    chunk_ids: list[str] = field(default_factory=list)


def is_indexable(rel_path: str) -> bool:
    name = rel_path.rsplit("/", 1)[-1]
    return name in TEXT_NAMES or Path(name).suffix.lower() in TEXT_SUFFIXES


# --- outlines ----------------------------------------------------------------------------------

def outline_python(source: str) -> list[Symbol]:
    """Classes, functions and methods via ``ast`` (raises ``SyntaxError`` on invalid code)."""
    tree = ast.parse(source)
    symbols: list[Symbol] = []

    def visit(nodes: list[ast.stmt], parent: str | None) -> None:
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start = min([node.lineno, *(d.lineno for d in node.decorator_list)])
                end = getattr(node, "end_lineno", None) or node.lineno
                if isinstance(node, ast.ClassDef):
                    kind = "class"
                else:
                    kind = "method" if parent and parent[0].isupper() else "function"
                symbols.append(Symbol(node.name, kind, start, end, parent))
                if isinstance(node, ast.ClassDef):
                    visit(node.body, node.name)

    visit(tree.body, None)
    return symbols


_REGEX_OUTLINES: dict[str, list[tuple[re.Pattern[str], str]]] = {
    "js": [
        (re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+([A-Za-z_$][\w$]*)"), "class"),
        (re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s*\*?\s*([A-Za-z_$][\w$]*)"), "function"),
        (re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*=>"), "function"),
        (re.compile(r"^\s*(?:export\s+)?(?:interface|type|enum)\s+([A-Za-z_$][\w$]*)"), "type"),
    ],
    "java": [
        (re.compile(r"^\s*(?:public|private|protected|internal|abstract|final|static|sealed|data|open|\s)*\s*(?:class|interface|enum|record|object)\s+(\w+)"), "class"),
        (re.compile(r"^\s*(?:public|private|protected|static|final|synchronized|abstract|override|async|\s)+[\w<>\[\],\s]+\s+(\w+)\s*\([^;]*$"), "function"),
        (re.compile(r"^\s*(?:override\s+)?fun\s+(?:<[^>]+>\s*)?(\w+)\s*\("), "function"),
    ],
    "go": [
        (re.compile(r"^func\s+(?:\([^)]*\)\s*)?(\w+)\s*\("), "function"),
        (re.compile(r"^type\s+(\w+)\s+(?:struct|interface)"), "class"),
    ],
    "rust": [
        (re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+(\w+)"), "function"),
        (re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:struct|enum|trait|impl(?:<[^>]*>)?)\s+(\w+)"), "class"),
    ],
    "ruby": [
        (re.compile(r"^\s*(?:class|module)\s+([A-Z]\w*)"), "class"),
        (re.compile(r"^\s*def\s+(?:self\.)?(\w+[?!]?)"), "function"),
    ],
    "php": [
        (re.compile(r"^\s*(?:abstract\s+|final\s+)?(?:class|interface|trait)\s+(\w+)"), "class"),
        (re.compile(r"^\s*(?:public|private|protected|static|\s)*function\s+(\w+)"), "function"),
    ],
}
_SUFFIX_LANG = {
    ".js": "js", ".jsx": "js", ".ts": "js", ".tsx": "js", ".mjs": "js", ".cjs": "js", ".vue": "js", ".svelte": "js",
    ".java": "java", ".kt": "java", ".kts": "java", ".cs": "java", ".scala": "java", ".swift": "java",
    ".c": "java", ".cpp": "java", ".cc": "java", ".h": "java", ".hpp": "java",
    ".go": "go", ".rs": "rust", ".rb": "ruby", ".php": "php",
}


def outline_regex(source: str, suffix: str) -> list[Symbol]:
    """Approximate outline: a symbol runs until the next symbol at the same or lower indentation."""
    patterns = _REGEX_OUTLINES.get(_SUFFIX_LANG.get(suffix.lower(), ""), [])
    if not patterns:
        return []
    lines = source.splitlines()
    found: list[tuple[int, int, str, str]] = []  # (line, indent, name, kind)
    for number, line in enumerate(lines, start=1):
        for pattern, kind in patterns:
            match = pattern.match(line)
            if match and match.group(1) not in {"if", "for", "while", "switch", "catch", "return", "new"}:
                found.append((number, len(line) - len(line.lstrip()), match.group(1), kind))
                break
    symbols: list[Symbol] = []
    for i, (start, indent, name, kind) in enumerate(found):
        end = len(lines)
        for later_start, later_indent, _n, _k in found[i + 1:]:
            if later_indent <= indent:
                end = later_start - 1
                break
        while end > start and not lines[end - 1].strip():
            end -= 1
        parent = next((s.name for s in reversed(symbols) if s.kind == "class" and s.start_line < start <= s.end_line), None)
        symbols.append(Symbol(name, "method" if parent and kind == "function" else kind, start, end, parent))
    return symbols


def outline(path: str, source: str) -> list[Symbol]:
    suffix = Path(path).suffix.lower()
    if suffix in (".py", ".pyi"):
        try:
            return outline_python(source)
        except (SyntaxError, ValueError):
            pass  # fall through to the regex outline for broken files
        return [
            Symbol(m.group(2), "class" if m.group(1) == "class" else "function", n, n)
            for n, line in enumerate(source.splitlines(), start=1)
            if (m := re.match(r"\s*(?:async\s+)?(def|class)\s+(\w+)", line))
        ]
    return outline_regex(source, suffix)


# --- chunking ----------------------------------------------------------------------------------

def _windows(start: int, end: int, size: int) -> list[tuple[int, int]]:
    return [(s, min(end, s + size - 1)) for s in range(start, end + 1, size)]


def chunk_source(path: str, source: str, symbols: list[Symbol], *, max_lines: int = MAX_CHUNK_LINES) -> list[Chunk]:
    """Split at top-level symbol boundaries; code between symbols becomes "module" chunks."""
    lines = source.splitlines()
    total = len(lines)
    if total == 0:
        return []
    top = sorted((s for s in symbols if s.parent is None), key=lambda s: s.start_line)
    spans: list[tuple[int, int, str | None]] = []
    cursor = 1
    for sym in top:
        if sym.start_line < cursor:
            continue  # overlapping (regex outline); already covered
        if sym.start_line > cursor:
            spans.append((cursor, sym.start_line - 1, None))
        spans.append((sym.start_line, min(sym.end_line, total), sym.name))
        cursor = min(sym.end_line, total) + 1
    if cursor <= total:
        spans.append((cursor, total, None))

    chunks: list[Chunk] = []
    for start, end, name in spans:
        for w_start, w_end in _windows(start, end, max_lines):
            text = "\n".join(lines[w_start - 1:w_end])
            if not text.strip():
                continue
            text = redact(text)
            chunk_id = f"{path}:{w_start}-{w_end}:{sha256_text(text)[:8]}"
            chunks.append(Chunk(chunk_id, path, w_start, w_end, name, text))
    return chunks


# --- file tree ---------------------------------------------------------------------------------

def file_tree(sandbox: Sandbox, *, max_entries: int = 300) -> str:
    """Indented tree of readable files (directories first appear when their first file does)."""
    lines: list[str] = []
    seen_dirs: set[str] = set()
    count = 0
    for rel in sandbox.iter_files(max_files=max_entries * 4):
        parts = rel.split("/")
        for depth in range(1, len(parts)):
            directory = "/".join(parts[:depth])
            if directory not in seen_dirs:
                seen_dirs.add(directory)
                lines.append("  " * (depth - 1) + parts[depth - 1] + "/")
        lines.append("  " * (len(parts) - 1) + parts[-1])
        count += 1
        if count >= max_entries:
            lines.append(f"... [truncated after {max_entries} files]")
            break
    return "\n".join(lines) if lines else "(empty workspace)"


# --- the index ---------------------------------------------------------------------------------

class CodeIndex:
    """In-memory index of one workspace (thread-safe for concurrent readers)."""

    def __init__(self, workspace_id: str) -> None:
        self.workspace_id = workspace_id
        self.files: dict[str, FileEntry] = {}
        self.chunks: dict[str, Chunk] = {}
        self.vectors: dict[str, np.ndarray] = {}
        self.status = "empty"  # empty | indexing | ready | error
        self.error: str | None = None
        self.updated_at: float | None = None
        self.embedded = False
        self._lock = threading.RLock()
        self._matrix: tuple[list[str], object] | None = None

    def summary(self) -> dict[str, object]:
        with self._lock:
            return {
                "status": self.status,
                "files": len(self.files),
                "chunks": len(self.chunks),
                "embedded_chunks": len(self.vectors),
                "error": self.error,
                "updated_at": self.updated_at,
            }

    def symbols_for(self, path: str) -> list[Symbol] | None:
        with self._lock:
            entry = self.files.get(path)
            return list(entry.symbols) if entry else None

    def find_symbol(self, name: str, *, limit: int = 20) -> list[tuple[str, Symbol]]:
        lowered = name.lower()
        exact, partial = [], []
        with self._lock:
            for path, entry in self.files.items():
                for sym in entry.symbols:
                    if sym.name == name:
                        exact.append((path, sym))
                    elif lowered in sym.name.lower():
                        partial.append((path, sym))
        return (exact + partial)[:limit]

    def chunk_at(self, path: str, line: int) -> Chunk | None:
        with self._lock:
            entry = self.files.get(path)
            if not entry:
                return None
            for chunk_id in entry.chunk_ids:
                chunk = self.chunks.get(chunk_id)
                if chunk and chunk.start_line <= line <= chunk.end_line:
                    return chunk
        return None

    def _build_matrix(self) -> tuple[list[str], object] | None:
        if not self.vectors:
            return None
        ids = list(self.vectors)
        matrix = np.vstack([self.vectors[i] for i in ids]).astype("float32")
        if faiss is not None:
            index = faiss.IndexFlatIP(matrix.shape[1])
            index.add(matrix)
            return ids, index
        return ids, matrix

    def search_vectors(self, query: list[float], k: int) -> list[tuple[Chunk, float]]:
        """Cosine similarity (vectors are L2-normalised on insert)."""
        with self._lock:
            if self._matrix is None:
                self._matrix = self._build_matrix()
            if self._matrix is None:
                return []
            ids, store = self._matrix
            q = _normalise(np.asarray(query, dtype="float32")).reshape(1, -1)
            if q.shape[1] != (store.d if faiss is not None and hasattr(store, "d") else store.shape[1]):  # type: ignore[union-attr]
                return []
            k = min(k, len(ids))
            if faiss is not None and hasattr(store, "search"):
                scores, positions = store.search(q, k)  # type: ignore[union-attr]
                pairs = zip(positions[0].tolist(), scores[0].tolist())
            else:
                sims = (store @ q.T).ravel()  # type: ignore[operator]
                top = np.argsort(-sims)[:k]
                pairs = ((int(i), float(sims[i])) for i in top)
            return [(self.chunks[ids[pos]], float(score)) for pos, score in pairs if pos >= 0 and ids[pos] in self.chunks]

    def invalidate_matrix(self) -> None:
        with self._lock:
            self._matrix = None


def _normalise(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm else vec


class Indexer:
    """Builds, persists and incrementally refreshes ``CodeIndex`` objects."""

    def __init__(
        self, index_dir: Path, *, embed_fn: EmbedFn | None = None, max_file_bytes: int = 1_000_000,
        embedding_model: str = "", legacy_embedding_model: str = "",
    ) -> None:
        self.index_dir = index_dir
        self.embed_fn = embed_fn
        # The model that makes the vectors. A saved index from a different one (e.g. after
        # switching LLM_PROVIDER) has vectors of another size/space, so it is rebuilt.
        # Indexes saved before this was recorded count as ``legacy_embedding_model``.
        self.embedding_model = embedding_model
        self.legacy_embedding_model = legacy_embedding_model or embedding_model
        self.max_file_bytes = max_file_bytes
        self._indexes: dict[str, CodeIndex] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._lock = threading.Lock()

    def get(self, workspace: Workspace) -> CodeIndex:
        """The workspace's index (loaded from disk on first use; possibly empty)."""
        with self._lock:
            index = self._indexes.get(workspace.id)
            if index is None:
                index = self._load(workspace.id) or CodeIndex(workspace.id)
                self._indexes[workspace.id] = index
            return index

    def build(self, workspace: Workspace) -> CodeIndex:
        """(Re)index synchronously; only files whose hash changed are re-chunked and re-embedded."""
        index = self.get(workspace)
        sandbox = Sandbox(workspace.root, max_file_bytes=self.max_file_bytes)
        started = time.monotonic()
        with index._lock:
            index.status, index.error = "indexing", None
        try:
            current: dict[str, str] = {}
            sources: dict[str, str] = {}
            for rel in sandbox.iter_files():
                if not is_indexable(rel):
                    continue
                try:
                    text = sandbox.read_text(rel)
                except (SandboxError, OSError):
                    continue
                current[rel] = sha256_text(text)
                sources[rel] = text
                if len(current) >= MAX_INDEXED_FILES:
                    break

            with index._lock:
                removed = [p for p in index.files if p not in current]
                changed = [p for p, h in current.items() if p not in index.files or index.files[p].sha256 != h]
                for path in removed + changed:
                    entry = index.files.pop(path, None)
                    for chunk_id in entry.chunk_ids if entry else []:
                        index.chunks.pop(chunk_id, None)
                        index.vectors.pop(chunk_id, None)
                new_chunks: list[Chunk] = []
                for path in changed:
                    symbols = outline(path, sources[path])
                    chunks = chunk_source(path, sources[path], symbols)
                    index.files[path] = FileEntry(current[path], symbols, [c.id for c in chunks])
                    for chunk in chunks:
                        index.chunks[chunk.id] = chunk
                    new_chunks.extend(chunks)

            error = self._embed(index, new_chunks)
            with index._lock:
                index.invalidate_matrix()
                index.status, index.error = "ready", error
                index.updated_at = time.time()
            self._save(index)
            logger.info(
                "Indexed workspace %s: %d files (%d changed, %d removed), %d chunks in %.1fs",
                workspace.id, len(current), len(changed), len(removed), len(index.chunks), time.monotonic() - started,
            )
        except Exception as exc:  # noqa: BLE001 - indexing failure must not crash the app
            logger.exception("Indexing failed for workspace %s", workspace.id)
            with index._lock:
                index.status, index.error = "error", f"{type(exc).__name__}: {exc}"
        return index

    def build_async(self, workspace: Workspace) -> bool:
        """Start a background re-index unless one is already running. Returns True if started."""
        with self._lock:
            running = self._threads.get(workspace.id)
            if running is not None and running.is_alive():
                return False
            thread = threading.Thread(target=self.build, args=(workspace,), name=f"ca-index-{workspace.id}", daemon=True)
            self._threads[workspace.id] = thread
        self.get(workspace).status = "indexing"
        thread.start()
        return True

    def _embed(self, index: CodeIndex, chunks: list[Chunk]) -> str | None:
        if not chunks:
            return None
        if self.embed_fn is None:
            return "embeddings disabled (no OpenAI client); keyword search only"
        texts = [f"{c.path}\n{c.symbol or ''}\n{c.text}" for c in chunks]
        try:
            vectors = self.embed_fn(texts)
        except Exception as exc:  # noqa: BLE001 - fall back to keyword-only search
            logger.warning("Embedding failed for workspace %s: %s", index.workspace_id, exc)
            return f"embedding failed ({type(exc).__name__}); keyword search only"
        with index._lock:
            for chunk, vector in zip(chunks, vectors):
                index.vectors[chunk.id] = _normalise(np.asarray(vector, dtype="float32"))
            index.embedded = bool(index.vectors)
        return None

    # --- persistence --------------------------------------------------------------------------

    def _dir(self, workspace_id: str) -> Path:
        return self.index_dir / workspace_id

    def _save(self, index: CodeIndex) -> None:
        directory = self._dir(index.workspace_id)
        try:
            directory.mkdir(parents=True, exist_ok=True)
            with index._lock:
                meta = {
                    "version": INDEX_VERSION,
                    "files": {p: {"sha256": e.sha256, "symbols": [asdict(s) for s in e.symbols], "chunk_ids": e.chunk_ids}
                              for p, e in index.files.items()},
                    "chunks": [asdict(c) for c in index.chunks.values()],
                    "vector_ids": list(index.vectors),
                    "embedding_model": self.embedding_model,
                    "updated_at": index.updated_at,
                    "error": index.error,
                }
                matrix = np.vstack(list(index.vectors.values())) if index.vectors else np.zeros((0, 0), dtype="float32")
            tmp = directory / "meta.json.tmp"
            tmp.write_text(json.dumps(meta), encoding="utf-8")
            np.save(directory / "vectors.npy", matrix)
            os.replace(tmp, directory / "meta.json")
        except OSError:
            logger.exception("Could not persist index for workspace %s", index.workspace_id)

    def _load(self, workspace_id: str) -> CodeIndex | None:
        directory = self._dir(workspace_id)
        try:
            meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
            if meta.get("version") != INDEX_VERSION:
                return None
            if meta.get("embedding_model", self.legacy_embedding_model) != self.embedding_model:
                logger.info("Rebuilding index for workspace %s: embedding model changed to %s",
                            workspace_id, self.embedding_model)
                return None
            matrix = np.load(directory / "vectors.npy")
        except (OSError, ValueError):
            return None
        index = CodeIndex(workspace_id)
        for chunk in meta["chunks"]:
            index.chunks[chunk["id"]] = Chunk(**chunk)
        for path, entry in meta["files"].items():
            index.files[path] = FileEntry(entry["sha256"], [Symbol(**s) for s in entry["symbols"]], entry["chunk_ids"])
        for chunk_id, row in zip(meta["vector_ids"], matrix):
            index.vectors[chunk_id] = row.astype("float32")
        index.embedded = bool(index.vectors)
        index.status = "ready"
        index.updated_at = meta.get("updated_at")
        index.error = meta.get("error")
        return index
