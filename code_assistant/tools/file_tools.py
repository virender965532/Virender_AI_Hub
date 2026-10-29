"""READ tools: list_dir, read_file, search_code, find_symbol, get_file_outline.

All paths go through the sandbox. read_file returns line-numbered text in
ranges (large files must be read in pieces), shows the agent's own proposed
edits (the working set) and caches disk reads per request (invalidated when the
file's mtime changes or the agent edits it).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from ..workspace.indexer import outline
from ..workspace.sandbox import SandboxError, is_secret_path, redact
from .registry import RiskLevel, Tool, ToolContext, ToolError


def _disk_text(ctx: ToolContext, rel: str) -> str:
    real = ctx.sandbox.resolve(rel, must_exist=True)
    key = (rel, real.stat().st_mtime)
    with ctx.lock:
        cached = ctx.read_cache.get(key)
    if cached is not None:
        return cached
    text = ctx.sandbox.read_text(rel)
    with ctx.lock:
        ctx.read_cache[key] = text
    return text


def current_text(ctx: ToolContext, path: str) -> tuple[str, str, bool]:
    """``(relative_path, text, is_proposed)``: the working-set version if the agent edited it, else disk."""
    real = ctx.sandbox.resolve(path)
    rel = ctx.sandbox.relative(real)
    if is_secret_path(rel):
        raise SandboxError(f"refusing to read a secret file: {rel}")
    proposed = ctx.working_set.get(rel)
    if proposed is not None:
        return rel, proposed, True
    return rel, _disk_text(ctx, rel), False


def list_dir(ctx: ToolContext, path: str = ".", depth: int = 1) -> str:
    """List a directory (non-ignored entries only), optionally a few levels deep."""
    depth = max(1, min(int(depth), 4))
    base = ctx.sandbox.resolve(path, must_exist=True)
    if not base.is_dir():
        raise ToolError(f"not a directory: {path}")
    lines: list[str] = []

    def walk(directory: Path, level: int) -> None:
        try:
            entries = sorted(os.scandir(directory), key=lambda e: (not e.is_dir(follow_symlinks=False), e.name.lower()))
        except OSError as exc:
            lines.append(f"{'  ' * level}[unreadable: {exc.strerror}]")
            return
        for entry in entries:
            rel = ctx.sandbox.relative(Path(entry.path))
            is_dir = entry.is_dir(follow_symlinks=False)
            if ctx.sandbox.is_ignored(rel, is_dir=is_dir) or (not is_dir and is_secret_path(rel)):
                continue
            if len(lines) >= 500:
                return
            if is_dir:
                lines.append(f"{'  ' * level}{entry.name}/")
                if level + 1 < depth:
                    walk(Path(entry.path), level + 1)
            else:
                lines.append(f"{'  ' * level}{entry.name}")
    walk(base, 0)
    for rel in ctx.working_set.created_paths():
        if os.path.dirname(rel).replace("\\", "/") == ("" if ctx.sandbox.relative(base) == "." else ctx.sandbox.relative(base)):
            lines.append(f"{os.path.basename(rel)}  (proposed new file)")
    if len(lines) >= 500:
        lines.append("...[truncated]")
    return "\n".join(lines) or "(empty)"


def read_file(ctx: ToolContext, path: str, start_line: int = 1, end_line: int | None = None) -> str:
    """Line-numbered file contents for a range (max ``max_read_lines`` lines per call)."""
    rel, text, proposed = current_text(ctx, path)
    lines = text.splitlines()
    total = len(lines)
    start = max(1, int(start_line))
    limit = ctx.settings.max_read_lines
    end = min(total, int(end_line) if end_line else start + limit - 1, start + limit - 1)
    if ctx.session is not None:
        ctx.session.note_file_read(rel)
    if total == 0:
        return f"{rel}: (empty file)"
    if start > total:
        raise ToolError(f"start_line {start} is past the end of {rel} ({total} lines)")
    body = "\n".join(f"{n:>5}| {lines[n - 1]}" for n in range(start, end + 1))
    header = f"{rel} lines {start}-{end} of {total}"
    if proposed:
        header += " (includes your proposed, not-yet-approved edits)"
    footer = f"\n...[truncated: {total - end} more lines; call read_file with start_line={end + 1}]" if end < total else ""
    return f"{header}\n{redact(body)}{footer}"


def search_code(ctx: ToolContext, query: str, mode: str = "hybrid", max_results: int = 8) -> str:
    """Hybrid (identifiers + meaning), exact substring or regex search across the workspace."""
    if ctx.retriever is None:
        raise ToolError("search is unavailable for this workspace")
    max_results = max(1, min(int(max_results), 20))
    if mode == "hybrid":
        hits = ctx.retriever.search(query, k=max_results)
        return "\n\n".join(h.render() for h in hits) or "No matches."
    rows = ctx.retriever.keyword(query, regex=(mode == "regex"), max_results=max_results * 5)
    return "\n".join(f"{p}:{n}: {t.strip()}" for p, n, t in rows) or "No matches."


def find_symbol(ctx: ToolContext, name: str) -> str:
    """Definitions of a class/function/method by name (index first, then a regex scan)."""
    if not re.fullmatch(r"[A-Za-z_$][\w$\.]*", name):
        raise ToolError("name must be an identifier such as 'MyClass' or 'my_function'")
    short = name.rsplit(".", 1)[-1]
    rows: list[str] = []
    if ctx.index is not None and ctx.index.files:
        for path, sym in ctx.index.find_symbol(short):
            parent = f"{sym.parent}." if sym.parent else ""
            rows.append(f"{path}:{sym.start_line}-{sym.end_line}  {sym.kind} {parent}{sym.name}")
    if not rows and ctx.retriever is not None:
        pattern = rf"(?:def|class|function|interface|type|struct|fn|func)\s+{re.escape(short)}\b|\b{re.escape(short)}\s*=\s*(?:async\s*)?\("
        rows = [f"{p}:{n}  {t.strip()}" for p, n, t in ctx.retriever.keyword(pattern, regex=True, ignore_case=False, max_results=20)]
    return "\n".join(rows) or f"No definition of {name!r} found."


def get_file_outline(ctx: ToolContext, path: str) -> str:
    """Classes / functions with line ranges for one file."""
    rel, text, proposed = current_text(ctx, path)
    symbols = None if proposed or ctx.index is None else ctx.index.symbols_for(rel)
    if symbols is None:
        symbols = outline(rel, text)
    total = len(text.splitlines())
    if not symbols:
        return f"{rel} ({total} lines): no classes or functions found."
    rows = [f"{rel} ({total} lines)"]
    for sym in symbols:
        indent = "  " if sym.parent else ""
        rows.append(f"{indent}{sym.kind} {sym.name}  lines {sym.start_line}-{sym.end_line}")
    return "\n".join(rows)


TOOLS = [
    Tool(
        "list_dir", "List files and folders in a workspace directory (ignored and secret files are hidden).",
        {"type": "object", "properties": {
            "path": {"type": "string", "description": "Workspace-relative directory; '.' for the root."},
            "depth": {"type": "integer", "description": "1-4 levels (default 1)."},
        }, "required": []},
        RiskLevel.READ, list_dir,
    ),
    Tool(
        "read_file", "Read a file with line numbers. Large files must be read in ranges.",
        {"type": "object", "properties": {
            "path": {"type": "string"},
            "start_line": {"type": "integer", "description": "1-based first line (default 1)."},
            "end_line": {"type": "integer", "description": "Inclusive last line (default: up to 400 lines)."},
        }, "required": ["path"]},
        RiskLevel.READ, read_file,
    ),
    Tool(
        "search_code", "Search the workspace. mode=hybrid (default) mixes identifier and semantic search; "
        "mode=exact finds a literal string; mode=regex a regular expression. Returns paths with line ranges.",
        {"type": "object", "properties": {
            "query": {"type": "string"},
            "mode": {"type": "string", "enum": ["hybrid", "exact", "regex"]},
            "max_results": {"type": "integer"},
        }, "required": ["query"]},
        RiskLevel.READ, search_code,
    ),
    Tool(
        "find_symbol", "Find where a class, function or method is defined.",
        {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
        RiskLevel.READ, find_symbol,
    ),
    Tool(
        "get_file_outline", "List the classes and functions in a file with their line ranges.",
        {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        RiskLevel.READ, get_file_outline,
    ),
]
