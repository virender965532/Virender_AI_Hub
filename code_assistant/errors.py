"""Exceptions shared across the module.

Kept free of heavy imports so ``routes.py`` can reference them at app startup
without loading the pipeline (OpenAI, FAISS, git, ...), which is built lazily
on the first Code Assistant request.
"""

from __future__ import annotations


class SandboxError(ValueError):
    """A path or operation violates the workspace sandbox."""


class ConflictError(RuntimeError):
    """The file changed on disk after the change was proposed."""


class RollbackConflictError(RuntimeError):
    """A file was modified after the change was applied; rollback would lose that work."""


class SessionBusyError(RuntimeError):
    """Another request is already running for this session."""
