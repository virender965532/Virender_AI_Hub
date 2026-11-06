"""Sandbox: traversal, symlink escapes, secrets, ignore rules, redaction, workspace allowlist."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from code_assistant.workspace.sandbox import Sandbox, SandboxError, WorkspaceRegistry, is_secret_path, redact


@pytest.mark.parametrize("bad", ["../../etc/passwd", "pkg/../../x", "..\\..\\windows\\win.ini", "/etc/passwd", "", "a\x00b"])
def test_path_traversal_blocked(workspace_dir, bad):
    with pytest.raises(SandboxError):
        Sandbox(workspace_dir).resolve(bad)


def test_absolute_path_inside_root_allowed(workspace_dir):
    sandbox = Sandbox(workspace_dir)
    assert sandbox.resolve(str(workspace_dir / "pkg" / "calc.py")) == Path(os.path.realpath(workspace_dir / "pkg" / "calc.py"))


def test_symlink_escape_blocked(workspace_dir, tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    link = workspace_dir / "escape.txt"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not permitted on this system")
    sandbox = Sandbox(workspace_dir)
    with pytest.raises(SandboxError, match="outside"):
        sandbox.read_text("escape.txt")
    assert "escape.txt" not in list(sandbox.iter_files())


def test_secret_files_never_read(workspace_dir):
    sandbox = Sandbox(workspace_dir)
    with pytest.raises(SandboxError, match="secret"):
        sandbox.read_text(".env")
    assert ".env" not in list(sandbox.iter_files())
    with pytest.raises(SandboxError, match="secret"):
        sandbox.check_writable("config/prod.pem")


@pytest.mark.parametrize("name,secret", [
    (".env", True), (".env.local", True), ("prod.env", True), ("server.key", True), ("id_rsa", True),
    ("aws/credentials", True), (".env.example", False), ("keyboard.py", False), ("tokens.py", False),
])
def test_is_secret_path(name, secret):
    assert is_secret_path(name) is secret


def test_ignore_rules(workspace_dir):
    files = list(Sandbox(workspace_dir).iter_files())
    assert "pkg/calc.py" in files and "tests/test_calc.py" in files
    assert not any(f.startswith("node_modules") for f in files)
    assert "debug.log" not in files  # from .gitignore


def test_redaction():
    text = (
        'OPENAI_API_KEY="sk-proj-abcdefghijklmnopqrstuvwxyz012345"\n'
        "token = get_token(user)\n"
        "password = request.form['password']\n"
        "DB_PASSWORD=hunter2hunter2x9\n"
        "url = 'postgres://admin:s3cretpw@db:5432/x'\n"
        "-----BEGIN RSA PRIVATE KEY-----\nabc\n-----END RSA PRIVATE KEY-----\n"
    )
    out = redact(text)
    assert "sk-proj-abcdef" not in out and "hunter2hunter2x9" not in out and "s3cretpw" not in out
    assert "abc\n-----END" not in out
    assert "token = get_token(user)" in out  # ordinary code is left alone
    assert "password = request.form['password']" in out


def test_workspace_registry_allowlist(workspace_dir, tmp_path):
    registry = WorkspaceRegistry((Path(os.path.realpath(tmp_path)),))
    ws = registry.register(str(workspace_dir))
    assert registry.get(ws.id).root == Path(os.path.realpath(workspace_dir))
    with pytest.raises(SandboxError, match="CA_WORKSPACE_ROOTS"):
        registry.register(str(Path(os.path.realpath(tmp_path)).parent))
    with pytest.raises(SandboxError, match="CA_WORKSPACE_ROOTS"):
        WorkspaceRegistry(()).register(str(workspace_dir))
