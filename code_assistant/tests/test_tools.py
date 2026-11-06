"""Tool registry, file/edit/exec/git tools, checkpoints and rollback."""

from __future__ import annotations

import subprocess
import sys

import pytest

from code_assistant.schemas import ProposedChange
from code_assistant.tools.edit_tools import ConflictError, PatchError, apply_change_to_disk, apply_unified_diff, make_unified_diff
from code_assistant.tools.exec_tools import CommandNotAllowedError, failure_signatures
from code_assistant.tools.git_tools import CheckpointManager, RollbackConflictError
from code_assistant.tools.registry import RiskLevel, build_default_registry

FIX_DIFF = "@@ -4,2 +4,2 @@\n def add(a, b):\n-    return a - b\n+    return a + b\n"


@pytest.fixture
def registry():
    return build_default_registry()


def test_unknown_tool_and_bad_args_return_errors(registry, tool_ctx):
    assert registry.dispatch("rm_rf", "{}", tool_ctx).startswith("Error: unknown tool 'rm_rf'")
    assert "not valid JSON" in registry.dispatch("read_file", "{bad", tool_ctx)
    assert "missing required" in registry.dispatch("read_file", "{}", tool_ctx)
    assert "unknown argument" in registry.dispatch("read_file", '{"path": "x", "foo": 1}', tool_ctx)
    assert "must be of type integer" in registry.dispatch("read_file", '{"path": "x", "start_line": "1"}', tool_ctx)
    assert registry.dispatch("apply_patch", '{"change_id": "x"}', tool_ctx).startswith("Error: unknown tool")


def test_risk_filter(registry, tool_ctx):
    out = registry.dispatch("propose_edit", {"path": "pkg/calc.py", "unified_diff": FIX_DIFF, "explanation": "x"},
                            tool_ctx, allowed={RiskLevel.READ})
    assert "not available in this mode" in out
    names = {t["function"]["name"] for t in registry.openai_tools({RiskLevel.READ})}
    assert names == {"list_dir", "read_file", "search_code", "find_symbol", "get_file_outline", "git_status", "git_diff"}


def test_read_file_ranges_and_secrets(registry, tool_ctx):
    out = registry.dispatch("read_file", {"path": "pkg/calc.py", "start_line": 4, "end_line": 5}, tool_ctx)
    assert "pkg/calc.py lines 4-5 of 10" in out and "    5|     return a - b" in out
    assert "secret" in registry.dispatch("read_file", {"path": ".env"}, tool_ctx)
    assert "outside" in registry.dispatch("read_file", {"path": "../../etc/passwd"}, tool_ctx) or \
        "not allowed" in registry.dispatch("read_file", {"path": "../../etc/passwd"}, tool_ctx)
    assert "pkg/calc.py" in tool_ctx.session.files_read


def test_tool_output_truncated(registry, tool_ctx, workspace_dir):
    (workspace_dir / "big.py").write_text("\n".join(f"x{i} = {'y' * 80}" for i in range(400)), encoding="utf-8")
    out = registry.dispatch("read_file", {"path": "big.py"}, tool_ctx)
    assert out.endswith("narrow the request)") and "[truncated]" in out


def test_list_dir_outline_symbol_search(registry, tool_ctx):
    listing = registry.dispatch("list_dir", {"path": "."}, tool_ctx)
    assert "pkg/" in listing and ".env" not in listing and "node_modules" not in listing
    outline = registry.dispatch("get_file_outline", {"path": "pkg/calc.py"}, tool_ctx)
    assert "function add  lines 4-5" in outline and "method total" in outline
    assert "pkg/calc.py:4-5  function add" in registry.dispatch("find_symbol", {"name": "add"}, tool_ctx)
    assert "pkg/calc.py" in registry.dispatch("search_code", {"query": "a - b", "mode": "exact"}, tool_ctx)


def test_propose_edit_is_not_written_and_is_visible_to_reads(registry, tool_ctx, workspace_dir):
    out = registry.dispatch("propose_edit", {"path": "pkg/calc.py", "unified_diff": FIX_DIFF, "explanation": "fix add"}, tool_ctx)
    assert "Proposed edit recorded" in out
    assert "a - b" in (workspace_dir / "pkg" / "calc.py").read_text(encoding="utf-8")  # disk untouched
    assert "return a + b" in registry.dispatch("read_file", {"path": "pkg/calc.py"}, tool_ctx)
    changes = tool_ctx.working_set.to_changes()
    assert len(changes) == 1 and changes[0].kind == "edit" and "+    return a + b" in changes[0].unified_diff


def test_create_file_and_secret_write_refused(registry, tool_ctx, workspace_dir):
    registry.dispatch("create_file", {"path": "pkg/util.py", "content": "X = 1\n", "explanation": "new"}, tool_ctx)
    assert not (workspace_dir / "pkg" / "util.py").exists()
    assert "already exists" in registry.dispatch("create_file", {"path": "pkg/calc.py", "content": "", "explanation": "x"}, tool_ctx)
    assert "secret" in registry.dispatch("create_file", {"path": ".env.local", "content": "A=1", "explanation": "x"}, tool_ctx)


def test_diff_applier_tolerates_llm_mistakes():
    original = "a\nb\nc\n\nd\ne\n"
    # wrong line numbers, unprefixed blank context line, no file headers
    diff = "@@ -40,4 +40,4 @@\n c\n\n d\n-e\n+E\n"
    assert apply_unified_diff(original, diff) == "a\nb\nc\n\nd\nE\n"
    crlf = "x = 1\r\ny = 2\r\n"
    assert apply_unified_diff(crlf, "@@ -1,2 +1,2 @@\n x = 1\n-y = 2\n+y = 3\n") == "x = 1\r\ny = 3\r\n"
    with pytest.raises(PatchError, match="re-read|Re-read"):
        apply_unified_diff(original, "@@ -1,1 +1,1 @@\n-zzz\n+q\n")
    assert apply_unified_diff("a\n", "@@ -1,0 +2,1 @@\n+b\n") == "a\nb\n"


def test_unified_diff_roundtrip():
    old, new = "one\ntwo\nthree\n", "one\n2\nthree\n"
    assert apply_unified_diff(old, make_unified_diff("f.txt", old, new)) == new


def test_command_allowlist(tool_ctx):
    runner = tool_ctx.runner
    root = tool_ctx.sandbox.root
    for bad in (["rm", "-rf", "/"], ["curl", "http://x"], ["python", "-c", "print(1)"], ["/usr/bin/pytest"]):
        with pytest.raises(CommandNotAllowedError):
            runner.validate(bad, root)
    with pytest.raises(CommandNotAllowedError, match="outside"):
        runner.validate(["pytest", str(root.parent)], root)
    with pytest.raises(CommandNotAllowedError):
        runner.validate(["pytest", "../other"], root)
    runner.validate(["pytest", "tests/test_calc.py::test_add", "-k", "add"], root)
    runner.validate(["python", "-m", "pytest", "-q"], root)


def test_command_timeout_kills_process(tool_ctx, monkeypatch, tmp_path):
    import dataclasses

    script = tmp_path / "proj" / "test_slow.py"
    script.write_text("import time\n\ndef test_slow():\n    time.sleep(30)\n", encoding="utf-8")
    tool_ctx.runner.settings = dataclasses.replace(tool_ctx.settings, command_timeout=1.0)
    result = tool_ctx.runner.run(["python", "-m", "pytest", "-q", "test_slow.py"], cwd=tool_ctx.sandbox.root)
    assert result.timed_out and result.exit_code is None and result.duration_ms < 20000


def test_command_env_has_no_secrets(tool_ctx, monkeypatch):
    from code_assistant.tools.exec_tools import child_env

    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-leak-0000000000000000")
    assert "OPENAI_API_KEY" not in child_env()


@pytest.mark.skipif(subprocess.run([sys.executable, "-m", "pytest", "--version"], capture_output=True).returncode != 0, reason="pytest missing")
def test_run_tests_uses_private_copy_with_proposals(registry, tool_ctx, workspace_dir):
    failing = registry.dispatch("run_tests", {}, tool_ctx)
    assert "Tests FAILED" in failing and "test_add" in failing
    assert "FAILED tests/test_calc.py::test_add" in failure_signatures(failing)
    registry.dispatch("propose_edit", {"path": "pkg/calc.py", "unified_diff": FIX_DIFF, "explanation": "fix"}, tool_ctx)
    assert "Tests PASSED" in registry.dispatch("run_tests", {}, tool_ctx)
    assert "a - b" in (workspace_dir / "pkg" / "calc.py").read_text(encoding="utf-8")
    copy = tool_ctx.verify_env.path
    assert copy is not None and not (copy / ".env").exists() and not (copy / "node_modules").exists()


def test_run_command_is_queued_not_executed(registry, tool_ctx):
    out = registry.dispatch("run_command", {"cmd": "pytest -q"}, tool_ctx)
    assert "has NOT run" in out
    [change] = tool_ctx.session.list_changes()
    assert change.kind == "command" and change.status == "pending" and change.command == ["pytest", "-q"]
    assert "not allowed" in registry.dispatch("run_command", {"cmd": "rm -rf ."}, tool_ctx)


def test_failure_signatures_ignore_line_numbers():
    a = failure_signatures("pkg/x.py:10:5: F401 'os' imported but unused\nFAILED tests/t.py::test_a - assert 1\n")
    b = failure_signatures("pkg/x.py:12:5: F401 'os' imported but unused\nFAILED tests/t.py::test_a - assert 2\n")
    assert a == b and len(a) == 2


def _change(workspace_dir, new: str) -> ProposedChange:
    from code_assistant.workspace.sandbox import sha256_text

    original = (workspace_dir / "pkg" / "calc.py").read_bytes().decode("utf-8")  # keep CRLF on Windows
    return ProposedChange(change_id="c1", path="pkg/calc.py", unified_diff="", explanation="", original_content=original,
                          new_content=new, original_sha256=sha256_text(original))


def test_apply_and_rollback_restore_original(tool_ctx, workspace_dir, tmp_path):
    original = (workspace_dir / "pkg" / "calc.py").read_text(encoding="utf-8")
    change = _change(workspace_dir, "NEW\n")
    manager = CheckpointManager(tmp_path / "cp")
    checkpoint = manager.create(tool_ctx.sandbox, ["pkg/calc.py", "pkg/new.py"])
    apply_change_to_disk(tool_ctx.sandbox, change)
    (workspace_dir / "pkg" / "new.py").write_text("created\n", encoding="utf-8")
    manager.mark_applied(checkpoint, tool_ctx.sandbox)
    assert (workspace_dir / "pkg" / "calc.py").read_text(encoding="utf-8") == "NEW\n"

    restored = manager.rollback(manager.load(checkpoint.id), tool_ctx.sandbox)
    assert sorted(restored) == ["pkg/calc.py", "pkg/new.py"]
    assert (workspace_dir / "pkg" / "calc.py").read_text(encoding="utf-8") == original
    assert not (workspace_dir / "pkg" / "new.py").exists()


def test_rollback_refuses_if_modified_after_apply(tool_ctx, workspace_dir, tmp_path):
    manager = CheckpointManager(tmp_path / "cp")
    checkpoint = manager.create(tool_ctx.sandbox, ["pkg/calc.py"], make_branch=False)
    apply_change_to_disk(tool_ctx.sandbox, _change(workspace_dir, "NEW\n"))
    manager.mark_applied(checkpoint, tool_ctx.sandbox)
    (workspace_dir / "pkg" / "calc.py").write_text("user edited\n", encoding="utf-8")
    with pytest.raises(RollbackConflictError):
        manager.rollback(checkpoint, tool_ctx.sandbox)
    manager.rollback(checkpoint, tool_ctx.sandbox, force=True)


def test_apply_refuses_when_file_changed_since_proposal(tool_ctx, workspace_dir):
    change = _change(workspace_dir, "NEW\n")
    (workspace_dir / "pkg" / "calc.py").write_text("changed by user\n", encoding="utf-8")
    with pytest.raises(ConflictError):
        apply_change_to_disk(tool_ctx.sandbox, change)


def test_git_tools_and_checkpoint_branch(registry, tool_ctx, workspace_dir, tmp_path):
    git = pytest.importorskip("git")
    repo = git.Repo.init(workspace_dir)
    repo.git.add("pkg", "tests", "README.md", ".gitignore")
    repo.index.commit("init")
    (workspace_dir / "pkg" / "calc.py").write_text("changed\n", encoding="utf-8")
    status = registry.dispatch("git_status", {}, tool_ctx)
    assert "pkg/calc.py" in status and ".env" not in status
    assert "+changed" in registry.dispatch("git_diff", {"path": "pkg/calc.py"}, tool_ctx)
    checkpoint = CheckpointManager(tmp_path / "cp").create(tool_ctx.sandbox, ["pkg/calc.py"])
    assert checkpoint.git_head == repo.head.commit.hexsha
    assert checkpoint.git_branch in [b.name for b in repo.branches]
    assert repo.active_branch.name != checkpoint.git_branch  # never checked out
    repo.close()
