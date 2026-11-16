import os
import shutil
from pathlib import Path

import pytest

from agent import ToolError, Workspace

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "buggy_calc"


@pytest.fixture
def ws(tmp_path):
    shutil.copytree(EXAMPLE, tmp_path / "proj")
    (tmp_path / "proj" / ".git").mkdir()
    (tmp_path / "proj" / ".git" / "config").write_text("[core]")
    (tmp_path / "proj" / ".env").write_text("GROQ_API_KEY=secret")
    (tmp_path / "outside.txt").write_text("private")
    return Workspace(tmp_path / "proj")


@pytest.mark.parametrize("path", ["../outside.txt", "tests/../../outside.txt"])
def test_cannot_escape_the_workspace(ws, path):
    with pytest.raises(ToolError, match="outside the workspace"):
        ws.read_file(path)


def test_backslash_paths_never_reach_outside_files(ws):
    # On Windows "..\\" is a separator (-> outside the workspace); on Linux it is just an odd
    # file name inside the workspace (-> not found). Either way the private file is never read.
    with pytest.raises(ToolError):
        ws.read_file("..\\outside.txt")


def test_absolute_paths_outside_are_blocked(ws, tmp_path):
    with pytest.raises(ToolError):
        ws.read_file(str(tmp_path / "outside.txt"))


def test_symlink_escape_is_blocked(ws, tmp_path):
    link = ws.root / "sneaky"
    try:
        os.symlink(tmp_path / "outside.txt", link)
    except (OSError, NotImplementedError):
        pytest.skip("creating symlinks needs extra privileges on this OS")
    with pytest.raises(ToolError, match="outside the workspace"):
        ws.read_file("sneaky")


def test_git_and_env_are_protected(ws):
    with pytest.raises(ToolError):
        ws.write_file(".git/config", "evil")
    with pytest.raises(ToolError):
        ws.read_file(".env")
    assert ".env" not in ws.list_files() and ".git" not in ws.list_files()


def test_read_file_has_line_numbers(ws):
    out = ws.read_file("calc.py", 4, 5)
    assert out.splitlines()[0].startswith("   4 | def mean")
    assert "(showing lines 4-5" in out


def test_edit_file_needs_exactly_one_match(ws):
    with pytest.raises(ToolError, match="not found"):
        ws.edit_file("calc.py", "this text is not there", "x")
    with pytest.raises(ToolError, match="appears"):
        ws.edit_file("calc.py", "raise ValueError", "x")
    assert ws.edit_file("calc.py", "(len(values) - 1)", "len(values)") == "edited calc.py"
    assert "/ len(values)" in (ws.root / "calc.py").read_text()


def test_search_code(ws):
    assert "calc.py:4: def mean" in ws.search_code(r"def mean")
    with pytest.raises(ToolError):
        ws.search_code("(unclosed")


def test_run_tests_reports_failures_and_hides_secrets(ws, monkeypatch):
    out = ws.run_tests()
    assert out.startswith("FAILED") and "failed" in out
    monkeypatch.setenv("GROQ_API_KEY", "super-secret")
    (ws.root / "tests" / "test_env.py").write_text("import os\ndef test_no_key():\n    assert 'GROQ_API_KEY' not in os.environ\n")
    assert ws.run_tests("tests/test_env.py").startswith("PASSED")


def test_approval_hook_can_reject_changes(tmp_path):
    shutil.copytree(EXAMPLE, tmp_path / "p")
    ws = Workspace(tmp_path / "p", approve_write=lambda path, content: False)
    with pytest.raises(ToolError, match="rejected"):
        ws.write_file("new.py", "x = 1")
    assert not (tmp_path / "p" / "new.py").exists()


def test_diff_shows_changes(ws):
    ws.edit_file("calc.py", "if n < 1:", "if n < 2:")
    diff = ws.diff()
    assert "-    if n < 1:" in diff and "+    if n < 2:" in diff
