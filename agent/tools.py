"""The agent's tools. Everything happens inside one workspace folder (a "jail").

Least privilege: the agent can list, read, search, write/edit files inside the
workspace and run the test suite. It can NOT run arbitrary shell commands,
touch files outside the workspace, or modify .git.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", ".pytest_cache"}
PROTECTED = {".git", ".env"}
MAX_READ_LINES = 400
MAX_WRITE_BYTES = 100_000
MAX_OUTPUT_CHARS = 4_000


class ToolError(Exception):
    """A tool failed in an expected way. The message is sent back to the model so it can fix its call."""


def truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n... [{len(text) - limit} characters cut] ...\n{text[-half:]}"


@dataclass
class Workspace:
    root: Path
    test_timeout: int = 60
    approve_write: Callable[[str, str], bool] | None = None  # human-in-the-loop hook
    snapshot: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.root = Path(self.root).resolve()
        if not self.root.is_dir():
            raise ValueError(f"workspace {self.root} does not exist")
        self.snapshot = {str(p.relative_to(self.root)): p.read_text(encoding="utf-8", errors="replace")
                         for p in self._iter_files()}

    # -------------------------------------------------------------- safety
    def resolve(self, rel: str, for_write: bool = False) -> Path:
        """Turn a model-supplied path into a real path INSIDE the workspace, or refuse."""
        if not rel or "\x00" in rel:
            raise ToolError("empty or invalid path")
        candidate = (self.root / rel).resolve()  # resolves "..", and follows symlinks
        if candidate != self.root and self.root not in candidate.parents:
            raise ToolError(f"path {rel!r} is outside the workspace")
        parts = set(candidate.relative_to(self.root).parts)
        if for_write and parts & PROTECTED:
            raise ToolError(f"writing to {rel!r} is not allowed")
        if parts & {".env"}:
            raise ToolError("reading secrets files is not allowed")
        return candidate

    def _iter_files(self):
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
            for name in sorted(filenames):
                p = Path(dirpath) / name
                if p.suffix in {".pyc"} or name == ".env":
                    continue
                yield p

    # --------------------------------------------------------------- tools
    def list_files(self, path: str = ".") -> str:
        base = self.resolve(path)
        files = [str(p.relative_to(self.root)).replace("\\", "/") for p in self._iter_files() if base in p.parents or p == base]
        if not files:
            return "(no files)"
        shown = files[:200]
        more = f"\n... and {len(files) - 200} more" if len(files) > 200 else ""
        return "\n".join(shown) + more

    def read_file(self, path: str, start_line: int = 1, end_line: int | None = None) -> str:
        p = self.resolve(path)
        if not p.is_file():
            raise ToolError(f"{path} is not a file")
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        start = max(1, start_line)
        end = min(len(lines), end_line or len(lines), start + MAX_READ_LINES - 1)
        numbered = [f"{i:4d} | {lines[i - 1]}" for i in range(start, end + 1)]
        footer = f"\n(showing lines {start}-{end} of {len(lines)})" if end < len(lines) or start > 1 else ""
        return "\n".join(numbered) + footer if numbered else "(empty file)"

    def search_code(self, pattern: str) -> str:
        try:
            rx = re.compile(pattern)
        except re.error as exc:
            raise ToolError(f"bad regex: {exc}") from exc
        hits = []
        for p in self._iter_files():
            try:
                for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
                    if rx.search(line):
                        hits.append(f"{p.relative_to(self.root).as_posix()}:{n}: {line.strip()}")
            except UnicodeDecodeError:
                continue
            if len(hits) >= 50:
                break
        return "\n".join(hits[:50]) or "no matches"

    def write_file(self, path: str, content: str) -> str:
        p = self.resolve(path, for_write=True)
        if len(content.encode()) > MAX_WRITE_BYTES:
            raise ToolError("file too large")
        if self.approve_write and not self.approve_write(path, content):
            raise ToolError("the user rejected this change")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"wrote {len(content.splitlines())} lines to {path}"

    def edit_file(self, path: str, old: str, new: str) -> str:
        """Exact search-and-replace. `old` must appear exactly once, so the edit is unambiguous."""
        p = self.resolve(path, for_write=True)
        if not p.is_file():
            raise ToolError(f"{path} does not exist (use write_file to create it)")
        text = p.read_text(encoding="utf-8")
        count = text.count(old) if old else 0
        if count == 0:
            raise ToolError("`old` text not found. Read the file again and copy the exact text, including spaces.")
        if count > 1:
            raise ToolError(f"`old` text appears {count} times. Include more surrounding lines to make it unique.")
        updated = text.replace(old, new, 1)
        if self.approve_write and not self.approve_write(path, updated):
            raise ToolError("the user rejected this change")
        p.write_text(updated, encoding="utf-8")
        return f"edited {path}"

    def run_tests(self, path: str = "") -> str:
        args = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--no-header"]
        if path:
            self.resolve(path)
            args.append(path)
        # Minimal environment: no API keys or other secrets leak into the code under test.
        env = {k: v for k, v in os.environ.items() if k in {"PATH", "SYSTEMROOT", "TEMP", "TMP", "HOME", "USERPROFILE"}}
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        try:
            proc = subprocess.run(args, cwd=self.root, env=env, capture_output=True, text=True,
                                  timeout=self.test_timeout, shell=False)
        except subprocess.TimeoutExpired:
            raise ToolError(f"tests took longer than {self.test_timeout}s and were stopped")
        status = "PASSED" if proc.returncode == 0 else f"FAILED (exit code {proc.returncode})"
        return f"{status}\n{truncate(proc.stdout + proc.stderr)}"

    # ---------------------------------------------------------------- diff
    def diff(self) -> str:
        import difflib

        out = []
        current = {str(p.relative_to(self.root)): p for p in self._iter_files()}
        for rel in sorted(set(current) | set(self.snapshot)):
            before = self.snapshot.get(rel, "")
            after = current[rel].read_text(encoding="utf-8", errors="replace") if rel in current else ""
            if before != after:
                name = rel.replace("\\", "/")
                out.extend(difflib.unified_diff(before.splitlines(True), after.splitlines(True), f"a/{name}", f"b/{name}"))
        return "".join(out)


# JSON schemas the model sees (OpenAI "function calling" format).
TOOL_SPECS: list[dict[str, Any]] = [
    {"name": "list_files", "description": "List files in the workspace (or a sub-folder).",
     "parameters": {"type": "object", "properties": {"path": {"type": "string", "default": "."}}}},
    {"name": "read_file", "description": "Read a file with line numbers. Use start_line/end_line for big files.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"}}, "required": ["path"]}},
    {"name": "search_code", "description": "Regex search across all files. Returns file:line: text.",
     "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]}},
    {"name": "edit_file", "description": "Replace an exact snippet in a file. `old` must match exactly once (copy it from read_file, without the line numbers).",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old": {"type": "string"},
                    "new": {"type": "string"}}, "required": ["path", "old", "new"]}},
    {"name": "write_file", "description": "Create or overwrite a whole file.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                    "required": ["path", "content"]}},
    {"name": "run_tests", "description": "Run the pytest suite (optionally one file). Returns PASSED/FAILED and output.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}},
    {"name": "finish", "description": "Call when the task is done (or impossible). Give a short summary of what you changed and why.",
     "parameters": {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}},
]
