"""Offline demo: replay a scripted agent run on a temporary copy of examples/buggy_calc.

No API key needed. The "model" here is a fixed script of tool calls, so you can
see exactly what the loop, the tools, the safety checks and the diff look like.
For a real run with an LLM:  python -m agent --workspace examples/buggy_calc "Make the tests pass"
"""

import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent import Agent, Reply, ScriptedModel, Workspace, call  # noqa: E402
from agent.__main__ import print_step  # noqa: E402

SCRIPT = [
    Reply("First, run the tests to see what is broken.", [call("run_tests")]),
    Reply(None, [call("read_file", path="calc.py")]),
    Reply(None, [call("read_file", path="../../secrets.txt")]),  # the sandbox blocks this
    Reply(None, [call("edit_file", path="calc.py", old="(len(values) - 1)", new="len(values)")]),
    Reply(None, [
        call("edit_file", path="calc.py", old="    n = len(values)\n", new="    values = sorted(values)\n    n = len(values)\n"),
        call("edit_file", path="calc.py", old="if n < 1:", new="if n < 2:"),
    ]),
    Reply(None, [call("run_tests")]),
    Reply(None, [call("finish", summary="mean divided by n-1, median did not sort, is_prime(1) was True. All tests pass.")]),
]

with tempfile.TemporaryDirectory() as tmp:
    project = Path(tmp) / "buggy_calc"
    shutil.copytree(ROOT / "examples" / "buggy_calc", project)
    result = Agent(ScriptedModel(SCRIPT), Workspace(project), on_step=print_step).run("Make the failing tests pass")
    print(f"\nstatus: {result.status} | tool calls: {len(result.steps)}")
    print(f"summary: {result.summary}\n")
    print(result.diff)
