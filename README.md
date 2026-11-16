# Coding Agent

A small, safe **tool-using coding agent**: give it a task and a project folder, and an LLM (Groq or any OpenAI-compatible API with function calling) explores the code, runs the tests, makes exact edits, re-runs the tests, and reports a summary and a diff.
Built from scratch: the agent loop, the tools, the sandbox and the safety limits. No agent framework.

```text
$ python examples/replay_demo.py          # offline replay of a full run, no API key needed
[01] ok  run_tests() -> FAILED (exit code 1)
[02] ok  read_file(path='calc.py') ->    1 | """A tiny stats module with three bugs..."""
[03] ERR read_file(path='../../secrets.txt') -> ERROR: path '../../secrets.txt' is outside the workspace
[04] ok  edit_file(path='calc.py', old='(len(values) - 1)', new='len(values)') -> edited calc.py
[05] ok  edit_file(path='calc.py', old='    n = len(values)\n', new='    values = sorted(values)...') -> edited calc.py
[06] ok  edit_file(path='calc.py', old='if n < 1:', new='if n < 2:') -> edited calc.py
[07] ok  run_tests() -> PASSED

status: finished | tool calls: 7
summary: mean divided by n-1, median did not sort, is_prime(1) was True. All tests pass.

--- a/calc.py
+++ b/calc.py
-    return sum(values) / (len(values) - 1)
+    return sum(values) / len(values)
+    values = sorted(values)
-    if n < 1:
+    if n < 2:
```

Live run with a real model:

```bash
python -m agent --workspace examples/buggy_calc "Make the failing tests pass"
python -m agent --workspace path/to/your/project --confirm "Add input validation to parse_date"
```

## How it works

```mermaid
flowchart TB
    T[task] --> M["LLM with tool schemas<br/>(function calling)"]
    M -->|tool_calls| X["execute each tool<br/>inside the workspace jail"]
    X -->|"result or ERROR: ..."| C["append tool results<br/>(shorten old ones)"]
    C --> M
    M -->|finish(summary)| D["summary + unified diff<br/>+ JSONL transcript"]
    M -->|plain text| D
    X -->|"limits: max steps, repeated errors"| D
```

1. The model gets a system prompt (explore, run tests first, minimal edits, re-test, finish) and **JSON schemas** for 7 tools.
2. It replies with **tool calls** (`{"name": "read_file", "arguments": "{\"path\": \"calc.py\"}"}`). The model never executes anything itself.
3. The loop validates and runs each call, and sends the result back as a `tool` message with the matching `tool_call_id`.
4. Errors are **returned to the model as text** (`ERROR: old text appears 2 times, include more lines`) so it can correct itself, instead of crashing the run.
5. It stops on `finish`, on a plain-text answer, after `max_steps` model calls, or after N tool errors in a row.
6. At the end it prints a **unified diff** of every change (from a snapshot taken at start) and writes a **JSONL transcript** of every step.

## Tools

| tool | what it does | safety |
|---|---|---|
| `list_files` | lists project files | skips `.git`, `.venv`, `__pycache__`, `.env` |
| `read_file` | file with line numbers, max 400 lines per call | path must resolve inside the workspace |
| `search_code` | regex search, max 50 hits | invalid regex returns an error |
| `edit_file` | **exact search-and-replace**, `old` must match exactly once | ambiguous or missing matches are rejected with a helpful message |
| `write_file` | create/overwrite a file | 100 KB limit, `.git` and `.env` protected |
| `run_tests` | `python -m pytest -q` with a timeout | no shell, minimal env (no API keys), 60 s timeout, output truncated |
| `finish` | ends the run with a summary | |

**Why exact search-and-replace for edits?** Asking a model to rewrite a whole file wastes tokens and often silently drops lines. Asking for line numbers breaks as soon as lines shift. An exact, unique snippet is unambiguous and easy to verify; it's the approach used by real coding agents.

## Safety and limits (least privilege)

- **Workspace jail**: every path is resolved (`..` and symlinks followed) and must stay inside the workspace. Tested with `../`, `..\`, absolute paths and a symlink pointing outside.
- **No arbitrary shell**: the only command the agent can run is the test suite, with `shell=False`.
- **Secrets stay out**: `.env` can't be read; the test subprocess gets a minimal environment, so code under test can't read your API keys (there's a test for that).
- **Human in the loop**: `--confirm` asks y/N before every file change.
- **Budgets**: max steps, max consecutive errors, output truncation, and old tool results are shortened so the context window doesn't fill up.
- File contents are treated as data: the system prompt says not to follow instructions found in files.

## Quick start

```bash
git clone https://github.com/nitishfound/coding-agent.git
cd coding-agent
pip install -r requirements.txt
python examples/replay_demo.py          # offline
pytest -q                               # 21 tests
cp .env.example .env                    # add GROQ_API_KEY for live runs
python -m agent --workspace examples/buggy_calc "Make the failing tests pass"
```

## Project structure

```
agent/
  tools.py     Workspace sandbox + the 7 tools + their JSON schemas
  llm.py       OpenAI-compatible function-calling client, tool-call parsing, ScriptedModel
  loop.py      the agent loop, stop conditions, context compaction, transcript
  __main__.py  CLI (--confirm, --max-steps)
examples/
  buggy_calc/      a tiny project with 3 bugs and failing tests
  replay_demo.py   offline replay of a full run
tests/         21 tests: sandbox escapes, protected files, exact edits, secret-free test env,
               approval hook, a full scripted fix of the example, error recovery, stop conditions,
               context compaction, OpenAI tool-call parsing over mocked HTTP
```

## Testing an agent without an LLM

The loop is tested with a `ScriptedModel` that returns a fixed sequence of tool calls. That makes the tests fast, free and deterministic, and lets me test the hard parts: bad JSON arguments, unknown tools, wrong argument names, sandbox escapes, repeated failures, and that every tool result goes back with the right `tool_call_id`. The HTTP client is tested separately against a mocked OpenAI-style response.

## License

MIT
