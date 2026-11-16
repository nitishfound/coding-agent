"""Run the coding agent.

    python -m agent --workspace path/to/project "Make the failing tests pass"
    python -m agent --workspace examples/buggy_calc --confirm "Fix the bugs"
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from .llm import OpenAICompatibleModel
from .loop import Agent, Step
from .tools import Workspace


def print_step(step: Step) -> None:
    mark = "ok " if step.ok else "ERR"
    args = ", ".join(f"{k}={str(v)[:40]!r}" for k, v in step.arguments.items())
    first = step.result.splitlines()[0][:100] if step.result else ""
    print(f"[{step.n:02d}] {mark} {step.tool}({args}) -> {first}")


def ask_approval(path: str, new_content: str) -> bool:
    print(f"\n--- agent wants to change {path} ({len(new_content.splitlines())} lines) ---")
    return input("allow? [y/N] ").strip().lower() == "y"


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    p = argparse.ArgumentParser(description="A small tool-using coding agent")
    p.add_argument("task")
    p.add_argument("--workspace", required=True)
    p.add_argument("--max-steps", type=int, default=25)
    p.add_argument("--confirm", action="store_true", help="ask before every file change")
    args = p.parse_args(argv)

    key = os.getenv("LLM_API_KEY") or os.getenv("GROQ_API_KEY")
    if not key:
        print("Set GROQ_API_KEY (free at https://console.groq.com/keys) or LLM_API_KEY in .env", file=sys.stderr)
        return 2
    model = OpenAICompatibleModel(
        os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"), key, os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    )
    ws = Workspace(Path(args.workspace), approve_write=ask_approval if args.confirm else None)
    transcript = Path("runs") / f"{datetime.now():%Y%m%d-%H%M%S}.jsonl"
    agent = Agent(model, ws, max_steps=args.max_steps, on_step=print_step, transcript=transcript)

    result = agent.run(args.task)
    print(f"\nstatus: {result.status} | tool calls: {len(result.steps)} | tokens: {result.tokens}")
    print(f"summary: {result.summary}\n")
    print(result.diff or "(no changes)")
    print(f"\ntranscript: {transcript}")
    return 0 if result.status == "finished" else 1


if __name__ == "__main__":
    sys.exit(main())
