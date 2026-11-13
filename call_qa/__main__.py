"""CLI: analyse transcripts, list calls waiting for review, resume them with a decision.

    python -m call_qa data/transcripts/*.json
    python -m call_qa data/transcripts/*.json --review confirm --reviewer qa-lead
"""
from __future__ import annotations
import argparse, json, pathlib
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from .graph import build_graph
from .llm import from_env


def main() -> None:
    ap = argparse.ArgumentParser(prog="call_qa")
    ap.add_argument("files", nargs="+")
    ap.add_argument("--review", choices=["confirm", "dismiss"], help="apply this decision to every flagged call")
    ap.add_argument("--reviewer", default="reviewer")
    args = ap.parse_args()

    graph = build_graph(from_env(), checkpointer=InMemorySaver())
    rows = []
    for path in args.files:
        transcript = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        cfg = {"configurable": {"thread_id": transcript["id"]}}
        out = graph.invoke({"transcript": transcript}, cfg)
        if "__interrupt__" in out:
            pending = out["__interrupt__"][0].value
            if not args.review:
                rows.append({"call_id": pending["call_id"], "status": "needs_review", "reasons": pending["reasons"]})
                continue
            out = graph.invoke(Command(resume={"decision": args.review, "reviewer": args.reviewer}), cfg)
        rows.append(out["report"])

    for r in rows:
        extra = r.get("reasons") or r.get("failed_checks") or []
        print(f"{r['call_id']:<26} {r['status']:<13} {str(r.get('score', '-')):>5}  {', '.join(extra)}")


if __name__ == "__main__":
    main()
