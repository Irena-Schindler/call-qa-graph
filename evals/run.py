"""Eval: compare the pipeline with human labels. Per-check precision/recall and routing accuracy.

    python -m evals.run                       # offline (ScriptedLLM)
    LLM_PROVIDER=azure python -m evals.run    # same gold set against a real model
Exits non-zero if routing accuracy or any check's recall drops below the threshold (used in CI).
"""
from __future__ import annotations
import json, pathlib, sys
import yaml
from langgraph.checkpoint.memory import InMemorySaver
from call_qa.graph import build_graph
from call_qa.llm import from_env

ROOT = pathlib.Path(__file__).resolve().parent.parent
CHECKS = ["ai_disclosure", "booking_confirmed", "handover_honoured", "no_dead_end", "frustration_followed_up"]
MIN_RECALL, MIN_ROUTING = 1.0, 0.9


def main() -> int:
    gold = yaml.safe_load((ROOT / "evals/gold.yaml").read_text())
    graph = build_graph(from_env(), checkpointer=InMemorySaver())
    tp = {c: 0 for c in CHECKS}; fp = dict(tp); fn = dict(tp)
    routed_ok, mismatches = 0, []
    for cid, label in gold.items():
        t = json.loads((ROOT / f"data/transcripts/{cid}.json").read_text(encoding="utf-8"))
        out = graph.invoke({"transcript": t}, {"configurable": {"thread_id": cid}})
        failed = {c["name"] for c in out["checks"] if not c["passed"]}
        expected = set(label["failed"])
        for c in CHECKS:
            tp[c] += c in failed and c in expected
            fp[c] += c in failed and c not in expected
            fn[c] += c not in failed and c in expected
        reviewed = "__interrupt__" in out
        routed_ok += reviewed == label["review"]
        if failed != expected or reviewed != label["review"]:
            mismatches.append(f"{cid}: got {sorted(failed)} review={reviewed}, expected {sorted(expected)} review={label['review']}")

    ok = True
    print(f"{'check':<26}{'precision':>10}{'recall':>8}")
    for c in CHECKS:
        p = tp[c] / (tp[c] + fp[c]) if tp[c] + fp[c] else 1.0
        r = tp[c] / (tp[c] + fn[c]) if tp[c] + fn[c] else 1.0
        ok &= r >= MIN_RECALL
        print(f"{c:<26}{p:>10.2f}{r:>8.2f}")
    routing = routed_ok / len(gold)
    ok &= routing >= MIN_ROUTING
    print(f"routing to human review: {routed_ok}/{len(gold)} correct ({routing:.0%})")
    for m in mismatches:
        print("  mismatch", m)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
