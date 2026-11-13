"""LangGraph pipeline: redact -> extract -> check -> (human review) -> report.

    START ─► redact ─► extract ─► check ──► route ──► report ─► END
                                             │          ▲
                                             └► review ─┘   (interrupt, resumes with a reviewer decision)

Calls are routed to a human when a critical rule fails, the model is unsure or the caller was
frustrated. Review is a real LangGraph interrupt: the run is checkpointed, survives a restart with a
persistent checkpointer, and continues with `Command(resume=...)`.
"""
from __future__ import annotations
from typing import Any, Literal, TypedDict
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from .checks import run_all, score
from .llm import LLM, ScriptedLLM
from .models import CallFacts, CheckResult, Review, Transcript
from .redact import redact

CONFIDENCE_FLOOR = 0.7


class CallState(TypedDict, total=False):
    transcript: dict[str, Any]
    facts: dict[str, Any]
    checks: list[dict[str, Any]]
    review_reasons: list[str]
    review: dict[str, Any] | None
    report: dict[str, Any]


def build_graph(llm: LLM | None = None, checkpointer: Any = None):
    llm = llm or ScriptedLLM()

    def redact_node(state: CallState) -> CallState:
        t = Transcript.model_validate(state["transcript"])
        for turn in t.turns:
            turn.text = redact(turn.text)
        return {"transcript": t.model_dump()}

    def extract_node(state: CallState) -> CallState:
        facts = llm.extract(Transcript.model_validate(state["transcript"]))
        return {"facts": facts.model_dump()}

    def check_node(state: CallState) -> CallState:
        t = Transcript.model_validate(state["transcript"])
        f = CallFacts.model_validate(state["facts"])
        results = run_all(t, f)
        reasons = [f"critical: {r.name}" for r in results if not r.passed and r.severity == "critical"]
        if f.confidence < CONFIDENCE_FLOOR:
            reasons.append(f"low confidence {f.confidence:.2f}")
        if f.caller_frustrated:
            reasons.append("caller frustrated")
        return {"checks": [r.model_dump() for r in results], "review_reasons": reasons, "review": None}

    def route(state: CallState) -> Literal["review", "report"]:
        return "review" if state["review_reasons"] else "report"

    def review_node(state: CallState) -> CallState:
        decision = interrupt({
            "call_id": state["transcript"]["id"],
            "reasons": state["review_reasons"],
            "failed": [c for c in state["checks"] if not c["passed"]],
            "summary": state["facts"]["summary"],
        })
        return {"review": Review.model_validate(decision).model_dump()}

    def report_node(state: CallState) -> CallState:
        results = [CheckResult.model_validate(c) for c in state["checks"]]
        review = state.get("review")
        failed = [r.name for r in results if not r.passed]
        if review:
            status = "flagged" if review["decision"] == "confirm" else "cleared"
        else:
            status = "issues" if failed else "ok"
        return {"report": {
            "call_id": state["transcript"]["id"],
            "status": status,
            "score": score(results),
            "intent": state["facts"]["intent"],
            "outcome": state["facts"]["outcome"],
            "failed_checks": failed,
            "review_reasons": state["review_reasons"],
            "reviewer": review["reviewer"] if review else None,
        }}

    g = StateGraph(CallState)
    g.add_node("redact", redact_node)
    g.add_node("extract", extract_node)
    g.add_node("check", check_node)
    g.add_node("review", review_node)
    g.add_node("report", report_node)
    g.add_edge(START, "redact")
    g.add_edge("redact", "extract")
    g.add_edge("extract", "check")
    g.add_conditional_edges("check", route, {"review": "review", "report": "report"})
    g.add_edge("review", "report")
    g.add_edge("report", END)
    return g.compile(checkpointer=checkpointer)
