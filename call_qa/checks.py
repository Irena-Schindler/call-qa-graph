"""Deterministic policy checks. They run on every call, cost nothing and are reproducible.

The LLM is only used where rules cannot reach (intent, outcome, frustration); everything that can be
decided from the transcript and the event log is decided here.
"""
from __future__ import annotations
import re
from .models import CallFacts, CheckResult, Transcript

DISCLOSURE = re.compile(r"\b(ki|künstliche[nr]? intelligenz|virtuelle|digitale assistentin|digitaler assistent|"
                        r"ai assistant|virtual assistant|automated assistant|an ai)\b", re.I)
ASK_CONFIRM = re.compile(r"(soll ich|bestätigen|ist das (so )?(korrekt|richtig)|passt das|"
                         r"shall i|should i|can you confirm|is that (correct|right))", re.I)
AFFIRM = re.compile(r"^\s*(ja|jawohl|genau|passt|gerne|richtig|korrekt|yes|yeah|correct|sure|please do)\b", re.I)
NEGATE = re.compile(r"\b(nein|nee|doch nicht|no|not)\b", re.I)
WANTS_HUMAN = re.compile(r"(mitarbeiter|einen menschen|echte[nr]? person|mit jemandem sprechen|"
                         r"human|real person|speak to someone|talk to someone)", re.I)
OFFERS_HANDOVER = re.compile(r"(verbinde|weiterleiten|rückruf|zurückrufen|ruft sie .* zurück|"
                             r"transfer|connect you|call you back|callback)", re.I)


def ai_disclosure(t: Transcript) -> CheckResult:
    first_agent = next((x.text for x in t.turns if x.role == "agent"), "")
    ok = bool(DISCLOSURE.search(first_agent))
    return CheckResult(name="ai_disclosure", passed=ok, severity="critical",
                       evidence="first agent turn discloses AI" if ok else f"first agent turn: {first_agent[:80]!r}")


def booking_confirmed(t: Transcript) -> CheckResult:
    """Every booking must be preceded by an explicit question and an explicit yes."""
    for ev in (e for e in t.events if e.type == "booking_created"):
        confirmed = False
        for j in range(min(ev.at_turn, len(t.turns) - 1), 0, -1):
            prev, cur = t.turns[j - 1], t.turns[j]
            if prev.role == "agent" and cur.role == "caller" and ASK_CONFIRM.search(prev.text):
                confirmed = bool(AFFIRM.search(cur.text)) and not NEGATE.search(cur.text)
                break
        if not confirmed:
            return CheckResult(name="booking_confirmed", passed=False, severity="critical",
                               evidence=f"booking after turn {ev.at_turn} without explicit confirmation")
    return CheckResult(name="booking_confirmed", passed=True, severity="critical",
                       evidence="no booking or every booking confirmed")


def handover_honoured(t: Transcript) -> CheckResult:
    asks = [i for i, x in enumerate(t.turns) if x.role == "caller" and WANTS_HUMAN.search(x.text)]
    if not asks:
        return CheckResult(name="handover_honoured", passed=True, severity="major", evidence="no request for a human")
    first = asks[0]
    offered = any(x.role == "agent" and OFFERS_HANDOVER.search(x.text) for x in t.turns[first + 1:])
    evented = any(e.type in ("transfer", "callback_created") and e.at_turn >= first for e in t.events)
    ok = offered or evented
    return CheckResult(name="handover_honoured", passed=ok, severity="major",
                       evidence=f"human requested in turn {first}; " + ("handover offered" if ok else "ignored"))


def no_dead_end(t: Transcript, f: CallFacts) -> CheckResult:
    rescued = any(e.type in ("transfer", "callback_created") for e in t.events)
    ok = f.outcome != "unresolved" or rescued
    return CheckResult(name="no_dead_end", passed=ok, severity="major",
                       evidence=f"outcome={f.outcome}, handover_event={rescued}")


def frustration_followed_up(t: Transcript, f: CallFacts) -> CheckResult:
    rescued = any(e.type in ("transfer", "callback_created") for e in t.events)
    ok = (not f.caller_frustrated) or rescued
    return CheckResult(name="frustration_followed_up", passed=ok, severity="minor",
                       evidence="caller frustrated, no follow-up" if not ok else "ok")


def run_all(t: Transcript, f: CallFacts) -> list[CheckResult]:
    return [ai_disclosure(t), booking_confirmed(t), handover_honoured(t),
            no_dead_end(t, f), frustration_followed_up(t, f)]


WEIGHTS = {"critical": 40, "major": 20, "minor": 10}


def score(results: list[CheckResult]) -> int:
    return max(0, 100 - sum(WEIGHTS[r.severity] for r in results if not r.passed))
