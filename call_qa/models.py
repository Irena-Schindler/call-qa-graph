"""Typed data that flows through the graph."""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field

Role = Literal["agent", "caller"]


class Turn(BaseModel):
    role: Role
    text: str


class Event(BaseModel):
    """Side effect the agent triggered, e.g. a booking written to the calendar."""
    type: Literal["booking_created", "transfer", "callback_created"]
    at_turn: int = Field(ge=0, description="index of the turn after which the event happened")


class Transcript(BaseModel):
    id: str
    language: Literal["de", "en"] = "de"
    turns: list[Turn]
    events: list[Event] = []


class CallFacts(BaseModel):
    """What the LLM extracts from a call. Kept small on purpose: every field is checked downstream."""
    intent: Literal["booking", "reschedule", "cancellation", "question", "complaint", "other"]
    outcome: Literal["resolved", "booked", "callback", "transferred", "unresolved"]
    human_requested: bool
    caller_frustrated: bool
    summary: str = Field(max_length=300)
    confidence: float = Field(ge=0, le=1)


class CheckResult(BaseModel):
    name: str
    passed: bool
    severity: Literal["critical", "major", "minor"]
    evidence: str


class Review(BaseModel):
    decision: Literal["confirm", "dismiss"]
    reviewer: str
    note: str = ""
