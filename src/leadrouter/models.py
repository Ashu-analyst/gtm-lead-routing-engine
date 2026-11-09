from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


class LeadIn(BaseModel):
    """An inbound lead as it arrives from a web form, an event list or an import."""

    email: str
    first_name: str = ""
    last_name: str = ""
    company: str = ""
    website: str = ""
    country: str = ""
    title: str = ""
    source: str = "web_form"
    message: str = ""
    page_views: int = 0
    demo_requested: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Firmographics(BaseModel):
    domain: str
    company: str
    industry: str
    employees: int
    country: str
    region: str


class ScoreBreakdown(BaseModel):
    fit: float
    intent: float
    total: float
    grade: str
    reasons: list[str]


class Qualification(BaseModel):
    budget: Optional[str] = None
    authority: Optional[str] = None
    need: Optional[str] = None
    timeline: Optional[str] = None
    summary: str = ""
    method: str = "rules"


class RoutingDecision(BaseModel):
    owner: str
    queue: str
    rule: str
    sla_minutes: int


class ProcessedLead(BaseModel):
    lead: LeadIn
    domain: str
    is_duplicate: bool
    duplicate_of: Optional[str] = None
    firmographics: Optional[Firmographics] = None
    score: ScoreBreakdown
    qualification: Qualification
    routing: RoutingDecision
