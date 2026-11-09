"""BANT qualification of the free-text message on a lead.

Uses the Claude API when ANTHROPIC_API_KEY is set; otherwise a deterministic rule parser,
so the pipeline, tests and CI never depend on a network call.
"""
from __future__ import annotations

import json
import os
import re

from .models import LeadIn, Qualification

_BUDGET = re.compile(r"(?:€|eur|euro|\$|usd)\s?(\d[\d.,]*\s?(?:k|m)?)|(\d[\d.,]*\s?(?:k|m)?)\s?(?:€|eur|euros?)|budget (?:is |of )?(approved|allocated|secured|planned)", re.I)
_TIMELINE = re.compile(r"\b(asap|immediately|this (?:month|quarter|year)|next (?:month|quarter|year)|q[1-4](?:\s?20\d\d)?|within \d+ (?:weeks?|months?)|by (?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*)\b", re.I)
_AUTH = re.compile(r"\b(i (?:decide|own the decision|sign off)|decision maker|our (?:cfo|ceo|board) (?:asked|wants)|on behalf of (?:our|the) (?:cfo|ceo|board))\b", re.I)
_NEED = re.compile(r"\b(replace|migrat\w*|automate|integrat\w*|consolidat\w*|audit|compliance|reporting|too manual|spreadsheets?|scale)\b", re.I)


def rules_qualify(lead: LeadIn) -> Qualification:
    msg = lead.message or ""
    b = _BUDGET.search(msg)
    t = _TIMELINE.search(msg)
    a = _AUTH.search(msg)
    needs = sorted({m.group(0).lower() for m in _NEED.finditer(msg)})
    budget = None
    if b:
        budget = (b.group(1) or b.group(2) or b.group(3) or "").strip()
    authority = a.group(0) if a else ("title suggests decision maker" if re.search(r"chief|vp|head|director|founder|owner", lead.title, re.I) else None)
    parts = [p for p in [
        f"budget: {budget}" if budget else None,
        f"timeline: {t.group(0)}" if t else None,
        f"need: {', '.join(needs)}" if needs else None,
        "authority: yes" if authority else None,
    ] if p]
    return Qualification(
        budget=budget or None, authority=authority, need=", ".join(needs) or None,
        timeline=t.group(0) if t else None,
        summary="; ".join(parts) or "no qualification signals in message", method="rules",
    )


PROMPT = """You qualify inbound B2B leads using BANT. Read the lead and return only JSON with keys
budget, authority, need, timeline (each a short string or null) and summary (one sentence).
Do not guess: use null when the message does not say it.

Title: {title}
Company: {company}
Message: {message}"""


def llm_qualify(lead: LeadIn, model: str = "claude-sonnet-4-5") -> Qualification:  # pragma: no cover - network
    import httpx

    resp = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01",
                 "content-type": "application/json"},
        json={"model": model, "max_tokens": 300,
              "messages": [{"role": "user", "content": PROMPT.format(title=lead.title, company=lead.company, message=lead.message)}]},
        timeout=30,
    )
    resp.raise_for_status()
    text = resp.json()["content"][0]["text"]
    data = json.loads(text[text.find("{"): text.rfind("}") + 1])
    return Qualification(**{k: data.get(k) for k in ("budget", "authority", "need", "timeline")},
                         summary=data.get("summary") or "", method="llm")


def qualify(lead: LeadIn) -> Qualification:
    if os.environ.get("ANTHROPIC_API_KEY") and lead.message:
        try:
            return llm_qualify(lead)
        except Exception:  # fall back rather than drop the lead
            pass
    return rules_qualify(lead)
