"""Rule-based lead routing with account matching and capacity-aware round robin.

Rules live in YAML so RevOps can change routing without a deploy. They are evaluated in order;
the first matching rule picks a queue, then the queue assigns a rep:

1. Existing account -> the account owner (keeps one owner per customer).
2. Otherwise the queue's reps in round-robin, skipping reps who are away or at daily capacity.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from .models import Firmographics, RoutingDecision, ScoreBreakdown


@dataclass
class Rep:
    """capacity = new leads per working day."""
    name: str
    capacity: int = 25
    away: bool = False
    assigned: int = 0
    today: str = ""
    today_count: int = 0

    def has_room(self, day: str) -> bool:
        if self.today != day:
            self.today, self.today_count = day, 0
        return not self.away and self.today_count < self.capacity

    def take(self) -> None:
        self.assigned += 1
        self.today_count += 1


@dataclass
class Queue:
    name: str
    reps: list[Rep]
    sla_minutes: int
    _next: int = 0

    def assign(self, day: str = "") -> Optional[Rep]:
        n = len(self.reps)
        for i in range(n):
            rep = self.reps[(self._next + i) % n]
            if rep.has_room(day):
                self._next = (self._next + i + 1) % n
                rep.take()
                return rep
        return None


@dataclass
class Rule:
    name: str
    queue: str
    when: dict[str, Any] = field(default_factory=dict)

    def matches(self, ctx: dict[str, Any]) -> bool:
        for key, expected in self.when.items():
            value = ctx.get(key)
            if isinstance(expected, list):
                if value not in expected:
                    return False
            elif isinstance(expected, dict):
                if "gte" in expected and not (value is not None and value >= expected["gte"]):
                    return False
                if "lt" in expected and not (value is not None and value < expected["lt"]):
                    return False
            elif value != expected:
                return False
        return True


class Router:
    def __init__(self, queues: dict[str, Queue], rules: list[Rule], account_owners: dict[str, str],
                 fallback_queue: str):
        self.queues = queues
        self.rules = rules
        self.account_owners = account_owners
        self.fallback_queue = fallback_queue

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Router":
        cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        queues = {
            q["name"]: Queue(q["name"], [Rep(**r) for r in q["reps"]], q["sla_minutes"])
            for q in cfg["queues"]
        }
        rules = [Rule(r["name"], r["queue"], r.get("when", {})) for r in cfg["rules"]]
        return cls(queues, rules, cfg.get("account_owners", {}), cfg["fallback_queue"])

    @staticmethod
    def context(domain: str, firmo: Optional[Firmographics], score: ScoreBreakdown, source: str,
                day: str = "") -> dict[str, Any]:
        emp = firmo.employees if firmo else 0
        segment = "Enterprise" if emp >= 2000 else "Mid-Market" if emp >= 200 else "SMB" if firmo else "Unknown"
        return {
            "domain": domain, "region": firmo.region if firmo else "Unknown", "segment": segment,
            "industry": firmo.industry if firmo else "Unknown", "grade": score.grade,
            "score": score.total, "source": source, "known_company": firmo is not None, "day": day,
        }

    def route(self, ctx: dict[str, Any]) -> RoutingDecision:
        owner = self.account_owners.get(ctx["domain"])
        if owner:
            return RoutingDecision(owner=owner, queue="account-owner", rule="existing_account", sla_minutes=60)
        for rule in self.rules:
            if rule.matches(ctx):
                queue = self.queues[rule.queue]
                rep = queue.assign(ctx.get("day", ""))
                if rep:
                    return RoutingDecision(owner=rep.name, queue=queue.name, rule=rule.name, sla_minutes=queue.sla_minutes)
        fb = self.queues[self.fallback_queue]
        rep = fb.assign(ctx.get("day", ""))
        return RoutingDecision(owner=rep.name if rep else "unassigned", queue=fb.name, rule="fallback",
                               sla_minutes=fb.sla_minutes)

    def load(self) -> dict[str, int]:
        return {r.name: r.assigned for q in self.queues.values() for r in q.reps}
