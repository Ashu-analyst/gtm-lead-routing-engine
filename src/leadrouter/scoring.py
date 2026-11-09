"""Lead scoring: an explainable fit + intent score, plus a calibrated model trained on history.

Fit answers "is this the kind of company we win?" (industry, size, region, seniority).
Intent answers "are they ready to talk now?" (demo request, page views, source, message).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .models import Firmographics, LeadIn, ScoreBreakdown

SENIOR = re.compile(r"\b(chief|cxo|ceo|cfo|coo|cto|cio|vp|vice president|head|director|founder|owner|geschäftsführer|leiter)\b", re.I)
MANAGER = re.compile(r"\b(manager|lead|teamlead|principal|senior)\b", re.I)
URGENT = re.compile(r"\b(asap|urgent|this quarter|this month|q[1-4]|deadline|audit|budget approved|rfp|tender)\b", re.I)

FEATURES = [
    "icp_industry", "size_mid", "size_enterprise", "core_region", "senior", "manager",
    "demo", "views_log", "src_referral", "src_event", "src_paid", "urgent_msg", "free_mail",
]


@dataclass
class ICP:
    industries: set[str] = field(default_factory=lambda: {"Manufacturing", "Logistics", "Retail", "Chemicals", "Automotive"})
    core_regions: set[str] = field(default_factory=lambda: {"DACH", "Benelux", "Nordics"})
    mid_size: tuple[int, int] = (200, 2000)
    enterprise_min: int = 2000


def features(lead: LeadIn, firmo: Optional[Firmographics], free_mail: bool, icp: ICP) -> dict[str, float]:
    emp = firmo.employees if firmo else 0
    src = lead.source.lower()
    return {
        "icp_industry": float(bool(firmo and firmo.industry in icp.industries)),
        "size_mid": float(icp.mid_size[0] <= emp < icp.mid_size[1]),
        "size_enterprise": float(emp >= icp.enterprise_min),
        "core_region": float(bool(firmo and firmo.region in icp.core_regions)),
        "senior": float(bool(SENIOR.search(lead.title))),
        "manager": float(bool(MANAGER.search(lead.title)) and not SENIOR.search(lead.title)),
        "demo": float(lead.demo_requested),
        "views_log": math.log1p(max(lead.page_views, 0)),
        "src_referral": float(src == "referral"),
        "src_event": float(src == "event"),
        "src_paid": float(src == "paid_search"),
        "urgent_msg": float(bool(URGENT.search(lead.message))),
        "free_mail": float(free_mail),
    }


# Hand-set starting weights (points). These are what a RevOps team would agree in a workshop;
# `ScoreModel.fit` replaces them with weights learned from closed-won history.
DEFAULT_POINTS = {
    "icp_industry": 20, "size_mid": 15, "size_enterprise": 20, "core_region": 10, "senior": 15,
    "manager": 7, "demo": 25, "views_log": 6, "src_referral": 12, "src_event": 6, "src_paid": 3,
    "urgent_msg": 10, "free_mail": -15,
}
FIT_KEYS = {"icp_industry", "size_mid", "size_enterprise", "core_region", "senior", "manager", "free_mail"}


POINT_GRADES = (70, 50, 30)        # hand-set points, 0-100
PROBABILITY_GRADES = (40, 25, 12)  # calibrated conversion probability in %, base rate ~15%


def grade(total: float, cuts: tuple[float, float, float] = POINT_GRADES) -> str:
    return "A" if total >= cuts[0] else "B" if total >= cuts[1] else "C" if total >= cuts[2] else "D"


class ScoreModel:
    """Logistic regression on the feature set; outputs a 0-100 conversion score with reasons."""

    def __init__(self, points: Optional[dict[str, float]] = None):
        self.points = dict(points or DEFAULT_POINTS)
        self.coef: Optional[np.ndarray] = None
        self.intercept: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 20000, lr: float = 0.3) -> "ScoreModel":
        """Plain gradient-descent logistic regression (no sklearn dependency at runtime)."""
        n, d = X.shape
        w, b = np.zeros(d), 0.0
        for _ in range(iters):
            p = 1 / (1 + np.exp(-(X @ w + b)))
            g = p - y
            w -= lr * (X.T @ g / n + l2 * w / n)
            b -= lr * g.mean()
        self.coef, self.intercept = w, b
        return self

    def vector(self, f: dict[str, float]) -> np.ndarray:
        return np.array([f[k] for k in FEATURES])

    def probability(self, f: dict[str, float]) -> float:
        if self.coef is None:
            raise RuntimeError("model not fitted")
        z = float(self.vector(f) @ self.coef + self.intercept)
        return 1 / (1 + math.exp(-z))

    def score(self, f: dict[str, float]) -> ScoreBreakdown:
        if self.coef is not None:
            contrib = dict(zip(FEATURES, self.vector(f) * self.coef))
            total = round(100 * self.probability(f), 1)
        else:
            contrib = {k: f[k] * self.points[k] for k in FEATURES}
            total = round(max(0.0, min(100.0, sum(contrib.values()))), 1)
        fit = sum(v for k, v in contrib.items() if k in FIT_KEYS)
        intent = sum(v for k, v in contrib.items() if k not in FIT_KEYS)
        top = sorted(((v, k) for k, v in contrib.items() if abs(v) > 1e-9), reverse=True)
        reasons = [f"+{k}" if v > 0 else f"-{k}" for v, k in top[:3]]
        if any(v < 0 for v, _ in top):
            reasons.append(f"-{min(top)[1]}")
        cuts = PROBABILITY_GRADES if self.coef is not None else POINT_GRADES
        return ScoreBreakdown(fit=round(fit, 2), intent=round(intent, 2), total=total,
                              grade=grade(total, cuts), reasons=list(dict.fromkeys(reasons)))


def auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """ROC AUC via the rank statistic (ties counted half)."""
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    for s in np.unique(scores):
        idx = scores == s
        ranks[idx] = ranks[idx].mean()
    pos = y_true == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
