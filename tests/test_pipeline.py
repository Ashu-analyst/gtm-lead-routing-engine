import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from leadrouter.enrich import DuplicateDetector, FirmographicIndex, lead_domain, normalise_company, normalise_domain
from leadrouter.models import Firmographics, LeadIn, ScoreBreakdown
from leadrouter.qualifier import rules_qualify
from leadrouter.routing import Queue, Rep, Router
from leadrouter.scoring import ICP, ScoreModel, auc, features, grade

ROOT = Path(__file__).resolve().parents[1]


def lead(**kw):
    base = dict(email="lena.mueller@hansa-logistik.de", first_name="Lena", last_name="Müller",
                company="Hansa Logistik GmbH", title="Head of Operations")
    base.update(kw)
    return LeadIn(**base)


# ---------- enrichment ----------
@pytest.mark.parametrize("raw,expected", [
    ("https://www.Example.com/about?x=1", "example.com"),
    ("jo@Example.DE", "example.de"),
    ("www.rijn-chemie.nl", "rijn-chemie.nl"),
])
def test_normalise_domain(raw, expected):
    assert normalise_domain(raw) == expected


def test_free_mail_uses_website():
    assert lead_domain(lead(email="lena@gmail.com", website="https://acme.de")) == "acme.de"


def test_company_name_normalisation():
    assert normalise_company("Rijn Chemie B.V.") == normalise_company("rijn chemie")


def test_fuzzy_company_lookup():
    idx = FirmographicIndex([Firmographics(domain="nordic-pack.se", company="Nordic Pack AB", industry="Manufacturing",
                                           employees=2600, country="SE", region="Nordics")])
    assert idx.lookup("", "Nordic-Pack").domain == "nordic-pack.se"
    assert idx.lookup("", "Totally Different") is None


def test_duplicate_detection_variants():
    d = DuplicateDetector()
    assert d.check(lead(), "hansa-logistik.de") is None
    assert d.check(lead(email="Lena.Mueller+webinar@hansa-logistik.de"), "hansa-logistik.de")
    assert d.check(lead(email="lmueller@hansa-logistik.de"), "hansa-logistik.de")  # same name, same company
    assert d.check(lead(email="max@hansa-logistik.de", first_name="Max", last_name="Koch"), "hansa-logistik.de") is None


def test_gmail_dots_are_ignored():
    d = DuplicateDetector()
    assert d.check(lead(email="lena.mueller@gmail.com"), "") is None
    assert d.check(lead(email="lenamueller@googlemail.com"), "")


# ---------- scoring ----------
def test_points_score_rewards_demo_and_penalises_free_mail():
    m, icp = ScoreModel(), ICP()
    firmo = Firmographics(domain="a.de", company="A", industry="Manufacturing", employees=800, country="DE", region="DACH")
    hot = m.score(features(lead(demo_requested=True, page_views=20), firmo, False, icp))
    cold = m.score(features(lead(title="Working Student"), None, True, icp))
    assert hot.total > cold.total and hot.grade in "AB" and cold.grade == "D"
    assert "+demo" in hot.reasons


def test_grade_bands():
    assert [grade(x) for x in (80, 60, 35, 5)] == ["A", "B", "C", "D"]


def test_auc_perfect_and_random():
    y = np.array([0, 0, 1, 1])
    assert auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert auc(y, np.array([0.5, 0.5, 0.5, 0.5])) == 0.5


def test_logistic_fit_recovers_signal():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 13))
    y = (rng.random(2000) < 1 / (1 + np.exp(-(2 * X[:, 6] - 1)))).astype(int)
    m = ScoreModel().fit(X, y)
    assert np.argmax(np.abs(m.coef)) == 6 and m.coef[6] > 1.0


# ---------- qualification ----------
def test_rules_qualifier_extracts_bant():
    q = rules_qualify(lead(message="Audit coming up, need compliance reporting asap. Budget of 40k EUR. I own the decision."))
    assert q.budget and "40k" in q.budget
    assert q.timeline == "asap"
    assert "compliance" in q.need and q.authority


def test_rules_qualifier_no_signals():
    q = rules_qualify(lead(title="Analyst", message="Just browsing."))
    assert q.budget is None and q.timeline is None and "no qualification" in q.summary


# ---------- routing ----------
def test_round_robin_respects_capacity_and_away():
    q = Queue("q", [Rep("a", capacity=1), Rep("b", away=True), Rep("c", capacity=2)], 60)
    assert [q.assign("d1").name for _ in range(3)] == ["a", "c", "c"]
    assert q.assign("d1") is None
    assert q.assign("d2").name == "a"  # capacity resets the next day


def test_router_rules_and_account_owner():
    r = Router.from_yaml(ROOT / "config" / "routing.yaml")
    firmo = Firmographics(domain="x.de", company="X", industry="Logistics", employees=5000, country="DE", region="DACH")
    s = ScoreBreakdown(fit=1, intent=1, total=55, grade="A", reasons=[])
    assert r.route(Router.context("x.de", firmo, s, "web_form", "2026-11-01")).queue == "enterprise-dach"
    assert r.route(Router.context("hansa-logistik.de", firmo, s, "web_form")).owner == "Lena Weber"
    d = ScoreBreakdown(fit=0, intent=0, total=3, grade="D", reasons=[])
    assert r.route(Router.context("y.com", None, d, "web_form")).queue == "nurture"


# ---------- API ----------
def test_api_end_to_end():
    from leadrouter.api import app
    c = TestClient(app)
    assert c.get("/health").json()["status"] == "ok"
    payload = {"email": "jonas@hansa-logistik.de", "first_name": "Jonas", "last_name": "Kraus",
               "title": "VP Operations", "demo_requested": True, "message": "Budget approved, this quarter."}
    out = c.post("/leads", json=payload).json()
    assert out["routing"]["rule"] == "existing_account" and out["firmographics"]["industry"] == "Logistics"
    again = c.post("/leads", json=payload).json()
    assert again["is_duplicate"] and again["routing"]["queue"] == "duplicate-merge"


def test_committed_results_meet_targets():
    r = json.loads((ROOT / "docs" / "results.json").read_text())
    assert r["scoring"]["auc_calibrated"] >= r["scoring"]["auc_workshop_points"]
    assert r["dedupe"]["recall"] >= 0.95 and r["dedupe"]["precision"] >= 0.95
    assert r["routing"]["midmarket_balance_max_minus_min"] <= 1
