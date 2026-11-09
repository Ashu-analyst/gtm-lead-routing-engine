"""End-to-end pipeline: enrich -> de-duplicate -> score -> qualify -> route."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .enrich import FREE_MAIL, DuplicateDetector, FirmographicIndex, lead_domain, normalise_domain
from .models import LeadIn, ProcessedLead
from .qualifier import qualify
from .routing import Router
from .scoring import ICP, ScoreModel, features

ROOT = Path(__file__).resolve().parents[2]


class LeadPipeline:
    def __init__(self, index: FirmographicIndex, router: Router, model: Optional[ScoreModel] = None,
                 icp: Optional[ICP] = None):
        self.index = index
        self.router = router
        self.model = model or ScoreModel()
        self.icp = icp or ICP()
        self.dedupe = DuplicateDetector()

    @classmethod
    def default(cls, model: Optional[ScoreModel] = None) -> "LeadPipeline":
        return cls(
            FirmographicIndex.from_csv(ROOT / "data" / "firmographics.csv"),
            Router.from_yaml(ROOT / "config" / "routing.yaml"),
            model,
        )

    def process(self, lead: LeadIn) -> ProcessedLead:
        domain = lead_domain(lead)
        dup_of = self.dedupe.check(lead, domain)
        firmo = self.index.lookup(domain, lead.company)
        free_mail = normalise_domain(lead.email) in FREE_MAIL
        f = features(lead, firmo, free_mail, self.icp)
        score = self.model.score(f)
        qual = qualify(lead)
        if dup_of:
            # a duplicate goes back to whoever owns the original record, never into round-robin
            from .models import RoutingDecision
            routing = RoutingDecision(owner="owner-of-original", queue="duplicate-merge",
                                      rule="duplicate", sla_minutes=1440)
        else:
            routing = self.router.route(Router.context(firmo.domain if firmo else domain, firmo, score, lead.source,
                                                        lead.created_at.date().isoformat()))
        return ProcessedLead(lead=lead, domain=domain, is_duplicate=dup_of is not None, duplicate_of=dup_of,
                             firmographics=firmo, score=score, qualification=qual, routing=routing)
