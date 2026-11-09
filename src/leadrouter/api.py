"""FastAPI service: POST a lead, get it back enriched, scored, qualified and routed."""
from __future__ import annotations

import json

from fastapi import FastAPI

from .models import LeadIn, ProcessedLead
from .pipeline import ROOT, LeadPipeline
from .scoring import ScoreModel
import numpy as np

app = FastAPI(title="GTM Lead Routing Engine", version="1.0.0")


def _load_model() -> ScoreModel:
    path = ROOT / "data" / "model.json"
    model = ScoreModel()
    if path.exists():
        m = json.loads(path.read_text())
        model.coef, model.intercept = np.array(m["coef"]), m["intercept"]
    return model


pipeline = LeadPipeline.default(_load_model())


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": "calibrated" if pipeline.model.coef is not None else "points"}


@app.post("/leads", response_model=ProcessedLead)
def create_lead(lead: LeadIn) -> ProcessedLead:
    return pipeline.process(lead)


@app.get("/rep-load")
def rep_load() -> dict:
    return pipeline.router.load()
