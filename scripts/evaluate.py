"""Train the score model on history, then run the whole pipeline on new inbound leads.

Writes data/model.json, docs/results.json and the charts used in the README.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from leadrouter.enrich import FREE_MAIL, FirmographicIndex, lead_domain, normalise_domain  # noqa: E402
from leadrouter.models import LeadIn  # noqa: E402
from leadrouter.pipeline import LeadPipeline  # noqa: E402
from leadrouter.routing import Router  # noqa: E402
from leadrouter.scoring import FEATURES, ICP, ScoreModel, auc, features  # noqa: E402


def load_history(index: FirmographicIndex):
    X, y = [], []
    icp = ICP()
    with open(ROOT / "data" / "history.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            lead = LeadIn(**{k: v for k, v in r.items() if k != "converted"},)
            dom = lead_domain(lead)
            firmo = index.lookup(dom, lead.company)
            fv = features(lead, firmo, normalise_domain(lead.email) in FREE_MAIL, icp)
            X.append([fv[k] for k in FEATURES])
            y.append(int(r["converted"]))
    return np.array(X), np.array(y)


def lift_top_decile(y: np.ndarray, s: np.ndarray) -> float:
    k = max(1, len(s) // 10)
    top = np.argsort(-s)[:k]
    return float(y[top].mean() / y.mean())


def main() -> dict:
    index = FirmographicIndex.from_csv(ROOT / "data" / "firmographics.csv")
    X, y = load_history(index)
    cut = int(len(y) * 0.8)  # time-ordered split: train on the past, test on the most recent 20%
    Xtr, ytr, Xte, yte = X[:cut], y[:cut], X[cut:], y[cut:]

    points = ScoreModel()
    pts = np.array([sum(x[i] * points.points[k] for i, k in enumerate(FEATURES)) for x in Xte])
    model = ScoreModel().fit(Xtr, ytr)
    prob = 1 / (1 + np.exp(-(Xte @ model.coef + model.intercept)))

    res: dict = {"scoring": {
        "train_rows": int(cut), "test_rows": int(len(yte)), "base_rate": round(float(yte.mean()), 3),
        "auc_workshop_points": round(auc(yte, pts), 3), "auc_calibrated": round(auc(yte, prob), 3),
        "lift_top10_points": round(lift_top_decile(yte, pts), 2),
        "lift_top10_calibrated": round(lift_top_decile(yte, prob), 2),
        "weights": {k: round(float(w), 3) for k, w in zip(FEATURES, model.coef)},
    }}
    (ROOT / "data" / "model.json").write_text(json.dumps(
        {"features": FEATURES, "coef": model.coef.tolist(), "intercept": model.intercept}, indent=2))

    # ---- run the full pipeline on inbound leads ----
    pipe = LeadPipeline(index, Router.from_yaml(ROOT / "config" / "routing.yaml"), model)
    tp = fp = fn = 0
    grades, queues, rules, slas = Counter(), Counter(), Counter(), []
    with open(ROOT / "data" / "inbound.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    for r in rows:
        truth = r.pop("_duplicate")
        out = pipe.process(LeadIn(**r))
        tp += truth and out.is_duplicate
        fp += (not truth) and out.is_duplicate
        fn += truth and not out.is_duplicate
        if not out.is_duplicate:
            grades[out.score.grade] += 1
            queues[out.routing.queue] += 1
            rules[out.routing.rule] += 1
            slas.append(out.routing.sla_minutes)
    load = pipe.router.load()
    mm = [load[r.name] for r in pipe.router.queues["midmarket"].reps]
    res["dedupe"] = {"planted": int(tp + fn), "found": int(tp), "false_positives": int(fp),
                     "precision": round(tp / max(tp + fp, 1), 3), "recall": round(tp / max(tp + fn, 1), 3)}
    res["routing"] = {"leads_routed": int(sum(queues.values())), "by_queue": dict(queues), "by_rule": dict(rules),
                      "by_grade": dict(sorted(grades.items())), "rep_load": load,
                      "midmarket_balance_max_minus_min": int(max(mm) - min(mm))}
    (ROOT / "docs" / "results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    charts(res, yte, pts, prob)
    print(json.dumps({k: v for k, v in res.items() if k != "routing"} | {"routing": {k: res["routing"][k] for k in ("leads_routed", "by_queue", "midmarket_balance_max_minus_min")}}, indent=2))
    return res


def charts(res: dict, y, pts, prob) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    docs = ROOT / "docs"
    # 1) conversion by score decile, workshop points vs calibrated
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for s, label, color in [(pts, "Workshop points", "#9aa5b1"), (prob, "Calibrated model", "#1f6feb")]:
        order = np.argsort(s)
        deciles = np.array_split(order, 10)
        ax.plot(range(1, 11), [y[d].mean() * 100 for d in deciles], marker="o", label=label, color=color)
    ax.set_xlabel("Score decile (1 = lowest)")
    ax.set_ylabel("Conversion rate %")
    ax.set_title(f"Hold-out conversion by score decile  (AUC {res['scoring']['auc_workshop_points']} → {res['scoring']['auc_calibrated']})")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(docs / "score_deciles.png", dpi=150)
    plt.close(fig)

    # 2) rep load
    load = res["routing"]["rep_load"]
    load = {k: v for k, v in load.items() if k != "Marketing Nurture"}
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.barh(list(load), list(load.values()), color="#1f6feb")
    ax.invert_yaxis()
    ax.set_xlabel("Leads assigned")
    ax.set_title("Assigned leads per rep (round-robin with capacity)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(docs / "rep_load.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
