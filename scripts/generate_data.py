"""Generate a reproducible synthetic dataset.

- data/firmographics.csv : 400 companies (domain, industry, size, country)
- data/history.csv       : 6,000 past leads with a closed-won outcome (for model training)
- data/inbound.jsonl     : 1,500 new inbound leads incl. ~8% planted duplicates (with ground truth)

Outcomes come from a hidden "true" conversion process that the model has to recover;
the hand-set points in scoring.py deliberately disagree with it in places (as real
workshop weights do), so calibration has something to fix.
"""
from __future__ import annotations

import csv
import json
import math
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from leadrouter.enrich import REGIONS  # noqa: E402

rng = random.Random(42)

INDUSTRIES = ["Manufacturing", "Logistics", "Retail", "Chemicals", "Automotive", "Software",
              "Financial Services", "Healthcare", "Public Sector", "Education"]
COUNTRIES = ["DE"] * 8 + ["AT", "CH", "NL", "NL", "BE", "SE", "DK", "FR", "ES", "IT", "GB", "PL", "US"]
SYL = ["nor", "rijn", "alp", "hansa", "kraft", "linde", "vela", "terra", "nova", "ost", "west", "berg",
       "stern", "blau", "grün", "lux", "fjord", "delta", "arc", "mira"]
SUF = ["tech", "logistik", "chemie", "pack", "werke", "parts", "trade", "systems", "group", "labs"]
LEGAL = {"DE": "GmbH", "AT": "GmbH", "CH": "AG", "NL": "B.V.", "BE": "NV", "SE": "AB", "DK": "A/S",
         "FR": "SAS", "ES": "S.L.", "IT": "S.r.l.", "GB": "Ltd", "PL": "Sp. z o.o.", "US": "Inc."}
TLD = {"DE": "de", "AT": "at", "CH": "ch", "NL": "nl", "BE": "be", "SE": "se", "DK": "dk", "FR": "fr",
       "ES": "es", "IT": "it", "GB": "co.uk", "PL": "pl", "US": "com"}
TITLES = ["CEO", "Founder", "Head of Operations", "VP Sales", "Director IT", "Operations Manager",
          "Sales Manager", "Senior Analyst", "Analyst", "Working Student", "Assistant", "CFO",
          "Head of Revenue Operations", "Team Lead Customer Service", "Procurement Specialist"]
FIRST = ["Lena", "Jonas", "Mia", "Paul", "Sofie", "Erik", "Anna", "Tom", "Clara", "Lukas", "Emma",
         "Noah", "Hannah", "Felix", "Laura", "Max", "Julia", "David", "Sara", "Leon"]
LAST = ["Müller", "Schmidt", "Schneider", "Fischer", "Weber", "Meyer", "Wagner", "Becker", "Jansen",
        "de Vries", "Lindqvist", "Kowalski", "Rossi", "Garcia", "Smith", "Nowak", "Hoffmann", "Koch"]
SOURCES = ["web_form", "web_form", "web_form", "paid_search", "paid_search", "event", "referral", "content_download"]
MESSAGES = [
    "", "", "Just browsing.", "Please send pricing.",
    "We need to automate our reporting, spreadsheets are too manual. Budget approved for Q1.",
    "Looking to replace our current CRM this quarter, I own the decision.",
    "Audit coming up, need compliance reporting asap. Budget of 40k EUR.",
    "Interested in integrating with our ERP within 3 months.",
    "Student research project.",
    "Our CFO asked us to consolidate tools by next year.",
]

FREE = ["gmail.com", "gmx.de", "web.de", "outlook.com"]


def companies(n: int = 400) -> list[dict]:
    out, seen = [], set()
    while len(out) < n:
        name = (rng.choice(SYL) + rng.choice(SUF)).capitalize()
        c = rng.choice(COUNTRIES)
        dom = f"{name.lower().replace('ü', 'ue')}.{TLD[c]}"
        if dom in seen:
            continue
        seen.add(dom)
        emp = int(math.exp(rng.uniform(math.log(8), math.log(20000))))
        out.append({"domain": dom, "company": f"{name} {LEGAL[c]}", "industry": rng.choice(INDUSTRIES),
                    "employees": emp, "country": c})
    # existing customers named in routing.yaml
    out += [
        {"domain": "hansa-logistik.de", "company": "Hansa Logistik GmbH", "industry": "Logistics", "employees": 3400, "country": "DE"},
        {"domain": "nordic-pack.se", "company": "Nordic Pack AB", "industry": "Manufacturing", "employees": 2600, "country": "SE"},
        {"domain": "rijn-chemie.nl", "company": "Rijn Chemie B.V.", "industry": "Chemicals", "employees": 5100, "country": "NL"},
    ]
    return out


def true_logit(c: dict | None, title: str, source: str, views: int, demo: bool, msg: str, free: bool) -> float:
    """The hidden conversion process. Note: Retail converts badly here, mid-size beats enterprise,
    'paid_search' is weaker than the workshop believed, and referrals are much stronger."""
    z = -4.2
    if c:
        z += {"Manufacturing": 1.0, "Logistics": 1.1, "Chemicals": 0.9, "Automotive": 0.6,
              "Retail": -0.2, "Software": 0.2}.get(c["industry"], -0.4)
        e = c["employees"]
        z += 0.9 if 200 <= e < 2000 else 0.5 if e >= 2000 else -0.6
        z += 0.6 if REGIONS.get(c["country"]) in {"DACH", "Benelux", "Nordics"} else -0.2
    else:
        z -= 0.8
    t = title.lower()
    z += 0.9 if any(k in t for k in ["ceo", "founder", "vp", "head", "director", "cfo"]) else 0.3 if "manager" in t or "lead" in t else -0.5
    if "student" in t or "assistant" in t:
        z -= 0.8
    z += {"referral": 1.4, "event": 0.5, "paid_search": -0.1, "content_download": -0.3}.get(source, 0.0)
    z += 0.35 * math.log1p(views)
    z += 1.5 if demo else 0.0
    z += 0.8 if any(k in msg.lower() for k in ["budget", "this quarter", "asap", "q1", "decision"]) else 0.0
    z -= 0.7 if "student" in msg.lower() else 0.0
    z -= 0.6 if free else 0.0
    return z


def make_lead(comps: list[dict], when: datetime) -> tuple[dict, float]:
    c = rng.choice(comps) if rng.random() > 0.12 else None
    first, last = rng.choice(FIRST), rng.choice(LAST)
    free = c is None or rng.random() < 0.08
    local = f"{first}.{last}".lower().replace(" ", "").replace("ü", "ue").replace("ö", "oe")
    email = f"{local}@{rng.choice(FREE) if free else c['domain']}"
    title = rng.choice(TITLES)
    source = rng.choice(SOURCES)
    views = int(rng.expovariate(1 / 4))
    demo = rng.random() < 0.18
    msg = rng.choice(MESSAGES)
    lead = {
        "email": email, "first_name": first, "last_name": last,
        "company": c["company"] if c else rng.choice(["", "Freelance", "Private"]),
        "website": f"https://www.{c['domain']}" if c and free else "",
        "country": c["country"] if c else "", "title": title, "source": source, "message": msg,
        "page_views": views, "demo_requested": demo, "created_at": when.isoformat(),
    }
    return lead, true_logit(c, title, source, views, demo, msg, free)


def main() -> None:
    data = ROOT / "data"
    data.mkdir(exist_ok=True)
    comps = companies()
    with open(data / "firmographics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(comps[0]))
        w.writeheader()
        w.writerows(comps)

    start = datetime(2025, 1, 1)
    with open(data / "history.csv", "w", newline="", encoding="utf-8") as f:
        w = None
        for i in range(6000):
            lead, z = make_lead(comps, start + timedelta(hours=i * 1.4))
            lead["converted"] = int(rng.random() < 1 / (1 + math.exp(-z)))
            if w is None:
                w = csv.DictWriter(f, fieldnames=list(lead))
                w.writeheader()
            w.writerow(lead)

    inbound, originals, seen_emails = [], [], set()
    t0 = datetime(2026, 10, 5, 8)
    for i in range(1500):
        when = t0 + timedelta(minutes=i * 9)
        if originals and rng.random() < 0.08:
            o = dict(rng.choice(originals))
            kind = rng.choice(["same", "plus", "case", "name_other_email"])
            local, dom = o["email"].split("@")
            if kind == "plus":
                o["email"] = f"{local}+webinar@{dom}"
            elif kind == "case":
                o["email"] = o["email"].upper()
            elif kind == "name_other_email":
                if dom in FREE:  # can't match a name at a company domain for free-mail leads
                    o["email"] = o["email"]
                else:
                    o["email"] = f"{o['first_name'][0].lower()}{o['last_name'].lower().replace(' ', '')}@{dom}"
            o["created_at"] = when.isoformat()
            o["_duplicate"] = True
            inbound.append(o)
            continue
        lead, _ = make_lead(comps, when)
        name_key = (lead["first_name"], lead["last_name"], lead["email"].split("@")[1])
        if lead["email"] in seen_emails or name_key in seen_emails:
            continue  # a genuinely new lead must be a new person
        seen_emails.update({lead["email"], name_key})
        lead["_duplicate"] = False
        originals.append(lead)
        inbound.append(lead)
    with open(data / "inbound.jsonl", "w", encoding="utf-8") as f:
        for row in inbound:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"companies={len(comps)} history=6000 inbound={len(inbound)} duplicates={sum(r['_duplicate'] for r in inbound)}")


if __name__ == "__main__":
    main()
