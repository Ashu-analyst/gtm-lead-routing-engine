"""Domain normalisation, firmographic enrichment and duplicate detection."""
from __future__ import annotations

import csv
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable, Optional

from .models import Firmographics, LeadIn

FREE_MAIL = {
    "gmail.com", "googlemail.com", "yahoo.com", "hotmail.com", "outlook.com", "live.com",
    "gmx.de", "gmx.net", "web.de", "t-online.de", "icloud.com", "proton.me", "protonmail.com",
}

REGIONS = {
    "DE": "DACH", "AT": "DACH", "CH": "DACH",
    "NL": "Benelux", "BE": "Benelux", "LU": "Benelux",
    "SE": "Nordics", "DK": "Nordics", "NO": "Nordics", "FI": "Nordics",
    "FR": "Southern Europe", "ES": "Southern Europe", "IT": "Southern Europe", "PT": "Southern Europe",
    "GB": "UK&I", "IE": "UK&I",
    "PL": "CEE", "CZ": "CEE", "HU": "CEE", "RO": "CEE",
}

_LEGAL = re.compile(r"\b(gmbh|ag|se|kg|bv|nv|ltd|limited|inc|llc|sa|sas|sl|srl|ab|oy|as|co|sp z o o)\b", re.I)


def normalise_domain(value: str) -> str:
    """Turn 'https://www.Example.com/about' or 'jo@Example.com' into 'example.com'."""
    v = (value or "").strip().lower()
    if "@" in v:
        v = v.split("@", 1)[1]
    v = re.sub(r"^https?://", "", v)
    v = v.split("/", 1)[0].split("?", 1)[0]
    if v.startswith("www."):
        v = v[4:]
    return v


def lead_domain(lead: LeadIn) -> str:
    """Company domain for a lead: the email domain unless it is a free-mail provider."""
    email_domain = normalise_domain(lead.email)
    if email_domain and email_domain not in FREE_MAIL:
        return email_domain
    return normalise_domain(lead.website)


def normalise_company(name: str) -> str:
    n = (name or "").lower().replace(".", "").replace("/", "")
    n = re.sub(r"[^a-z0-9äöüß ]", " ", n)
    n = _LEGAL.sub(" ", n)
    return re.sub(r"\s+", " ", n).strip()


class FirmographicIndex:
    """Lookup of company data by domain, with a fuzzy company-name fallback."""

    def __init__(self, rows: Iterable[Firmographics]):
        self.by_domain = {r.domain: r for r in rows}
        self.by_name = {normalise_company(r.company): r for r in self.by_domain.values()}

    @classmethod
    def from_csv(cls, path: str | Path) -> "FirmographicIndex":
        with open(path, newline="", encoding="utf-8") as f:
            rows = [
                Firmographics(
                    domain=r["domain"], company=r["company"], industry=r["industry"],
                    employees=int(r["employees"]), country=r["country"],
                    region=REGIONS.get(r["country"], "Rest of World"),
                )
                for r in csv.DictReader(f)
            ]
        return cls(rows)

    def lookup(self, domain: str, company: str = "") -> Optional[Firmographics]:
        if domain and domain in self.by_domain:
            return self.by_domain[domain]
        key = normalise_company(company)
        if not key:
            return None
        if key in self.by_name:
            return self.by_name[key]
        best, best_ratio = None, 0.0
        for name, row in self.by_name.items():
            ratio = SequenceMatcher(None, key, name).ratio()
            if ratio > best_ratio:
                best, best_ratio = row, ratio
        return best if best_ratio >= 0.88 else None


class DuplicateDetector:
    """Flags a lead as duplicate when the same person (email) or the same name at the same
    company domain was already seen. Keeps the first record as the master."""

    def __init__(self) -> None:
        self.by_email: dict[str, str] = {}
        self.by_name_domain: dict[tuple[str, str], str] = {}

    @staticmethod
    def _email_key(email: str) -> str:
        local, _, domain = email.strip().lower().partition("@")
        local = local.split("+", 1)[0]
        if domain in {"gmail.com", "googlemail.com"}:
            local = local.replace(".", "")
            domain = "gmail.com"
        return f"{local}@{domain}"

    def check(self, lead: LeadIn, domain: str) -> Optional[str]:
        ekey = self._email_key(lead.email)
        nkey = (f"{lead.first_name} {lead.last_name}".strip().lower(), domain)
        if ekey in self.by_email:
            return self.by_email[ekey]
        if nkey[0] and domain and nkey in self.by_name_domain:
            return self.by_name_domain[nkey]
        self.by_email[ekey] = lead.email
        if nkey[0] and domain:
            self.by_name_domain[nkey] = lead.email
        return None
