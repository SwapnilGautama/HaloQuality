# question_engine/parser.py
from __future__ import annotations
import re
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import Dict, List, Tuple, Optional

from rapidfuzz import fuzz, process
from .lexicon import DIM_CANON, DIM_SYNONYMS, METRIC_CANON, METRIC_SYNONYMS, DOMAIN_HINTS

# Month → "MMM YY"
def _to_month_str(dt: datetime) -> str:
    return dt.strftime("%b %y")

def parse_month_range(text: str) -> Tuple[Optional[str], Optional[str]]:
    t = text.lower()

    # relative: last N months
    m = re.search(r"last\s+(\d+)\s+months?", t)
    if m:
        k = int(m.group(1))
        end = datetime.utcnow().replace(day=1)
        start = end - relativedelta(months=k)
        return _to_month_str(start), _to_month_str(end)

    if "last month" in t:
        end = datetime.utcnow().replace(day=1) - relativedelta(months=1)
        return _to_month_str(end), _to_month_str(end)

    # absolute: find two month-like tokens
    MONTH_NAMES = "(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"
    # e.g., Jun 2025 / June 25
    pats = [
        rf"({MONTH_NAMES})\s+(\d{{2,4}})",
        r"(\d{4})-(\d{2})",
    ]
    months: List[str] = []
    for pat in pats:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            try:
                if len(m.groups()) == 2 and m.re.pattern.startswith("("):
                    mon, y = m.group(1)[:3].title(), m.group(2)
                    if len(y) == 2: y = "20"+y
                    dt = datetime.strptime(f"{mon} {y}", "%b %Y")
                else:
                    y, mm = int(m.group(1)), int(m.group(2))
                    dt = datetime(y, mm, 1)
                months.append(_to_month_str(dt))
            except Exception:
                pass
    if len(months) >= 2:
        months.sort(key=lambda s: datetime.strptime(s, "%b %y"))
        return months[0], months[-1]
    if len(months) == 1:
        return months[0], months[0]
    return None, None

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())

def resolve_dimension(token: str) -> Optional[str]:
    t = _norm(token)
    if t in DIM_CANON: return DIM_CANON[t]
    if t in DIM_SYNONYMS: return DIM_SYNONYMS[t]
    # fuzzy resolve
    cand, score, _ = process.extractOne(t, list(DIM_CANON.keys())+list(DIM_SYNONYMS.keys()), scorer=fuzz.partial_ratio)
    if score >= 85:
        return DIM_CANON.get(cand, DIM_SYNONYMS.get(cand))
    return None

def resolve_metric(token: str) -> Optional[str]:
    t = _norm(token)
    if t in METRIC_CANON: return METRIC_CANON[t]
    if t in METRIC_SYNONYMS: return METRIC_SYNONYMS[t]
    cand, score, _ = process.extractOne(t, list(METRIC_CANON.keys())+list(METRIC_SYNONYMS.keys()), scorer=fuzz.partial_ratio)
    if score >= 85:
        return METRIC_CANON.get(cand, METRIC_SYNONYMS.get(cand))
    return None

def infer_domain(text: str) -> str:
    t = text.lower()
    scores = {}
    for dom, keys in DOMAIN_HINTS.items():
        scores[dom] = sum(1 for k in keys if k in t)
    # default priority: complaints > fpa > cases (tweakable)
    ordered = sorted(scores.items(), key=lambda x: (-x[1], {"complaints":0,"fpa":1,"cases":2}[x[0]]))
    return ordered[0][0] if ordered else "complaints"

def parse_group_by(text: str) -> List[str]:
    """
    Captures 'by …' lists, e.g., 'show rca1 by portfolio, process'
    """
    g = []
    m = re.search(r"\bby\b(.+)", text, flags=re.IGNORECASE)
    if not m:
        return g
    tail = m.group(1)
    # stop at 'for', 'where', 'top', 'in', 'over', 'between'
    tail = re.split(r"\b(for|where|top|in|over|between)\b", tail, flags=re.IGNORECASE)[0]
    parts = re.split(r"[,/|]+| and ", tail, flags=re.IGNORECASE)
    for p in parts:
        dim = resolve_dimension(p)
        if dim and dim not in g:
            g.append(dim)
    return g

def parse_filters(text: str) -> Dict[str, List[str]]:
    """
    Simple dim:value parser. Accepts:
      - for process X
      - where portfolio = london, scheme alpha
      - portfolio london chichester
      - process "member enquiry"
    """
    out: Dict[str, List[str]] = {}
    # explicit 'for/where' segments
    for kw in ["for", "where", "in"]:
        m = re.search(rf"\b{kw}\b(.+)", text, flags=re.IGNORECASE)
        if not m: continue
        seg = m.group(1)
        seg = re.split(r"\b(by|top|between|over)\b", seg, flags=re.IGNORECASE)[0]
        # look for dim:value pairs first
        for pair in re.finditer(r"([A-Za-z][A-Za-z\s]+)\s*[:=]\s*\"?([A-Za-z0-9\-\s]+)\"?", seg):
            dim = resolve_dimension(pair.group(1))
            if dim:
                out.setdefault(dim, []).append(pair.group(2).strip())
        # then loose 'dim value1 value2'
        tokens = re.findall(r"\"[^\"]+\"|[A-Za-z0-9\-\_]+", seg)
        i = 0
        while i < len(tokens)-1:
            dim = resolve_dimension(tokens[i].strip('"'))
            if dim:
                vals = []
                i += 1
                while i < len(tokens):
                    nxt = tokens[i].strip('"')
                    if resolve_dimension(nxt): break
                    vals.append(nxt); i += 1
                if vals: out.setdefault(dim, []).extend(vals)
            else:
                i += 1
    # de-dup
    for k in list(out.keys()):
        out[k] = list(dict.fromkeys(out[k]))
    return out

def parse_topn(text: str) -> Optional[int]:
    m = re.search(r"\btop\s+(\d+)", text, flags=re.IGNORECASE)
    return int(m.group(1)) if m else None

def parse_sort(text: str) -> Optional[str]:
    if "ascending" in text.lower() or "asc" in text.lower(): return "asc"
    if "descending" in text.lower() or "desc" in text.lower(): return "desc"
    return None
