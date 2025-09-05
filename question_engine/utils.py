# question_engine/utils.py
from __future__ import annotations
import re
from typing import Dict, Iterable, Optional, Tuple

from dateutil.relativedelta import relativedelta
from datetime import datetime

# Month tokens like 2025-06, Jun 2025, June 25, 2025/06, etc.
MONTH_PATTERNS = [
    r"(?P<ym>\d{4}-\d{2})",
    r"(?P<my>\d{2}/\d{4})",
    r"(?P<mname>(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*)\s+(?P<y>\d{2,4})",
]

def parse_month(expr: str) -> Optional[str]:
    s = expr.strip()
    for pat in MONTH_PATTERNS:
        m = re.search(pat, s, flags=re.IGNORECASE)
        if not m:
            continue
        if m.lastgroup == "ym":
            return m.group("ym")
        if m.lastgroup == "my":
            mm, yyyy = m.group("my").split("/")
            return f"{yyyy}-{mm.zfill(2)}"
        if m.lastgroup == "mname":
            mname = m.group("mname")[:3].title()
            y = m.group("y")
            if len(y) == 2:
                y = f"20{y}"
            dt = datetime.strptime(f"{mname} {y}", "%b %Y")
            return dt.strftime("%Y-%m")
    return None

def parse_month_range(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Try to pull an inclusive month range from text.
    Examples:
      "from 2025-04 to 2025-06"
      "Apr 2025 to Jun 2025"
      "last month" / "last 3 months"
    """
    t = text.lower()

    # relative phrases
    now = datetime.utcnow().replace(day=1)
    if "last month" in t:
        last = now - relativedelta(months=1)
        return last.strftime("%Y-%m"), last.strftime("%Y-%m")
    m = re.search(r"last\s+(\d+)\s+months?", t)
    if m:
        k = int(m.group(1))
        start = (now - relativedelta(months=k)).strftime("%Y-%m")
        end = (now - relativedelta(months=1)).strftime("%Y-%m")
        return start, end

    # explicit range
    # find two months in text
    months = []
    tmp = text
    for _ in range(2):
        m = None
        for pat in MONTH_PATTERNS:
            m = re.search(pat, tmp, flags=re.IGNORECASE)
            if m:
                mo = parse_month(m.group(0))
                if mo:
                    months.append(mo)
                tmp = tmp[m.end():]
                break
        if not m:
            break
    if len(months) == 2:
        months.sort()
        return months[0], months[1]

    # single explicit month
    one = parse_month(text)
    if one:
        return one, one

    return None, None


def parse_dim_filters(text: str, alias_map: Dict[str, str]) -> Dict[str, Iterable[str]]:
    """
    Heuristic exact-match filter parser. Usage:
      parse_dim_filters("portfolio london onshore manual critical yes", DIM_CANONICAL)
    Returns: {"Portfolio_std": ["London"], "Shore": ["Onshore"], "Automation": ["Manual"], "Critical": ["Yes"]}
    """
    words = re.findall(r"[A-Za-z0-9\-\_']+", text)
    out: Dict[str, list] = {}

    # greedy two-word keys first (e.g., "process group")
    keys_sorted = sorted(alias_map.keys(), key=lambda k: -len(k))
    i = 0
    while i < len(words):
        # look ahead for 2-word and 1-word keys
        joined2 = " ".join(words[i:i+2]).lower()
        joined1 = words[i].lower()

        col = None
        if joined2 in alias_map:
            col = alias_map[joined2]
            i += 2
        elif joined1 in alias_map:
            col = alias_map[joined1]
            i += 1

        if col:
            # next token(s) until we hit another key or end
            vals = []
            while i < len(words):
                look2 = " ".join(words[i:i+2]).lower()
                look1 = words[i].lower()
                if look2 in alias_map or look1 in alias_map:
                    break
                vals.append(words[i])
                i += 1
            if vals:
                out.setdefault(col, []).append(" ".join(vals))
            continue

        i += 1

    # boolean normalization for known Y/N-like columns
    for k in ["Critical", "WithinSLA", "Consented", "MercerConsented", "VulnerableCustomer"]:
        if k in out:
            out[k] = [ "Yes" if v.lower().startswith(("y","yes","true","1")) else
                       "No" if v.lower().startswith(("n","no","false","0")) else v
                       for v in out[k] ]

    return out
