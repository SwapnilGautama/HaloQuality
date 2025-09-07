# =========================
# UPDATED FAIL-REASON MODEL
# =========================
import re
import pandas as pd

# Prioritised & broader ruleset (order matters)
# Each entry: ("Display name", compiled_regex)
REASON_RULES: list[tuple[str, re.Pattern]] = [
    (
        "Data entry / setup",
        re.compile(
            r"\b(record|update|amend|correct|details|address|dob|ni|nino|national insurance|"
            r"surname|forename|name|typo|keyed|data entry|case (?:not )?created|workitem|workflow)\b",
            re.I,
        ),
    ),
    (
        "Communication / update",
        re.compile(
            r"\b(email|letter|call|chase|follow.?up|advise|inform|update|send|issue|reminder)\b",
            re.I,
        ),
    ),
    (
        "Scheme / rules",
        re.compile(
            r"\b(rule|eligib|ineligib|protected|section|basis|gmp|spouse pension|benefit basis|trust deed)\b",
            re.I,
        ),
    ),
    (
        "Bank / payment",
        re.compile(
            r"\b(bank|payment|paid|refund|cheque|bacs|overpayment|underpayment)\b",
            re.I,
        ),
    ),
    (
        "Manual calculation",
        re.compile(
            r"\b(manual calc|recalc|calculate|calculation|calc\b|factor|projection|quote|gmp)\b",
            re.I,
        ),
    ),
    (
        "QA / workflow reject",
        re.compile(
            r"\b(reject|invalid|valid reject|self reject|qa|checklist|snapshot|event type|action per)\b",
            re.I,
        ),
    ),
    (
        "Retirement options",
        re.compile(
            r"\b(retire|retirement|drawdown|annuity|ufpls|pension commencement|commutation)\b",
            re.I,
        ),
    ),
    (
        "Procedure / documents",
        re.compile(
            r"\b(form|document|docs|evidence|certificate|proof|id\b|passport|driving licence|signature|consent|authority)\b",
            re.I,
        ),
    ),
    (
        "Death benefits",
        re.compile(
            r"\b(death|bereave|deceased|probate)\b",
            re.I,
        ),
    ),
    (
        "Transfer",
        re.compile(
            r"\b(transfer|cetv|transferring)\b",
            re.I,
        ),
    ),
    (
        "Postal / dispatch",
        re.compile(
            r"\b(post|posted|dispatch|mail|returned)\b",
            re.I,
        ),
    ),
    (
        "Contributions / fees",
        re.compile(
            r"\b(contribution|fee count|fees|charges|levy)\b",
            re.I,
        ),
    ),
    (
        "Pension increase",
        re.compile(
            r"\b(pension increase|uprate|uprating|pi)\b",
            re.I,
        ),
    ),
    (
        "System / workflow",
        re.compile(
            r"\b(system|aptia|workflow|error code|bug|crash)\b",
            re.I,
        ),
    ),
    (
        "Waiting on member/TPA",
        re.compile(
            r"\b(member to|member has to|await(?:ing)? member|waiting on member|tpa|third party admin)\b",
            re.I,
        ),
    ),
]


def _label_reason(comments: pd.Series) -> pd.Series:
    """
    Assign exactly one reason per comment using a prioritised rule order.
    Falls back to 'Other' if nothing matches.
    Vectorised for speed and deterministic (first match wins).
    """
    s = comments.fillna("").astype(str).str.lower()
    reason = pd.Series("Other", index=s.index)
    unmatched = pd.Series(True, index=s.index)

    for name, rx in REASON_RULES:
        hits = s.str.contains(rx)
        # only assign to rows that are still unmatched
        newly_matched = unmatched & hits
        if newly_matched.any():
            reason.loc[newly_matched] = name
            unmatched &= ~hits
        if not unmatched.any():
            break

    return reason


def _top80_pareto(counts: pd.Series, total: int) -> pd.DataFrame:
    """
    Correct Pareto for 'top 80% + Other':
      - Sort categories by count desc
      - Keep adding categories until cumulative percent >= 80% (or run out)
      - Append 'Other' row for the remainder (if any)
      - Return reason/count/percent/cum_percent
    """
    df = counts.sort_values(ascending=False).reset_index()
    df.columns = ["reason", "count"]
    if total <= 0:
        # Avoid divide by zero; return empty DF
        return pd.DataFrame(columns=["reason", "count", "percent", "cum_percent"])

    df["percent"] = (df["count"] / total) * 100.0

    kept_rows = []
    coverage = 0.0
    for _, r in df.iterrows():
        kept_rows.append(r)
        coverage += r["percent"]
        if coverage >= 80.0:
            break

    top = pd.DataFrame(kept_rows)
    # If nothing matched (should not happen), keep the top category at least
    if top.empty and not df.empty:
        top = df.iloc[:1].copy()

    other_percent = 100.0 - top["percent"].sum()
    other_count = max(0, int(round(total * other_percent / 100.0)))

    # Only append Other if there's a remainder
    if other_count > 0:
        top = pd.concat(
            [
                top,
                pd.DataFrame(
                    [{"reason": "Other", "count": other_count, "percent": other_percent}]
                ),
            ],
            ignore_index=True,
        )

    top["cum_percent"] = top["percent"].cumsum()
    # Tidy presentation
    top["percent"] = top["percent"].round(1)
    top["cum_percent"] = top["cum_percent"].round(1)
    return top
