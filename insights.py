"""Pattern extraction, heuristic prioritisation and recommendations.

Everything here is computed from the analysed reviews; nothing is invented.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from review_engine import SEVERITY_RANK

HEURISTIC_NOTE = (
    "Priority score (0-100) is a transparent heuristic, not an objectively validated metric: "
    "40% review frequency (relative to the most-reported theme), 30% share of negative reviews within the theme, "
    "30% average severity of its negative reviews. Use it to start a discussion, not to replace judgement."
)

ACTIONS = {
    "Bug": "Reproduce and triage these reports first (check crash/error logs for the latest release), ship a fix or hotfix, and reply to affected reviewers.",
    "Performance": "Profile load times and battery/CPU use on mid-range devices, set a performance budget, and fix the slowest screens first.",
    "UX": "Run a quick usability test on the affected flow, simplify it, and A/B test the revised design against the current one.",
    "Pricing": "Audit billing and refund failures, review how plans and prices are communicated, and test a clearer plan comparison or lower-cost tier.",
    "Content Request": "Add this to the roadmap discussion and validate demand with a short in-app survey before committing engineering time.",
    "Other": "Review the underlying reviews manually, assign an owner, and decide whether it needs a product fix or a support-process change.",
}

SUMMARY_COLUMNS = [
    "theme", "category", "reviews", "negative", "positive", "neutral", "pain_reviews", "feature_requests",
    "avg_rating", "avg_severity", "high_severity", "negative_share", "priority_score", "priority_rank",
]


def pain_mask(df: pd.DataFrame) -> pd.Series:
    """Negative reviews that are complaints (not pure feature requests)."""
    return (df["sentiment"] == "Negative") & (df["category"] != "Content Request")


def feature_mask(df: pd.DataFrame) -> pd.Series:
    return df["is_feature_request"].astype(bool) | (df["category"] == "Content Request")


def build_theme_summary(df: pd.DataFrame, min_support: int = 3) -> pd.DataFrame:
    rows = []
    for theme, grp in df.groupby("theme"):
        pain = grp[pain_mask(grp)]
        rows.append({
            "theme": theme,
            "category": grp["category"].mode().iat[0],
            "reviews": len(grp),
            "negative": int((grp["sentiment"] == "Negative").sum()),
            "positive": int((grp["sentiment"] == "Positive").sum()),
            "neutral": int((grp["sentiment"] == "Neutral").sum()),
            "pain_reviews": len(pain),
            "feature_requests": int(feature_mask(grp).sum()),
            "avg_rating": round(float(grp["rating"].mean()), 2) if grp["rating"].notna().any() else np.nan,
            "avg_severity": round(float(pain["severity"].map(SEVERITY_RANK).mean()), 2) if len(pain) else np.nan,
            "high_severity": int((pain["severity"] == "High").sum()),
            "negative_share": round(len(pain) / len(grp), 3),
        })
    ts = pd.DataFrame(rows, columns=[c for c in SUMMARY_COLUMNS if c not in ("priority_score", "priority_rank")])
    if ts.empty:
        return ts.assign(priority_score=np.nan, priority_rank=np.nan)

    eligible = ts["pain_reviews"] >= min_support
    max_pain = ts.loc[eligible, "pain_reviews"].max() if eligible.any() else 1
    freq = ts["pain_reviews"] / max_pain
    sev = ((ts["avg_severity"].fillna(1) - 1) / 2).clip(0, 1)
    score = 100 * (0.4 * freq + 0.3 * ts["negative_share"] + 0.3 * sev)
    ts["priority_score"] = score.where(eligible).round(1)
    ts = ts.sort_values(["priority_score", "reviews"], ascending=[False, False], na_position="last").reset_index(drop=True)
    ts["priority_rank"] = ts["priority_score"].rank(ascending=False, method="first")
    return ts


def _top_counts(sub: pd.DataFrame, n: int) -> pd.DataFrame:
    if sub.empty:
        return pd.DataFrame(columns=["theme", "reviews"])
    out = sub.groupby("theme").size().sort_values(ascending=False).head(n).reset_index()
    out.columns = ["theme", "reviews"]
    return out


def top_problems(df: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    return _top_counts(df[pain_mask(df)], n)


def top_requests(df: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    return _top_counts(df[feature_mask(df)], n)


def top_positives(df: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    return _top_counts(df[(df["sentiment"] == "Positive") & ~feature_mask(df)], n)


def high_priority_pain_points(ts: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    return ts[ts["priority_score"].notna()].head(n)


def find_emerging(df: pd.DataFrame, min_support: int = 3) -> tuple[pd.DataFrame, str]:
    """Themes whose share of negative reviews is clearly higher in the latest third of the date range."""
    cols = ["theme", "recent_reviews", "earlier_reviews", "recent_share", "earlier_share"]
    empty = pd.DataFrame(columns=cols)
    dated = df[df["date"].notna()]
    if len(dated) < 20:
        return empty, "Emerging patterns need at least 20 reviews with valid dates."
    span = (dated["date"].max() - dated["date"].min()).days
    if span < 28:
        return empty, "Emerging patterns need a date range of at least 4 weeks."
    cutoff = dated["date"].max() - pd.Timedelta(days=span / 3)
    pain = dated[pain_mask(dated)]
    recent, earlier = pain[pain["date"] > cutoff], pain[pain["date"] <= cutoff]
    if len(recent) < 5 or len(earlier) < 5:
        return empty, "Too few negative reviews in the recent or earlier period to compare."
    rows = []
    for theme in recent["theme"].unique():
        r, e = int((recent["theme"] == theme).sum()), int((earlier["theme"] == theme).sum())
        rs, es = r / len(recent), e / len(earlier)
        if r >= min_support and rs >= 2 * es:
            rows.append({"theme": theme, "recent_reviews": r, "earlier_reviews": e,
                         "recent_share": round(rs, 3), "earlier_share": round(es, 3)})
    note = (f"Recent period: after {cutoff:%Y-%m-%d} ({len(recent)} negative reviews) vs. earlier period "
            f"({len(earlier)} negative reviews). A theme is flagged when its share of negative reviews at least doubled.")
    out = pd.DataFrame(rows, columns=cols).sort_values("recent_reviews", ascending=False).reset_index(drop=True)
    return out, note


def _evidence(df: pd.DataFrame, theme: str, problem: bool, k: int = 2) -> list[str]:
    sub = df[df["theme"] == theme]
    sub = sub[pain_mask(sub)] if problem else sub[feature_mask(sub)]
    if problem:
        sub = sub.assign(_s=sub["severity"].map(SEVERITY_RANK)).sort_values("_s", ascending=False, kind="stable")
    return [t if len(t) <= 160 else t[:157] + "..." for t in sub["review"].head(k)]


def build_recommendations(df: pd.DataFrame, ts: pd.DataFrame, emerging: pd.DataFrame,
                          min_support: int = 3, max_problems: int = 4, max_requests: int = 2) -> list[dict]:
    """Only themes with at least `min_support` supporting reviews produce a recommendation."""
    recs: list[dict] = []
    rising = set(emerging["theme"]) if not emerging.empty else set()
    for _, row in high_priority_pain_points(ts, max_problems).iterrows():
        why = [f"{row['negative_share']:.0%} of reviews on this theme are negative complaints"]
        if row["high_severity"]:
            why.append(f"{int(row['high_severity'])} reported as high severity")
        if pd.notna(row["avg_rating"]):
            why.append(f"average rating {row['avg_rating']:.1f}/5")
        if row["theme"] in rising:
            why.append("its share of complaints is rising in recent reviews")
        recs.append({
            "type": "Problem", "theme": row["theme"], "category": row["category"],
            "observed": f"{int(row['pain_reviews'])} of {int(row['reviews'])} reviews about \"{row['theme']}\" are complaints.",
            "supporting_reviews": int(row["pain_reviews"]),
            "why": "; ".join(why) + f". Priority score {row['priority_score']:.0f}/100 (heuristic).",
            "action": ACTIONS.get(row["category"], ACTIONS["Other"]),
            "evidence": _evidence(df, row["theme"], True),
        })
    for _, row in top_requests(df, 10).iterrows():
        if len([r for r in recs if r["type"] == "Opportunity"]) >= max_requests:
            break
        if row["reviews"] < min_support:
            continue
        recs.append({
            "type": "Opportunity", "theme": row["theme"], "category": "Content Request",
            "observed": f"{int(row['reviews'])} reviews request \"{row['theme']}\".",
            "supporting_reviews": int(row["reviews"]),
            "why": "Repeated explicit requests signal unmet demand and can reduce churn if addressed.",
            "action": ACTIONS["Content Request"],
            "evidence": _evidence(df, row["theme"], False),
        })
    return recs


def build_report(df: pd.DataFrame, cleaning, analysis, ts: pd.DataFrame, emerging: pd.DataFrame,
                 recs: list[dict], source_label: str, min_support: int) -> str:
    n = len(df)
    sent = df["sentiment"].value_counts()
    lines = [
        "APP REVIEW INSIGHTS REPORT",
        "=" * 60,
        f"Generated: {datetime.now():%Y-%m-%d %H:%M}",
        f"Data source: {source_label}",
        f"Reviews analyzed: {n}",
        f"Cleaning: {cleaning.input_rows} rows in, {cleaning.empty_removed} empty removed, "
        f"{cleaning.duplicates_removed} duplicates removed, {cleaning.final_rows} kept.",
        f"Analysis method: {analysis.gemini_count} reviews by Gemini ({analysis.model}), "
        f"{analysis.fallback_count} by rule-based fallback.",
    ]
    for note in analysis.notes:
        lines.append(f"NOTE: {note}")
    lines += ["", "SENTIMENT"]
    lines += [f"  {s}: {int(sent.get(s, 0))} ({sent.get(s, 0) / n:.0%})" for s in ("Positive", "Neutral", "Negative")]

    def block(title: str, frame: pd.DataFrame, col: str = "reviews"):
        lines.extend(["", title])
        if frame.empty:
            lines.append("  (none found)")
        for _, r in frame.iterrows():
            lines.append(f"  - {r['theme']}: {int(r[col])} reviews")

    block("MOST REPORTED PROBLEMS", top_problems(df))
    block("MOST REQUESTED FEATURES", top_requests(df))
    block("COMMON POSITIVE FEEDBACK", top_positives(df))

    lines += ["", f"HIGH-PRIORITY PAIN POINTS (min {min_support} reviews per theme)", f"  {HEURISTIC_NOTE}"]
    top = high_priority_pain_points(ts, 8)
    if top.empty:
        lines.append("  (no theme has enough supporting reviews)")
    for _, r in top.iterrows():
        lines.append(f"  #{int(r['priority_rank'])} {r['theme']} [{r['category']}] - score {r['priority_score']:.0f}, "
                     f"{int(r['pain_reviews'])} complaints, {int(r['high_severity'])} high severity")

    lines += ["", "EMERGING PATTERNS"]
    if emerging.empty:
        lines.append("  (none detected or not enough dated data)")
    for _, r in emerging.iterrows():
        lines.append(f"  - {r['theme']}: {int(r['recent_reviews'])} recent vs {int(r['earlier_reviews'])} earlier negative reviews")

    lines += ["", "RECOMMENDATIONS"]
    if not recs:
        lines.append("  (not enough supporting data to make recommendations)")
    for i, r in enumerate(recs, 1):
        lines += [f"  {i}. [{r['type']}] {r['theme']}",
                  f"     Observed: {r['observed']}",
                  f"     Supporting reviews: {r['supporting_reviews']}",
                  f"     Why it matters: {r['why']}",
                  f"     Suggested action: {r['action']}"]
    lines += ["", "-" * 60,
              "Themes, severity and sentiment come from automated analysis and may contain errors; "
              "verify against the raw reviews before acting."]
    return "\n".join(lines)
