"""AI App Review Insight & Pattern Extractor - Streamlit dashboard."""
from __future__ import annotations

import os

import pandas as pd
import plotly.express as px
import streamlit as st

import data_sources as ds
import insights as ins
import review_engine as eng

st.set_page_config(page_title="App Review Insight Extractor", page_icon="📊", layout="wide")

SENT_COLORS = {"Positive": "#2E9E6B", "Neutral": "#9AA3B2", "Negative": "#D64545"}
ACCENT = "#3B6FE0"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def default_api_key() -> str:
    """Key from Streamlit secrets or environment (never hard-coded, never displayed)."""
    try:
        key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        key = ""
    return (key or os.environ.get("GEMINI_API_KEY", "")).strip()


def style(fig, height: int = 340):
    fig.update_layout(
        template="plotly_white", height=height, margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title_text=""),
        font=dict(size=13),
    )
    return fig


def run_pipeline(raw: pd.DataFrame, label: str, api_key: str, model: str, batch_size: int, max_reviews: int) -> dict:
    cleaned, stats = eng.clean_reviews(raw)
    if cleaned.empty:
        raise ValueError("No usable reviews remain after cleaning (all rows were empty or duplicates).")
    total_clean = len(cleaned)
    subset = cleaned
    note = ""
    if api_key and total_clean > max_reviews:
        # Cap Gemini usage: keep the most recent reviews (original order preserved).
        subset = cleaned.sort_values("date", ascending=False, na_position="last", kind="stable").head(max_reviews)
        subset = subset.sort_values("review_id")
        note = f"Gemini analysis is capped at {max_reviews} reviews; the most recent {max_reviews} of {total_clean} were analyzed."
    bar = st.progress(0.0, text="Analyzing reviews...")
    analyzed, report = eng.analyze_reviews(
        subset, api_key or None, model, batch_size,
        progress=lambda p: bar.progress(min(p, 1.0), text=f"Analyzing reviews... {p:.0%}"),
    )
    bar.empty()
    return {"df": analyzed, "cleaning": stats, "report": report, "label": label,
            "total_clean": total_clean, "cap_note": note}


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("Review Insights")
    st.caption("Turn raw app reviews into prioritized product insights.")

    source = st.radio("Data source", ["Sample dataset", "Upload CSV", "Google Play (optional)"])
    uploaded, app_id, lang, country, count = None, "", "en", "us", 200
    if source == "Upload CSV":
        uploaded = st.file_uploader("CSV file", type=["csv"])
        st.caption("Required: a review text column. Optional: rating, date, app name.")
    elif source.startswith("Google Play"):
        app_id = st.text_input("App package ID", "com.duolingo")
        c1, c2 = st.columns(2)
        lang = c1.text_input("Language", "en")
        country = c2.text_input("Country", "us")
        count = st.slider("Reviews to fetch", 50, 500, 200, 50)
    else:
        st.caption("Built-in synthetic reviews for a fictional app. Works with no setup.")

    st.divider()
    st.subheader("Analysis")
    typed_key = st.text_input("Gemini API key (optional)", type="password",
                              help="Used only for this session. You can also set GEMINI_API_KEY in Streamlit secrets or the environment.")
    api_key = typed_key.strip() or default_api_key()
    st.caption("Gemini key detected." if api_key else "No key: rule-based fallback analysis will be used.")
    model = st.selectbox("Gemini model", eng.MODEL_OPTIONS)
    max_reviews = st.slider("Max reviews sent to Gemini", 20, 500, 150, 10)
    batch_size = st.slider("Batch size", 10, 40, 20, 5)
    min_support = st.slider("Minimum reviews per theme", 2, 10, 3,
                            help="Themes with fewer supporting reviews are not ranked or used in recommendations.")
    run = st.button("Run analysis", type="primary")

    with st.expander("About"):
        st.markdown(
            "**AI App Review Insight & Pattern Extractor** cleans app reviews, classifies them "
            "(sentiment, category, theme, severity) with Gemini, and ranks recurring problems and requests.\n\n"
            "- Without an API key a simple keyword heuristic is used and clearly labelled.\n"
            "- AI output can be wrong; verify against the raw reviews.\n"
            "- Priority scores are heuristics, not validated metrics.\n"
            "- Review text is sent to Google's Gemini API only when a key is provided."
        )

# --------------------------------------------------------------------------- #
# Run analysis
# --------------------------------------------------------------------------- #
def load_raw() -> tuple[pd.DataFrame, str]:
    if source == "Sample dataset":
        return ds.load_sample(), "Sample dataset (synthetic)"
    if source == "Upload CSV":
        if uploaded is None:
            raise ValueError("Please upload a CSV file first.")
        return ds.read_uploaded_csv(uploaded), f"Uploaded CSV: {uploaded.name}"
    return ds.fetch_play_reviews(app_id.strip(), lang.strip() or "en", country.strip() or "us", count), f"Google Play: {app_id.strip()}"


if run:
    try:
        with st.spinner("Loading reviews..."):
            raw, label = load_raw()
        st.session_state["result"] = run_pipeline(raw, label, api_key, model, batch_size, max_reviews)
    except Exception as exc:  # show a friendly message instead of a stack trace
        st.error(f"Could not complete the analysis: {exc}")
elif "result" not in st.session_state:
    # First load: analyze the bundled sample with the free rule-based fallback (no API usage).
    with st.spinner("Preparing sample data..."):
        st.session_state["result"] = run_pipeline(ds.load_sample(), "Sample dataset (synthetic)", "", model, batch_size, max_reviews)
    st.session_state["first_load"] = True

res = st.session_state["result"]
df: pd.DataFrame = res["df"]
cleaning, report = res["cleaning"], res["report"]

ts = ins.build_theme_summary(df, min_support)
emerging, emerging_note = ins.find_emerging(df, min_support)
recs = ins.build_recommendations(df, ts, emerging, min_support)

# --------------------------------------------------------------------------- #
# Header + status banners
# --------------------------------------------------------------------------- #
st.title("AI App Review Insight & Pattern Extractor")
st.caption(f"Source: {res['label']}")

if report.mode == "rule_based":
    st.warning("AI analysis is unavailable for this run, so a simple keyword-based fallback was used. "
               "Themes and severity are approximate. Add a Gemini API key in the sidebar and click **Run analysis** for AI analysis.")
elif report.mode == "mixed":
    st.warning(f"Gemini analyzed {report.gemini_count} reviews; {report.fallback_count} used the rule-based fallback after errors.")
else:
    st.success(f"All {report.gemini_count} reviews analyzed with Gemini ({report.model}).")
for note in report.notes:
    if report.mode != "rule_based" or "No Gemini API key" not in note:
        st.caption(f"Note: {note}")
if res["label"].startswith("Sample"):
    st.info("You are viewing a synthetic sample dataset written for demonstration; it does not represent a real app.")
if res["cap_note"]:
    st.info(res["cap_note"])

with st.expander("Data cleaning summary"):
    m = st.columns(5)
    m[0].metric("Rows in file", cleaning.input_rows)
    m[1].metric("Empty removed", cleaning.empty_removed)
    m[2].metric("Duplicates removed", cleaning.duplicates_removed)
    m[3].metric("Kept after cleaning", cleaning.final_rows)
    m[4].metric("Analyzed", len(df))
    if cleaning.invalid_dates or cleaning.invalid_ratings:
        st.caption(f"Unparseable dates set to blank: {cleaning.invalid_dates}. Ratings outside 1-5 or non-numeric ignored: {cleaning.invalid_ratings}.")

tab_over, tab_prio, tab_recs, tab_reviews, tab_export = st.tabs(
    ["Overview", "Priorities", "Recommendations", "Reviews", "Export"])

# --------------------------------------------------------------------------- #
# Overview
# --------------------------------------------------------------------------- #
with tab_over:
    n = len(df)
    neg_share = (df["sentiment"] == "Negative").mean()
    k = st.columns(4)
    k[0].metric("Reviews analyzed", f"{n:,}")
    k[1].metric("Negative share", f"{neg_share:.0%}")
    k[2].metric("High-severity reviews", int((df["severity"] == "High").sum()))
    k[3].metric("Themes found", df["theme"].nunique())

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Sentiment distribution")
        sent = df["sentiment"].value_counts().reindex(["Positive", "Neutral", "Negative"]).fillna(0).reset_index()
        sent.columns = ["sentiment", "reviews"]
        fig = px.pie(sent, names="sentiment", values="reviews", hole=0.55, color="sentiment",
                     color_discrete_map=SENT_COLORS)
        fig.update_traces(textinfo="percent+value", sort=False)
        st.plotly_chart(style(fig))
    with c2:
        st.subheader("Category distribution")
        cat = df.groupby(["category", "sentiment"]).size().reset_index(name="reviews")
        order = df["category"].value_counts().index.tolist()
        fig = px.bar(cat, x="category", y="reviews", color="sentiment", color_discrete_map=SENT_COLORS,
                     category_orders={"category": order})
        fig.update_layout(xaxis_title="", yaxis_title="Reviews")
        st.plotly_chart(style(fig))

    st.subheader("Top themes")
    top_themes = ts.sort_values("reviews", ascending=False).head(10)["theme"].tolist()
    tt = df[df["theme"].isin(top_themes)].groupby(["theme", "sentiment"]).size().reset_index(name="reviews")
    fig = px.bar(tt, x="reviews", y="theme", color="sentiment", orientation="h", color_discrete_map=SENT_COLORS,
                 category_orders={"theme": top_themes[::-1]})
    fig.update_layout(xaxis_title="Reviews", yaxis_title="")
    st.plotly_chart(style(fig, 380))

    st.subheader("Review trend by date")
    dated = df[df["date"].notna()]
    if dated.empty:
        st.caption("No valid dates in the data, so no trend is shown.")
    else:
        span = (dated["date"].max() - dated["date"].min()).days
        freq, label = ("W", "week") if span > 60 else ("D", "day")
        trend = (dated.assign(period=dated["date"].dt.to_period(freq).dt.start_time)
                 .groupby(["period", "sentiment"]).size().reset_index(name="reviews"))
        fig = px.bar(trend, x="period", y="reviews", color="sentiment", color_discrete_map=SENT_COLORS)
        fig.update_layout(xaxis_title="", yaxis_title=f"Reviews per {label}")
        st.plotly_chart(style(fig))

# --------------------------------------------------------------------------- #
# Priorities
# --------------------------------------------------------------------------- #
with tab_prio:
    st.subheader("Priority-ranked pain points")
    st.caption(ins.HEURISTIC_NOTE)
    ranked = ins.high_priority_pain_points(ts, 10)
    if ranked.empty:
        st.info(f"No theme has at least {min_support} complaint reviews. Lower the minimum in the sidebar or add more data.")
    else:
        fig = px.bar(ranked.iloc[::-1], x="priority_score", y="theme", orientation="h",
                     color_discrete_sequence=[ACCENT], text="priority_score",
                     hover_data=["category", "pain_reviews", "high_severity"])
        fig.update_layout(xaxis_title="Priority score (heuristic, 0-100)", yaxis_title="", xaxis_range=[0, 100])
        st.plotly_chart(style(fig, 80 + 38 * len(ranked)))
        view = ranked[["priority_rank", "theme", "category", "pain_reviews", "negative_share", "avg_severity",
                       "high_severity", "avg_rating", "priority_score"]].rename(columns={
            "priority_rank": "Rank", "theme": "Theme", "category": "Category", "pain_reviews": "Complaints",
            "negative_share": "Negative share", "avg_severity": "Avg severity (1-3)", "high_severity": "High severity",
            "avg_rating": "Avg rating", "priority_score": "Score"})
        st.dataframe(view, hide_index=True, column_config={
            "Rank": st.column_config.NumberColumn(format="%d"),
            "Negative share": st.column_config.NumberColumn(format="percent")})

    c1, c2, c3 = st.columns(3)
    for col, title, frame in ((c1, "Most reported problems", ins.top_problems(df)),
                              (c2, "Most requested features", ins.top_requests(df)),
                              (c3, "Common positive feedback", ins.top_positives(df))):
        with col:
            st.subheader(title)
            if frame.empty:
                st.caption("Nothing found in this data.")
            else:
                st.dataframe(frame, hide_index=True, column_config={
                    "theme": "Theme",
                    "reviews": st.column_config.ProgressColumn("Reviews", min_value=0,
                                                               max_value=int(frame["reviews"].max()), format="%d")})

    st.subheader("Emerging patterns")
    st.caption(emerging_note)
    if not emerging.empty:
        st.dataframe(emerging.rename(columns={
            "theme": "Theme", "recent_reviews": "Recent negative reviews", "earlier_reviews": "Earlier negative reviews",
            "recent_share": "Recent share", "earlier_share": "Earlier share"}), hide_index=True,
            column_config={"Recent share": st.column_config.NumberColumn(format="percent"),
                           "Earlier share": st.column_config.NumberColumn(format="percent")})
    elif "Recent period" in emerging_note:
        st.caption("No theme met the threshold for an emerging pattern.")

# --------------------------------------------------------------------------- #
# Recommendations
# --------------------------------------------------------------------------- #
with tab_recs:
    st.subheader("Actionable recommendations")
    st.caption("Each recommendation is generated from the analyzed reviews and cites its supporting review count. "
               "Suggested actions are standard playbook steps per category, not predictions of impact.")
    if not recs:
        st.info(f"Not enough supporting data: no theme reaches {min_support} reviews. No recommendations were generated.")
    for i, r in enumerate(recs, 1):
        with st.container(border=True):
            st.markdown(f"**{i}. {r['theme'].title()}**  ·  {r['type']}  ·  {r['category']}")
            st.markdown(f"**Observed:** {r['observed']}")
            st.markdown(f"**Supporting reviews:** {r['supporting_reviews']}")
            st.markdown(f"**Why it matters:** {r['why']}")
            st.markdown(f"**Suggested action:** {r['action']}")
            if r["evidence"]:
                with st.expander("Example reviews"):
                    for e in r["evidence"]:
                        st.markdown(f"> {e}")

# --------------------------------------------------------------------------- #
# Reviews table
# --------------------------------------------------------------------------- #
with tab_reviews:
    st.subheader("Review explorer")
    f1, f2, f3, f4 = st.columns(4)
    sel_sent = f1.multiselect("Sentiment", eng.SENTIMENTS)
    sel_cat = f2.multiselect("Category", eng.CATEGORIES)
    sel_sev = f3.multiselect("Severity", eng.SEVERITIES)
    sel_theme = f4.multiselect("Theme", sorted(df["theme"].unique()))
    query = st.text_input("Search review text, theme or reason")
    view = df
    if sel_sent:
        view = view[view["sentiment"].isin(sel_sent)]
    if sel_cat:
        view = view[view["category"].isin(sel_cat)]
    if sel_sev:
        view = view[view["severity"].isin(sel_sev)]
    if sel_theme:
        view = view[view["theme"].isin(sel_theme)]
    if query:
        q = query.strip()
        hit = (view["review"].str.contains(q, case=False, regex=False, na=False)
               | view["theme"].str.contains(q, case=False, regex=False, na=False)
               | view["reason"].str.contains(q, case=False, regex=False, na=False))
        view = view[hit]
    st.caption(f"Showing {len(view):,} of {len(df):,} reviews")
    st.dataframe(
        view[["date", "app_name", "rating", "sentiment", "category", "theme", "severity", "review", "reason", "analysis_source"]],
        hide_index=True,
        column_config={"date": st.column_config.DateColumn("Date"), "review": st.column_config.TextColumn("Review", width="large"),
                       "reason": st.column_config.TextColumn("Reason", width="medium")},
    )

# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
with tab_export:
    st.subheader("Download results")
    export_df = df.copy()
    export_df["date"] = export_df["date"].dt.strftime("%Y-%m-%d")
    theme_export = ts.copy()
    report_txt = ins.build_report(df, cleaning, report, ts, emerging, recs, res["label"], min_support)
    d1, d2, d3 = st.columns(3)
    d1.download_button("Analyzed reviews (CSV)", export_df.to_csv(index=False).encode("utf-8"),
                       "analyzed_reviews.csv", "text/csv")
    d2.download_button("Theme summary (CSV)", theme_export.to_csv(index=False).encode("utf-8"),
                       "theme_summary.csv", "text/csv")
    d3.download_button("Insights report (TXT)", report_txt.encode("utf-8"),
                       "insights_report.txt", "text/plain")
    with st.expander("Preview report"):
        st.text(report_txt)
