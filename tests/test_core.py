import io
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import data_sources as ds
import insights as ins
import review_engine as eng


def test_cleaning_removes_empty_and_duplicates():
    raw = pd.DataFrame({
        "Review Text": ["Great app", "great   app ", "", None, "Crashes a lot", "N/A"],
        "Stars": [5, 5, 3, 3, "bad", 9],
        "Date": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "not a date", "2026-01-05"],
    })
    df, st = eng.clean_reviews(raw)
    assert list(df["review"]) == ["Great app", "Crashes a lot"]
    assert (st.input_rows, st.empty_removed, st.duplicates_removed, st.final_rows) == (6, 3, 1, 2)
    assert st.invalid_dates == 1 and st.invalid_ratings == 1
    assert df["app_name"].iloc[0] == "Unknown app"


def test_missing_review_column_raises():
    with pytest.raises(ValueError, match="No review text column"):
        eng.clean_reviews(pd.DataFrame({"foo": [1]}))


def test_fallback_analysis():
    r = eng.fallback_analyze("App crashes every time I open it, unusable", 1)
    assert (r["sentiment"], r["category"], r["severity"]) == ("Negative", "Bug", "High")
    r = eng.fallback_analyze("Please add dark mode", 4)
    assert r["category"] == "Content Request" and r["is_feature_request"]
    r = eng.fallback_analyze("Love it, very easy to use", 5)
    assert r["sentiment"] == "Positive" and r["severity"] == "Low"


def _frame(n=4):
    raw = pd.DataFrame({"review": [f"review {i}" for i in range(n)], "rating": [1] * n})
    return eng.clean_reviews(raw)[0]


def test_no_api_key_uses_fallback_and_says_so():
    out, rep = eng.analyze_reviews(_frame(), api_key=None)
    assert rep.mode == "rule_based" and set(out["analysis_source"]) == {eng.SOURCE_FALLBACK}
    assert any("No Gemini API key" in n for n in rep.notes)


def test_gemini_success_and_partial_fallback():
    df = _frame(3)
    reply = '[{"id":1,"sentiment":"negative","category":"Bug","theme":"Login Failure!","severity":"high","reason":"x","is_feature_request":false},' \
            '{"id":2,"sentiment":"BAD","category":"Bug","theme":"t","severity":"High"}]'
    with patch.object(eng, "_call_gemini", return_value=reply):
        out, rep = eng.analyze_reviews(df, api_key="k")
    assert rep.mode == "mixed" and rep.gemini_count == 1 and rep.fallback_count == 2
    first = out.iloc[0]
    assert first["theme"] == "login failure" and first["sentiment"] == "Negative" and first["analysis_source"] == "Gemini"


def test_gemini_fatal_error_falls_back_without_leaking_key():
    class Resp:
        status_code = 403
        text = "bad key SECRET123"
        def json(self):
            return {"error": {"message": "API key SECRET123 not valid"}}
    with patch.object(eng.requests, "post", return_value=Resp()):
        out, rep = eng.analyze_reviews(_frame(25), api_key="SECRET123", batch_size=10)
    assert rep.mode == "rule_based" and len(out) == 25
    assert "SECRET123" not in " ".join(rep.notes)
    assert sum("Batch" in n for n in rep.notes) == 1  # stopped after the fatal error


def test_retry_on_rate_limit():
    class Rate:
        status_code = 429
        text = ""
        def json(self): return {"error": {"message": "quota"}}
    with patch.object(eng.requests, "post", return_value=Rate()) as p:
        with pytest.raises(eng.GeminiError):
            eng._call_gemini("k", "m", "p", sleep=lambda s: None)
    assert p.call_count == 3


def test_sample_pipeline_and_insights():
    df, st = eng.clean_reviews(ds.load_sample())
    assert st.empty_removed == 3 and st.duplicates_removed == 4
    out, _ = eng.analyze_reviews(df)
    ts = ins.build_theme_summary(out, 3)
    assert ts["priority_score"].dropna().between(0, 100).all()
    emerging, _ = ins.find_emerging(out, 3)
    recs = ins.build_recommendations(out, ts, emerging, 3)
    assert recs and all(r["supporting_reviews"] >= 3 for r in recs)
    # no recommendations when support threshold is unreachable
    ts_hi = ins.build_theme_summary(out, 1000)
    assert ins.build_recommendations(out, ts_hi, emerging, 1000) == []
    assert "RECOMMENDATIONS" in ins.build_report(out, st, eng.AnalysisReport(), ts, emerging, recs, "test", 3)


def test_upload_reader_latin1():
    buf = io.BytesIO("review,rating\ncafé está bien,5\n".encode("latin-1"))
    assert len(ds.read_uploaded_csv(buf)) == 1
