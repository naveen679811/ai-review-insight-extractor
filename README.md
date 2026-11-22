# AI App Review Insight & Pattern Extractor

A lightweight Streamlit tool that turns raw mobile app reviews into prioritized product insights: it cleans the data, classifies each review with the Gemini API (sentiment, category, theme, severity), finds recurring patterns, and generates data-backed recommendations.

**Stack:** Python · Streamlit · Pandas · Plotly · Gemini API · CSV

## Features
- **Input:** upload a CSV, use the built-in synthetic sample, or (optionally) fetch Google Play reviews.
- **Cleaning:** removes empty and duplicate reviews, normalizes text/dates/ratings, reports what was removed.
- **AI analysis:** batched Gemini calls extract sentiment, category (Bug, Performance, Content Request, UX, Pricing, Other), theme, severity and a short reason.
- **Fallback:** with no API key (or if Gemini fails) a transparent keyword heuristic is used and clearly labelled in the UI and in the exports (`analysis_source` column).
- **Insights:** most reported problems, most requested features, positive feedback, emerging patterns (recent vs. earlier negative reviews), priority-ranked pain points.
- **Recommendations:** only for themes with enough supporting reviews; each shows observed problem, review count, why it matters, suggested action and example reviews.
- **Dashboard:** KPIs, sentiment/category/theme charts, date trend, searchable and filterable review table.
- **Export:** analyzed reviews (CSV), theme summary (CSV), insights report (TXT).

> The priority score (0-100) is a heuristic: 40% frequency, 30% negative share, 30% average severity. It is not a validated metric. AI output can contain errors; verify against raw reviews.

## Project structure
```
app.py              Streamlit dashboard
review_engine.py    Cleaning, Gemini integration, rule-based fallback
insights.py         Themes, priority heuristic, recommendations, report
data_sources.py     Sample / CSV / Google Play loaders
data/sample_reviews.csv   Synthetic demo data (fictional app)
tests/test_core.py  Unit tests (Gemini calls are mocked)
.streamlit/         config.toml and secrets.toml.example
```

## Run locally
```bash
git clone https://github.com/<your-username>/ai-review-insight-extractor.git
cd ai-review-insight-extractor
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
The app opens at http://localhost:8501 and works immediately with the sample data (rule-based mode).

## Gemini API setup (optional)
1. Create a free key at https://aistudio.google.com/apikey.
2. Provide it in **one** of these ways (never commit it):
   - Paste it into the sidebar field (kept only for the session), or
   - `cp .streamlit/secrets.toml.example .streamlit/secrets.toml` and edit it (git-ignored), or
   - `export GEMINI_API_KEY="your-key"` (Windows PowerShell: `$env:GEMINI_API_KEY="your-key"`).
3. Click **Run analysis**. The banner at the top shows whether Gemini or the fallback was used.

Free-tier rate limits apply. The sidebar caps how many reviews are sent to Gemini (default 150, most recent first) and lets you set the batch size. Review text is sent to Google's API only when a key is provided.

## CSV format
Required: a review text column (`review`, `content`, `text`, `comment`, ...). Optional: `rating` (1-5), `date`, `app_name`. Common alternative column names are detected automatically.

## Deploy on Streamlit Community Cloud
1. Push the project to a public GitHub repository (see below).
2. Go to https://share.streamlit.io, sign in with GitHub, click **Create app**.
3. Select the repository, branch `main`, main file `app.py`.
4. Open **Advanced settings → Secrets** and add: `GEMINI_API_KEY = "your-key"` (optional; without it the app runs in fallback mode).
5. Click **Deploy**. Add the live URL to your resume and README.

## GitHub repository setup
```bash
git init
git add .
git commit -m "Initial commit: AI App Review Insight & Pattern Extractor"
git branch -M main
git remote add origin https://github.com/<your-username>/ai-review-insight-extractor.git
git push -u origin main
```
Check `git status` first to confirm `.streamlit/secrets.toml` is not listed (it is in `.gitignore`).

## How to test
```bash
pip install -r requirements-dev.txt
python -m pytest -q        # 9 tests: cleaning, fallback, mocked Gemini, insights
```
Manual checks:
1. Launch the app: the sample loads in rule-based mode with a warning banner. "Data cleaning summary" should show 3 empty and 4 duplicate rows removed.
2. Open every tab; try the filters and search in **Reviews**; download all three exports.
3. Upload your own CSV (or one with columns renamed, e.g. `content`, `score`, `at`) and click **Run analysis**.
4. Upload a CSV without a review column to see the error message.
5. Add a Gemini key and click **Run analysis**: the banner should say reviews were analyzed with Gemini. Try an invalid key to confirm the app falls back and reports the error.
6. Optional: choose **Google Play**, enter a package ID such as `com.duolingo`, and run.

## Limitations
- The sample dataset is synthetic; the fallback analysis is keyword-based and ignores negation and sarcasm.
- Theme names from Gemini can vary between batches; the app passes earlier themes to the model to improve consistency but does not guarantee it.
- Google Play fetching depends on an unofficial library and may break if Google changes its pages.

## Resume description
**AI App Review Insight & Pattern Extractor** | Python, Streamlit, Pandas, Plotly, Gemini API
- Built and deployed a Streamlit tool that cleans app-review data (deduplication, normalization) and uses the Gemini API in batches to classify sentiment, category, theme and severity, with a labelled rule-based fallback when AI is unavailable.
- Designed a transparent priority heuristic (frequency, negative share, severity) plus recent-vs-earlier trend detection to surface recurring pain points, feature requests and emerging issues for product teams.
- Delivered an interactive Plotly dashboard with filters, search, data-backed recommendations and CSV/TXT exports, with unit tests, secret-safe API handling and one-click Streamlit Cloud deployment.
