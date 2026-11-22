"""Loading review data: bundled sample, uploaded CSV, optional Google Play fetch."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

SAMPLE_PATH = Path(__file__).parent / "data" / "sample_reviews.csv"


def load_sample() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_PATH)


def read_uploaded_csv(file) -> pd.DataFrame:
    """Read an uploaded CSV, trying UTF-8 first and then Latin-1."""
    try:
        df = pd.read_csv(file)
    except UnicodeDecodeError:
        file.seek(0)
        df = pd.read_csv(file, encoding="latin-1")
    except pd.errors.EmptyDataError:
        raise ValueError("The uploaded file is empty.")
    except pd.errors.ParserError as exc:
        raise ValueError(f"Could not parse the CSV file: {exc}")
    if df.empty:
        raise ValueError("The uploaded CSV has no rows.")
    return df


def fetch_play_reviews(app_id: str, lang: str = "en", country: str = "us", count: int = 200) -> pd.DataFrame:
    """Fetch the newest Google Play reviews. Optional: the rest of the app works without it."""
    if not app_id:
        raise ValueError("Enter an app package ID, e.g. com.duolingo.")
    try:
        from google_play_scraper import Sort, reviews
    except ImportError as exc:
        raise RuntimeError("google-play-scraper is not installed (pip install google-play-scraper).") from exc
    try:
        result, _ = reviews(app_id, lang=lang, country=country, sort=Sort.NEWEST, count=count)
    except Exception as exc:  # library raises assorted network/parsing errors
        raise RuntimeError(
            f"Could not fetch reviews for '{app_id}' ({type(exc).__name__}). "
            "Check the package ID and your connection, or use the sample/CSV option instead."
        ) from exc
    if not result:
        raise RuntimeError(f"No reviews returned for '{app_id}'.")
    df = pd.DataFrame(result)[["content", "score", "at"]]
    df.columns = ["review", "rating", "date"]
    df["app_name"] = app_id
    return df
