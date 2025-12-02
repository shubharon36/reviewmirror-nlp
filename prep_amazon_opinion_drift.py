#!/usr/bin/env python3
# Prep Amazon Reviews (2018) for Opinion-Drift Analysis.
# Usage:
#   python prep_amazon_opinion_drift.py --input path/to/Electronics_5.json.gz --outdir data/ --min_reviews 5 --alpha 0.7
# It produces:
#   - reviews.parquet : cleaned row-level reviews
#   - user_trajectories.parquet : per-user time-ordered sequences with sentiment & drift stats

import argparse, os, gzip, json, math
from datetime import datetime, timezone
from typing import Iterator, Dict, Any, List, Tuple

import pandas as pd
import numpy as np

# --- Sentiment (baseline) ---
def safe_vader_compound(text: str) -> float:
    '''
    Returns VADER compound in [-1,1] if available; otherwise a simple heuristic fallback.
    '''
    try:
        # requires: pip install vaderSentiment
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        if not hasattr(safe_vader_compound, "_an"):
            safe_vader_compound._an = SentimentIntensityAnalyzer()
        return float(safe_vader_compound._an.polarity_scores(text or "")["compound"])
    except Exception:
        # fallback: naive lexicon
        text = (text or "").lower()
        pos = sum(text.count(w) for w in ["good", "great", "excellent", "amazing", "love", "perfect", "happy"])
        neg = sum(text.count(w) for w in ["bad", "terrible", "awful", "hate", "poor", "broken", "angry"])
        if pos + neg == 0:
            return 0.0
        return (pos - neg) / (pos + neg)

def parse_gz_json(path: str) -> Iterator[Dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def stars_to_unit(stars: float) -> float:
    '''Map 1..5 -> [-1,1].'''
    try:
        s = float(stars)
    except Exception:
        return np.nan
    return (s - 3.0) / 2.0  # 1->-1, 3->0, 5->+1

def month_floor(ts: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(year=ts.year, month=ts.month, day=1, tz="UTC")

def build_reviews_df(path: str) -> pd.DataFrame:
    rows = []
    for rec in parse_gz_json(path):
        rows.append({
            "user_id": rec.get("reviewerID"),
            "item_id": rec.get("asin"),
            "text": rec.get("reviewText"),
            "stars": rec.get("overall"),
            "ts": pd.to_datetime(rec.get("unixReviewTime"), unit="s", utc=True) if rec.get("unixReviewTime") else pd.NaT,
            "summary": rec.get("summary"),
            "helpful_votes": rec.get("vote")
        })
    df = pd.DataFrame(rows)
    # Basic cleaning
    df = df.dropna(subset=["user_id", "item_id", "ts"]).reset_index(drop=True)
    # English-only quick filter (heuristic): keep ASCII-dominant texts; adjust later if needed
    def is_ascii_major(text):
        if not isinstance(text, str): return True
        ascii_chars = sum(1 for ch in text if ord(ch) < 128)
        return ascii_chars >= 0.9 * len(text) if len(text) > 0 else True
    df = df[df["text"].map(is_ascii_major)]
    return df

def compute_features(df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    df = df.copy()
    df["sent_text"] = df["text"].map(safe_vader_compound).astype(float)
    df["sent_stars"] = df["stars"].map(stars_to_unit).astype(float)
    df["sent_hybrid"] = alpha * df["sent_text"] + (1.0 - alpha) * df["sent_stars"]
    # style proxies
    def style_feats(txt: str) -> Tuple[float, float, float]:
        if not isinstance(txt, str) or not txt:
            return (0.0, 0.0, 0.0)
        exclam = txt.count("!") / max(1, len(txt))
        first_person = sum(txt.lower().count(w) for w in [" i ", " i'm ", " i’ve ", " my ", " me "]) / max(1, len(txt))
        caps = sum(1 for ch in txt if ch.isupper()) / max(1, len(txt))
        return (exclam, first_person, caps)
    feats = df["text"].map(style_feats).tolist()
    df[["style_exclam_rate", "style_firstperson_rate", "style_caps_rate"]] = pd.DataFrame(feats, index=df.index)
    # time bins
    df["month"] = df["ts"].map(month_floor)
    return df

def build_user_trajectories(df: pd.DataFrame, min_reviews: int = 5) -> pd.DataFrame:
    df = df.sort_values(["user_id", "ts"])
    # keep users with >= min_reviews
    vc = df["user_id"].value_counts()
    keep_users = set(vc[vc >= min_reviews].index)
    df = df[df["user_id"].isin(keep_users)].copy()
    # z-score within user for text sentiment to normalize style
    df["text_z_user"] = df.groupby("user_id")["sent_text"].transform(lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-8))
    # aggregate monthly
    agg = (df.groupby(["user_id", "month"])
             .agg(sent_text_mean=("sent_text", "mean"),
                  sent_hybrid_mean=("sent_hybrid", "mean"),
                  stars_mean=("stars", "mean"),
                  n_reviews=("text", "count"))
             .reset_index())
    # drift stats per user (slope via simple OLS on month index)
    def slope_for_user(g: pd.DataFrame) -> Dict[str, float]:
        g = g.sort_values("month")
        x = np.arange(len(g), dtype=float)
        y = g["sent_hybrid_mean"].values.astype(float)
        if len(x) < 2 or np.isnan(y).all():
            return {"drift_slope": np.nan, "drift_delta": np.nan}
        # simple least squares slope
        x = x - x.mean()
        denom = (x**2).sum() + 1e-8
        slope = float((x * (y - y.mean())).sum() / denom)
        delta = float(y[-1] - y[0])
        return {"drift_slope": slope, "drift_delta": delta}
    drift = (agg.groupby("user_id")
                .apply(slope_for_user)
                .apply(pd.Series)
                .reset_index())
    # assemble sequences as lists (optional)
    seqs = (agg.groupby("user_id")
              .agg(months=("month", list),
                   sent_hybrid_seq=("sent_hybrid_mean", list),
                   stars_seq=("stars_mean", list),
                   counts_seq=("n_reviews", list))
              .reset_index())
    traj = pd.merge(drift, seqs, on="user_id", how="left")
    return df, agg, traj

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to gz JSON (e.g., Electronics_5.json.gz)")
    ap.add_argument("--outdir", default="data", help="Output folder")
    ap.add_argument("--min_reviews", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.7, help="Weight on text vs stars in hybrid sentiment")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    print(f"[load] {args.input}")
    df = build_reviews_df(args.input)
    print(f"[rows] {len(df):,}")
    df = compute_features(df, alpha=args.alpha)
    print("[features] computed text sentiment, style proxies, month bins")
    df_clean, monthly, traj = build_user_trajectories(df, min_reviews=args.min_reviews)
    print(f"[users] {traj.shape[0]:,} users with >= {args.min_reviews} reviews")
    # save
    df_clean.to_parquet(os.path.join(args.outdir, "reviews.parquet"))
    monthly.to_parquet(os.path.join(args.outdir, "reviews_monthly.parquet"))
    traj.to_parquet(os.path.join(args.outdir, "user_trajectories.parquet"))
    # csv previews
    traj.head(1000).to_csv(os.path.join(args.outdir, "user_trajectories_preview.csv"), index=False)
    print("[done] wrote parquet and preview CSV to", args.outdir)

if __name__ == "__main__":
    main()