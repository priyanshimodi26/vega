import re
import csv
import json
import torch
import sqlite3
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
from sentence_transformers import SentenceTransformer, util

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
DB_PATH  = BASE_DIR / "data" / "vega.db"
LM_PATH  = BASE_DIR / "data" / "loughran_mcdonald.csv"

# ── Model config ─────────────────────────────────────────────────
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 64   # sentence embeddings are cheap; large batches are fine

# ── MiniLM risk anchors — 8 categories ───────────────────────────
# Each anchor phrase represents the semantic centre of a risk category.
# For each sentence in the transcript, we compute cosine similarity vs
# each anchor and store the max similarity per category.
RISK_ANCHORS = {
    "liquidity_risk":     "cash flow pressure, funding constraints, liquidity concerns",
    "demand_softness":    "demand weakness, volume decline, slower growth",
    "margin_compression": "margin pressure, cost increase, profitability headwinds",
    "regulatory_risk":    "regulatory uncertainty, compliance burden, policy risk",
    "macro_headwinds":    "macroeconomic uncertainty, inflation pressure, rate environment",
    "competitive_threat": "competitive intensity, market share loss, pricing pressure",
    "management_evasion": "we will update, cannot comment, too early to say",
    "overconfidence":     "extremely confident, absolutely certain, no doubt whatsoever",
}

# ── Loughran-McDonald column names ────────────────────────────────
# CSV has different column names depending on the version downloaded.
# We check both old-style (Positive, Negative, …) and new-style.
LM_COLUMN_ALIASES = {
    "uncertainty":      ["Uncertainty",      "UNCERTAINTY"],
    "litigious":        ["Litigious",        "LITIGIOUS"],
    "strong_modal":     ["Strong_Modal",     "StrongModal",  "STRONG_MODAL"],
    "weak_modal":       ["Weak_Modal",       "WeakModal",    "WEAK_MODAL"],
    "extreme_positive": ["Positive", "POSITIVE"],   # L-M "Positive" ≈ overconfident/extreme
}

# Minimum sentence length to consider for MiniLM (chars)
MIN_SENTENCE_LENGTH = 15

#Database initialiser
def init_db():
    """Create risk_scores table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS risk_scores (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            transcript_id       INTEGER REFERENCES transcripts(id),
            ticker              TEXT NOT NULL,
            earnings_date       TEXT NOT NULL,
            liquidity_risk      REAL,
            demand_softness     REAL,
            margin_compression  REAL,
            regulatory_risk     REAL,
            macro_headwinds     REAL,
            competitive_threat  REAL,
            management_evasion  REAL,
            overconfidence      REAL,
            lm_uncertainty      REAL,
            lm_litigious        REAL,
            lm_weak_modal       REAL,
            lm_strong_modal     REAL,
            lm_extreme_positive REAL,
            scored_at           TEXT NOT NULL,
            UNIQUE(transcript_id)
        )
    """)
    conn.commit()
    conn.close()

#Sentence splitter
def split_sentences(text: str) -> list[str]:
    """
    Split transcript text into sentences for embedding.
    Same logic used across all model files for consistency.
    """
    if not text or len(text.strip()) < 50:
        return []

    raw = re.split(r'(?<=[.!?])\s+', text)
    sentences = []
    for s in raw:
        s = s.strip()
        if len(s) < MIN_SENTENCE_LENGTH:
            continue
        # Skip bare speaker labels like "Operator:" or "John Smith:"
        if re.match(r'^[A-Z][a-zA-Z\s]{0,30}:\s*$', s):
            continue
        sentences.append(s)
    return sentences

#Loughran-McDonald loader
def load_lm_wordlists(lm_path: Path) -> dict[str, set[str]]:
    """
    Parse the Loughran-McDonald master dictionary CSV.
    Returns a dict mapping category name → set of lowercase words.

    Handles column name variations across L-M CSV versions.
    Words are flagged by a non-zero value in their respective column
    (L-M uses year of inclusion, not a 1/0 flag).
    """
    if not lm_path.exists():
        print(f"[WARN] L-M dictionary not found at {lm_path}")
        print("  Download from: https://sraf.nd.edu/loughranmcdonald-master-dictionary/")
        print("  Returning empty word lists — lm_* features will be 0.0")
        return {k: set() for k in LM_COLUMN_ALIASES}

    wordlists: dict[str, set[str]] = {k: set() for k in LM_COLUMN_ALIASES}

    with open(lm_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []

        # Resolve which actual column name to use for each category
        col_map: dict[str, str | None] = {}
        for category, aliases in LM_COLUMN_ALIASES.items():
            col_map[category] = next(
                (alias for alias in aliases if alias in headers), None
            )

        missing = [cat for cat, col in col_map.items() if col is None]
        if missing:
            print(f"  [WARN] L-M columns not found for: {missing}")
            print(f"  Available columns: {headers[:20]}")

        for row in reader:
            word = row.get("Word", row.get("WORD", "")).strip().lower()
            if not word:
                continue
            for category, col in col_map.items():
                if col and row.get(col, "0").strip() not in ("", "0"):
                    wordlists[category].add(word)

    counts = {k: len(v) for k, v in wordlists.items()}
    print(f"  [L-M] Word list sizes: {counts}")
    return wordlists


def compute_lm_scores(text: str, wordlists: dict[str, set[str]]) -> dict[str, float]:
    """
    Count Loughran-McDonald word occurrences in text, normalised by total words.
    Returns a dict of {category: normalised_count}.
    """
    if not text or not text.strip():
        return {f"lm_{k}": 0.0 for k in LM_COLUMN_ALIASES}

    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    total = len(words)
    if total == 0:
        return {f"lm_{k}": 0.0 for k in LM_COLUMN_ALIASES}

    scores = {}
    for category, wordlist in wordlists.items():
        count = sum(1 for w in words if w in wordlist)
        scores[f"lm_{category}"] = round(count / total, 6)

    return scores

#MiniLM risk scorer
def compute_minilm_scores(
    sentences: list[str],
    model: SentenceTransformer,
    anchor_embeddings: dict[str, "torch.Tensor"],
) -> dict[str, float]:
    """
    For each risk category, compute the MAX cosine similarity between
    any sentence and the category's anchor embedding.

    Returns {category_name: max_cosine_score} for all 8 risk categories.
    Scores are in [0, 1] — higher = stronger risk signal.
    """
    if not sentences:
        return {k: 0.0 for k in RISK_ANCHORS}

    # Embed all sentences in one batched call
    sentence_embeddings = model.encode(
        sentences,
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        convert_to_tensor=True,
        normalize_embeddings=True,   # pre-normalise → dot product = cosine sim
    )

    scores = {}
    for category, anchor_emb in anchor_embeddings.items():
        # cosine_similarity returns a [N] tensor of scores for each sentence
        sims = util.cos_sim(anchor_emb, sentence_embeddings)[0]  # shape [N]
        max_sim = float(sims.max().item())
        scores[category] = round(max(max_sim, 0.0), 4)  # clip to [0, 1]

    return scores

#Main processing loop
def flag_all_transcripts() -> dict:
    """
    Run the hybrid risk flagger on all unscored transcripts in the DB.
    Loads models once, processes all transcripts, saves to risk_scores.

    Returns {"scored": int, "skipped": int, "failed": int}
    """
    init_db()

    # ── Load L-M word lists ───────────────────────────────────────
    print("[RISK] Loading Loughran-McDonald word lists...")
    lm_wordlists = load_lm_wordlists(LM_PATH)

    # ── Load MiniLM model ─────────────────────────────────────────
    print(f"[RISK] Loading {MODEL_NAME}...")
    minilm = SentenceTransformer(MODEL_NAME)
    print(f"[RISK] Model loaded.")

    # Pre-embed anchor phrases once — expensive to repeat per transcript
    print("[RISK] Embedding anchor phrases...")
    anchor_embeddings = {}
    for category, phrase in RISK_ANCHORS.items():
        anchor_embeddings[category] = minilm.encode(
            phrase,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
    print(f"[RISK] {len(anchor_embeddings)} anchors embedded.")

    # ── Fetch unscored transcripts ────────────────────────────────
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT t.id, t.ticker, t.earnings_date,
               t.prepared_remarks, t.qa_section, t.full_text
        FROM transcripts t
        LEFT JOIN risk_scores r ON t.id = r.transcript_id
        WHERE r.transcript_id IS NULL
        ORDER BY t.ticker, t.earnings_date
    """).fetchall()
    conn.close()

    if not rows:
        print("[RISK] All transcripts already scored.")
        return {"scored": 0, "skipped": 0, "failed": 0}

    print(f"[RISK] {len(rows)} transcripts to score...")
    stats = {"scored": 0, "skipped": 0, "failed": 0}

    for i, (transcript_id, ticker, earnings_date,
            prepared_remarks, qa_section, full_text) in enumerate(rows):

        print(f"\n[{i+1}/{len(rows)}] {ticker} {earnings_date}")

        try:
            # Combine sections for MiniLM (we score on the full transcript)
            # We also run L-M on full_text for maximum word coverage
            combined_text = (prepared_remarks or "") + "\n" + (qa_section or "")
            sentences = split_sentences(combined_text)
            print(f"  Sentences for MiniLM: {len(sentences)}")

            # ── MiniLM scores ─────────────────────────────────────
            if sentences:
                minilm_scores = compute_minilm_scores(
                    sentences, minilm, anchor_embeddings
                )
            else:
                minilm_scores = {k: 0.0 for k in RISK_ANCHORS}

            # ── L-M scores ────────────────────────────────────────
            lm_scores = compute_lm_scores(full_text or combined_text, lm_wordlists)

            # ── Save to DB ────────────────────────────────────────
            conn = sqlite3.connect(DB_PATH)
            conn.execute("""
                INSERT OR IGNORE INTO risk_scores (
                    transcript_id, ticker, earnings_date,
                    liquidity_risk, demand_softness, margin_compression,
                    regulatory_risk, macro_headwinds, competitive_threat,
                    management_evasion, overconfidence,
                    lm_uncertainty, lm_litigious, lm_weak_modal,
                    lm_strong_modal, lm_extreme_positive,
                    scored_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                transcript_id, ticker, earnings_date,
                minilm_scores["liquidity_risk"],
                minilm_scores["demand_softness"],
                minilm_scores["margin_compression"],
                minilm_scores["regulatory_risk"],
                minilm_scores["macro_headwinds"],
                minilm_scores["competitive_threat"],
                minilm_scores["management_evasion"],
                minilm_scores["overconfidence"],
                lm_scores["lm_uncertainty"],
                lm_scores["lm_litigious"],
                lm_scores["lm_weak_modal"],
                lm_scores["lm_strong_modal"],
                lm_scores["lm_extreme_positive"],
                datetime.utcnow().isoformat()
            ))
            conn.commit()
            conn.close()

            # ── Print summary ─────────────────────────────────────
            top_minilm = max(minilm_scores.items(), key=lambda x: x[1])
            print(f"  MiniLM top risk: {top_minilm[0]} = {top_minilm[1]:.3f}")
            print(f"  L-M uncertainty={lm_scores['lm_uncertainty']:.4f}  "
                  f"weak_modal={lm_scores['lm_weak_modal']:.4f}  "
                  f"strong_modal={lm_scores['lm_strong_modal']:.4f}")

            stats["scored"] += 1

        except Exception as e:
            print(f"  [ERROR] {ticker} {earnings_date}: {e}")
            stats["failed"] += 1
            continue

    print(f"\n[DONE] Scored: {stats['scored']} | "
          f"Skipped: {stats['skipped']} | Failed: {stats['failed']}")
    return stats

#Entry point
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Usage: python models/risk_flagger.py TCS
        ticker_filter = sys.argv[1].upper()
        print(f"[FILTER] Scoring only: {ticker_filter}")

        init_db()

        print("[RISK] Loading Loughran-McDonald word lists...")
        lm_wordlists = load_lm_wordlists(LM_PATH)

        print(f"[RISK] Loading {MODEL_NAME}...")
        minilm = SentenceTransformer(MODEL_NAME)

        print("[RISK] Embedding anchor phrases...")
        anchor_embeddings = {
            cat: minilm.encode(phrase, convert_to_tensor=True, normalize_embeddings=True)
            for cat, phrase in RISK_ANCHORS.items()
        }

        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute("""
            SELECT t.id, t.ticker, t.earnings_date,
                   t.prepared_remarks, t.qa_section, t.full_text
            FROM transcripts t
            LEFT JOIN risk_scores r ON t.id = r.transcript_id
            WHERE r.transcript_id IS NULL AND t.ticker = ?
            ORDER BY t.earnings_date
        """, (ticker_filter,)).fetchall()
        conn.close()

        if not rows:
            print(f"No unscored transcripts for {ticker_filter}")
            sys.exit(0)

        for i, (transcript_id, ticker, earnings_date,
                prepared_remarks, qa_section, full_text) in enumerate(rows):
            print(f"\n[{i+1}/{len(rows)}] {ticker} {earnings_date}")

            combined = (prepared_remarks or "") + "\n" + (qa_section or "")
            sentences = split_sentences(combined)

            minilm_scores = (
                compute_minilm_scores(sentences, minilm, anchor_embeddings)
                if sentences else {k: 0.0 for k in RISK_ANCHORS}
            )
            lm_scores = compute_lm_scores(full_text or combined, lm_wordlists)

            conn = sqlite3.connect(DB_PATH)
            conn.execute("""
                INSERT OR IGNORE INTO risk_scores (
                    transcript_id, ticker, earnings_date,
                    liquidity_risk, demand_softness, margin_compression,
                    regulatory_risk, macro_headwinds, competitive_threat,
                    management_evasion, overconfidence,
                    lm_uncertainty, lm_litigious, lm_weak_modal,
                    lm_strong_modal, lm_extreme_positive,
                    scored_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                transcript_id, ticker, earnings_date,
                minilm_scores["liquidity_risk"],
                minilm_scores["demand_softness"],
                minilm_scores["margin_compression"],
                minilm_scores["regulatory_risk"],
                minilm_scores["macro_headwinds"],
                minilm_scores["competitive_threat"],
                minilm_scores["management_evasion"],
                minilm_scores["overconfidence"],
                lm_scores["lm_uncertainty"],
                lm_scores["lm_litigious"],
                lm_scores["lm_weak_modal"],
                lm_scores["lm_strong_modal"],
                lm_scores["lm_extreme_positive"],
                datetime.utcnow().isoformat()
            ))
            conn.commit()
            conn.close()

            top = max(minilm_scores.items(), key=lambda x: x[1])
            print(f"  top risk: {top[0]} = {top[1]:.3f}")
            print(f"  lm_uncertainty={lm_scores['lm_uncertainty']:.4f}  "
                  f"lm_weak_modal={lm_scores['lm_weak_modal']:.4f}")

    else:
        flag_all_transcripts()