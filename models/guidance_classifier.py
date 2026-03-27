import re
import json
import sqlite3
import warnings
from pathlib import Path
from datetime import datetime

import torch
from transformers import pipeline

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
DB_PATH  = BASE_DIR / "data" / "vega.db"

# ── Model config ─────────────────────────────────────────────────
MODEL_NAME = "yiyanghkust/finbert-fls"
DEVICE     = 0 if torch.cuda.is_available() else -1
BATCH_SIZE = 16
MAX_LENGTH = 512

# Labels that count as forward-looking
FLS_LABELS = {"Specific FLS", "Non-specific FLS"}

# How many top FLS sentences to store per transcript section
TOP_N_SENTENCES = 5

#Database initialiser and sentence splitter
def init_db():
    """Create guidance_scores table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS guidance_scores (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            transcript_id       INTEGER REFERENCES transcripts(id),
            ticker              TEXT NOT NULL,
            earnings_date       TEXT NOT NULL,
            fls_ratio_prepared  REAL,
            fls_ratio_qa        REAL,
            specific_fls_ratio  REAL,
            top_fls_sentences   TEXT,
            scored_at           TEXT NOT NULL,
            UNIQUE(transcript_id)
        )
    """)
    conn.commit()
    conn.close()


def split_sentences(text: str) -> list[str]:
    """Split transcript text into sentences, filtering noise."""
    if not text or len(text.strip()) < 50:
        return []

    raw = re.split(r'(?<=[.!?])\s+', text)
    sentences = []
    for s in raw:
        s = s.strip()
        if len(s) < 10:
            continue
        if re.match(r'^[A-Z][a-zA-Z\s]{0,30}:\s*$', s):
            continue
        sentences.append(s)
    return sentences

#Core classification function
def classify_sentences(sentences: list[str], classifier) -> dict:
    """
    Run FinBERT-FLS classification on a list of sentences.

    Returns:
        {
            "fls_ratio":       float  (% sentences that are any FLS)
            "specific_ratio":  float  (% sentences that are Specific FLS)
            "top_sentences":   list   (top TOP_N_SENTENCES forward-looking sentences)
            "n_sentences":     int
        }
    """
    if not sentences:
        return {
            "fls_ratio": 0.0,
            "specific_ratio": 0.0,
            "top_sentences": [],
            "n_sentences": 0
        }

    truncated = [s[:1800] for s in sentences]
    fls_count      = 0
    specific_count = 0
    fls_sentences  = []  # list of (score, sentence) for sorting

    for i in range(0, len(truncated), BATCH_SIZE):
        batch     = truncated[i:i + BATCH_SIZE]
        originals = sentences[i:i + BATCH_SIZE]
        try:
            results = classifier(batch)
            for j, sentence_scores in enumerate(results):
                top = max(sentence_scores, key=lambda x: x["score"])

                if top["label"] in FLS_LABELS:
                    fls_count += 1
                    fls_sentences.append((top["score"], originals[j]))

                if top["label"] == "Specific FLS":
                    specific_count += 1

        except Exception as e:
            print(f"  [WARN] Batch failed: {e} — skipping")
            continue

    total = len(sentences)

    # Sort by confidence score, take top N
    fls_sentences.sort(reverse=True)
    top_sentences = [s for _, s in fls_sentences[:TOP_N_SENTENCES]]

    return {
        "fls_ratio":      round(fls_count      / total, 4) if total else 0.0,
        "specific_ratio": round(specific_count / total, 4) if total else 0.0,
        "top_sentences":  top_sentences,
        "n_sentences":    total
    }

#Main Processing Loop
def classify_all_transcripts() -> dict:
    """
    Run FinBERT-FLS on all unclassified transcripts in the DB.
    Returns {"scored": int, "skipped": int, "failed": int}
    """
    init_db()

    print(f"[FLS] Loading {MODEL_NAME}...")
    classifier = pipeline(
        "text-classification",
        model=MODEL_NAME,
        top_k=None,
        device=DEVICE,
        truncation=True,
        max_length=MAX_LENGTH,
    )
    print(f"[FLS] Model loaded on {'GPU' if DEVICE == 0 else 'CPU'}")

    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT t.id, t.ticker, t.earnings_date,
               t.prepared_remarks, t.qa_section
        FROM transcripts t
        LEFT JOIN guidance_scores g ON t.id = g.transcript_id
        WHERE g.transcript_id IS NULL
        ORDER BY t.ticker, t.earnings_date
    """).fetchall()
    conn.close()

    if not rows:
        print("[FLS] All transcripts already classified.")
        return {"scored": 0, "skipped": 0, "failed": 0}

    print(f"[FLS] {len(rows)} transcripts to classify...")
    stats = {"scored": 0, "skipped": 0, "failed": 0}

    for i, (transcript_id, ticker, earnings_date,
            prepared_remarks, qa_section) in enumerate(rows):

        print(f"\n[{i+1}/{len(rows)}] {ticker} {earnings_date}")

        try:
            # Classify prepared remarks
            prep_sentences = split_sentences(prepared_remarks or "")
            prep_result    = classify_sentences(prep_sentences, classifier)

            # Classify Q&A section
            qa_sentences = split_sentences(qa_section or "")
            qa_result    = classify_sentences(qa_sentences, classifier)

            # Combine top sentences from both sections
            all_top = prep_result["top_sentences"] + qa_result["top_sentences"]
            top_json = json.dumps(all_top[:TOP_N_SENTENCES])

            # Overall specific FLS ratio across both sections
            total_sents   = prep_result["n_sentences"] + qa_result["n_sentences"]
            prep_specific = prep_result["specific_ratio"] * prep_result["n_sentences"]
            qa_specific   = qa_result["specific_ratio"]  * qa_result["n_sentences"]
            overall_specific = round(
                (prep_specific + qa_specific) / total_sents, 4
            ) if total_sents else 0.0

            # Save to DB
            conn = sqlite3.connect(DB_PATH)
            conn.execute("""
                INSERT OR IGNORE INTO guidance_scores (
                    transcript_id, ticker, earnings_date,
                    fls_ratio_prepared, fls_ratio_qa,
                    specific_fls_ratio, top_fls_sentences,
                    scored_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                transcript_id, ticker, earnings_date,
                prep_result["fls_ratio"],
                qa_result["fls_ratio"],
                overall_specific,
                top_json,
                datetime.utcnow().isoformat()
            ))
            conn.commit()
            conn.close()

            print(f"  Prepared → fls={prep_result['fls_ratio']:.3f}  "
                  f"specific={prep_result['specific_ratio']:.3f}  "
                  f"({prep_result['n_sentences']} sentences)")
            print(f"  Q&A      → fls={qa_result['fls_ratio']:.3f}  "
                  f"specific={qa_result['specific_ratio']:.3f}  "
                  f"({qa_result['n_sentences']} sentences)")

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
        ticker_filter = sys.argv[1].upper()
        print(f"[FILTER] Classifying only: {ticker_filter}")
        init_db()

        print(f"[FLS] Loading {MODEL_NAME}...")
        classifier = pipeline(
            "text-classification",
            model=MODEL_NAME,
            top_k=None,
            device=DEVICE,
            truncation=True,
            max_length=MAX_LENGTH,
        )
        print(f"[FLS] Model loaded on {'GPU' if DEVICE == 0 else 'CPU'}")

        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute("""
            SELECT t.id, t.ticker, t.earnings_date,
                   t.prepared_remarks, t.qa_section
            FROM transcripts t
            LEFT JOIN guidance_scores g ON t.id = g.transcript_id
            WHERE g.transcript_id IS NULL AND t.ticker = ?
            ORDER BY t.earnings_date
        """, (ticker_filter,)).fetchall()
        conn.close()

        if not rows:
            print(f"No unclassified transcripts for {ticker_filter}")
        else:
            for i, (transcript_id, ticker, earnings_date,
                    prepared_remarks, qa_section) in enumerate(rows):
                print(f"\n[{i+1}/{len(rows)}] {ticker} {earnings_date}")
                prep_result = classify_sentences(
                    split_sentences(prepared_remarks or ""), classifier)
                qa_result   = classify_sentences(
                    split_sentences(qa_section or ""), classifier)

                all_top  = prep_result["top_sentences"] + qa_result["top_sentences"]
                top_json = json.dumps(all_top[:TOP_N_SENTENCES])

                total_sents   = prep_result["n_sentences"] + qa_result["n_sentences"]
                prep_specific = prep_result["specific_ratio"] * prep_result["n_sentences"]
                qa_specific   = qa_result["specific_ratio"]  * qa_result["n_sentences"]
                overall_specific = round(
                    (prep_specific + qa_specific) / total_sents, 4
                ) if total_sents else 0.0

                conn = sqlite3.connect(DB_PATH)
                conn.execute("""
                    INSERT OR IGNORE INTO guidance_scores (
                        transcript_id, ticker, earnings_date,
                        fls_ratio_prepared, fls_ratio_qa,
                        specific_fls_ratio, top_fls_sentences,
                        scored_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    transcript_id, ticker, earnings_date,
                    prep_result["fls_ratio"],
                    qa_result["fls_ratio"],
                    overall_specific,
                    top_json,
                    datetime.utcnow().isoformat()
                ))
                conn.commit()
                conn.close()

                print(f"  prep fls={prep_result['fls_ratio']:.3f} "
                      f"specific={prep_result['specific_ratio']:.3f}")
                print(f"  qa   fls={qa_result['fls_ratio']:.3f} "
                      f"specific={qa_result['specific_ratio']:.3f}")
    else:
        classify_all_transcripts()