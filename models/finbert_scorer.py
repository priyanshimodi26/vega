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
MODEL_NAME  = "ProsusAI/finbert"
DEVICE      = 0 if torch.cuda.is_available() else -1  # GPU if available, else CPU
BATCH_SIZE  = 16    # sentences per batch — reduce to 8 if you get memory errors
MAX_LENGTH  = 512   # FinBERT's maximum token input length
MIN_SENTENCE_LENGTH = 10  # skip very short fragments

#Database initialiser and sentence splitter
def init_db():
    """Create sentiment_scores table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sentiment_scores (
            id                   INTEGER PRIMARY KEY AUTOINCREMENT,
            transcript_id        INTEGER REFERENCES transcripts(id),
            ticker               TEXT NOT NULL,
            earnings_date        TEXT NOT NULL,
            prepared_positive    REAL,
            prepared_negative    REAL,
            prepared_uncertainty REAL,
            qa_positive          REAL,
            qa_negative          REAL,
            qa_uncertainty       REAL,
            scored_at            TEXT NOT NULL,
            UNIQUE(transcript_id)
        )
    """)
    conn.commit()
    conn.close()


def split_sentences(text: str) -> list[str]:
    """
    Split transcript text into sentences for FinBERT inference.
    Uses simple punctuation splitting — good enough for earnings call text
    which is already well-structured speech.

    Filters out very short fragments that would add noise.
    """
    if not text or len(text.strip()) < 50:
        return []

    # Split on sentence-ending punctuation
    import re
    raw = re.split(r'(?<=[.!?])\s+', text)

    sentences = []
    for s in raw:
        s = s.strip()
        # Skip very short fragments, page artifacts, speaker labels
        if len(s) < MIN_SENTENCE_LENGTH:
            continue
        # Skip lines that are just speaker names e.g. "Operator:" or "John Smith:"
        if re.match(r'^[A-Z][a-zA-Z\s]{0,30}:\s*$', s):
            continue
        sentences.append(s)

    return sentences

#Core Scoring Function
def score_sentences(sentences: list[str], scorer) -> dict:
    """
    Run FinBERT inference on a list of sentences.
    Returns aggregated document-level scores.

    Args:
        sentences: list of sentence strings
        scorer:    HuggingFace pipeline object (loaded once, passed in)

    Returns:
        {
            "positive":    float (% of sentences scored positive)
            "negative":    float (% of sentences scored negative)
            "uncertainty": float (% of sentences scored neutral)
            "n_sentences": int   (total sentences scored)
        }
    """
    if not sentences:
        return {"positive": 0.0, "negative": 0.0, "uncertainty": 0.0, "n_sentences": 0}

    # Truncate sentences to MAX_LENGTH tokens to avoid FinBERT errors
    # Simple char-based truncation — 512 tokens ≈ 1800 chars for English
    truncated = [s[:1800] for s in sentences]

    counts = {"positive": 0, "negative": 0, "neutral": 0}

    # Process in batches
    for i in range(0, len(truncated), BATCH_SIZE):
        batch = truncated[i:i + BATCH_SIZE]
        try:
            results = scorer(batch)
            for sentence_scores in results:
                # sentence_scores is a list of {label, score} dicts
                top = max(sentence_scores, key=lambda x: x["score"])
                counts[top["label"]] += 1
        except Exception as e:
            print(f"  [WARN] Batch scoring failed: {e} — skipping batch")
            continue

    total = sum(counts.values())
    if total == 0:
        return {"positive": 0.0, "negative": 0.0, "uncertainty": 0.0, "n_sentences": 0}

    return {
        "positive":    round(counts["positive"] / total, 4),
        "negative":    round(counts["negative"] / total, 4),
        "uncertainty": round(counts["neutral"]  / total, 4),
        "n_sentences": total
    }

#Main Processing Loop
def score_all_transcripts() -> dict:
    """
    Run FinBERT scoring on all unscored transcripts in the DB.
    Loads the model once, processes all transcripts, saves to sentiment_scores.

    Returns {"scored": int, "skipped": int, "failed": int}
    """
    init_db()

    # Load model once — expensive operation, do it outside the loop
    print(f"[FINBERT] Loading {MODEL_NAME}...")
    scorer = pipeline(
        "text-classification",
        model=MODEL_NAME,
        top_k=None,
        device=DEVICE,
        truncation=True,
        max_length=MAX_LENGTH,
    )
    print(f"[FINBERT] Model loaded on {'GPU' if DEVICE == 0 else 'CPU'}")

    # Fetch all transcripts not yet scored
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT t.id, t.ticker, t.earnings_date,
               t.prepared_remarks, t.qa_section
        FROM transcripts t
        LEFT JOIN sentiment_scores s ON t.id = s.transcript_id
        WHERE s.transcript_id IS NULL
        ORDER BY t.ticker, t.earnings_date
    """).fetchall()
    conn.close()

    if not rows:
        print("[FINBERT] All transcripts already scored.")
        return {"scored": 0, "skipped": 0, "failed": 0}

    print(f"[FINBERT] {len(rows)} transcripts to score...")
    stats = {"scored": 0, "skipped": 0, "failed": 0}

    for i, (transcript_id, ticker, earnings_date,
            prepared_remarks, qa_section) in enumerate(rows):

        print(f"\n[{i+1}/{len(rows)}] {ticker} {earnings_date}")

        try:
            # Score prepared remarks
            prep_sentences = split_sentences(prepared_remarks or "")
            print(f"  Prepared remarks: {len(prep_sentences)} sentences")
            prep_scores = score_sentences(prep_sentences, scorer)

            # Score Q&A section
            qa_sentences = split_sentences(qa_section or "")
            print(f"  Q&A section:      {len(qa_sentences)} sentences")
            qa_scores = score_sentences(qa_sentences, scorer)

            # Save to DB
            conn = sqlite3.connect(DB_PATH)
            conn.execute("""
                INSERT OR IGNORE INTO sentiment_scores (
                    transcript_id, ticker, earnings_date,
                    prepared_positive, prepared_negative, prepared_uncertainty,
                    qa_positive, qa_negative, qa_uncertainty,
                    scored_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                transcript_id, ticker, earnings_date,
                prep_scores["positive"],
                prep_scores["negative"],
                prep_scores["uncertainty"],
                qa_scores["positive"],
                qa_scores["negative"],
                qa_scores["uncertainty"],
                datetime.utcnow().isoformat()
            ))
            conn.commit()
            conn.close()

            print(f"  Prepared → pos={prep_scores['positive']:.3f}  "
                  f"neg={prep_scores['negative']:.3f}  "
                  f"unc={prep_scores['uncertainty']:.3f}")
            print(f"  Q&A      → pos={qa_scores['positive']:.3f}  "
                  f"neg={qa_scores['negative']:.3f}  "
                  f"unc={qa_scores['uncertainty']:.3f}")

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

    # Optional: pass a ticker to score only that company
    # Usage: python models/finbert_scorer.py TCS
    if len(sys.argv) > 1:
        ticker_filter = sys.argv[1].upper()
        print(f"[FILTER] Scoring only: {ticker_filter}")

        # Temporarily patch to filter by ticker
        init_db()
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute("""
            SELECT t.id, t.ticker, t.earnings_date,
                   t.prepared_remarks, t.qa_section
            FROM transcripts t
            LEFT JOIN sentiment_scores s ON t.id = s.transcript_id
            WHERE s.transcript_id IS NULL AND t.ticker = ?
            ORDER BY t.earnings_date
        """, (ticker_filter,)).fetchall()
        conn.close()

        if not rows:
            print(f"No unscored transcripts found for {ticker_filter}")
        else:
            print(f"[FINBERT] Loading {MODEL_NAME}...")
            scorer = pipeline(
                "text-classification",
                model=MODEL_NAME,
                top_k=None,
                device=DEVICE,
                truncation=True,
                max_length=MAX_LENGTH,
            )
            print(f"[FINBERT] Model loaded on {'GPU' if DEVICE == 0 else 'CPU'}")
            init_db()

            for i, (transcript_id, ticker, earnings_date,
                    prepared_remarks, qa_section) in enumerate(rows):
                print(f"\n[{i+1}/{len(rows)}] {ticker} {earnings_date}")
                prep_scores = score_sentences(split_sentences(prepared_remarks or ""), scorer)
                qa_scores   = score_sentences(split_sentences(qa_section or ""), scorer)

                conn = sqlite3.connect(DB_PATH)
                conn.execute("""
                    INSERT OR IGNORE INTO sentiment_scores (
                        transcript_id, ticker, earnings_date,
                        prepared_positive, prepared_negative, prepared_uncertainty,
                        qa_positive, qa_negative, qa_uncertainty,
                        scored_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    transcript_id, ticker, earnings_date,
                    prep_scores["positive"], prep_scores["negative"], prep_scores["uncertainty"],
                    qa_scores["positive"],  qa_scores["negative"],  qa_scores["uncertainty"],
                    datetime.utcnow().isoformat()
                ))
                conn.commit()
                conn.close()
                print(f"  prep pos={prep_scores['positive']:.3f} neg={prep_scores['negative']:.3f} unc={prep_scores['uncertainty']:.3f}")
                print(f"  qa   pos={qa_scores['positive']:.3f} neg={qa_scores['negative']:.3f} unc={qa_scores['uncertainty']:.3f}")
    else:
        score_all_transcripts()