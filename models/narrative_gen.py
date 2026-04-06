import os
import re
import time
import hashlib
import sqlite3
import warnings
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from google import genai

load_dotenv()

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
DB_PATH  = BASE_DIR / "data" / "vega.db"

# ── Model config ─────────────────────────────────────────────────
MODEL_NAME    = "gemini-2.5-flash"
MAX_TOKENS    = 1024
TEMPERATURE   = 0.3   # low temperature = more consistent structured output
TRANSCRIPT_CHARS = 12000  # truncate transcript to this length before sending

# ── Rate limiting ─────────────────────────────────────────────────
SLEEP_BETWEEN_CALLS = 5  # seconds — free tier is rate limited
# ── DB initialiser ────────────────────────────────────────────────
def init_db():
    """Create narratives table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS narratives (
            transcript_hash  TEXT PRIMARY KEY,
            transcript_id    INTEGER REFERENCES transcripts(id),
            ticker           TEXT NOT NULL,
            earnings_date    TEXT NOT NULL,
            narrative        TEXT NOT NULL,
            model            TEXT NOT NULL,
            generated_at     TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def make_hash(text: str) -> str:
    """SHA-256 hash of truncated transcript — used as cache key."""
    return hashlib.sha256(text[:TRANSCRIPT_CHARS].encode()).hexdigest()


# ── Prompt builder ────────────────────────────────────────────────
def build_prompt(transcript_text: str, ticker: str, earnings_date: str) -> str:
    """
    Build the Gemini prompt for analyst note generation.
    Instructs the model to produce exactly 3 sections.
    """
    truncated = transcript_text[:TRANSCRIPT_CHARS]

    return f"""You are a senior equity analyst reading an earnings call transcript.
Analyse the transcript below for {ticker} (earnings date: {earnings_date}).

Write a structured analyst note with EXACTLY these three sections and headers:

MANAGEMENT TONE
In 2-3 sentences, describe the overall confidence level of management.
Cite one specific phrase from the transcript as evidence (in quotes).

FORWARD GUIDANCE
In 2-3 sentences, summarise the most specific forward-looking statements made.
Include any numbers, percentages, or directional guidance mentioned.

TOP RISK SIGNAL
In 1-2 sentences, identify the single most important risk or concern raised.
Be specific — name the risk category and what was said about it.

---TRANSCRIPT---
{truncated}
---END TRANSCRIPT---

Respond with only the three sections above. No preamble, no conclusion."""


# ── Core generation function ──────────────────────────────────────
def generate_narrative(
    client,
    transcript_id: int,
    ticker: str,
    earnings_date: str,
    full_text: str,
) -> str | None:
    """
    Generate an analyst note for one transcript using Gemini.
    Checks cache first — only calls API if not already generated.

    Returns the narrative string, or None if generation failed.
    """
    transcript_hash = make_hash(full_text)

    # ── Cache check ───────────────────────────────────────────────
    conn = sqlite3.connect(DB_PATH)
    cached = conn.execute(
        "SELECT narrative FROM narratives WHERE transcript_hash = ?",
        (transcript_hash,)
    ).fetchone()
    conn.close()

    if cached:
        return cached[0]

    # ── Build prompt and call API ─────────────────────────────────
    prompt = build_prompt(full_text, ticker, earnings_date)

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
        )
        narrative = response.text.strip()

        if not narrative:
            print(f"  [WARN] Empty response for {ticker} {earnings_date}")
            return None

        # ── Save to DB ────────────────────────────────────────────
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""
            INSERT OR IGNORE INTO narratives
                (transcript_hash, transcript_id, ticker, earnings_date,
                 narrative, model, generated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            transcript_hash, transcript_id, ticker, earnings_date,
            narrative, MODEL_NAME, datetime.utcnow().isoformat()
        ))
        conn.commit()
        conn.close()

        return narrative

    except Exception as e:
        print(f"  [ERROR] Gemini call failed: {e}")
        return None


# ── Main processing loop ──────────────────────────────────────────
def generate_all_narratives() -> dict:
    """
    Generate analyst notes for all transcripts not yet in narratives table.
    Returns {"generated": int, "cached": int, "failed": int}
    """
    init_db()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("[ERROR] GEMINI_API_KEY not set. Run: export GEMINI_API_KEY='your_key'")
        return {"generated": 0, "cached": 0, "failed": 0}

    client = genai.Client(api_key=api_key)

    # Fetch all transcripts not yet in narratives table
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT t.id, t.ticker, t.earnings_date, t.full_text
        FROM transcripts t
        WHERE NOT EXISTS (
            SELECT 1 FROM narratives n
            WHERE n.transcript_id = t.id
        )
        ORDER BY t.ticker, t.earnings_date
    """).fetchall()
    conn.close()

    if not rows:
        print("[NARRATIVE] All transcripts already have narratives.")
        return {"generated": 0, "cached": 0, "failed": 0}

    print(f"[NARRATIVE] {len(rows)} transcripts to process...")
    stats = {"generated": 0, "cached": 0, "failed": 0}

    for i, (transcript_id, ticker, earnings_date, full_text) in enumerate(rows):
        print(f"\n[{i+1}/{len(rows)}] {ticker} {earnings_date}")

        if not full_text or len(full_text.strip()) < 100:
            print(f"  [SKIP] Text too short")
            stats["failed"] += 1
            continue

        narrative = generate_narrative(
            client, transcript_id, ticker, earnings_date, full_text
        )

        if narrative:
            # Print first 200 chars as preview
            preview = narrative[:200].replace("\n", " ")
            print(f"  ✓ {preview}...")
            stats["generated"] += 1
        else:
            stats["failed"] += 1

        # Rate limit — free tier allows ~15 requests/minute
        time.sleep(SLEEP_BETWEEN_CALLS)

    print(f"\n[DONE] Generated: {stats['generated']} | "
          f"Cached: {stats['cached']} | Failed: {stats['failed']}")
    return stats


# ── Entry point ───────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        ticker_filter = sys.argv[1].upper()
        print(f"[FILTER] Generating narratives for: {ticker_filter}")
        init_db()

        api_key = os.environ.get("GEMINI_API_KEY")
        client = genai.Client(api_key=api_key)

        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute("""
            SELECT t.id, t.ticker, t.earnings_date, t.full_text
            FROM transcripts t
            WHERE t.ticker = ?
            AND NOT EXISTS (
                SELECT 1 FROM narratives n WHERE n.transcript_id = t.id
            )
            ORDER BY t.earnings_date
        """, (ticker_filter,)).fetchall()
        conn.close()

        if not rows:
            print(f"No unprocessed transcripts for {ticker_filter}")
            sys.exit(0)

        for i, (transcript_id, ticker, earnings_date, full_text) in enumerate(rows):
            print(f"\n[{i+1}/{len(rows)}] {ticker} {earnings_date}")
            narrative = generate_narrative(
                client, transcript_id, ticker, earnings_date, full_text or ""
            )
            if narrative:
                print(narrative[:400])
            time.sleep(SLEEP_BETWEEN_CALLS)
    else:
        generate_all_narratives()