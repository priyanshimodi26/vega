#Imports and Constants
import os
import tempfile
import requests
import pdfplumber
import sqlite3
import json
import time
import re
from datetime import datetime
from pathlib import Path
from io import BytesIO

# ── Paths ────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent.parent
DATA_DIR  = BASE_DIR / "data"
DB_PATH   = DATA_DIR / "vega.db"
REGISTRY  = BASE_DIR / "pipeline" / "transcript_registry.json"

# ── Request headers ──────────────────────────────────────────────
# We identify ourselves honestly as a research project
HEADERS = {
    "User-Agent": "VEGA-Research/1.0 (Academic project; priyanshi5930@gmail.com)",
    "Accept": "application/pdf",
}

# ── Rate limiting ────────────────────────────────────────────────
DELAY_BETWEEN_REQUESTS = 2  # seconds — be polite to IR websites

#Database Initialiser
def init_db():
    """Create database tables if they don't exist."""
    DATA_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS transcripts (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker           TEXT NOT NULL,
            bse_code         TEXT NOT NULL,
            company_name     TEXT NOT NULL,
            earnings_date    TEXT NOT NULL,
            fiscal_quarter   TEXT,
            prepared_remarks TEXT,
            qa_section       TEXT,
            full_text        TEXT,
            pdf_url          TEXT,
            fetched_at       TEXT NOT NULL,
            UNIQUE(bse_code, earnings_date)
        )
    """)
    conn.commit()
    conn.close()
    print(f"[DB] Initialised at {DB_PATH}")

#PDF Downloader
def download_pdf(url: str) -> BytesIO | None:
    """
    Download a PDF using a headless Chrome browser via Selenium.
    Bypasses server-side blocks on NSE/BSE that reject non-browser requests.
    Returns PDF as in-memory bytes object, or None if download fails.
    """
    import tempfile
    import os
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    from webdriver_manager.chrome import ChromeDriverManager

    # Configure headless Chrome to download PDFs to a temp directory
    tmpdir = tempfile.mkdtemp()

    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-software-rasterizer")
    chrome_options.add_argument("--remote-debugging-port=9222")
    chrome_options.binary_location = "/usr/bin/chromium-browser"
    chrome_options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )

    # Tell Chrome to auto-download PDFs instead of opening them
    prefs = {
    "download.default_directory": tmpdir,
    "download.prompt_for_download": False,
    "download.directory_upgrade": True,
    "plugins.always_open_pdf_externally": True,
    "plugins.plugins_disabled": ["Chrome PDF Viewer"],
    }
    chrome_options.add_experimental_option("prefs", prefs)
    chrome_options.add_argument("--disable-plugins")

    driver = None
    try:
        service = Service("/usr/bin/chromedriver")
        driver = webdriver.Chrome(service=service, options=chrome_options)

        print(f"  [BROWSER] Fetching: {url}")
        driver.get(url)

        # Wait for PDF to download (max 30 seconds)
        driver.get(url)

        # For direct PDF URLs, use CDP to fetch content as bytes directly
        import base64
        result = driver.execute_cdp_cmd("Page.captureScreenshot", {}) 

        # Use requests session with selenium cookies instead
        cookies = driver.get_cookies()
        driver.quit()
        driver = None

        session = __import__('requests').Session()
        for cookie in cookies:
            session.cookies.set(cookie['name'], cookie['value'])

        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36",
            "Referer": "https://www.nseindia.com/"
        })

        response = session.get(url, timeout=30)
        if response.status_code == 200 and len(response.content) > 1000:
            size_kb = len(response.content) / 1024
            print(f"  [OK] Downloaded {size_kb:.1f} KB")
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)
            return BytesIO(response.content)
        else:
            print(f"  [ERROR] Failed after cookie extraction: HTTP {response.status_code}")
            return None

    except Exception as e:
        print(f"  [ERROR] Selenium download failed: {e}")
        return None

    finally:
        if driver:
            driver.quit()

#Text Extarctor and section splitter
# Phrases that mark the start of the Q&A section
QA_MARKERS = [
    "question and answer",
    "q&a session",
    "we will now begin the question",
    "open the floor for questions",
    "operator: we will now",
    "moderator: we will now",
    "we will now take questions",
    "floor is now open for questions",
]

def clean_text(text: str) -> str:
    """
    Clean raw PDF-extracted text.
    Removes page numbers, excessive whitespace, and repeated headers.
    """
    # Remove standalone page numbers (e.g. "Page 1 of 12" or just "1")
    text = re.sub(r'\bPage \d+ of \d+\b', '', text)
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)

    # Collapse multiple blank lines into one
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Remove excessive spaces
    text = re.sub(r'[ \t]{2,}', ' ', text)

    return text.strip()


def extract_and_split(pdf_bytes: BytesIO) -> dict:
    """
    Extract text from PDF and split into prepared remarks and Q&A section.

    Returns:
        {
            "full_text":        str,
            "prepared_remarks": str,
            "qa_section":       str  (empty string if no Q&A marker found)
        }
    """
    full_text = ""

    with pdfplumber.open(pdf_bytes) as pdf:
        # Skip first 2 pages — usually cover page and participants list
        pages_to_read = pdf.pages[2:]
        for page in pages_to_read:
            page_text = page.extract_text()
            if page_text:
                full_text += page_text + "\n"

    full_text = clean_text(full_text)

    # Find Q&A boundary
    lower_text = full_text.lower()
    split_index = -1

    for marker in QA_MARKERS:
        idx = lower_text.find(marker)
        if idx != -1:
            # Take the earliest marker found
            if split_index == -1 or idx < split_index:
                split_index = idx

    if split_index == -1:
        # No Q&A section found — store everything as prepared remarks
        print("  [WARN] No Q&A marker found — storing as prepared remarks only")
        return {
            "full_text":        full_text,
            "prepared_remarks": full_text,
            "qa_section":       ""
        }

    return {
        "full_text":        full_text,
        "prepared_remarks": full_text[:split_index].strip(),
        "qa_section":       full_text[split_index:].strip()
    }

#Database Writer
def save_transcript(
    ticker:        str,
    bse_code:      str,
    company_name:  str,
    earnings_date: str,
    fiscal_quarter: str,
    extracted:     dict,
    pdf_url:       str
) -> bool:
    """
    Save extracted transcript to SQLite.
    Returns True if inserted, False if already existed.
    """
    conn = sqlite3.connect(DB_PATH)

    cursor = conn.execute("""
        INSERT OR IGNORE INTO transcripts (
            ticker, bse_code, company_name, earnings_date,
            fiscal_quarter, prepared_remarks, qa_section,
            full_text, pdf_url, fetched_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        ticker,
        bse_code,
        company_name,
        earnings_date,
        fiscal_quarter,
        extracted["prepared_remarks"],
        extracted["qa_section"],
        extracted["full_text"],
        pdf_url,
        datetime.utcnow().isoformat()
    ))

    conn.commit()
    inserted = cursor.rowcount > 0
    conn.close()

    if inserted:
        print(f"  [DB] Saved: {company_name} {fiscal_quarter}")
    else:
        print(f"  [DB] Already exists, skipped: {company_name} {fiscal_quarter}")

    return inserted

#Registry loader and main scrape function
def load_registry() -> list:
    """
    Load the transcript registry JSON file.
    Returns a list of transcript metadata dicts.
    """
    if not REGISTRY.exists():
        print(f"[ERROR] Registry not found at {REGISTRY}")
        print("  Create pipeline/transcript_registry.json first.")
        return []

    with open(REGISTRY, "r") as f:
        registry = json.load(f)

    print(f"[REGISTRY] Loaded {len(registry)} entries")
    return registry


def scrape_all(ticker_filter: str = None) -> dict:
    """
    Main scrape function. Iterates over registry and fetches
    all transcripts not already in the database.

    Args:
        ticker_filter: If provided, only scrape this ticker.
                      e.g. "INFY" to scrape Infosys only.

    Returns:
        {"fetched": int, "skipped": int, "failed": int}
    """
    init_db()
    registry = load_registry()

    if not registry:
        return {"fetched": 0, "skipped": 0, "failed": 0}

    # Apply ticker filter if provided
    if ticker_filter:
        registry = [r for r in registry if r["ticker"] == ticker_filter.upper()]
        print(f"[FILTER] Filtered to {len(registry)} entries for {ticker_filter.upper()}")

    stats = {"fetched": 0, "skipped": 0, "failed": 0}

    for i, entry in enumerate(registry):
        ticker        = entry["ticker"]
        bse_code      = entry["bse_code"]
        company_name  = entry["company_name"]
        earnings_date = entry["earnings_date"]
        fiscal_quarter = entry["fiscal_quarter"]
        pdf_url       = entry["pdf_url"]

        print(f"\n[{i+1}/{len(registry)}] {company_name} — {fiscal_quarter}")
        print(f"  URL: {pdf_url}")

        # Download PDF
        pdf_bytes = download_pdf(pdf_url)
        if pdf_bytes is None:
            stats["failed"] += 1
            continue

        # Extract and split text
        try:
            extracted = extract_and_split(pdf_bytes)
        except Exception as e:
            print(f"  [ERROR] PDF extraction failed: {e}")
            stats["failed"] += 1
            continue

        # Sanity check — skip if extracted text is too short
        if len(extracted["full_text"]) < 500:
            print(f"  [WARN] Extracted text too short ({len(extracted['full_text'])} chars) — skipping")
            stats["failed"] += 1
            continue

        # Save to DB
        inserted = save_transcript(
            ticker, bse_code, company_name,
            earnings_date, fiscal_quarter,
            extracted, pdf_url
        )

        if inserted:
            stats["fetched"] += 1
        else:
            stats["skipped"] += 1

        # Be polite — wait between requests
        if i < len(registry) - 1:
            time.sleep(DELAY_BETWEEN_REQUESTS)

    print(f"\n[DONE] Fetched: {stats['fetched']} | Skipped: {stats['skipped']} | Failed: {stats['failed']}")
    return stats

#Entry point
if __name__ == "__main__":
    import sys

    # Optional: pass a ticker to scrape only that company
    # Usage: python pipeline/scraper.py INFY
    ticker_filter = sys.argv[1] if len(sys.argv) > 1 else None

    results = scrape_all(ticker_filter=ticker_filter)