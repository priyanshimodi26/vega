# VEGA — Pipeline Documentation

Complete technical specification of every component in the VEGA pipeline.
This is the single source of truth for implementation. Each section maps
to exactly one file in the codebase.

---

## Overview

VEGA processes Indian NIFTY 50 earnings concall transcripts through four
sequential stages: ingestion, NLP signal extraction, quantitative analysis,
and presentation. Every stage writes to SQLite so any stage can be re-run
independently without reprocessing upstream data.

```
NSE corporate filings  ──┐
 (Selenium download)     ├──► [1] Ingestion ──► SQLite DB
NSE price data (nsepy) ──┘          │
                                    │
                         ┌──────────▼──────────────────────┐
                         │      [2] NLP Signal Engine       │
                         │  FinBERT · FinBERT-FLS · MiniLM │
                         │  L-M word lists · Gemini Flash   │
                         └──────────┬──────────────────────┘
                                    │
                         ┌──────────▼──────────┐
                         │  [3] Backtest        │
                         │  Multi-dim OLS       │
                         │  Fundamental controls│
                         └──────────┬──────────┘
                                    │
                         ┌──────────▼──────────┐
                         │  [4] Dashboard       │
                         │  Plotly Dash         │
                         │  Render.com deploy   │
                         └─────────────────────┘
```

---

## Stage 1 — Ingestion
### Files: `pipeline/scraper.py`, `pipeline/price_fetcher.py`, `pipeline/run_pipeline.py`

---

### 1A. Transcript scraper
**File:** `pipeline/scraper.py`

**What it does:**
Fetches earnings concall transcript PDFs from NSE corporate filings for
a defined list of NIFTY 50 tickers. Parses and cleans the PDF text.
Splits transcript into two sections. Stores everything in SQLite.

**Data source — two layer architecture:**

Layer 1 — Static registry (pipeline/transcript_registry.json)
Manually curated NSE filing URLs for 18 NIFTY 50 companies
covering Q1FY24 through Q3FY26. Used for backtest. Stable
and reproducible.

Layer 2 — Dynamic discovery (pipeline/auto_discover.py)
Selenium visits NSE announcements page per ticker, scrapes
latest filing URLs, appends new entries to registry.
Built on Day 15. Keeps dashboard current going forward.

**PDF download method:**
Selenium headless Chromium establishes a browser session
on nseindia.com, extracts session cookies, then uses those
cookies in a requests.Session() to download the PDF directly.
Required because NSE/BSE block all non-browser server-side
requests at infrastructure level.
```

**Ticker universe — NIFTY 50 subset with reliable English transcripts:**
Focus on these sectors first (most consistent transcript availability):
- IT: TCS, INFY, WIPRO, HCLTECH, TECHM
- Banking: HDFCBANK, ICICIBANK, SBIN
- Consumer: HINDUNILVR, ASIANPAINT, NESTLEIND
- Energy: RELIANCE
- Telecom: BHARTIARTL
- Automobile: MARUTI, TATAMOTORS
- Pharma: SUNPHARMA, DRREDDY

**PDF parsing approach:**
Use `pdfplumber` to extract text page by page. Concall transcripts from
BSE follow a loose but recognisable structure:
- Pages 1–2: Cover page, participants list (skip these)
- Pages 3–N: Prepared remarks by management
- Pages N+1–end: Q&A section with analysts

**Section splitting logic:**
Search for these marker phrases to detect the Q&A boundary:
```python
QA_MARKERS = [
    "question and answer",
    "q&a session",
    "we will now begin the question",
    "open the floor for questions",
    "operator: we will now",
]
```
Everything before the first marker → `prepared_remarks`
Everything after → `qa_section`
If no marker found → store full text in `prepared_remarks`, leave `qa_section` empty.

**SQLite schema — transcripts table:**
```sql
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
```

**Key implementation notes:**
- Always check if (bse_code, earnings_date) already exists before fetching
- NSE archive URLs
- Achieved: 155 transcripts across 16 companies (Q1FY24–Q4FY26)

---

### 1B. Price fetcher and abnormal return calculator
**File:** `pipeline/price_fetcher.py`

**What it does:**
Fetches historical OHLCV price data for each ticker using nsepy. Computes
two abnormal return windows per earnings event. Fetches year-on-year
earnings variables from Screener.in as fundamental controls.

**Price data source:** yfinance (`.NS` suffix for NSE stocks, `^NSEI` for NIFTY 50 benchmark)
**Fundamental controls source:** Screener.in company financials page
  (HTML scrape — no official API, but structured and stable)

**Abnormal return calculation:**
Abnormal return = Stock return − Expected return
Expected return = Beta × NIFTY 50 index return over same window

Compute for two windows per earnings event:
- `ar_3d`: 3-day post-announcement (day 0 to day +3)
- `ar_yoy`: stock return on announcement date vs. same date one year prior
  (captures year-on-year market-implied earnings expectation)

Beta estimation: rolling 252-day OLS regression of stock returns on
NIFTY 50 returns, computed fresh for each earnings event.
Note: nsepy was originally planned but is broken due to NSE SSL infrastructure
changes. yfinance is the working replacement.

**Fundamental controls (from Screener.in):**
- `rev_yoy`: revenue year-on-year % change for that quarter
- `pat_yoy`: profit after tax year-on-year % change for that quarter

These are used as control variables in the backtest regression (Change 5
from research synthesis — motivated by Kundu & Banerjee 2021).

**SQLite schema — prices table:**
```sql
CREATE TABLE IF NOT EXISTS prices (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker         TEXT NOT NULL,
    bse_code       TEXT NOT NULL,
    earnings_date  TEXT NOT NULL,
    ar_3d          REAL,
    ar_yoy         REAL,
    beta           REAL,
    rev_yoy        REAL,
    pat_yoy        REAL,
    computed_at    TEXT NOT NULL,
    UNIQUE(bse_code, earnings_date)
)
```

---

### 1C. Pipeline orchestrator
**File:** `pipeline/run_pipeline.py`

**What it does:**
Single entry point that runs the full ingestion pipeline in order:
1. For each ticker in the universe → scraper fetches transcripts
2. For each transcript in DB → price_fetcher computes returns
3. Prints a summary: N transcripts fetched, N prices computed, N gaps

Run with: `python pipeline/run_pipeline.py`

Supports a `--ticker` flag to run for a single company during development:
`python pipeline/run_pipeline.py --ticker 532540`

---

## Stage 2 — NLP Signal Engine
### Files: `models/finbert_scorer.py`, `models/guidance_classifier.py`, `models/risk_flagger.py`, `models/narrative_gen.py`, `models/run_models.py`

---

### 2A. FinBERT sentiment scorer
**File:** `models/finbert_scorer.py`

**What it does:**
Runs ProsusAI/FinBERT inference on both transcript sections separately.
Produces three scores per section: positive%, negative%, uncertainty%.
Six total features per transcript.

**Model:** `ProsusAI/finbert` (HuggingFace)
**Task:** Sequence classification → positive / negative / neutral per sentence

**Why FinBERT over general BERT:**
FinBERT was pretrained on 4.9B tokens of financial text (Reuters, earnings
reports, 10-K filings). It understands financial hedging language, modal
verbs in a financial context, and forward-looking statement structure in
a way general BERT does not. Validated in Yang et al. (2020).

**Inference approach:**
- Chunk text into 512-token windows (FinBERT's max input length)
- Score each sentence individually using a sliding sentence splitter
- Aggregate to document level: count sentences in each class / total sentences
- Run separately on `prepared_remarks` and `qa_section`

**Output features per transcript:**
```
prepared_positive    # % of prepared remark sentences scored positive
prepared_negative    # % scored negative
prepared_uncertainty # % scored neutral (maps to uncertainty)
qa_positive
qa_negative
qa_uncertainty
```

Motivated by Bollen et al. (2011): preserving dimensions separately
rather than collapsing to one polarity score, since individual dimensions
have different predictive power.

**SQLite schema — sentiment_scores table:**
```sql
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
```

---

### 2B. Forward guidance classifier
**File:** `models/guidance_classifier.py`

**What it does:**
Classifies each sentence as forward-looking or not using FinBERT-FLS.
Extracts the top forward-looking sentences from each section.
Produces a guidance_score (ratio of FLS sentences) per transcript.

**Model:** `yiyanghkust/finbert-fls` (HuggingFace)
**Task:** Three-class classification → "Specific FLS" / "Non-specific FLS" / "Not FLS" per sentence

**Why FinBERT-FLS over BART zero-shot:**
FinBERT-FLS is a supervised model fine-tuned specifically on forward-
looking statements from analyst reports — trained for exactly this task.
BART-large-mnli is a general zero-shot model not trained on financial
language. Domain-specific supervised > general zero-shot for this task.
Supported by FinBERT2 paper's finding that fine-tuned BERT outperforms
general models by 9.7–12.3% on financial classification.

**Output per transcript:**
- `fls_ratio_prepared`: ratio of FLS sentences in prepared remarks
- `fls_ratio_qa`: ratio of FLS sentences in Q&A
- `top_fls_sentences`: JSON list of top 5 forward-looking sentences
  (stored as text, used in dashboard transcript viewer)
- `specific_fls_ratio`: ratio of Specific FLS sentences (concrete guidance) across both sections

**SQLite schema — guidance_scores table:**
```sql
CREATE TABLE IF NOT EXISTS guidance_scores (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    transcript_id       INTEGER REFERENCES transcripts(id),
    ticker              TEXT NOT NULL,
    earnings_date       TEXT NOT NULL,
    fls_ratio_prepared  REAL,
    fls_ratio_qa        REAL,
    top_fls_sentences   TEXT,
    scored_at           TEXT NOT NULL,
    UNIQUE(transcript_id)
)
```

---

### 2C. Risk flagger
**File:** `models/risk_flagger.py`

**What it does:**
Detects risk signals using a hybrid of two approaches:
1. MiniLM cosine similarity against risk anchor phrases (semantic)
2. Loughran-McDonald word list counts (lexical)

Produces one risk score per category per transcript.

**Models:**
- `sentence-transformers/all-MiniLM-L6-v2` for semantic similarity
- Loughran-McDonald master dictionary CSV for lexical counting
  (download from: https://sraf.nd.edu/loughranmcdonald-master-dictionary/)

**Why hybrid approach:**
MiniLM catches semantically similar risk language even when exact keywords
aren't used. L-M word lists catch validated financial risk vocabulary that
has been shown in prior literature (Larcker & Zakolyukina 2012, L-M 2011)
to predict financial outcomes. Together they cover both semantic and
lexical dimensions of risk.

**Risk categories and MiniLM anchor phrases:**
```python
RISK_ANCHORS = {
    "liquidity_risk":      "cash flow pressure, funding constraints, liquidity concerns",
    "demand_softness":     "demand weakness, volume decline, slower growth",
    "margin_compression":  "margin pressure, cost increase, profitability headwinds",
    "regulatory_risk":     "regulatory uncertainty, compliance burden, policy risk",
    "macro_headwinds":     "macroeconomic uncertainty, inflation pressure, rate environment",
    "competitive_threat":  "competitive intensity, market share loss, pricing pressure",
    "management_evasion":  "we will update, cannot comment, too early to say",
    "overconfidence":      "extremely confident, absolutely certain, no doubt whatsoever",
}
```

**Loughran-McDonald features (motivated by Larcker & Zakolyukina 2012):**
```
lm_uncertainty_count   # anxiety/uncertainty words normalized by length
lm_litigious_count     # legal risk words
lm_weak_modal_count    # hedging language (may, might, could, possibly)
lm_strong_modal_count  # confident language (will, must, shall)
lm_extreme_positive    # overconfidence signal
```

**SQLite schema — risk_scores table:**
```sql
CREATE TABLE IF NOT EXISTS risk_scores (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    transcript_id         INTEGER REFERENCES transcripts(id),
    ticker                TEXT NOT NULL,
    earnings_date         TEXT NOT NULL,
    liquidity_risk        REAL,
    demand_softness       REAL,
    margin_compression    REAL,
    regulatory_risk       REAL,
    macro_headwinds       REAL,
    competitive_threat    REAL,
    management_evasion    REAL,
    overconfidence        REAL,
    lm_uncertainty        REAL,
    lm_litigious          REAL,
    lm_weak_modal         REAL,
    lm_strong_modal       REAL,
    lm_extreme_positive   REAL,
    scored_at             TEXT NOT NULL,
    UNIQUE(transcript_id)
)
```

---

### 2D. Narrative generator
**File:** `models/narrative_gen.py`

**What it does:**
Calls Google Gemini Flash API to generate a structured 3-section analyst
note per transcript. Results are cached in SQLite — API is only called
once per transcript ever.

**Model:** gemini-2.5-flash via google-genai SDK (free tier: 20 req/day per model)

**Prompt structure:**
Instructs the model to produce exactly three sections:
1. MANAGEMENT TONE — confidence level with one cited phrase as evidence
2. FORWARD GUIDANCE — specific numbers or directional statements mentioned
3. TOP RISK SIGNAL — the single most important risk raised

**Caching:** SHA-256 hash of truncated transcript text used as cache key.
If hash exists in `narratives` table, return cached result immediately.

**SQLite schema — narratives table:**
```sql
CREATE TABLE IF NOT EXISTS narratives (
    transcript_hash  TEXT PRIMARY KEY,
    ticker           TEXT NOT NULL,
    earnings_date    TEXT NOT NULL,
    narrative        TEXT NOT NULL,
    model            TEXT NOT NULL,
    generated_at     TEXT NOT NULL
)

**Note on quota:** gemini-2.5-flash has a 20 req/day free tier limit. Run
`python models/narrative_gen.py` daily until all 156 narratives are cached.
Already-cached narratives are skipped automatically via SHA-256 hash lookup.
The deprecated google-generativeai package has been replaced with google-genai.
```

---

### 2E. Model orchestrator
**File:** `models/run_models.py`

**What it does:**
Runs all four NLP models in sequence over every unscored transcript
in the DB. Checks each model's table for existing scores before running
to avoid reprocessing. Prints progress.

Run with: `python models/run_models.py`

Supports `--model` flag to run one module at a time during development:
`python models/run_models.py --model finbert`
`python models/run_models.py --model guidance`
`python models/run_models.py --model risk`
`python models/run_models.py --model narrative`

---

## Stage 3 — Backtest
### File: `analysis/backtest.py`

**What it does:**
Joins all signal tables with the prices table. Runs multi-dimensional
OLS regression. Computes and prints correlation matrix, R², p-values,
and coefficient table. Saves results for dashboard display.

**Regression design (motivated by Bollen et al. 2011 + Kundu & Banerjee 2021):**

```
ar_3d ~ β1(prepared_positive)
      + β2(prepared_uncertainty)
      + β3(qa_positive)
      + β4(qa_uncertainty)
      + β5(fls_ratio_qa)
      + β6(lm_uncertainty)
      + β7(lm_extreme_positive)
      + β8(lm_weak_modal)
      + β9(rev_yoy)          ← fundamental control
      + β10(pat_yoy)         ← fundamental control
      + ε
```

Run the same regression with `ar_yoy` as the dependent variable for
the medium-term signal test.

**Key outputs to report:**
- R² for NLP-only model vs. fundamentals-only model vs. combined model
- Coefficient + p-value for each NLP feature
- Which section (prepared vs. Q&A) has stronger predictive power
- Pearson correlation matrix of all features vs. returns

**SQLite schema — backtest_results table:**
```sql
CREATE TABLE IF NOT EXISTS backtest_results (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_date      TEXT NOT NULL,
    n_events      INTEGER,
    r_squared_3d  REAL,
    r_squared_yoy REAL,
    coefficients  TEXT,   -- JSON: {feature: {coef, pvalue, ci_lower, ci_upper}}
    correlation   TEXT    -- JSON: correlation matrix
)
```

---

## Stage 4 — Dashboard
### Files: `dashboard/app.py`, `dashboard/layout.py`, `dashboard/callbacks.py`

**What it does:**
Plotly Dash web application with four panels. Reads exclusively from
SQLite — zero API calls at render time.

**Panel 1 — Sentiment timeline:**
Line chart showing prepared_positive, qa_positive, and qa_uncertainty
across all earnings dates for a selected ticker. Price return overlaid
on secondary y-axis. Vertical lines mark earnings dates.

**Panel 2 — Risk flag heatmap:**
Heatmap where y-axis = risk categories, x-axis = earnings dates,
cell color = risk score intensity. Hover tooltip shows the flagged
sentences. Visually striking — the most memorable chart in the app.

**Panel 3 — Transcript viewer + analyst note:**
Left: scrollable transcript with sentences color-coded by category
(green = positive, red = risk flag, blue = forward guidance).
Right: Gemini-generated analyst note displayed as a styled card.
Toggle between prepared remarks and Q&A sections.

**Panel 4 — Signal vs. returns scatter:**
Scatter plot: x = qa_uncertainty score, y = ar_3d abnormal return.
One dot per earnings event, colored by ticker. Regression line with
R² annotation. This is your "money chart."

**Deployment:** Render.com free tier.
`Procfile`: `web: gunicorn dashboard.app:server`
Add `gunicorn` to requirements.txt.

---

## Database summary

All tables live in `data/vega.db`. Here is the full entity relationship:

```
transcripts
    │
    ├──► sentiment_scores   (1:1 via transcript_id)
    ├──► guidance_scores    (1:1 via transcript_id)
    ├──► risk_scores        (1:1 via transcript_id)
    └──► narratives         (1:1 via transcript_hash)

prices                      (joined on ticker + earnings_date)

backtest_results            (derived from all above)
```

---

## Full file map

```
vega/
├── pipeline/
│   ├── scraper.py           NSE transcript fetcher + section splitter
│   ├── price_fetcher.py     nsepy prices + abnormal returns + fundamentals
│   ├── run_pipeline.py      orchestrator (--ticker flag supported)
│   └── auto_discover.py     Day 15 — dynamic NSE URL discovery
├── models/
│   ├── finbert_scorer.py    ProsusAI/FinBERT → 6 sentiment features
│   ├── guidance_classifier.py  yiyanghkust/finbert-fls → FLS ratio + top sentences
│   ├── risk_flagger.py      MiniLM + Loughran-McDonald → 13 risk features
│   ├── narrative_gen.py     Gemini Flash → cached analyst note
│   └── run_models.py        orchestrator (--model flag supported)
├── analysis/
│   └── backtest.py          multi-dim OLS + correlation + results to DB
├── dashboard/
│   ├── app.py               Dash entry point + server object
│   ├── layout.py            4-panel layout definition
│   └── callbacks.py         reactive callbacks for all interactivity
├── data/
│   ├── vega.db              SQLite database (gitignored)
│   └── loughran_mcdonald.csv  L-M master dictionary (gitignored)
├── notebooks/
│   └── eda.ipynb            exploratory analysis + backtest visualisation
├── research_papers/
│   ├── NOTES.md             research synthesis
│   └── *.pdf                source papers
├── requirements.txt
├── Procfile
└── README.md
```

---

## Requirements — final list

```
# Core data
requests
pdfplumber
yfinance
pandas
numpy

# NLP
transformers
torch
sentence-transformers
scipy

# Stats
statsmodels
scikit-learn

# LLM
google-generativeai

# Dashboard
plotly
dash
gunicorn

# Utilities
python-dotenv
tqdm

selenium
webdriver-manager
chromium-browser      # system package, not pip
chromium-chromedriver # system package, not pip
```

---

*This document is updated as the build progresses.
Last updated: Day 10 complete. Scraper: 156 transcripts. Price fetcher: 156 abnormal return pairs. FinBERT scorer: 156 scored. Guidance classifier: 156 classified. Risk flagger: 156 scored. Narrative generator: 44/156 cached (running daily due to quota). Backtesting complete. OLS backtest — ar_3d R²=0.130 (p=0.014), ar_yoy R²=0.155 (p=0.004), specific_fls and lm_extreme_positive significant. Plotly Dash dashboard made.*
