# VEGA — Volatility & Earnings Guidance Analyzer

> *Do concall transcripts of NIFTY 50 companies contain predictive signals about short-term price movements? VEGA is built to find out.*

---

## What is VEGA?

Every quarter, executives at NIFTY 50 companies spend 60–90 minutes on earnings calls — choosing their words carefully. They hedge. They signal. They bury risk disclosures in subordinate clauses and front-load confidence where confidence may not be warranted.

VEGA is an NLP pipeline that reads those transcripts the way a quant analyst would — — extracting sentiment, forward guidance signals, and risk flags at scale — then tests whether those signals have statistically meaningful predictive power over 3-day abnormal stock returns for NIFTY 50 listed companies.

The output is a live, interactive dashboard where you can look up any covered company, see its earnings sentiment history, inspect flagged risk sentences, and read an AI-generated analyst note — all updated each earnings cycle.

---

## Status

> 🔧 **Active development.** Pipeline scaffolding complete. NLP engine and dashboard in progress.

| Component | Status |
|---|---|
| Transcript scraper (NSE + Selenium) | ✅ Complete |
| Transcript registry (155 transcripts, 16 companies) | ✅ Completes |
| Price fetcher + abnormal returns | 🔧 In progress |
| FinBERT sentiment scorer | ⏳ Pending |
| FinBERT-FLS guidance classifier | ⏳ Pending |
| Risk flagger (MiniLM + L-M word lists) | ⏳ Pending |
| Gemini Flash narrative generator | ⏳ Pending |
| Backtest (multi-dim OLS regression) | ⏳ Pending |
| Plotly Dash dashboard | ⏳ Pending |
| Deployment (Render.com) | ⏳ Pending |

*Results and key findings will be added here as the backtest is completed.*

---

## Methodology

### 1. Data collection
Earnings concall transcripts are sourced from NSE corporate filings (Analysts/Institutional Investor Meet/Con. Call Updates category) for 16 NIFTY 50 companies (KOTAKBANK and MARUTI pending) with consistent English transcript availability. A Selenium-based download pipeline handles NSE's infrastructure-level blocking of non-browser requests. The historical dataset covers Q1FY24 through Q3FY26 (~155 transcripts across 16 companies). A dynamic discovery layer (auto_discover.py) keeps the registry current as new quarters are filed.

### 2. NLP signal extraction
Each transcript is processed through a three-layer NLP stack:

- **Sentiment scoring** — [ProsusAI/FinBERT](https://huggingface.co/ProsusAI/finbert), a BERT model fine-tuned on financial news, scores each sentence as positive, negative, or neutral. Scores are aggregated into document-level uncertainty, positivity, and negativity percentages.

- **Forward guidance classification** — yya518/FinBERT-FLS, a FinBERT variant fine-tuned specifically on forward-looking statements, classifies sentences as forward-looking or not.

- **Risk flag detection** — Sentence embeddings from `all-MiniLM-L6-v2` are used to compute cosine similarity against predefined risk signal anchors across 8 categories (liquidity risk, demand softness, margin compression, regulatory concern, and others).

### 3. Narrative generation
Google Gemini Flash (free tier) generates a structured three-section analyst note per transcript: management tone, forward guidance summary, and top risk signal. Outputs are cached in SQLite to avoid redundant API calls.

### 4. Composite signal score
Sentiment percentages, guidance tone, and risk flag intensities are combined into a single composite score per earnings event. The weighting scheme is documented in `analysis/backtest.py`.

### 5. Backtest
The composite score is regressed against 3-day abnormal returns (stock return minus beta-adjusted sector ETF return) using OLS. Pearson correlation, R², and p-values are reported. Results will be published here.

---

## Architecture

```
NSE corporate filings     ──┐
(Selenium download)         ├──► Scraper & parser ──► SQLite DB
NSE price data (nsepy)    ──┘                              │
                                                           ▼
                                           ┌─────────────────────────────┐
                                           │      NLP Signal Engine      │
                                           │  FinBERT · BART · MiniLM    │
                                           │  Gemini Flash (narratives)  │
                                           └──────────────┬──────────────┘
                                                          │
                                                          ▼
                                                  Composite scorer
                                                          │
                                               ┌─────────────────────┐
                                               │   OLS backtest      │
                                               │   R² · p-value      │
                                               └──────────┬──────────┘
                                                          │
                                                          ▼
                                               Plotly Dash dashboard
                                              [deployed on Render.com]
```

---

## Project structure

```
vega/
├── pipeline/
│   ├── scraper.py            # BSE concall transcript fetcher
│   ├── price_fetcher.py      # yfinance + abnormal return computation
│   ├── run_pipeline.py       # one-command orchestrator
|   ├── transcript_registry.json
│   └── auto_discover.py      # Day 15 — dynamic NSE URL discovery
├── models/
│   ├── finbert_scorer.py     # sentence-level sentiment inference
│   ├── guidance_classifier.py
│   ├── risk_flagger.py       # MiniLM cosine similarity risk detection
│   └── narrative_gen.py      # Gemini Flash analyst note generation
├── analysis/
│   └── backtest.py           # OLS regression + correlation stats
├── dashboard/
│   ├── app.py                # Dash entry point
│   ├── layout.py
│   └── callbacks.py
├── data/                     # vega.db lives here (gitignored)
├── notebooks/                # EDA and backtest exploration
├── requirements.txt
├── Procfile                  # for Render.com deployment
└── README.md
```

---

## Running locally

```bash
# 1. Clone and enter
git clone https://github.com/yourusername/vega.git
cd vega

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your Gemini API key (free at aistudio.google.com)
export GEMINI_API_KEY="your_key_here"

# 5. Run the pipeline
python pipeline/run_pipeline.py
# System dependencies (WSL/Ubuntu)
sudo apt-get install -y chromium-browser chromium-chromedriver

# 6. Launch the dashboard
python models/run_models.py
python dashboard/app.py
```

---

## Tech stack

| Layer | Tool | Purpose |
|---|---|---|
| Sentiment model | ProsusAI/FinBERT | Finance-domain sentence sentiment |
| Forward guidance   | yya518/FinBERT-FLS | Fine-tuned forward-looking statement classifier |
| Embeddings | all-MiniLM-L6-v2 | Risk flag cosine similarity |
| Narrative generation | Google Gemini Flash | AI analyst note generation (free tier) |
| Price data | nsepy | NSE/BSE historical OHLCV + abnormal return calculation |
| Transcript source  | NSE corporate filings (Analysts/Institutional Investor Meet) | SEBI-mandated concall transcript filings |
| PDF download       | Selenium + Chromium + requests session | Bypasses NSE infrastructure-level blocks |
| Storage | SQLite + pandas | Transcript, score, and narrative cache |
| Dashboard | Plotly Dash | Interactive web application |
| Statistical analysis | statsmodels + scipy | OLS regression, p-values, confidence intervals |
| Deployment | Render.com | Public live URL (free tier) |
| Forward guidance | yya518/FinBERT-FLS | Forward-looking statement classification |
| Browser automation | Selenium + Chromium | NSE filing download (bypasses infrastructure-level blocks) |

---

## About

Built by [Priyanshi Modi](https://linkedin.com/in/priyanshi-modi-97bb67318) — CS + Economics undergraduate at BITS Pilani Goa, with research interests at the intersection of NLP and quantitative finance.

This project was developed independently as part of a broader effort to apply transformer-based NLP to financial signal extraction — a domain where the gap between publicly available information and actionable insight remains large.

---

*Inspired by the conviction that executive language is itself a financial signal — one that goes largely unquantified.*
