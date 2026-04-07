import sqlite3
import pandas as pd
from pathlib import Path
import dash
from dash import dcc, html

# ── Paths ─────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
DB_PATH  = BASE_DIR / "data" / "vega.db"

# ── App initialisation ────────────────────────────────────────────
app = dash.Dash(
    __name__,
    title="VEGA — Earnings Signal Analyzer",
    suppress_callback_exceptions=True,
)
server = app.server  # for gunicorn deployment

# ── Data loader ───────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    """
    Load the complete joined dataset from SQLite.
    Called once at startup and cached in memory.
    """
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT
            t.ticker, t.earnings_date, t.fiscal_quarter,
            t.prepared_remarks, t.qa_section,
            p.ar_3d, p.ar_yoy, p.beta,
            s.prepared_positive, s.prepared_negative, s.prepared_uncertainty,
            s.qa_positive, s.qa_negative, s.qa_uncertainty,
            g.fls_ratio_prepared, g.fls_ratio_qa,
            g.specific_fls_ratio, g.top_fls_sentences,
            r.liquidity_risk, r.demand_softness, r.margin_compression,
            r.regulatory_risk, r.macro_headwinds, r.competitive_threat,
            r.management_evasion, r.overconfidence,
            r.lm_uncertainty, r.lm_weak_modal, r.lm_strong_modal,
            r.lm_extreme_positive
        FROM transcripts t
        JOIN prices p
            ON t.ticker = p.ticker AND t.earnings_date = p.earnings_date
        JOIN sentiment_scores s ON t.id = s.transcript_id
        JOIN guidance_scores g  ON t.id = g.transcript_id
        JOIN risk_scores r      ON t.id = r.transcript_id
        ORDER BY t.ticker, t.earnings_date
    """, conn)
    conn.close()
    df["earnings_date"] = pd.to_datetime(df["earnings_date"])
    return df


def load_narratives() -> dict:
    """Load cached Gemini narratives as {ticker_date: narrative}."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT ticker, earnings_date, narrative FROM narratives"
    ).fetchall()
    conn.close()
    return {f"{r[0]}_{r[1]}": r[2] for r in rows}


# ── Load data at startup ──────────────────────────────────────────
DF = load_data()
NARRATIVES = load_narratives()
TICKERS = sorted(DF["ticker"].unique().tolist())

RISK_CATEGORIES = [
    "liquidity_risk", "demand_softness", "margin_compression",
    "regulatory_risk", "macro_headwinds", "competitive_threat",
    "management_evasion", "overconfidence",
]

RISK_LABELS = {
    "liquidity_risk":     "Liquidity Risk",
    "demand_softness":    "Demand Softness",
    "margin_compression": "Margin Compression",
    "regulatory_risk":    "Regulatory Risk",
    "macro_headwinds":    "Macro Headwinds",
    "competitive_threat": "Competitive Threat",
    "management_evasion": "Management Evasion",
    "overconfidence":     "Overconfidence",
}

# ── Import layout and callbacks after app is defined ─────────────
from layout import create_layout
from callbacks import register_callbacks

app.layout = create_layout(TICKERS)
register_callbacks(app, DF, NARRATIVES, TICKERS, RISK_CATEGORIES, RISK_LABELS)

if __name__ == "__main__":
    app.run(debug=True, port=8050)