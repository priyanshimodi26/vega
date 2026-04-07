import sqlite3
import json
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path
from datetime import datetime
from scipy import stats

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
DB_PATH  = BASE_DIR / "data" / "vega.db"

# ── Features used in regression ──────────────────────────────────
NLP_FEATURES = [
    "prepared_positive",
    "prepared_uncertainty",
    "qa_positive",
    "qa_uncertainty",
    "fls_ratio_qa",
    "specific_fls_ratio",
    "lm_uncertainty",
    "lm_extreme_positive",
    "lm_weak_modal",
]

FUNDAMENTAL_FEATURES = []  # rev_yoy / pat_yoy — add later if scraped from Screener.in

ALL_FEATURES = NLP_FEATURES + FUNDAMENTAL_FEATURES


# ── Database initialiser ──────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS backtest_results (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            run_date      TEXT NOT NULL,
            n_events      INTEGER,
            r_squared_3d  REAL,
            r_squared_yoy REAL,
            coefficients  TEXT,
            correlation   TEXT
        )
    """)
    conn.commit()
    conn.close()


# ── Data loader ───────────────────────────────────────────────────
def load_dataset() -> pd.DataFrame:
    """
    Join all signal tables with prices table.
    Returns a clean DataFrame ready for regression.
    """
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT
            t.ticker,
            t.earnings_date,
            t.fiscal_quarter,

            -- Dependent variables
            p.ar_3d,
            p.ar_yoy,
            p.beta,

            -- FinBERT sentiment features
            s.prepared_positive,
            s.prepared_negative,
            s.prepared_uncertainty,
            s.qa_positive,
            s.qa_negative,
            s.qa_uncertainty,

            -- Guidance features
            g.fls_ratio_prepared,
            g.fls_ratio_qa,
            g.specific_fls_ratio,

            -- MiniLM risk features
            r.liquidity_risk,
            r.demand_softness,
            r.margin_compression,
            r.regulatory_risk,
            r.macro_headwinds,
            r.competitive_threat,
            r.management_evasion,
            r.overconfidence,

            -- Loughran-McDonald features
            r.lm_uncertainty,
            r.lm_litigious,
            r.lm_weak_modal,
            r.lm_strong_modal,
            r.lm_extreme_positive

        FROM transcripts t
        JOIN prices p
            ON t.ticker = p.ticker AND t.earnings_date = p.earnings_date
        JOIN sentiment_scores s ON t.id = s.transcript_id
        JOIN guidance_scores g  ON t.id = g.transcript_id
        JOIN risk_scores r      ON t.id = r.transcript_id
        WHERE p.ar_3d IS NOT NULL
        ORDER BY t.ticker, t.earnings_date
    """, conn)
    conn.close()

    print(f"[BACKTEST] Dataset loaded: {len(df)} events, {len(df.columns)} columns")
    print(f"[BACKTEST] Tickers: {sorted(df['ticker'].unique().tolist())}")
    return df


# ── OLS regression ────────────────────────────────────────────────
def run_regression(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    label: str
) -> dict:
    """
    Run OLS regression of target on features.
    Returns dict with R², adjusted R², p-values, coefficients.
    """
    # Drop rows with any NaN in features or target
    cols = features + [target]
    clean = df[cols].dropna()

    if len(clean) < 20:
        print(f"  [WARN] Too few observations ({len(clean)}) for {label} — skipping")
        return {}

    X = sm.add_constant(clean[features])
    y = clean[target]

    model = sm.OLS(y, X).fit()

    print(f"\n{'='*60}")
    print(f"OLS: {target} ~ {label}")
    print(f"{'='*60}")
    print(f"  N observations : {len(clean)}")
    print(f"  R²             : {model.rsquared:.4f}")
    print(f"  Adjusted R²    : {model.rsquared_adj:.4f}")
    print(f"  F-statistic    : {model.fvalue:.4f}  (p={model.f_pvalue:.4f})")
    print(f"\n  {'Feature':<25} {'Coef':>10} {'p-value':>10} {'Sig':>5}")
    print(f"  {'-'*55}")

    coef_dict = {}
    for feature in features:
        coef  = model.params.get(feature, np.nan)
        pval  = model.pvalues.get(feature, np.nan)
        ci_lo = model.conf_int()[0].get(feature, np.nan)
        ci_hi = model.conf_int()[1].get(feature, np.nan)
        sig   = "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.1 else ""))
        print(f"  {feature:<25} {coef:>10.4f} {pval:>10.4f} {sig:>5}")
        coef_dict[feature] = {
            "coef": round(float(coef), 6),
            "pvalue": round(float(pval), 6),
            "ci_lower": round(float(ci_lo), 6),
            "ci_upper": round(float(ci_hi), 6),
        }

    return {
        "n": len(clean),
        "r_squared": round(model.rsquared, 6),
        "r_squared_adj": round(model.rsquared_adj, 6),
        "f_pvalue": round(model.f_pvalue, 6),
        "coefficients": coef_dict,
    }


# ── Correlation matrix ────────────────────────────────────────────
def compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Pearson correlation matrix between all NLP features
    and the two dependent variables.
    """
    cols = ALL_FEATURES + ["ar_3d", "ar_yoy"]
    available = [c for c in cols if c in df.columns]
    corr = df[available].dropna().corr(method="pearson")
    return corr


# ── Model comparison ──────────────────────────────────────────────
def compare_models(df: pd.DataFrame, target: str):
    """
    Compare three model specifications:
    1. NLP-only
    2. Fundamentals-only (if available)
    3. Combined

    Prints R² comparison table.
    """
    print(f"\n{'='*60}")
    print(f"Model comparison — dependent variable: {target}")
    print(f"{'='*60}")

    # NLP-only
    nlp_result = run_regression(df, NLP_FEATURES, target, "NLP-only")

    print(f"\n  Model R² summary for {target}:")
    print(f"  NLP-only R²      : {nlp_result.get('r_squared', 'N/A')}")
    if FUNDAMENTAL_FEATURES:
        fund_result = run_regression(df, FUNDAMENTAL_FEATURES, target, "Fundamentals-only")
        combined_result = run_regression(df, ALL_FEATURES, target, "Combined")
        print(f"  Fundamentals R²  : {fund_result.get('r_squared', 'N/A')}")
        print(f"  Combined R²      : {combined_result.get('r_squared', 'N/A')}")

    return nlp_result


# ── Pearson correlations vs returns ──────────────────────────────
def print_feature_correlations(df: pd.DataFrame):
    """Print correlation of each NLP feature vs ar_3d and ar_yoy."""
    print(f"\n{'='*60}")
    print("Feature correlations vs abnormal returns")
    print(f"{'='*60}")
    print(f"  {'Feature':<25} {'vs ar_3d':>10} {'vs ar_yoy':>10}")
    print(f"  {'-'*50}")

    for feat in ALL_FEATURES:
        if feat not in df.columns:
            continue
        clean_3d  = df[[feat, "ar_3d"]].dropna()
        clean_yoy = df[[feat, "ar_yoy"]].dropna()

        r_3d  = stats.pearsonr(clean_3d[feat],  clean_3d["ar_3d"])[0]  if len(clean_3d)  > 5 else np.nan
        r_yoy = stats.pearsonr(clean_yoy[feat], clean_yoy["ar_yoy"])[0] if len(clean_yoy) > 5 else np.nan

        print(f"  {feat:<25} {r_3d:>10.4f} {r_yoy:>10.4f}")


# ── Main backtest function ────────────────────────────────────────
def run_backtest() -> dict:
    """
    Main entry point. Loads data, runs regressions, saves results.
    Returns dict with key metrics.
    """
    init_db()
    df = load_dataset()

    # ── Descriptive stats ─────────────────────────────────────────
    print(f"\n[BACKTEST] Dependent variable summary:")
    print(f"  ar_3d  — mean={df['ar_3d'].mean():.4f}  std={df['ar_3d'].std():.4f}  "
          f"min={df['ar_3d'].min():.4f}  max={df['ar_3d'].max():.4f}")
    yoy_clean = df['ar_yoy'].dropna()
    print(f"  ar_yoy — mean={yoy_clean.mean():.4f}  std={yoy_clean.std():.4f}  "
          f"min={yoy_clean.min():.4f}  max={yoy_clean.max():.4f}")

    # ── Feature correlations ──────────────────────────────────────
    print_feature_correlations(df)

    # ── Regressions ───────────────────────────────────────────────
    result_3d  = compare_models(df, "ar_3d")
    result_yoy = compare_models(df, "ar_yoy")

    # ── Correlation matrix ────────────────────────────────────────
    corr = compute_correlations(df)

    # ── Save to DB ────────────────────────────────────────────────
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO backtest_results
            (run_date, n_events, r_squared_3d, r_squared_yoy,
             coefficients, correlation)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        datetime.utcnow().isoformat(),
        len(df),
        result_3d.get("r_squared"),
        result_yoy.get("r_squared"),
        json.dumps({
            "ar_3d":  result_3d.get("coefficients", {}),
            "ar_yoy": result_yoy.get("coefficients", {}),
        }),
        corr.to_json()
    ))
    conn.commit()
    conn.close()

    print(f"\n[BACKTEST] Results saved to DB.")
    print(f"[BACKTEST] ar_3d R²  = {result_3d.get('r_squared', 'N/A')}")
    print(f"[BACKTEST] ar_yoy R² = {result_yoy.get('r_squared', 'N/A')}")

    return {
        "r_squared_3d":  result_3d.get("r_squared"),
        "r_squared_yoy": result_yoy.get("r_squared"),
        "n_events": len(df),
    }


# ── Entry point ───────────────────────────────────────────────────
if __name__ == "__main__":
    run_backtest()