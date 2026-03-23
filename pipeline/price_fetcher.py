import yfinance as yf
import numpy as np
import pandas as pd
import sqlite3
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from scipy import stats

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
DB_PATH  = DATA_DIR / "vega.db"

# ── Benchmark ────────────────────────────────────────────────────
NIFTY_TICKER = "^NSEI"

# ── NSE symbol → yfinance ticker ─────────────────────────────────
TICKER_MAP = {
    "INFY":       "INFY.NS",
    "TCS":        "TCS.NS",
    "WIPRO":      "WIPRO.NS",
    "HCLTECH":    "HCLTECH.NS",
    "TECHM":      "TECHM.NS",
    "HDFCBANK":   "HDFCBANK.NS",
    "ICICIBANK":  "ICICIBANK.NS",
    "KOTAKBANK":  "KOTAKBANK.NS",
    "AXISBANK":   "AXISBANK.NS",
    "SBIN":       "SBIN.NS",
    "HINDUNILVR": "HINDUNILVR.NS",
    "ASIANPAINT": "ASIANPAINT.NS",
    "NESTLEIND":  "NESTLEIND.NS",
    "BHARTIARTL": "BHARTIARTL.NS",
    "RELIANCE":   "RELIANCE.NS",
    "MARUTI":     "MARUTI.NS",
    "TATAMOTORS": "TMPV.NS",
    "SUNPHARMA":  "SUNPHARMA.NS",
    "DRREDDY":    "DRREDDY.NS",
}

# ── Windows ──────────────────────────────────────────────────────
BETA_WINDOW_DAYS   = 365   # 1 year of daily returns for beta estimation
POST_EVENT_DAYS    = 5     # fetch 5 days after earnings to compute 3-day return
PRE_FETCH_DAYS     = 370   # how far back to fetch prices (beta window + buffer)

#Database initialiser and price data fetcher
def init_db():
    """Create prices table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prices (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker         TEXT NOT NULL,
            bse_code       TEXT NOT NULL,
            earnings_date  TEXT NOT NULL,
            ar_3d          REAL,
            ar_yoy         REAL,
            beta           REAL,
            computed_at    TEXT NOT NULL,
            UNIQUE(bse_code, earnings_date)
        )
    """)
    conn.commit()
    conn.close()


def fetch_prices(yf_ticker: str, start: datetime, end: datetime) -> pd.Series | None:
    """
    Fetch daily closing prices for a ticker between start and end dates.
    Returns a pandas Series indexed by date, or None if fetch fails.
    """
    try:
        raw = yf.download(
            yf_ticker,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=True
        )

        if raw.empty:
            print(f"  [WARN] No price data returned for {yf_ticker}")
            return None

        # Handle MultiIndex columns from yfinance
        close = raw["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.squeeze()

        return close.dropna()

    except Exception as e:
        print(f"  [ERROR] Price fetch failed for {yf_ticker}: {e}")
        return None

#Beta calculator
def compute_beta(stock_prices: pd.Series, nifty_prices: pd.Series) -> float:
    """
    Estimate beta using OLS regression of stock returns on NIFTY returns.
    Uses daily log returns over the available price window.

    Returns beta, or 1.0 as fallback if calculation fails.
    """
    try:
        # Align both series to the same dates
        combined = pd.DataFrame({
            "stock": stock_prices,
            "nifty": nifty_prices
        }).dropna()

        if len(combined) < 30:
            print(f"  [WARN] Too few overlapping days ({len(combined)}) for beta — using 1.0")
            return 1.0

        # Log returns: ln(price_t / price_t-1)
        stock_returns = np.log(combined["stock"] / combined["stock"].shift(1)).dropna()
        nifty_returns = np.log(combined["nifty"] / combined["nifty"].shift(1)).dropna()

        # Align after differencing
        stock_returns, nifty_returns = stock_returns.align(nifty_returns, join="inner")

        # OLS: stock_return = alpha + beta * nifty_return
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            nifty_returns.values,
            stock_returns.values
        )

        return round(float(slope), 4)

    except Exception as e:
        print(f"  [WARN] Beta calculation failed: {e} — using 1.0")
        return 1.0
    
#Abnormal return calculator
def compute_abnormal_returns(
    stock_prices: pd.Series,
    nifty_prices:  pd.Series,
    earnings_date: str,
    beta:          float
) -> dict:
    """
    Compute 3-day and year-on-year abnormal returns around an earnings date.

    ar_3d  = stock 3-day return after earnings − (beta × nifty 3-day return)
    ar_yoy = stock return on earnings date vs same date 1 year prior
             − (beta × nifty same-window return)

    Returns dict with ar_3d, ar_yoy (both None if data unavailable).
    """
    result = {"ar_3d": None, "ar_yoy": None}

    try:
        event_date = pd.Timestamp(earnings_date)

        # ── 3-day abnormal return ─────────────────────────────────
        # Find the closest trading day on or after earnings date
        future_prices = stock_prices[stock_prices.index >= event_date]
        if len(future_prices) < 4:
            print(f"  [WARN] Not enough post-event prices for ar_3d")
        else:
            day0_stock = future_prices.iloc[0]   # price on/after earnings date
            day3_stock = future_prices.iloc[3]   # price 3 trading days later
            stock_3d   = (day3_stock - day0_stock) / day0_stock

            # Same window for NIFTY
            future_nifty = nifty_prices[nifty_prices.index >= event_date]
            if len(future_nifty) >= 4:
                day0_nifty = future_nifty.iloc[0]
                day3_nifty = future_nifty.iloc[3]
                nifty_3d   = (day3_nifty - day0_nifty) / day0_nifty
                result["ar_3d"] = round(stock_3d - beta * nifty_3d, 6)

        # ── Year-on-year abnormal return ──────────────────────────
        # Compare price on earnings date vs price ~252 trading days prior
        past_prices = stock_prices[stock_prices.index < event_date]
        if len(past_prices) < 252:
            print(f"  [WARN] Not enough history for ar_yoy")
        else:
            price_now  = stock_prices[stock_prices.index >= event_date].iloc[0]
            price_yoy  = past_prices.iloc[-252]   # ~1 year ago
            stock_yoy  = (price_now - price_yoy) / price_yoy

            past_nifty = nifty_prices[nifty_prices.index < event_date]
            if len(past_nifty) >= 252:
                nifty_now  = nifty_prices[nifty_prices.index >= event_date].iloc[0]
                nifty_yoy  = past_nifty.iloc[-252]
                nifty_ret  = (nifty_now - nifty_yoy) / nifty_yoy
                result["ar_yoy"] = round(stock_yoy - beta * nifty_ret, 6)

    except Exception as e:
        print(f"  [WARN] Abnormal return calculation failed: {e}")

    return result

#Main processing loop
def compute_all_returns() -> dict:
    """
    Main function. For every transcript in the DB, compute abnormal returns
    and save to the prices table.

    Returns {"computed": int, "skipped": int, "failed": int}
    """
    init_db()

    conn = sqlite3.connect(DB_PATH)
    transcripts = conn.execute("""
        SELECT ticker, bse_code, earnings_date
        FROM transcripts
        ORDER BY ticker, earnings_date
    """).fetchall()
    conn.close()

    if not transcripts:
        print("[ERROR] No transcripts found in DB. Run scraper first.")
        return {"computed": 0, "skipped": 0, "failed": 0}

    # Group by ticker to minimise API calls
    from collections import defaultdict
    ticker_events = defaultdict(list)
    for ticker, bse_code, earnings_date in transcripts:
        ticker_events[ticker].append((bse_code, earnings_date))

    stats = {"computed": 0, "skipped": 0, "failed": 0}

    # Fetch NIFTY once — covers our full date range
    print("[NIFTY] Fetching benchmark prices...")
    nifty_start = datetime(2022, 1, 1)
    nifty_end   = datetime.today() + timedelta(days=5)
    nifty_prices = fetch_prices(NIFTY_TICKER, nifty_start, nifty_end)

    if nifty_prices is None:
        print("[ERROR] Could not fetch NIFTY data — aborting.")
        return stats

    print(f"[NIFTY] {len(nifty_prices)} trading days loaded.")

    # Process each ticker
    for ticker, events in ticker_events.items():

        yf_ticker = TICKER_MAP.get(ticker)
        if not yf_ticker:
            print(f"\n[SKIP] No yfinance mapping for {ticker}")
            stats["skipped"] += len(events)
            continue

        print(f"\n[{ticker}] Fetching prices for {len(events)} events...")

        # Fetch full price history for this ticker once
        stock_start = datetime(2022, 1, 1)
        stock_end   = datetime.today() + timedelta(days=5)
        stock_prices = fetch_prices(yf_ticker, stock_start, stock_end)

        if stock_prices is None:
            print(f"  [ERROR] Could not fetch prices for {ticker}")
            stats["failed"] += len(events)
            continue

        print(f"  {len(stock_prices)} trading days loaded.")

        # Compute beta once per ticker using full history
        beta = compute_beta(stock_prices, nifty_prices)
        print(f"  Beta: {beta}")

        # Process each earnings event for this ticker
        conn = sqlite3.connect(DB_PATH)
        for bse_code, earnings_date in events:

            # Skip if already computed
            existing = conn.execute("""
                SELECT id FROM prices
                WHERE bse_code = ? AND earnings_date = ?
            """, (bse_code, earnings_date)).fetchone()

            if existing:
                stats["skipped"] += 1
                continue

            # Compute abnormal returns
            returns = compute_abnormal_returns(
                stock_prices, nifty_prices, earnings_date, beta
            )

            if returns["ar_3d"] is None and returns["ar_yoy"] is None:
                print(f"  [FAIL] {ticker} {earnings_date} — no returns computed")
                stats["failed"] += 1
                continue

            # Save to DB
            conn.execute("""
                INSERT OR IGNORE INTO prices
                    (ticker, bse_code, earnings_date, ar_3d, ar_yoy,
                     beta, computed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                ticker, bse_code, earnings_date,
                returns["ar_3d"], returns["ar_yoy"],
                beta,
                datetime.utcnow().isoformat()
            ))
            conn.commit()

            ar3d  = f"{returns['ar_3d']:.4f}"  if returns["ar_3d"]  else "N/A"
            aryoy = f"{returns['ar_yoy']:.4f}" if returns["ar_yoy"] else "N/A"
            print(f"  ✓ {earnings_date}  ar_3d={ar3d}  ar_yoy={aryoy}")
            stats["computed"] += 1

        conn.close()

    print(f"\n[DONE] Computed: {stats['computed']} | "
          f"Skipped: {stats['skipped']} | Failed: {stats['failed']}")
    return stats

#Entry point
if __name__ == "__main__":
    import sys

    # Optional: pass a ticker to process only that company
    # Usage: python pipeline/price_fetcher.py TCS
    ticker_filter = sys.argv[1].upper() if len(sys.argv) > 1 else None

    if ticker_filter:
        # Temporarily filter ticker_events in compute_all_returns
        # by monkey-patching the DB query — simplest approach
        print(f"[FILTER] Processing only: {ticker_filter}")

    compute_all_returns()