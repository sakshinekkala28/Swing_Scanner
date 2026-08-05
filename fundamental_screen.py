"""
fundamental_screen.py
=====================
"NO-TRADE" fundamental filter for the swing scanner.

Runs BEFORE the technical scan and excludes structurally broken / governance-risky
companies from the 15%/30-day swing universe. Stocks that fail this gate never
reach the technical backtest — protecting you from the classic "technically
perfect but fundamentally doomed" trap (e.g. a bullish smallcap with 80%
promoter pledge and negative operating cash flow).

Philosophy — it is a NO-TRADE filter, not a stock picker.
  * Missing data = PASS with warning (never fail on absence, unless strict_mode)
  * Financials (banks/NBFCs/HFCs/insurers) skip leverage/interest-cover checks
  * Thresholds are LENIENT — kick out clearly broken, not marginally weak.
  * Governance data yfinance can't provide (promoter pledge, auditor qualification)
    is read from an optional `governance_overrides.csv` supplied by the user.

Data sources (priority order):
  1. governance_overrides.csv  (user-supplied — highest priority)
  2. yfinance .info             (valuation & quality metrics)
  3. yfinance quarterly_income_stmt / income_stmt (growth & interest cover)

Public API:
    load_overrides(path)              -> dict of manual data
    fetch_fundamentals(ticker)        -> raw fields for one stock (cached 24h)
    screen_universe(tickers, secmap, config, cb) -> (results, sector_medians)
    screen_fundamentals(bare, sector, fund, medians, ov, cfg) -> single-stock verdict
"""

import os
import io
import re
import time
import numpy as np
import pandas as pd
import streamlit as st

try:
    import yfinance as yf
except Exception:
    yf = None

# Screener.in scraping — reuses the session/prime helpers already in
# governance_fetcher.py (curl_cffi chrome124 impersonation bypasses Screener's
# bot filter). Fetches are gated behind HAVE_SCREENER; if the module isn't
# importable (missing curl_cffi), Screener enrichment silently skips and we
# fall back to yfinance-only data.
try:
    from governance_fetcher import (
        _make_session as _sc_session,
        _prime_screener as _sc_prime,
        _fetch_screener_with_retry as _sc_fetch,
    )
    HAVE_SCREENER = True
except Exception:
    HAVE_SCREENER = False


# ======================================================================================
#  1. CONFIGURATION — thresholds for the NO-TRADE gate
# ======================================================================================
DEFAULT_FUNDA_CONFIG = {
    # ---- master switches per pillar ----
    "valuation_enabled":   False,   # OFF: momentum swings can carry rich multiples
    "quality_enabled":     True,    # ON:  the real "broken business" filter
    "growth_enabled":      False,   # OFF: turnarounds can have neg growth
    "governance_enabled":  True,    # ON:  Indian smallcap-specific risks (pledge etc.)
    "ownership_enabled":   False,   # OFF: yfinance ownership data is unreliable for IN
    "trend_enabled":       True,    # ON:  NEW — catches chronic downtrenders + illiquid names

    # ---- Trend & Liquidity (NEW pillar, Aug-2026) ----
    # Rationale: 5 stocks (RBLBANK, SAIL, INDUSINDBK, PHOENIXLTD, IREDA) lost
    # money under every exit strategy in 5-strategy testing. The signal engine
    # kept firing on them because they had brief bounces above 200-DMA — but
    # they are secular downtrenders where every bounce fails. This pillar
    # rejects them at the fundamental gate BEFORE the technical scanner sees
    # them, using stock's OWN price behaviour (no yfinance fundamentals needed
    # for these three checks — they use the daily bars already in memory).
    "min_12m_return_%":    -15.0,   # secular downtrend floor (12mo total return)
    "min_sma200_slope_%":    0.0,   # 200-DMA must NOT be declining (over last 60 sess)
    "min_avg_turnover_cr":   5.0,   # ₹5 crore floor: illiquid stocks unexecutable at size

    # ---- Valuation ----
    "pe_absolute_max":        100.0,   # reject only if BOTH abs AND sector-rel fail
    "pe_sector_multiple_max":   3.0,   # P/E > 3× sector median AND > abs max → reject
    "pb_absolute_max":         15.0,   # warns (not rejects)
    "ev_ebitda_max":           30.0,   # EV/EBITDA absolute cap (with sector-rel check)
    "ev_ebitda_sector_multiple_max": 3.0,  # AND > 3× sector median → reject
    "peg_max":                  3.0,   # PEG > 3 → reject (growth doesn't justify multiple)

    # ---- Quality (non-financials get leverage checks too) ----
    "roe_min_%":                5.0,   # chronic sub-5% ROE → reject
    "roce_min_%":              10.0,   # capital destruction floor (non-financials only)
    "debt_to_equity_max":       3.0,   # extreme leverage (non-financials only)
    "interest_cover_min":       1.5,   # EBIT/Interest floor (non-financials only)
    "current_ratio_min":        0.8,   # liquidity floor (non-financials only)

    # ---- Growth ----
    "yoy_rev_decline_max_%":  -20.0,   # reject if YoY REV decline worse than this ...
    "yoy_rev_decline_streak":    2,    # ... for at least N consecutive quarters
    "pat_yoy_decline_max_%":  -25.0,   # reject if YoY PAT decline worse than this ...
    "pat_yoy_decline_streak":    2,    # ... for at least N consecutive quarters

    # ---- Governance (needs governance_overrides.csv for pledge / auditor / RPT) ----
    "promoter_pledge_max_%":   40.0,   # pledge > 40% → reject (strict = 25)
    "promoter_holding_min_%":  15.0,   # promoter < 15% → warn (widely-held / exited)
    "flag_auditor_qualified":  True,   # auditor qualification → reject
    "flag_rpt_concern":        True,   # related-party transactions flagged → reject

    # ---- Ownership flow (from override CSV — quarterly FII / DII / MF delta) ----
    "fii_delta_qoq_min_pp":    -3.0,   # FII holding drop > 3pp QoQ → warn
    "dii_delta_qoq_min_pp":    -3.0,   # DII holding drop > 3pp QoQ → warn
    "mf_delta_qoq_min_pp":     -3.0,   # MF  holding drop > 3pp QoQ → warn

    # ---- Missing-data policy ----
    "strict_mode": False,              # True = no data ⇒ reject; False = no data ⇒ warn
}

# NSE Industry values corresponding to LENDERS / INSURERS.
# For these, D/E, interest cover, current ratio checks are skipped.
# (Banks by definition run 8–12× "D/E"; the metric is meaningless for them.)
FINANCIAL_SECTORS = {
    "Financial Services", "Financial Services (Banks)", "Insurance",
    "Banks", "Housing Finance", "NBFC", "Non-Banking Financial Company",
    "Capital Markets",
}


# ======================================================================================
#  2. OVERRIDES — user-supplied governance data yfinance can't provide
# ======================================================================================
def load_overrides(path: str = "governance_overrides.csv") -> dict:
    """Load user-supplied governance data from a CSV next to this file.
    Missing file → {}. Never raises.

    Expected columns (only 'ticker' is required):
        ticker, promoter_pledge_pct, promoter_holding_pct,
        fii_delta_qoq, auditor_qualified, note

    Returns {TICKER_UPPER: {field: value}} — .NS / .BO suffix stripped.
    """
    full = path if os.path.isabs(path) else \
           os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    if not os.path.exists(full):
        return {}
    try:
        df = pd.read_csv(full, comment="#", skip_blank_lines=True)
        df.columns = [c.strip().lower() for c in df.columns]
        if "ticker" not in df.columns:
            return {}
        out = {}
        for _, row in df.iterrows():
            t = str(row["ticker"]).strip().upper() \
                                  .replace(".NS", "").replace(".BO", "")
            if not t:
                continue
            rec = {}
            for k in ("promoter_pledge_pct", "promoter_holding_pct",
                     "fii_delta_qoq", "dii_delta_qoq", "mf_delta_qoq",
                     "auditor_qualified", "rpt_concern"):
                v = row.get(k, np.nan)
                if pd.notna(v):
                    rec[k] = int(v) if k in ("auditor_qualified", "rpt_concern") \
                                    else float(v)
            out[t] = rec
        return out
    except Exception:
        return {}


# ======================================================================================
#  3. FETCH — pull raw fundamentals from yfinance (24-hour cache)
# ======================================================================================
def _num(x):
    """Coerce to float; return np.nan on failure or non-finite."""
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


# ======================================================================================
#  SCREENER.IN ENRICHMENT  (Aug-2026, Option C v3 — authoritative D/E, ROE, ROCE)
# ======================================================================================
# Screener.in is the go-to Indian-equity data source (matches Groww / Moneycontrol
# conventions). We already scrape it for governance (promoter pledge). This module
# extends that to fundamentals — ROE, ROCE come from the top-of-page "ratios card";
# D/E is computed from the balance-sheet section as Borrowings / (EqCapital + Reserves)
# which is exactly what Groww shows for consolidated D/E.
#
# Fetch strategy — try /consolidated/ FIRST (matches Groww's group-level view for
# stocks with subsidiaries like TVSMOTOR, ADANIGREEN, RELIANCE); fall back to the
# main /company/{slug}/ page for standalone-only companies (banks etc.).
#
# Failure modes handled: HAVE_SCREENER=False (curl_cffi missing), 4xx/5xx from
# Screener, HTML parse failure. In all failure modes, returns {} → upstream
# transparently falls back to yfinance BS-first (previous behavior).

_screener_session = None    # lazy-init singleton — created on first use
_screener_primed = False

def _get_screener_session():
    """Lazy-init the impersonated curl_cffi session. Reused across all lookups."""
    global _screener_session, _screener_primed
    if not HAVE_SCREENER:
        return None
    if _screener_session is None:
        _screener_session = _sc_session()
    if _screener_session is not None and not _screener_primed:
        _sc_prime(_screener_session)
        _screener_primed = True
    return _screener_session


_SCREENER_RATIO_RE = re.compile(
    r'<span[^>]*class="name"[^>]*>\s*([^<]+?)\s*</span>[\s\S]{0,200}?'
    r'<span[^>]*class="(?:number|value|nowrap)"[^>]*>\s*([^<]+?)\s*</span>'
)


def _parse_screener_top_ratios(html: str) -> dict:
    """Extract the top-of-page ratios card. Returns {label: value_str}."""
    out = {}
    for k, v in _SCREENER_RATIO_RE.findall(html):
        out[k.strip()] = v.strip()
    return out


def _parse_screener_bs_de(html: str) -> float:
    """Compute D/E from Screener's Balance Sheet section:
       D/E = Borrowings / (Equity Capital + Reserves).
       Returns np.nan on any parse failure."""
    try:
        tables = pd.read_html(io.StringIO(html))
    except Exception:
        return np.nan
    for tbl in tables:
        if tbl.shape[1] < 2 or tbl.shape[0] < 2:
            continue
        fc = tbl.iloc[:, 0].astype(str).str.strip().str.lower()
        if not (fc.str.contains("borrowings", na=False).any()
                and fc.str.contains("reserves", na=False).any()):
            continue
        latest_col = tbl.columns[-1]
        def _val(term):
            mask = fc.str.contains(term, na=False)
            if not mask.any():
                return None
            v = tbl.loc[mask.idxmax(), latest_col]
            try:
                return float(str(v).replace(",", "").replace("+", ""))
            except (ValueError, TypeError):
                return None
        b = _val("borrowings"); e = _val("equity capital"); r = _val("reserves")
        if b is not None and e is not None and r is not None and (e + r) > 0:
            return round(b / (e + r), 2)
        return np.nan
    return np.nan


def _parse_screener_quarterly(html: str) -> dict:
    """Parse the Quarterly Results table. Returns lists of last-4-quarters-of-
    latest-year YoY growth for Sales and Net Profit, matching the yfinance
    yoy_rev_growth_recent / yoy_pat_growth_recent structure.
    Also returns TTM sales & TTM PAT for optional downstream use."""
    out = {"yoy_rev_growth": [], "yoy_pat_growth": [],
           "ttm_sales": np.nan, "ttm_pat": np.nan}
    try:
        tables = pd.read_html(io.StringIO(html))
    except Exception:
        return out
    for tbl in tables:
        if tbl.shape[1] < 6 or tbl.shape[0] < 3:
            continue
        fc = tbl.iloc[:, 0].astype(str).str.strip().str.lower()
        has_rev = fc.str.contains("sales", na=False).any() or fc.str.contains("revenue", na=False).any()
        has_np  = fc.str.contains("net profit", na=False).any()
        if not (has_rev and has_np):
            continue

        def _row_series(term):
            mask = fc.str.contains(term, na=False)
            if not mask.any():
                return None
            row = tbl.iloc[mask.idxmax(), 1:]           # skip label column
            vals = []
            for v in row:
                try:
                    vals.append(float(str(v).replace(",", "").replace("+", "").replace("%", "")))
                except (ValueError, TypeError):
                    vals.append(np.nan)
            return pd.Series(vals)

        sales = _row_series("sales") if fc.str.contains("sales", na=False).any() \
                else _row_series("revenue")
        pat   = _row_series("net profit")
        if sales is None or pat is None:
            break

        # YoY growth for latest up-to-4 quarters (need 5+ quarters available)
        def _yoy_series(s):
            s = s.dropna().reset_index(drop=True)
            if len(s) < 5:
                return []
            yoy = []
            n = len(s)
            for i in range(n - 1, 3, -1):
                prev, cur = s.iloc[i - 4], s.iloc[i]
                if pd.notna(prev) and prev != 0:
                    if cur < 0 and prev < 0:
                        # both negative — signed change vs |prev|
                        yoy.append(float((cur - prev) / abs(prev) * 100))
                    else:
                        yoy.append(float((cur - prev) / abs(prev) * 100))
                if len(yoy) >= 4:
                    break
            return yoy

        out["yoy_rev_growth"] = _yoy_series(sales)
        out["yoy_pat_growth"] = _yoy_series(pat)

        # TTM (last 4 quarters)
        if sales.dropna().shape[0] >= 4:
            out["ttm_sales"] = float(sales.dropna().iloc[-4:].sum())
        if pat.dropna().shape[0] >= 4:
            out["ttm_pat"]   = float(pat.dropna().iloc[-4:].sum())
        break
    return out


def _pct_str_to_float(s: str) -> float:
    """Screener values like '34.4' or '-30.2 %' → float, np.nan on fail."""
    if not s or not isinstance(s, str):
        return np.nan
    x = s.strip().replace("%", "").replace(",", "").replace("+", "")
    if x in ("", "-", "--"):
        return np.nan
    try:
        return float(x)
    except ValueError:
        return np.nan


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def fetch_screener_fundamentals(ticker_bare: str) -> dict:
    """Fetch fundamentals from Screener.in for one stock. Returns:
       {"de": float, "roe": float, "roce": float,        (quality)
        "pe": float, "pb": float, "book_value": float,   (valuation)
        "yoy_rev_growth": list, "yoy_pat_growth": list,  (growth — last 4 Q)
        "ttm_sales": float, "ttm_pat": float,
        "src": "screener_consol" | "screener_std" | ...}
       Empty {} on any failure — caller falls back to yfinance.
    """
    if not HAVE_SCREENER:
        return {}
    session = _get_screener_session()
    if session is None:
        return {}
    slug = ticker_bare.upper().strip().replace(".NS", "").replace(".BO", "")

    for url_suffix, src_tag in (("/consolidated/", "screener_consol"),
                                 ("/",               "screener_std")):
        url = f"https://www.screener.in/company/{slug}{url_suffix}"
        try:
            r = session.get(url, timeout=25)
        except Exception:
            continue
        if r.status_code != 200 or len(r.text) < 5000:
            continue
        html = r.text
        top = _parse_screener_top_ratios(html)

        # Quality (already existed)
        roe  = _pct_str_to_float(top.get("ROE"))
        roce = _pct_str_to_float(top.get("ROCE"))
        de   = _parse_screener_bs_de(html)

        # Valuation (NEW)
        pe   = _pct_str_to_float(top.get("Stock P/E"))
        book_value = _pct_str_to_float(top.get("Book Value"))
        cur_price  = _pct_str_to_float(top.get("Current Price"))
        pb = np.nan
        if pd.notna(cur_price) and pd.notna(book_value) and book_value > 0:
            pb = round(cur_price / book_value, 2)

        # Growth (NEW)
        q = _parse_screener_quarterly(html)

        # Return if we got anything useful
        if any(pd.notna(v) for v in (roe, roce, de, pe)) or q["yoy_rev_growth"]:
            return {
                "de": de, "roe": roe, "roce": roce,
                "pe": pe, "pb": pb, "book_value": book_value,
                "yoy_rev_growth": q["yoy_rev_growth"],
                "yoy_pat_growth": q["yoy_pat_growth"],
                "ttm_sales": q["ttm_sales"], "ttm_pat": q["ttm_pat"],
                "src": src_tag,
            }
    return {}


# ======================================================================================
#  BS-FIRST HELPERS  (Aug-2026, Option C — data source fix)
# ======================================================================================
def _first_present(df: pd.DataFrame, keys: tuple, col):
    """Return the first non-null value in `df` for the first matching row-label
    from `keys`, at column `col`. Robust to yfinance's label variants."""
    if df is None or df.empty:
        return np.nan
    for k in keys:
        if k in df.index:
            v = _num(df.loc[k, col])
            if pd.notna(v):
                return v
    return np.nan


# Label variants Yahoo has used across versions — try them in preference order.
BS_TOTAL_DEBT_KEYS       = ("Total Debt", "TotalDebt")
BS_LONG_TERM_DEBT_KEYS   = ("Long Term Debt", "LongTermDebt")
BS_SHORT_TERM_DEBT_KEYS  = ("Current Debt", "Short Long Term Debt", "ShortLongTermDebt")
BS_STOCK_EQUITY_KEYS     = ("Stockholders Equity", "StockholdersEquity",
                            "Common Stock Equity", "CommonStockEquity",
                            "Total Stockholder Equity")
BS_TOTAL_ASSETS_KEYS     = ("Total Assets", "TotalAssets")
BS_CURR_LIAB_KEYS        = ("Current Liabilities", "Total Current Liabilities",
                            "CurrentLiabilities")
BS_CURR_ASSETS_KEYS      = ("Current Assets", "Total Current Assets",
                            "CurrentAssets")
ANN_NET_INCOME_KEYS      = ("Net Income", "Net Income Common Stockholders",
                            "NetIncome", "Net Income Continuous Operations",
                            "Net Income From Continuing Operations")


def _metrics_from_bs_ann(t, info: dict, ticker_bare: str = "") -> dict:
    """Compute D/E, ROE, current_ratio. Priority order per field:
        1. SCREENER.IN (consolidated) — matches Groww / Moneycontrol exactly
        2. QUARTERLY yfinance (TTM)   — matches Groww on most stocks
        3. ANNUAL yfinance            — fallback if quarterly not enough
        4. yfinance INFO              — final fallback

    Each field's origin is tagged in `src_*` for audit.

    OPTION-C-v2 (Aug-2026): switched ROE to TTM (last 4 quarterly NI sum
    over latest quarterly equity) — matches Groww's convention. Same for
    interest_cover downstream. TTM catches recent turnarounds/deteriorations
    that annual reports lag 6-15 months behind on. Also adds a `data_stale`
    flag when quarterly history < 4 quarters (post-listing / post-demerger
    stubs like TMPV) so upstream can DEMOTE quality-check rejections to warn.

    Formulas (Groww convention):
        D/E            = Total Debt / Stockholders Equity
        ROE (TTM)      = Sum(Net Income, last 4Q) / latest_quarterly_equity * 100
        Current Ratio  = Current Assets / Current Liabilities
    """
    de = roe = cr = np.nan
    src_de = src_roe = src_cr = "unavail"
    data_stale = False                    # NEW: True for post-demerger / new-listing

    # ---- Priority 1: Screener.in (matches Groww) ---------------------------
    # Only D/E, ROE, ROCE — Screener doesn't consistently expose current_ratio.
    # ROCE is computed here and returned via the dict; caller stores it.
    roce_screener = np.nan
    if ticker_bare:
        try:
            sc = fetch_screener_fundamentals(ticker_bare)
        except Exception:
            sc = {}
        if sc:
            if pd.notna(sc.get("de")) and sc["de"] > 0:
                de = sc["de"]; src_de = sc.get("src", "screener")
            if pd.notna(sc.get("roe")):
                roe = sc["roe"]; src_roe = sc.get("src", "screener")
            if pd.notna(sc.get("roce")):
                roce_screener = sc["roce"]

    # ---- yfinance sources for anything Screener didn't provide -------------
    try:
        bs = t.balance_sheet
        ann = t.income_stmt
        q_inc = t.quarterly_income_stmt
        q_bs = t.quarterly_balance_sheet
    except Exception:
        bs = ann = q_inc = q_bs = None

    # Prefer quarterly BS for D/E and CR (more recent than annual)
    bs_use = q_bs if (q_bs is not None and not q_bs.empty) else bs

    # --- D/E (only if Screener didn't provide) ---
    if pd.isna(de) and bs_use is not None and not bs_use.empty:
        latest_b = bs_use.columns[0]
        total_debt = _first_present(bs_use, BS_TOTAL_DEBT_KEYS, latest_b)
        if pd.isna(total_debt):
            lt = _first_present(bs_use, BS_LONG_TERM_DEBT_KEYS, latest_b)
            st_ = _first_present(bs_use, BS_SHORT_TERM_DEBT_KEYS, latest_b)
            if pd.notna(lt) or pd.notna(st_):
                total_debt = (lt if pd.notna(lt) else 0.0) + (st_ if pd.notna(st_) else 0.0)
        equity = _first_present(bs_use, BS_STOCK_EQUITY_KEYS, latest_b)
        if pd.notna(total_debt) and pd.notna(equity) and equity > 0:
            de = round(float(total_debt) / float(equity), 2)
            src_de = "bs_q" if bs_use is q_bs else "bs_ann"

    if pd.isna(de):
        de_raw = _num(info.get("debtToEquity"))
        if pd.notna(de_raw):
            de = round(de_raw / 100.0, 2)
            src_de = "info"

    # --- ROE (only if Screener didn't provide — TTM preferred) ---
    ttm_ni = np.nan
    n_quarters = 0
    if pd.notna(roe):
        # Screener supplied ROE. Still run outlier detection on quarterly NI
        # for the stale-data flag (needed to demote quality checks on TMPV).
        if q_inc is not None and not q_inc.empty:
            for k in ANN_NET_INCOME_KEYS:
                if k in q_inc.index:
                    series = q_inc.loc[k].dropna()
                    if len(series) >= 4:
                        q_abs = series.iloc[:4].abs()
                        med_abs = float(q_abs.median()); max_abs = float(q_abs.max())
                        if med_abs > 0 and max_abs / med_abs > 5.0:
                            data_stale = True
                    elif len(series) >= 1:
                        data_stale = True
                    break
    elif q_inc is not None and not q_inc.empty:
        for k in ANN_NET_INCOME_KEYS:
            if k in q_inc.index:
                series = q_inc.loc[k].dropna()
                if len(series) >= 4:
                    quarters = series.iloc[:4]
                    ttm_ni = float(quarters.sum())     # newest 4 quarters
                    n_quarters = 4
                    # OUTLIER DETECTION: post-demerger / spin-off / restructure
                    # gains often appear as a single-quarter NI 5-10× the median.
                    # E.g. TMPV Q1 net income was ₹80,000+ Cr from demerger
                    # opening-balance gain — pollutes TTM ROE massively.
                    q_abs = quarters.abs()
                    med_abs = float(q_abs.median())
                    max_abs = float(q_abs.max())
                    if med_abs > 0 and max_abs / med_abs > 5.0:
                        data_stale = True
                elif len(series) >= 1:
                    # Post-listing stub — flag as stale but still compute
                    ttm_ni = float(series.sum())
                    n_quarters = len(series)
                    data_stale = True
                break

    # Equity for the ROE denominator: latest quarterly if available, else annual avg
    eq_denom = np.nan
    if bs_use is not None and not bs_use.empty:
        eq_denom = _first_present(bs_use, BS_STOCK_EQUITY_KEYS, bs_use.columns[0])
    # If we're using annual BS with 2+ years, average for a better denominator
    if (bs_use is bs) and bs is not None and bs.shape[1] >= 2:
        eq_prev = _first_present(bs, BS_STOCK_EQUITY_KEYS, bs.columns[1])
        if pd.notna(eq_denom) and pd.notna(eq_prev):
            eq_denom = (eq_denom + eq_prev) / 2.0

    if pd.notna(ttm_ni) and pd.notna(eq_denom) and eq_denom > 0 and n_quarters >= 1:
        roe = round(float(ttm_ni) / float(eq_denom) * 100.0, 2)
        src_roe = f"ttm(q={n_quarters})" if n_quarters >= 4 else f"stub(q={n_quarters})"

    # Fallback: annual NI / annual equity
    if pd.isna(roe) and ann is not None and not ann.empty and bs is not None and not bs.empty:
        ni_ann = _first_present(ann, ANN_NET_INCOME_KEYS, ann.columns[0])
        eq_ann_last = _first_present(bs, BS_STOCK_EQUITY_KEYS, bs.columns[0])
        eq_ann_avg = eq_ann_last
        if bs.shape[1] >= 2:
            eq_prev = _first_present(bs, BS_STOCK_EQUITY_KEYS, bs.columns[1])
            if pd.notna(eq_ann_last) and pd.notna(eq_prev):
                eq_ann_avg = (eq_ann_last + eq_prev) / 2.0
        if pd.notna(ni_ann) and pd.notna(eq_ann_avg) and eq_ann_avg > 0:
            roe = round(float(ni_ann) / float(eq_ann_avg) * 100.0, 2)
            src_roe = "annual"

    if pd.isna(roe):
        roe_raw = _num(info.get("returnOnEquity"))
        if pd.notna(roe_raw):
            roe = round(roe_raw * 100.0, 2)
            src_roe = "info"

    # Sanity guard: absurd ROE magnitude (e.g. post-demerger stubs) → mark stale
    if pd.notna(roe) and abs(roe) > 100.0:
        data_stale = True

    # --- Current Ratio (quarterly BS preferred) ---
    if bs_use is not None and not bs_use.empty:
        latest_b = bs_use.columns[0]
        ca = _first_present(bs_use, BS_CURR_ASSETS_KEYS, latest_b)
        cl = _first_present(bs_use, BS_CURR_LIAB_KEYS, latest_b)
        if pd.notna(ca) and pd.notna(cl) and cl > 0:
            cr = round(float(ca) / float(cl), 2)
            src_cr = "bs_q" if bs_use is q_bs else "bs_ann"

    if pd.isna(cr):
        cr_raw = _num(info.get("currentRatio"))
        if pd.notna(cr_raw):
            cr = round(cr_raw, 2)
            src_cr = "info"

    return {"de": de, "roe": roe, "current_ratio": cr,
            "roce_screener": roce_screener,           # NEW — used to override yfinance ROCE
            "src_de": src_de, "src_roe": src_roe, "src_cr": src_cr,
            "data_stale": data_stale}


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)  # 24h — fundamentals don't move intraday
def fetch_fundamentals(ticker_yahoo: str, include_trend: bool = True) -> dict:
    """Pull the raw fundamental fields we need for the gate. Never raises;
    on failure returns {'_error': ...}. Missing fields are np.nan.

    Note: yfinance ROE is decimal form (0.15 = 15%); D/E is percent form
    (128 = 1.28). We normalise both to human-readable units below.
    """
    if yf is None:
        return {"_error": "yfinance unavailable"}
    out = {}
    try:
        t = yf.Ticker(ticker_yahoo)
        info = t.info if hasattr(t, "info") else {}
    except Exception as e:
        return {"_error": f"info fetch: {str(e)[:60]}"}

    # ---- Fetch Screener bundle ONCE (cached 24h). Reused for valuation,
    #      quality (via _metrics_from_bs_ann), and growth blocks below. ----
    _bare = ticker_yahoo.upper().replace(".NS", "").replace(".BO", "")
    try:
        _sc = fetch_screener_fundamentals(_bare)
    except Exception:
        _sc = {}
    out["_source_valuation"] = "screener" if _sc else "yfinance"
    out["_source_growth"]    = "screener" if _sc and _sc.get("yoy_rev_growth") else "yfinance"

    # ---- Valuation — Screener FIRST, yfinance fallback ----
    if _sc and pd.notna(_sc.get("pe")):
        out["pe"] = _sc["pe"]
    else:
        out["pe"] = _num(info.get("trailingPE"))
    if _sc and pd.notna(_sc.get("pb")):
        out["pb"] = _sc["pb"]
    else:
        out["pb"] = _num(info.get("priceToBook"))
    out["fwd_pe"]    = _num(info.get("forwardPE"))
    out["ev_ebitda"] = _num(info.get("enterpriseToEbitda"))    # EV/EBITDA still yfinance
    out["book_value"] = _sc.get("book_value") if _sc else np.nan
    # C6 FIX (Aug-2026): yfinance `pegRatio` is often stale/missing for NSE
    # names (Yahoo computes it from US-style 5-yr trailing EPS growth that
    # doesn't fit Indian quarterly reporting). Kept as a fallback but the
    # gate now prefers `peg_self` computed below from OUR own PAT growth.
    out["peg_yf"]       = _num(info.get("pegRatio"))
    out["peg_self"]     = np.nan                              # populated after PAT growth
    out["trailing_eps"] = _num(info.get("trailingEps"))
    out["forward_eps"]  = _num(info.get("forwardEps"))
    out["market_cap"]   = _num(info.get("marketCap"))

    # ---- Quality — dual-source: BALANCE SHEET FIRST, yfinance-info fallback ----
    # C6/OptC FIX (Aug-2026): yfinance's `info.debtToEquity` uses gross equity
    # (includes minority interest) → systematically UNDERSTATES D/E vs Groww /
    # Screener / moneycontrol (which use pure shareholders' equity). Concrete:
    # ADANIGREEN — yfinance info 3.47, real from BS 5.08, Groww 5.19.
    # We now compute D/E, ROE, current_ratio, ROCE from the balance sheet
    # DIRECTLY (using the same denominator convention as the reference sites)
    # and only fall back to `info` values if the BS parse fails. Every field
    # gets a `_source` tag ("bs" / "info" / "unavail") so rejection audits are
    # trustworthy. See helpers `_metric_from_bs` below fetch_fundamentals().
    _bare = ticker_yahoo.upper().replace(".NS", "").replace(".BO", "")
    _metrics = _metrics_from_bs_ann(t, info, ticker_bare=_bare)   # Screener-first helper
    out["roe"]           = _metrics["roe"]
    out["de"]            = _metrics["de"]
    out["current_ratio"] = _metrics["current_ratio"]
    out["_source_roe"]   = _metrics["src_roe"]
    out["_source_de"]    = _metrics["src_de"]
    out["_source_cr"]    = _metrics["src_cr"]
    out["_data_stale"]   = _metrics["data_stale"]
    # Screener-provided ROCE (matches Groww) — takes precedence over the
    # yfinance BS-computed ROCE further below.
    out["_roce_screener"] = _metrics.get("roce_screener", np.nan)
    # Non-critical margins/FCF stay from info (informational only)
    out["op_margin"]     = _num(info.get("operatingMargins"))
    out["profit_margin"] = _num(info.get("profitMargins"))
    out["fcf"]           = _num(info.get("freeCashflow"))

    # ---- Growth: YoY REVENUE AND PAT over recent quarters ----
    # SCREENER FIRST — its Quarterly Results table exactly matches what
    # Groww/Moneycontrol show. yfinance's quarterly_income_stmt is a fallback
    # (often standalone-only, sometimes missing recent quarters — see INDIGO
    # diagnostic where yfinance TTM NI showed only -48 Cr vs Groww's -4800 Cr).
    out["yoy_rev_growth_recent"] = []
    out["yoy_pat_growth_recent"] = []
    if _sc and _sc.get("yoy_rev_growth"):
        out["yoy_rev_growth_recent"] = list(_sc["yoy_rev_growth"])
    if _sc and _sc.get("yoy_pat_growth"):
        out["yoy_pat_growth_recent"] = list(_sc["yoy_pat_growth"])
    # If Screener didn't supply either series, fall back to yfinance quarterly
    _need_rev = not out["yoy_rev_growth_recent"]
    _need_pat = not out["yoy_pat_growth_recent"]
    if _need_rev or _need_pat:
        try:
            q = t.quarterly_income_stmt
            if q is not None and not q.empty:
                # Revenue YoY
                if _need_rev and "Total Revenue" in q.index:
                    rev = q.loc["Total Revenue"].dropna().sort_index()
                    if len(rev) >= 5:
                        yoy = []
                        for i in range(len(rev) - 1, 3, -1):
                            prev, cur = rev.iloc[i - 4], rev.iloc[i]
                            if pd.notna(prev) and prev != 0:
                                yoy.append(float((cur / prev - 1) * 100))
                            if len(yoy) >= 4:
                                break
                        out["yoy_rev_growth_recent"] = yoy
                # PAT YoY
                if _need_pat:
                    for pat_key in ("Net Income", "Net Income Common Stockholders",
                                    "NetIncome", "Net Income From Continuing Operations"):
                        if pat_key in q.index:
                            ni = q.loc[pat_key].dropna().sort_index()
                            if len(ni) >= 5:
                                yoy = []
                                for i in range(len(ni) - 1, 3, -1):
                                    prev, cur = ni.iloc[i - 4], ni.iloc[i]
                                    if pd.notna(prev) and pd.notna(cur) and prev != 0:
                                        yoy.append(float((cur - prev) / abs(prev) * 100))
                                    if len(yoy) >= 4:
                                        break
                                out["yoy_pat_growth_recent"] = yoy
                            break
        except Exception:
            pass

    # ---- C6 FIX: SELF-COMPUTED PEG from our own PAT growth ----
    # Formula: PEG = trailing PE / (avg PAT growth over recent quarters).
    # Avoids yfinance's unreliable pegRatio. We only compute when growth is
    # (a) available (>= 2 quarters) and (b) positive (PEG for negative growth
    # is a meaningless number in classic Peter Lynch formulation).
    try:
        pat_growth_list = out.get("yoy_pat_growth_recent") or []
        pe_val = out.get("pe")
        if (pd.notna(pe_val) and pe_val > 0 and len(pat_growth_list) >= 2):
            g_avg = float(np.mean(pat_growth_list[:4]))     # avg of last (up to) 4 quarters
            if g_avg > 0:
                out["peg_self"] = round(float(pe_val) / g_avg, 2)
    except Exception:
        pass

    # ---- Annual income & balance sheet: for Interest Cover AND ROCE ----
    out["interest_cover"] = np.nan
    out["roce"] = np.nan                        # NEW
    try:
        ann = t.income_stmt
        bs  = t.balance_sheet
        # ---- Interest Cover: EBIT / |Interest Expense| ----
        ebit_val = np.nan
        if ann is not None and not ann.empty:
            latest_i = ann.columns[0]
            for k in ("EBIT", "Operating Income", "Ebit"):
                if k in ann.index:
                    ebit_val = _num(ann.loc[k, latest_i])
                    if pd.notna(ebit_val):
                        break
            int_exp = np.nan
            for k in ("Interest Expense", "InterestExpense",
                      "Interest Expense Non Operating"):
                if k in ann.index:
                    int_exp = _num(ann.loc[k, latest_i])
                    if pd.notna(int_exp):
                        break
            if pd.notna(ebit_val) and pd.notna(int_exp) and int_exp != 0:
                out["interest_cover"] = float(ebit_val) / abs(float(int_exp))

        # ---- ROCE (NEW): EBIT / Capital Employed × 100 ----
        # Capital Employed = Total Assets − Current Liabilities
        if pd.notna(ebit_val) and bs is not None and not bs.empty:
            latest_b = bs.columns[0]
            total_assets = np.nan
            for k in ("Total Assets", "TotalAssets"):
                if k in bs.index:
                    total_assets = _num(bs.loc[k, latest_b])
                    if pd.notna(total_assets):
                        break
            curr_liab = np.nan
            for k in ("Current Liabilities", "Total Current Liabilities",
                      "CurrentLiabilities", "Other Current Liabilities"):
                if k in bs.index:
                    curr_liab = _num(bs.loc[k, latest_b])
                    if pd.notna(curr_liab):
                        break
            if pd.notna(total_assets) and pd.notna(curr_liab):
                cap_employed = total_assets - curr_liab
                if cap_employed > 0:
                    out["roce"] = (float(ebit_val) / cap_employed) * 100
    except Exception:
        pass

    # Override yfinance-computed ROCE with Screener's if available (matches
    # Groww/Moneycontrol convention). See fetch_screener_fundamentals.
    if pd.notna(out.get("_roce_screener", np.nan)):
        out["roce"] = float(out["_roce_screener"])

    # ---- Ownership (weak yfinance proxies — real data comes from override CSV) ----
    out["insider_pct_yf"]     = _num(info.get("heldPercentInsiders"))
    out["institution_pct_yf"] = _num(info.get("heldPercentInstitutions"))

    # ---- TREND & LIQUIDITY (NEW, Aug-2026) ----
    # M3 FIX (Aug-2026): only fetch 14-month history when the trend pillar is
    # actually enabled. Prior version ran an extra t.history() call on EVERY
    # stock even when trend_enabled=False, adding ~30-60 min per 2000-stock
    # AllNSE run and heavy yfinance rate-limit pressure.
    out["ret_12m_%"]         = np.nan
    out["sma200_slope_%"]    = np.nan
    out["avg_turnover_cr"]   = np.nan
    if include_trend:
        try:
            hist = t.history(period="14mo", interval="1d", auto_adjust=True)
            if hist is not None and not hist.empty and "Close" in hist.columns:
                close = hist["Close"].dropna()
                vol = hist["Volume"].dropna() if "Volume" in hist.columns else None

                # (1) 12-month total return using ~252 trading days ago as anchor.
                # M4 FIX (Aug-2026): NO first-bar fallback for short-history stocks.
                # Anchoring an IPO to its listing price shows spurious "bullish"
                # returns from post-listing lockup selloffs. If we don't have a
                # full year of history, leave ret_12m_% as NaN → gate treats it
                # as "unknown" (skip check) rather than falsely-bullish (pass).
                if len(close) >= 252:
                    ret_12m = float((close.iloc[-1] / close.iloc[-252]) - 1) * 100
                    out["ret_12m_%"] = ret_12m

                # (2) 200-DMA slope over the last 60 sessions, expressed as % change
                if len(close) >= 200 + 60:
                    sma200 = close.rolling(200).mean().dropna()
                    if len(sma200) >= 60:
                        slope = float((sma200.iloc[-1] / sma200.iloc[-60]) - 1) * 100
                        out["sma200_slope_%"] = slope

                # (3) 20-day average TURNOVER in ₹ crores (Close × Volume / 1e7)
                if vol is not None and len(vol) >= 20:
                    turn = (close * vol).tail(20).mean() / 1e7
                    out["avg_turnover_cr"] = float(turn)
        except Exception:
            pass

    return out


# ======================================================================================
#  4. SECTOR MEDIANS — used for relative-valuation checks
# ======================================================================================
def compute_sector_medians(fundamentals: dict, sector_by_bare: dict) -> dict:
    """From {bare_ticker: raw_fund_dict} + {bare_ticker: sector}, compute the
    per-sector median for the metrics used in relative-valuation checks.

    H9 FIX (Aug-2026): the previous `_ALL` universe-wide fallback for sectors
    with <5 samples mixed banks + FMCG + pharma into one "peer" median — the
    resulting sector-relative check was actively misleading (e.g. a small
    speciality-chemicals name being judged against a universe median dragged
    down by low-PE public-sector banks). The `_ALL` bucket is no longer
    computed. Small-sector stocks (< 5 peers) simply skip the sector-relative
    leg of each check — the absolute-cap leg still applies, but a low
    absolute-cap alone never rejects (screen_fundamentals still uses AND
    logic: BOTH abs cap AND sector-relative must fail).
    """
    fields = ("pe", "pb", "ev_ebitda", "roe", "roce", "de")
    rows = []
    for t, f in fundamentals.items():
        if not isinstance(f, dict) or "_error" in f:
            continue
        rec = {"ticker": t, "sector": sector_by_bare.get(t, "UNKNOWN")}
        for k in fields:
            rec[k] = f.get(k, np.nan)
        rows.append(rec)
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    med = {}
    for sec, sub in df.groupby("sector"):
        if len(sub) < 5 or sec == "UNKNOWN":
            continue                            # H9: small / unknown sectors get no median
        med[sec] = {k: float(sub[k].median()) for k in fields
                    if pd.notna(sub[k].median())}
    return med


# ======================================================================================
#  5. THE GATE — apply the no-trade filter to a single stock
# ======================================================================================
def screen_fundamentals(ticker_bare: str, sector: str,
                        fund: dict, sector_medians: dict,
                        overrides: dict, config: dict) -> dict:
    """Return {status, reasons, warnings, data} where:
       status ∈ {"pass", "reject", "pass_no_data"}
       reasons = list of hard-reject rationales (empty ⇒ pass)
       warnings = list of soft flags (never trigger reject alone)
       data = the numeric fundamentals we actually evaluated
    """
    reasons, warns, data = [], [], {}
    cfg = config
    is_financial = sector in FINANCIAL_SECTORS
    ov = overrides.get(ticker_bare.upper(), {})

    # ---- 5.0 Missing-data policy ----
    if not isinstance(fund, dict) or "_error" in fund:
        if cfg.get("strict_mode"):
            return {"status": "reject", "reasons": ["no fundamental data"],
                    "warnings": [], "data": {}}
        return {"status": "pass_no_data", "reasons": [],
                "warnings": ["no fundamental data (yfinance)"], "data": {}}

    # ---- 5.0b: STALE-DATA POLICY (Option-C-v2) ----
    # If this stock has post-demerger / new-listing stub financials, the
    # quality-pillar numbers can be wildly wrong (e.g. TMPV ROE 72% from
    # accounting artifact, INDIGO TTM NI missing recent losses). Instead of
    # hard-rejecting on questionable data, we DEMOTE quality-pillar rejects
    # to warnings so the stock at least reaches the technical scan.
    # Governance rejects (pledge, auditor) still fire — they use overrides
    # CSV, not yfinance.
    _quality_data_stale = bool(fund.get("_data_stale", False))

    # ---- 5a. VALUATION ----
    if cfg.get("valuation_enabled"):
        # H9 FIX: sector_medians has NO `_ALL` fallback anymore. Small sectors
        # (< 5 peers) get an empty dict — `sec_med.get("pe")` returns None,
        # which short-circuits the sector-relative leg of each check via the
        # `and sec_pe` guards below. Absolute-cap-alone can't reject on its
        # own (the check requires BOTH caps to fail), so a small-sector stock
        # simply passes the valuation gate — the intended behaviour, since we
        # have no reliable peer basis for a relative judgement.
        sec_med = sector_medians.get(sector, {})
        # P/E: absolute + sector-relative
        pe = fund.get("pe")
        if pd.notna(pe) and pe > 0:
            data["pe"] = round(pe, 2)
            sec_pe = sec_med.get("pe")
            if (pe > cfg["pe_absolute_max"] and sec_pe
                    and pe > cfg["pe_sector_multiple_max"] * sec_pe):
                reasons.append(f"P/E {pe:.0f} > abs cap {cfg['pe_absolute_max']:.0f} "
                               f"AND {pe/sec_pe:.1f}× sector median ({sec_pe:.0f})")
        # P/B: absolute (warn)
        pb = fund.get("pb")
        if pd.notna(pb):
            data["pb"] = round(pb, 2)
            if pb > cfg["pb_absolute_max"]:
                warns.append(f"P/B {pb:.1f} > {cfg['pb_absolute_max']:.0f} (rich)")
        # EV/EBITDA: absolute + sector-relative (NEW gate)
        eb = fund.get("ev_ebitda")
        if pd.notna(eb) and eb > 0:
            data["ev_ebitda"] = round(eb, 2)
            sec_eb = sec_med.get("ev_ebitda")
            if (eb > cfg["ev_ebitda_max"] and sec_eb
                    and eb > cfg["ev_ebitda_sector_multiple_max"] * sec_eb):
                reasons.append(f"EV/EBITDA {eb:.0f} > abs cap {cfg['ev_ebitda_max']:.0f} "
                               f"AND {eb/sec_eb:.1f}× sector median ({sec_eb:.0f})")
        # PEG — growth-adjusted P/E
        # C6 FIX (Aug-2026): prefer OUR self-computed PEG (from our own quarterly
        # PAT growth) over yfinance's `pegRatio`, which is often stale/missing/
        # wrongly-scaled for NSE names. Only self-computed PEG triggers a hard
        # REJECT. If we only have yfinance's PEG, we warn instead of rejecting —
        # the data is not trustworthy enough to fail real trades on.
        peg_self = fund.get("peg_self")
        peg_yf   = fund.get("peg_yf")
        if pd.notna(peg_self) and peg_self > 0:
            data["peg_self"] = round(peg_self, 2)
            if peg_self > cfg["peg_max"]:
                reasons.append(f"PEG(self) {peg_self:.1f} > {cfg['peg_max']:.1f} "
                               f"(PE ÷ recent PAT growth — growth doesn't justify multiple)")
        elif pd.notna(peg_yf) and peg_yf > 0:
            data["peg_yf"] = round(peg_yf, 2)
            if peg_yf > cfg["peg_max"]:
                warns.append(f"yfinance PEG {peg_yf:.1f} > {cfg['peg_max']:.1f} "
                             f"(low-confidence — yfinance PEG is unreliable for NSE)")

    # ---- 5b. QUALITY ----
    # Option-C-v2: `_add_quality` demotes REJECT → WARN when quality data is
    # stale (post-demerger / post-listing stubs where yfinance numbers can be
    # wrong by 10-100×). Governance still hard-rejects — pledge/auditor come
    # from override CSV and are reliable regardless of yfinance state.
    def _add_quality(msg):
        if _quality_data_stale:
            warns.append("[stale data] " + msg + " — demoted to warn "
                         "(post-demerger / new-listing yfinance data unreliable)")
        else:
            reasons.append(msg)

    if cfg.get("quality_enabled"):
        roe = fund.get("roe")
        if pd.notna(roe):
            data["roe_%"] = round(roe, 2)
            if roe < cfg["roe_min_%"]:
                _add_quality(f"ROE {roe:.1f}% < {cfg['roe_min_%']:.0f}% floor")
        # Leverage / solvency / capital-efficiency checks — non-financials only
        if not is_financial:
            # ROCE — capital-destruction check
            roce = fund.get("roce")
            if pd.notna(roce):
                data["roce_%"] = round(roce, 2)
                if roce < cfg["roce_min_%"]:
                    _add_quality(f"ROCE {roce:.1f}% < {cfg['roce_min_%']:.0f}% floor "
                                 f"(capital destruction)")
            de = fund.get("de")
            if pd.notna(de):
                data["de"] = round(de, 2)
                if de > cfg["debt_to_equity_max"]:
                    _add_quality(f"D/E {de:.1f} > {cfg['debt_to_equity_max']:.1f} ceiling")
            ic = fund.get("interest_cover")
            if pd.notna(ic):
                data["interest_cover"] = round(ic, 2)
                # M5 FIX (Aug-2026): negative interest_cover means EBIT is
                # negative (loss year). Reason message previously read like a
                # low-but-positive coverage — confusing to the user. Now clearly
                # differentiated.
                if ic <= 0:
                    _add_quality(f"EBIT is negative ({ic:.1f}× interest cover) "
                                 f"— company can't service debt from operations")
                elif ic < cfg["interest_cover_min"]:
                    _add_quality(f"Interest cover {ic:.1f}× < "
                                 f"{cfg['interest_cover_min']:.1f}× (debt-service risk)")
            cr = fund.get("current_ratio")
            if pd.notna(cr):
                data["current_ratio"] = round(cr, 2)
                if cr < cfg["current_ratio_min"]:
                    _add_quality(f"Current ratio {cr:.2f} < "
                                 f"{cfg['current_ratio_min']:.2f} (liquidity strain)")

    # ---- 5c. GROWTH ----
    if cfg.get("growth_enabled"):
        # Revenue decline streak
        yoy = fund.get("yoy_rev_growth_recent", [])
        if yoy:
            data["yoy_rev_%_recent"] = [round(x, 1) for x in yoy[:4]]
            floor = cfg["yoy_rev_decline_max_%"]
            need  = cfg["yoy_rev_decline_streak"]
            streak = 0
            for g in yoy:                       # leading (most-recent) run
                if g < floor:
                    streak += 1
                else:
                    break
            if streak >= need:
                reasons.append(f"Revenue YoY worse than {floor:.0f}% for "
                               f"{streak} consecutive quarters")
        # PAT decline streak (NEW gate) — earnings trajectory
        yoy_pat = fund.get("yoy_pat_growth_recent", [])
        if yoy_pat:
            data["yoy_pat_%_recent"] = [round(x, 1) for x in yoy_pat[:4]]
            floor_p = cfg["pat_yoy_decline_max_%"]
            need_p  = cfg["pat_yoy_decline_streak"]
            streak_p = 0
            for g in yoy_pat:
                if pd.notna(g) and g < floor_p:
                    streak_p += 1
                else:
                    break
            if streak_p >= need_p:
                reasons.append(f"PAT YoY worse than {floor_p:.0f}% for "
                               f"{streak_p} consecutive quarters")

    # ---- 5d. GOVERNANCE (needs governance_overrides.csv for pledge / auditor / RPT) ----
    if cfg.get("governance_enabled"):
        pledge    = ov.get("promoter_pledge_pct")
        prom_hold = ov.get("promoter_holding_pct")
        if pledge is not None:
            data["promoter_pledge_%"] = round(pledge, 2)
            if pledge > cfg["promoter_pledge_max_%"]:
                reasons.append(f"Promoter pledge {pledge:.0f}% > "
                               f"{cfg['promoter_pledge_max_%']:.0f}% (governance risk)")
        else:
            warns.append("promoter pledge unknown (add to governance_overrides.csv)")
        if prom_hold is not None:
            data["promoter_holding_%"] = round(prom_hold, 2)
            if prom_hold < cfg["promoter_holding_min_%"]:
                warns.append(f"Promoter holding {prom_hold:.0f}% < "
                             f"{cfg['promoter_holding_min_%']:.0f}% (widely-held or exited)")
        if cfg.get("flag_auditor_qualified") and ov.get("auditor_qualified"):
            reasons.append("Auditor has qualified opinion (from override CSV)")
        # RPT concern (NEW) — user-flagged related-party transaction risk
        if cfg.get("flag_rpt_concern") and ov.get("rpt_concern"):
            reasons.append("Related-party transactions flagged as concerning "
                           "(from override CSV)")

    # ---- 5e. OWNERSHIP FLOW (override CSV only — yfinance IN data too flaky) ----
    if cfg.get("ownership_enabled"):
        fii_d = ov.get("fii_delta_qoq")
        if fii_d is not None:
            data["fii_delta_qoq_pp"] = round(fii_d, 2)
            if fii_d < cfg["fii_delta_qoq_min_pp"]:
                warns.append(f"FII cut holding by {abs(fii_d):.1f}pp QoQ")
        # DII delta (NEW)
        dii_d = ov.get("dii_delta_qoq")
        if dii_d is not None:
            data["dii_delta_qoq_pp"] = round(dii_d, 2)
            if dii_d < cfg["dii_delta_qoq_min_pp"]:
                warns.append(f"DII cut holding by {abs(dii_d):.1f}pp QoQ")
        # MF delta (NEW)
        mf_d = ov.get("mf_delta_qoq")
        if mf_d is not None:
            data["mf_delta_qoq_pp"] = round(mf_d, 2)
            if mf_d < cfg["mf_delta_qoq_min_pp"]:
                warns.append(f"MF cut holding by {abs(mf_d):.1f}pp QoQ")

    # ---- 5f. TREND & LIQUIDITY (NEW pillar, Aug-2026) ----
    # Purpose — reject stocks that are chronic downtrenders or too illiquid to
    # execute. Directly targets the 5 stocks (RBLBANK, SAIL, INDUSINDBK,
    # PHOENIXLTD, IREDA) that lost money in every exit strategy in the 5-way
    # backtest study.
    #
    # Uses stock's OWN price history (not yfinance fundamentals) so it works
    # even when info fields are missing/stale. All three gates independently
    # fireable; ANY one triggers reject.
    if cfg.get("trend_enabled"):
        # (1) 12-month total return — the secular-downtrend killer
        r12 = fund.get("ret_12m_%")
        if pd.notna(r12):
            data["ret_12m_%"] = round(r12, 1)
            if r12 < cfg["min_12m_return_%"]:
                reasons.append(f"12mo return {r12:+.0f}% < {cfg['min_12m_return_%']:+.0f}% "
                               f"floor (secular downtrend)")
        # (2) 200-DMA slope over last 60 sessions — the "trend has turned" check
        slope = fund.get("sma200_slope_%")
        if pd.notna(slope):
            data["sma200_slope_%"] = round(slope, 2)
            if slope < cfg["min_sma200_slope_%"]:
                reasons.append(f"200-DMA slope {slope:+.1f}% over 60 sessions "
                               f"< {cfg['min_sma200_slope_%']:+.1f}% (200-DMA "
                               f"turning down)")
        # (3) Avg 20-day turnover — the illiquidity filter
        turn = fund.get("avg_turnover_cr")
        if pd.notna(turn):
            data["avg_turnover_cr"] = round(turn, 2)
            if turn < cfg["min_avg_turnover_cr"]:
                reasons.append(f"Avg turnover ₹{turn:.1f}cr < "
                               f"₹{cfg['min_avg_turnover_cr']:.1f}cr floor "
                               f"(too illiquid to execute at size)")

    status = "reject" if reasons else "pass"
    return {"status": status, "reasons": reasons, "warnings": warns, "data": data}


# ======================================================================================
#  6. BATCH — run the gate over the whole universe (two-pass, sector-median-aware)
# ======================================================================================
def _bare(ticker_yahoo: str) -> str:
    return ticker_yahoo.replace(".NS", "").replace(".BO", "").upper()


def screen_universe(tickers_yahoo: list, sector_map: dict, config: dict,
                    progress_cb=None) -> tuple:
    """Two-pass screen over a universe.
       Pass 1: fetch fundamentals for every stock (24h cached).
       Pass 2: compute sector medians, then apply gate to each stock.

    Args
        tickers_yahoo : ["HUDCO.NS", "IRFC.NS", ...]
        sector_map    : {"HUDCO": "Financial Services", ...}  (bare-ticker keys)
        config        : dict, typically DEFAULT_FUNDA_CONFIG with user overrides
        progress_cb   : optional callable(k, n, symbol) for a Streamlit progress bar

    Returns
        (results, sector_medians)
        results = {bare_ticker: {status, reasons, warnings, data}}
        sector_medians = {sector_name: {pe: x, pb: y, ...}}  (empty if valuation off)
    """
    overrides = load_overrides()
    fundamentals = {}
    n = len(tickers_yahoo)
    # M3 FIX: only pull 14mo history when trend pillar is on.
    include_trend = bool(config.get("trend_enabled", True))
    for k, ty in enumerate(tickers_yahoo):
        if progress_cb:
            progress_cb(k, n, ty)
        fundamentals[_bare(ty)] = fetch_fundamentals(ty, include_trend=include_trend)

    sec_by_bare = {_bare(ty): sector_map.get(_bare(ty), "UNKNOWN")
                   for ty in tickers_yahoo}

    medians = (compute_sector_medians(fundamentals, sec_by_bare)
               if config.get("valuation_enabled") else {})

    results = {}
    for ty in tickers_yahoo:
        b = _bare(ty)
        results[b] = screen_fundamentals(b, sec_by_bare[b],
                                         fundamentals.get(b, {}),
                                         medians, overrides, config)
    return results, medians


# ======================================================================================
#  7. SUMMARY helpers for the UI (optional but handy)
# ======================================================================================
def summarize_results(results: dict) -> dict:
    """Roll-up for the sidebar / results header."""
    n = len(results)
    if n == 0:
        return {"total": 0, "pass": 0, "reject": 0, "no_data": 0, "warn_only": 0}
    passed  = sum(1 for r in results.values() if r["status"] == "pass")
    rejected = sum(1 for r in results.values() if r["status"] == "reject")
    no_data  = sum(1 for r in results.values() if r["status"] == "pass_no_data")
    warn_only = sum(1 for r in results.values()
                    if r["status"] == "pass" and r["warnings"])
    return {"total": n, "pass": passed, "reject": rejected,
            "no_data": no_data, "warn_only": warn_only}


def rejects_to_dataframe(results: dict) -> pd.DataFrame:
    """Build a DataFrame of the rejected names for display in the UI."""
    rows = []
    for tick, r in results.items():
        if r["status"] != "reject":
            continue
        rows.append({
            "ticker": tick,
            "reasons": " | ".join(r["reasons"]),
            "warnings": " | ".join(r["warnings"]),
            **r["data"],
        })
    return pd.DataFrame(rows)