"""
governance_fetcher.py
=====================
Auto-populate governance_overrides.csv from public web sources —
now with FULL NSE universe support.

WHAT CHANGED FROM v1
--------------------
1. Universe integration — pulls the full NSE list (~2000) from
   universe_loader.py. Optional bucket filter (LargeCap / MidCap /
   SmallCap / Nifty500 / AllNSE).
2. Smart ticker resolution — Screener.in uses different slugs than NSE
   for some stocks (ETERNAL is still ZOMATO on Screener, HIMADRI is
   HSCL, etc). Now resolves via:
     - hardcoded aliases (known renames)
     - Screener's live search API (auto-discovers correct slug)
     - fallback to raw ticker
3. Retry with backoff — transient timeouts (like TATAMOTORS in v1) now
   retried up to 3× with exponential backoff.
4. Progressive cache persistence — cache saved every 25 stocks so a
   crash mid-way through a 2000-stock run doesn't lose progress.
5. Resume-from-cache — re-running after a crash picks up where it left
   off (previously fetched tickers still have fresh cache).
6. Adaptive rate limiting — slows down on 429/503, speeds up on success.
7. Failure log — .governance_failures.json records which tickers failed
   and why, so you can inspect after a bulk run.

USAGE
-----
CLI:
    # Refresh Nifty 500 (recommended weekly cron target)
    python governance_fetcher.py --bucket Nifty500

    # Refresh entire NSE (~2000 stocks, ~45 min)
    python governance_fetcher.py --all

    # Refresh specific watchlist (backward-compatible)
    python governance_fetcher.py --tickers TATAMOTORS,VEDL,ADANIENT

    # Force refresh (ignore cache — use after quarterly filing deadlines)
    python governance_fetcher.py --bucket Nifty500 --force

Streamlit:
    from governance_fetcher import refresh_with_streamlit_progress
    refresh_with_streamlit_progress(bare_tickers)   # unchanged API

Python (headless):
    from governance_fetcher import (
        refresh_governance_overrides,
        refresh_full_nse_universe,
    )
    refresh_full_nse_universe(bucket="Nifty500")
"""

import os

# ============================================================================
# VERSION SENTINEL — used to confirm this file contains the pledge-parser fix.
# Verify by running:
#   python -c "import governance_fetcher; print(governance_fetcher.PARSER_VERSION)"
# Expected output: "AUG-2026-PLEDGE-FIX"
# If output is different (or AttributeError), you are running the OLD file.
# ============================================================================
PARSER_VERSION = "AUG-2026-PLEDGE-FIX-V2-BULLET-REGEX"
import io
import json
import time
import re
import random
import argparse
import pandas as pd
import numpy as np

try:
    from curl_cffi import requests as curl_requests
    HAVE_CURL_CFFI = True
except Exception:
    HAVE_CURL_CFFI = False

try:
    import streamlit as st
except Exception:
    st = None

# Universe loader — imported lazily so watchlist mode still works even
# if universe_loader has problems.
try:
    from universe_loader import load_full_universe as _load_universe
    HAVE_UNIVERSE_LOADER = True
except Exception:
    HAVE_UNIVERSE_LOADER = False


# ======================================================================================
#  CONFIG
# ======================================================================================
_HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(_HERE, "governance_overrides.csv")
CACHE_FILE = os.path.join(_HERE, ".governance_cache.json")
FAILURES_LOG = os.path.join(_HERE, ".governance_failures.json")

CACHE_TTL_SEC = 7 * 24 * 60 * 60          # weekly refresh (shareholding is quarterly)
INCREMENTAL_SAVE_EVERY = 25               # save cache every N stocks — crash safety

SCREENER_URL         = "https://www.screener.in/company/{slug}/"
SCREENER_URL_CONSOL  = "https://www.screener.in/company/{slug}/consolidated/"
SCREENER_SEARCH_URL  = "https://www.screener.in/api/company/search/?q={q}"
NSE_SHAREHOLDING_URL = ("https://www.nseindia.com/api/corporate-shareholdings-master"
                       "?index=equities&symbol={ticker}")

BASE_REQUEST_GAP_SEC = 1.2
NSE_REQUEST_GAP_SEC  = 0.5
RETRY_ATTEMPTS       = 3
RETRY_BASE_BACKOFF_S = 2.0                # 2s, 4s, 8s

# Known ticker aliases: NSE symbol → correct Screener slug.
# Add as you discover new mismatches, or let auto-search find them.
SCREENER_ALIASES = {
    # M13 EXPANDED (Aug-2026): known NSE↔Screener slug mismatches. Add here
    # whenever you find another. Live search still catches unknowns as a
    # fallback (see _search_screener_slug), so this list stays a hot-path
    # optimisation, not an exhaustive registry.
    "ETERNAL":       "ZOMATO",                 # rename kept old Screener slug
    "HIMADRI":       "HSCL",                   # Himadri Speciality Chemical
    "MOTHERSON":     "MSUMI",                  # Motherson Sumi Wiring
    "LTIM":          "LTIM",                   # LTI+Mindtree merger
    "VODAFONEIDEA":  "IDEA",                   # NSE symbol is IDEA
    "IDEA":          "IDEA",
    "ADANIENSOL":    "ADANIENSOL",             # canonical
    "TATAMOTORSDVR": "TATAMTRDVR",             # DVR (delisted post-demerger)
}

AUTO_FIELDS   = ("promoter_pledge_pct", "promoter_holding_pct",
                 "fii_delta_qoq", "dii_delta_qoq", "mf_delta_qoq")
MANUAL_FIELDS = ("auditor_qualified", "rpt_concern", "note")
ALL_FIELDS    = ("ticker",) + AUTO_FIELDS + MANUAL_FIELDS


# ======================================================================================
#  CACHE
# ======================================================================================
def _read_cache() -> dict:
    if not os.path.exists(CACHE_FILE):
        return {}
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _write_cache(cache: dict) -> None:
    """Atomic write — prevents cache corruption if killed mid-write."""
    try:
        tmp = CACHE_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
        os.replace(tmp, CACHE_FILE)
    except Exception:
        pass


def _write_failures_log(failures: dict) -> None:
    try:
        with open(FAILURES_LOG, "w", encoding="utf-8") as f:
            json.dump(failures, f, indent=2)
    except Exception:
        pass


def _cache_fresh(entry: dict, ttl_sec: int = CACHE_TTL_SEC) -> bool:
    return bool(entry) and (time.time() - entry.get("ts", 0) < ttl_sec)


# ======================================================================================
#  SESSION
# ======================================================================================
def _make_session():
    if not HAVE_CURL_CFFI:
        return None
    return curl_requests.Session(impersonate="chrome124")


def _prime_screener(session) -> None:
    try:
        session.get("https://www.screener.in/", timeout=10)
        time.sleep(0.3)
    except Exception:
        pass


def _prime_nse(session) -> None:
    try:
        session.get("https://www.nseindia.com", timeout=10)
        time.sleep(0.3)
    except Exception:
        pass


# ======================================================================================
#  SCREENER TICKER RESOLUTION
# ======================================================================================
_slug_cache = {}                          # in-process cache of ticker → slug


def _resolve_screener_slug(ticker: str, cache: dict) -> str:
    """Best-guess Screener slug for an NSE symbol.
    Order: in-memory → hardcoded alias → disk cache → raw ticker."""
    tick_u = ticker.upper().strip()
    if tick_u in _slug_cache:
        return _slug_cache[tick_u]
    if tick_u in SCREENER_ALIASES:
        slug = SCREENER_ALIASES[tick_u]
        _slug_cache[tick_u] = slug
        return slug
    cached_entry = cache.get(tick_u, {})
    if cached_entry.get("slug"):
        slug = cached_entry["slug"]
        _slug_cache[tick_u] = slug
        return slug
    return tick_u


def _search_screener_slug(session, ticker: str) -> str:
    """Live search Screener for the correct slug. Returns None on failure."""
    if session is None:
        return None
    try:
        r = session.get(SCREENER_SEARCH_URL.format(q=ticker), timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        if not isinstance(data, list) or not data:
            return None
        for result in data[:3]:
            url_field = result.get("url", "")
            m = re.match(r"^/company/([^/]+)/?$", url_field)
            if m:
                return m.group(1)
    except Exception:
        pass
    return None


# ======================================================================================
#  SCREENER FETCHER (retry + adaptive backoff + search fallback)
# ======================================================================================
def _fetch_screener_with_retry(session, slug: str,
                               attempts: int = RETRY_ATTEMPTS) -> tuple:
    """Fetch a Screener page with retry. Returns (html_text, error_reason).

    M14 FIX (Aug-2026): 404 is now a HARD FAIL — no retry. A 404 means the
    slug is wrong (URL doesn't exist), not a transient network issue; retrying
    3×2 = 6 more times just wastes ~24s per bad ticker. Callers should treat
    404 as "try search fallback" — which _fetch_screener already does.
    Transient errors (429, 503, connection errors) are still retried."""
    last_err = None
    for attempt in range(attempts):
        for url_template in (SCREENER_URL, SCREENER_URL_CONSOL):
            url = url_template.format(slug=slug)
            try:
                r = session.get(url, timeout=25)
                if r.status_code == 200 and len(r.text) > 5000:
                    return r.text, None
                if r.status_code == 404:
                    # M14: short-circuit — don't retry a wrong URL
                    return None, "404 (slug wrong?)"
                if r.status_code in (429, 503):
                    time.sleep(RETRY_BASE_BACKOFF_S * (2 ** attempt) + random.random())
                    last_err = f"HTTP {r.status_code}"
                else:
                    last_err = f"HTTP {r.status_code}"
            except Exception as e:
                last_err = str(e)[:60]
        if attempt < attempts - 1:
            time.sleep(RETRY_BASE_BACKOFF_S * (2 ** attempt))
    return None, last_err or "unknown"


def _fetch_screener(session, ticker: str, cache: dict) -> tuple:
    """High-level Screener fetch: resolve slug, retry, search fallback.
    Returns (data_dict, error_str, slug_used)."""
    slug = _resolve_screener_slug(ticker, cache)
    html, err = _fetch_screener_with_retry(session, slug)

    # If direct fetch failed AND we used the raw ticker, try live search
    if html is None and slug == ticker.upper():
        search_slug = _search_screener_slug(session, ticker)
        if search_slug and search_slug.upper() != slug:
            slug = search_slug
            _slug_cache[ticker.upper()] = slug   # remember for this run
            html, err = _fetch_screener_with_retry(session, slug)

    if html is None:
        return {}, err or "fetch failed", slug

    data = _parse_screener_shareholding(html)
    if not data:
        return {}, "parse yielded nothing", slug
    return data, None, slug


# ======================================================================================
#  SCREENER HTML PARSER
# ======================================================================================
#  Pledge on Screener.in lives NOT in any table but as a text bullet in the
#  top-of-page Pros/Cons summary, e.g.
#      <li>Promoters have pledged 73.0% of their holding.</li>
#  The bullet appears only when pledge is > 0. Absence therefore means 0.0%
#  (Screener would flag any material pledge in the "cons" block by convention).
_PLEDGE_BULLET_RE = re.compile(
    r"Promoters?\s+have\s+pledged\s+([\d.]+)\s*%\s+of\s+their\s+holding",
    re.IGNORECASE,
)


def _parse_screener_shareholding(html: str) -> dict:
    """Extract shareholding data from Screener.in HTML.

    BUG FIX (Aug-2026 v2): The original pledge parser (v1) scanned HTML
    tables for a "pledge" row and defaulted to 0 on miss — Screener has
    NEVER exposed pledge in a table. v1a scanned every table but still
    couldn't find it (because the row does not exist). This v2 rewrite:

      (1) Pulls promoter / FII / DII / MF composition from the quarterly
          shareholding table (same as v1a — that part works).
      (2) Parses PLEDGE from the top-of-page Pros/Cons `<ul>`, using regex
          on the raw HTML (`Promoters have pledged X% of their holding.`).
          This is where Screener actually puts it.
      (3) If pledge bullet is absent BUT promoter holding was successfully
          read from the table, infers pledge = 0.0 — Screener's Pros/Cons
          block only mentions pledge when non-zero, so silence == zero.
          If neither is available (empty page / suspended stock), leaves
          the field OUT so upstream shows "unknown pledge".
    """
    out = {}
    try:
        tables = pd.read_html(io.StringIO(html))
    except Exception:
        tables = []

    # ---- Pass 1: primary shareholding table (promoter/FII/DII/mutual/public breakdown) ----
    # M12 FIX (Aug-2026): also detect the "widely-held" case — a shareholding
    # table with FII/DII/Public rows but NO promoter row (LT, HDFC AMC, IEX etc).
    # For such stocks, a "no pledge bullet on page" reading previously left
    # pledge as unknown → gate warned noisily on 30+ stocks per run. Widely-held
    # by definition means pledge = 0 (no promoter → nothing to pledge).
    primary_table_found = False
    widely_held_detected = False
    for tbl in tables:
        if tbl.shape[1] < 2 or tbl.shape[0] < 2:
            continue
        first_col = tbl.iloc[:, 0].astype(str).str.strip().str.lower()
        has_fii_or_pub = (first_col.str.contains("fii", na=False).any() or
                          first_col.str.contains("public", na=False).any())
        has_promoter = first_col.str.contains("promoter", na=False).any()
        # Widely-held: FII/Public rows exist but promoter row does NOT.
        if has_fii_or_pub and not has_promoter:
            widely_held_detected = True
        # Original composition-table filter — must have BOTH.
        if not (has_promoter and has_fii_or_pub):
            continue

        quarter_cols = [c for c in tbl.columns[1:]
                       if _looks_like_quarter_header(str(c))]
        if len(quarter_cols) < 1:
            quarter_cols = list(tbl.columns[1:])[-2:]
        if not quarter_cols:
            continue

        latest = quarter_cols[-1]
        prev = quarter_cols[-2] if len(quarter_cols) >= 2 else None

        for lbl_key, our_key in (("promoter", "promoter_holding_pct"),
                                 ("fii",      "fii"),
                                 ("dii",      "dii"),
                                 ("mutual",   "mf"),
                                 ("pledged",  "promoter_pledge_pct")):
            # Pledge may live outside this table — handle in Pass 2 below.
            if lbl_key == "pledged":
                continue
            mask = first_col.str.contains(lbl_key, na=False)
            if not mask.any():
                continue
            row_idx = first_col[mask].index[0]
            cur = _pct_to_float(tbl.at[row_idx, latest])
            if lbl_key == "promoter":
                if cur is not None:
                    out["promoter_holding_pct"] = cur
            else:
                if cur is None or prev is None:
                    continue
                prv = _pct_to_float(tbl.at[row_idx, prev])
                if prv is None:
                    continue
                out[f"{our_key}_delta_qoq"] = round(cur - prv, 2)
        primary_table_found = True
        break                                  # only ONE primary composition table

    # ---- Pass 2 (v2): pledge — regex on raw HTML bullet ----
    # Screener puts pledge as TEXT in the top Pros/Cons list, e.g.:
    #   <li>Promoters have pledged 73.0% of their holding.</li>
    # It is NOT in any <table>, which is why pd.read_html-based scans
    # (v1 and v1a) never found it. Regex on raw HTML is the only reliable
    # extractor. See PARSER_VERSION sentinel and the top-of-file docstring.
    m = _PLEDGE_BULLET_RE.search(html)
    if m:
        try:
            out["promoter_pledge_pct"] = round(float(m.group(1)), 2)
        except ValueError:
            pass
    elif "promoter_holding_pct" in out:
        # ---- Pass 2b: infer zero when bullet is absent ----
        # Screener's Pros/Cons block only mentions pledge when it is > 0
        # (pledge is only listed as a "cons" flag). If the shareholding
        # table parsed cleanly (we got a promoter_holding_pct) but no
        # pledge bullet appears, the real-world pledge is 0.0%.
        out["promoter_pledge_pct"] = 0.0
    elif widely_held_detected:
        # ---- Pass 2c (M12 FIX): widely-held → pledge is definitionally 0 ----
        # Stocks like LT, HDFC AMC, IEX have no promoter row on the
        # shareholding table at all — 100% widely held. There's nothing to
        # pledge, so pledge = 0.0. Also populate promoter_holding_pct = 0.0
        # so downstream code doesn't classify these as "data missing".
        out["promoter_pledge_pct"] = 0.0
        out["promoter_holding_pct"] = 0.0

    return out


def _looks_like_quarter_header(s: str) -> bool:
    return bool(re.match(r"^\s*(Mar|Jun|Sep|Dec)\s+\d{4}\s*$", s, re.IGNORECASE))


def _pct_to_float(x):
    if pd.isna(x):
        return None
    s = str(x).strip().replace("%", "").replace(",", "")
    if not s or s.lower() in ("nan", "-", "--", ""):
        return None
    try:
        return round(float(s), 2)
    except ValueError:
        return None


# ======================================================================================
#  NSE FALLBACK
# ======================================================================================
def _fetch_nse_shareholding(session, ticker: str) -> dict:
    url = NSE_SHAREHOLDING_URL.format(ticker=ticker)
    try:
        r = session.get(url, timeout=15)
        if r.status_code != 200 or not r.text:
            return {}
        data = r.json() if r.headers.get("content-type", "").startswith("application/json") \
               else json.loads(r.text)
    except Exception:
        return {}
    out = {}
    records = None
    if isinstance(data, dict):
        for key in ("data", "records", "shareholding", "shareholdingPattern"):
            if isinstance(data.get(key), list) and data[key]:
                records = data[key]
                break
    if not records:
        return {}
    latest = records[0]
    for k in ("pr_and_prgrp", "promoterAndPromoterGroup",
              "promoter_and_promoter_group_percentage"):
        v = latest.get(k)
        if v is not None:
            try:
                out["promoter_holding_pct"] = round(float(v), 2)
                break
            except (TypeError, ValueError):
                pass
    return out


# ======================================================================================
#  CSV MERGE / WRITE
# ======================================================================================
def _load_existing_csv() -> pd.DataFrame:
    if not os.path.exists(CSV_PATH):
        return pd.DataFrame(columns=list(ALL_FIELDS))
    try:
        df = pd.read_csv(CSV_PATH, comment="#", skip_blank_lines=True)
        df.columns = [c.strip().lower() for c in df.columns]
        for col in ALL_FIELDS:
            if col not in df.columns:
                df[col] = np.nan
        df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
        return df[list(ALL_FIELDS)]
    except Exception:
        return pd.DataFrame(columns=list(ALL_FIELDS))


def _read_header_comments() -> str:
    if not os.path.exists(CSV_PATH):
        return _default_header_comment()
    try:
        with open(CSV_PATH, "r", encoding="utf-8") as f:
            content = f.read()
        parts = []
        for line in content.splitlines():
            if line.strip().startswith("#") or not line.strip():
                parts.append(line)
            else:
                break
        return "\n".join(parts) + ("\n" if parts else "")
    except Exception:
        return _default_header_comment()


def _default_header_comment() -> str:
    return (
        "# --------------------------------------------------------------------------------------\n"
        "# governance_overrides.csv — AUTO-MAINTAINED by governance_fetcher.py\n"
        "# --------------------------------------------------------------------------------------\n"
        "# AUTO-fetched (overwritten on every refresh — do not edit by hand):\n"
        "#   promoter_pledge_pct, promoter_holding_pct, fii_delta_qoq, dii_delta_qoq, mf_delta_qoq\n"
        "#\n"
        "# USER-MAINTAINED (preserved across refreshes):\n"
        "#   auditor_qualified  : 1 if auditor issued qualified/adverse/disclaimer opinion\n"
        "#   rpt_concern        : 1 if related-party-transaction concerns exist\n"
        "#   note               : free text\n"
        "# --------------------------------------------------------------------------------------\n"
    )


def _merge_rows(existing: pd.DataFrame, fresh: dict) -> pd.DataFrame:
    existing = existing.copy()
    existing_tickers = set(existing["ticker"].str.upper()) if len(existing) else set()
    rows_to_add = []
    for tick, fields in fresh.items():
        tick_u = tick.upper().strip()
        if tick_u in existing_tickers:
            row_mask = existing["ticker"].str.upper() == tick_u
            for k in AUTO_FIELDS:
                if k in fields and fields[k] is not None:
                    existing.loc[row_mask, k] = fields[k]
        else:
            new_row = {"ticker": tick_u}
            for k in AUTO_FIELDS:
                new_row[k] = fields.get(k, np.nan)
            for k in MANUAL_FIELDS:
                new_row[k] = np.nan
            rows_to_add.append(new_row)
    if rows_to_add:
        existing = pd.concat([existing, pd.DataFrame(rows_to_add)],
                             ignore_index=True)
    return existing.sort_values("ticker").reset_index(drop=True)


def _write_csv(df: pd.DataFrame, header_comment: str) -> None:
    buf = io.StringIO()
    df.to_csv(buf, index=False, columns=list(ALL_FIELDS))
    lines = buf.getvalue().splitlines()
    with open(CSV_PATH, "w", encoding="utf-8") as f:
        f.write(lines[0] + "\n")     # column header
        f.write(header_comment)      # comment block
        for line in lines[1:]:
            f.write(line + "\n")


# ======================================================================================
#  PUBLIC API #1 — refresh a specific ticker list (backward-compatible)
# ======================================================================================
def _empty_stats() -> dict:
    return {"refreshed": 0, "from_cache": 0, "failed": 0,
            "source_breakdown": {"screener": 0, "nse_fallback": 0, "none": 0},
            "notes": []}


def refresh_governance_overrides(tickers: list,
                                 force_refresh: bool = False,
                                 progress_cb=None) -> dict:
    """Refresh governance data for `tickers` — WRITE to governance_overrides.csv.

    Backward-compatible with v1. The heavy lifting is here; both the CLI and
    refresh_full_nse_universe() delegate to this function.
    """
    tickers = sorted({str(t).upper().strip()
                      .replace(".NS", "").replace(".BO", "")
                      for t in tickers if t})
    if not tickers:
        return _empty_stats()

    cache = _read_cache()
    fresh_data = {}
    stats = _empty_stats()
    failures = {}

    session = _make_session()
    screener_primed = False
    nse_primed = False
    adaptive_gap = BASE_REQUEST_GAP_SEC

    total = len(tickers)
    for k, tick in enumerate(tickers):
        cached_entry = cache.get(tick, {})
        if not force_refresh and _cache_fresh(cached_entry):
            fresh_data[tick] = cached_entry["data"]
            stats["from_cache"] += 1
            src = cached_entry.get("source", "screener")
            stats["source_breakdown"][src] = stats["source_breakdown"].get(src, 0) + 1
            if progress_cb:
                progress_cb(k, total, tick, "cache")
            continue

        data, source, slug = {}, None, None
        if session is not None:
            if not screener_primed:
                _prime_screener(session)
                screener_primed = True
            data, err, slug = _fetch_screener(session, tick, cache)
            if data:
                source = "screener"
                adaptive_gap = max(BASE_REQUEST_GAP_SEC, adaptive_gap * 0.95)
            else:
                if err and ("429" in err or "503" in err):
                    adaptive_gap = min(10.0, adaptive_gap * 1.5)

            if not data:
                if not nse_primed:
                    _prime_nse(session)
                    nse_primed = True
                nse_data = _fetch_nse_shareholding(session, tick)
                if nse_data:
                    data = nse_data
                    source = "nse_fallback"
                time.sleep(NSE_REQUEST_GAP_SEC)

            time.sleep(adaptive_gap)

            if not data:
                failures[tick] = err or "no data from any source"

        if data:
            fresh_data[tick] = data
            entry = {"ts": time.time(), "data": data, "source": source}
            if source == "screener" and slug:
                entry["slug"] = slug
            cache[tick] = entry
            stats["refreshed"] += 1
            stats["source_breakdown"][source] = \
                stats["source_breakdown"].get(source, 0) + 1
            if progress_cb:
                progress_cb(k, total, tick, source)
        else:
            if cached_entry:
                fresh_data[tick] = cached_entry["data"]
                stats["notes"].append(f"{tick}: kept stale cache")
            else:
                stats["failed"] += 1
                stats["source_breakdown"]["none"] += 1
                stats["notes"].append(f"{tick}: no data from any source")
            if progress_cb:
                progress_cb(k, total, tick, "failed")

        # Progressive cache save — critical for long runs
        if (k + 1) % INCREMENTAL_SAVE_EVERY == 0:
            _write_cache(cache)

    _write_cache(cache)
    _write_failures_log(failures)

    existing_df = _load_existing_csv()
    header = _read_header_comments()
    merged = _merge_rows(existing_df, fresh_data)
    _write_csv(merged, header)
    return stats


# ======================================================================================
#  PUBLIC API #2 — refresh a whole NSE universe bucket
# ======================================================================================
def refresh_full_nse_universe(bucket: str = "Nifty500",
                              force_refresh: bool = False,
                              progress_cb=None,
                              max_tickers: int = None) -> dict:
    """Refresh governance data for an entire NSE universe bucket.

    bucket : "LargeCap" (100), "MidCap" (150), "SmallCap" (250),
             "Nifty500" (500), or "AllNSE" (~2000)

    Estimated time (fresh, no cache, ~1.2 sec/stock):
      LargeCap  ~2 min       Nifty500  ~12 min
      MidCap    ~3 min       AllNSE    ~40 min
      SmallCap  ~5 min
    """
    if not HAVE_UNIVERSE_LOADER:
        raise RuntimeError(
            "universe_loader.py not available. Place it next to "
            "governance_fetcher.py, or use refresh_governance_overrides() "
            "with an explicit ticker list."
        )
    bundle = _load_universe()
    buckets = bundle.get("buckets", {})
    if bucket not in buckets:
        avail = list(buckets.keys())
        raise ValueError(f"Unknown bucket '{bucket}'. Available: {avail}")
    tickers = list(buckets[bucket])
    if max_tickers:
        tickers = tickers[:max_tickers]
    return refresh_governance_overrides(tickers, force_refresh, progress_cb)


# ======================================================================================
#  STREAMLIT WRAPPER
# ======================================================================================
def refresh_with_streamlit_progress(tickers: list,
                                    force_refresh: bool = False,
                                    silent: bool = False) -> dict:
    if st is None or silent:
        return refresh_governance_overrides(tickers, force_refresh)
    prog = st.progress(0.0)
    stat = st.empty()

    def _cb(k, n, tick, src):
        stat.write(f"Governance refresh: {tick} ({k+1}/{n}) — {src}")
        prog.progress((k + 1) / n)

    result = refresh_governance_overrides(tickers, force_refresh, _cb)
    prog.empty(); stat.empty()
    if result["refreshed"] or result["from_cache"]:
        st.success(
            f"✅ Governance data: {result['refreshed']} fetched, "
            f"{result['from_cache']} from cache, {result['failed']} failed. "
            f"Sources: {result['source_breakdown']}"
        )
    return result


# ======================================================================================
#  CLI
# ======================================================================================
def _main():
    parser = argparse.ArgumentParser(
        description="Auto-populate governance_overrides.csv from public sources.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python governance_fetcher.py --bucket Nifty500       # recommended weekly cron
  python governance_fetcher.py --all                    # full NSE (~45 min)
  python governance_fetcher.py --bucket Nifty500 --max 50   # test on 50
  python governance_fetcher.py --tickers TATAMOTORS,VEDL    # specific
  python governance_fetcher.py --bucket Nifty500 --force    # ignore cache
        """,
    )
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--all", action="store_true",
                     help="Refresh entire NSE universe (~2000 stocks)")
    grp.add_argument("--bucket",
                     choices=["LargeCap", "MidCap", "SmallCap",
                              "Nifty500", "AllNSE"],
                     help="Refresh one universe bucket")
    grp.add_argument("--tickers", type=str,
                     help="Comma-separated ticker list")
    parser.add_argument("--force", action="store_true",
                        help="Ignore cache — refresh from source")
    parser.add_argument("--max", type=int, default=None,
                        help="Cap tickers (useful for testing)")
    args = parser.parse_args()

    start = time.time()

    def _cb(k, n, tick, src):
        # For long runs, print every 10th plus all failures.
        # For short runs (<= 50), print every ticker.
        if n <= 50 or (k + 1) % 10 == 0 or src in ("failed", "none"):
            print(f"  [{k+1:>4}/{n}] {tick:14s} — {src}")

    if args.tickers:
        tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
        print(f"Refreshing {len(tickers)} tickers (force={args.force})...")
        result = refresh_governance_overrides(tickers, args.force, _cb)
    else:
        bucket = "AllNSE" if args.all else args.bucket
        print(f"Refreshing bucket '{bucket}' (force={args.force}, cap={args.max})...")
        result = refresh_full_nse_universe(bucket, args.force, _cb, args.max)

    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"Done in {elapsed/60:.1f} min.")
    print(f"  Refreshed:  {result['refreshed']}")
    print(f"  From cache: {result['from_cache']}")
    print(f"  Failed:     {result['failed']}")
    print(f"  Sources:    {result['source_breakdown']}")
    if result["failed"]:
        print(f"\nSee .governance_failures.json for per-ticker failure reasons.")


if __name__ == "__main__":
    _main()