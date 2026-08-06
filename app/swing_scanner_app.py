"""
NSE Market-Wide Swing Scanner  (daily, after-market)
====================================================
Runs the single-stock engine (260706_swing_screener_app.py) across a whole universe of NSE
stocks, backtests each over its available history, and shortlists the names whose
signal fires *today* — i.e. candidates to buy at tomorrow's open for a ~10%+ swing in
7-15 days. Each candidate carries its historical backtest stats as a confidence measure.

Run with:  streamlit run swing_scanner_app.py
  (keep 260706_swing_screener_app.py in the same folder - this imports its engine)

Optional: place a `universe.csv` next to this file with columns [ticker, bucket]
          (bucket = LargeCap / MidCap / SmallCap / Nifty500) to override the built-in list.
"""

import os
import time
import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st

import importlib.util

# --- Locate the strategy engine (the screener file) ---
# Auto-discover any "*screener*.py" beside this file and use the MOST RECENTLY MODIFIED one,
# so the scanner always runs the current engine. Uses os.listdir (not glob) so that spaces or
# special characters in the folder path can never break the match.
_here = os.path.dirname(os.path.abspath(__file__))
_self = os.path.basename(os.path.abspath(__file__)).lower()

def _find_engine(folder: str):
    """M11 FIX (Aug-2026): tighter safety filter. Excludes any file whose name
    contains 'old', 'backup', 'bak', 'buggy', 'deprecated', 'wip', 'copy' — so
    a renamed `swing_screener_app.OLD_BUGGY.py` will not be silently picked up
    as newest and loaded as the live engine."""
    _EXCLUDE_TOKENS = ("old", "backup", "bak", "buggy", "deprecated", "wip",
                       "copy", "draft", "test")
    try:
        names = os.listdir(folder)
    except Exception:
        names = []
    cands = []
    for nm in names:
        low = nm.lower()
        if not low.endswith(".py"):
            continue
        if low == _self:                 # never import ourselves
            continue
        if "screener" not in low:        # case-insensitive match
            continue
        # M11: reject anything smelling like an archived / backup file
        stem = low[:-3]                  # strip .py suffix for token matching
        if any(tok in stem for tok in _EXCLUDE_TOKENS):
            continue
        full = os.path.join(folder, nm)
        if os.path.isfile(full):
            cands.append(full)
    return cands, names

_matches, _seen = _find_engine(_here)
if not _matches:
    _pys = [n for n in _seen if n.lower().endswith(".py")] or ["(none)"]
    raise FileNotFoundError(
        "No screener engine found.\n"
        f"  Looking in : {_here}\n"
        f"  This file  : {_self}\n"
        f"  .py files seen here: {', '.join(_pys)}\n"
        "  Fix: put the swing screener .py file (any name containing 'screener') in this exact "
        "folder, and launch streamlit from this folder."
    )
_ENGINE_PATH = max(_matches, key=os.path.getmtime)   # newest = current
ENGINE_FILE = os.path.basename(_ENGINE_PATH)
_spec = importlib.util.spec_from_file_location("engine", _ENGINE_PATH)
engine = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(engine)

# --- Fundamental "no-trade" gate ---
from screeners.fundamental_screen import (
    screen_universe as fs_screen_universe,
    summarize_results as fs_summarize,
    rejects_to_dataframe as fs_rejects_df,
    DEFAULT_FUNDA_CONFIG,
)

# --- News & Event risk (Aug-2026) ---
try:
    from news.nse_events import event_risk as _event_risk
    from news.news_sentiment import fetch_news_score as _news_score
    HAVE_NEWS = True
except Exception:
    HAVE_NEWS = False

# --- Universe loader: chained live sources + 24h disk cache (no CSV maintenance) ---
from scanner.data.universe_loader import load_full_universe as _ul_load

try:
    import yfinance as yf
except Exception:
    yf = None


# ======================================================================================
#  UNIVERSE  - built-in default lists (override with universe.csv if present)
# ======================================================================================
BUILTIN_UNIVERSE = {
    "LargeCap": [
        "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "HINDUNILVR", "ITC", "SBIN",
        "BHARTIARTL", "KOTAKBANK", "LT", "AXISBANK", "BAJFINANCE", "ASIANPAINT", "MARUTI",
        "SUNPHARMA", "TITAN", "ULTRACEMCO", "WIPRO", "NESTLEIND", "ONGC", "NTPC", "POWERGRID",
        "M&M", "TATAMOTORS", "TATASTEEL", "JSWSTEEL", "ADANIENT", "ADANIPORTS", "COALINDIA",
        "HCLTECH", "BAJAJFINSV", "TECHM", "GRASIM", "HINDALCO", "DRREDDY", "CIPLA", "BPCL",
        "BRITANNIA", "EICHERMOT", "DIVISLAB", "HEROMOTOCO", "INDUSINDBK", "APOLLOHOSP",
        "TATACONSUM", "BAJAJ-AUTO", "SBILIFE", "HDFCLIFE", "LTIM", "SHRIRAMFIN",
    ],
    "MidCap": [
        "HUDCO", "IRFC", "RVNL", "BEL", "BHEL", "IOC", "GAIL", "PFC", "RECLTD", "IRCTC",
        "ABCAPITAL", "ASHOKLEY", "AUROPHARMA", "BANKBARODA", "CANBK", "CGPOWER", "CONCOR",
        "COFORGE", "CUMMINSIND", "DLF", "GODREJPROP", "HAVELLS", "INDHOTEL", "JUBLFOOD",
        "LICHSGFIN", "LUPIN", "MRF", "NMDC", "OBEROIRLTY", "PAGEIND", "PERSISTENT",
        "PIIND", "POLYCAB", "SAIL", "SUZLON", "TATAPOWER", "TORNTPHARM", "TRENT", "VBL",
        "YESBANK", "IDFCFIRSTB", "PNB", "UNIONBANK", "MAXHEALTH", "LODHA", "HINDZINC",
    ],
    "SmallCap": [
        "IREDA", "MAZDOCK", "COCHINSHIP", "GRSE", "HAL", "BDL", "MIDHANI", "RITES",
        "IRCON", "NBCC", "ENGINERSIN", "HFCL", "GMRINFRA", "JWL", "KALYANKJIL", "KAYNES",
        "TATATECH", "ZOMATO", "NYKAA", "PAYTM", "POLICYBZR", "DELHIVERY", "MAPMYINDIA",
        "IEX", "CDSL", "BSE", "ANGELONE", "CAMS", "KFINTECH", "MCX", "INTELLECT",
        "TANLA", "ROUTE", "HAPPSTMNDS", "LATENTVIEW", "SONACOMS", "OLECTRA",
    ],
}
BUILTIN_UNIVERSE["Nifty500"] = sorted(set(sum(BUILTIN_UNIVERSE.values(), [])))


# Universe fetch and sector map are now delegated to universe_loader.py — see the
# `load_universe()` and `fetch_sector_map()` wrappers further down. That module owns
# the Chrome-TLS impersonation (curl_cffi), the chained live sources, and the 24h
# disk cache. This scanner file no longer talks to NSE directly.


def fetch_sector_map() -> dict:
    """Sector map is bundled with the universe loader — one call fetches both
    universe and sector data, using the same 24h disk cache."""
    return _ul_load().get("sector_map", {})


def apply_sector_caps(cand: pd.DataFrame, max_per_sector: int) -> tuple:
    """Keep at most `max_per_sector` names per sector, preferring the highest-ranked.
    Assumes `cand` is already sorted best-first.

    M8 FIX (Aug-2026): UNKNOWN is now treated as a sector like any other. Prior
    behaviour ("UNKNOWN never capped") silently disabled diversification whenever
    NSE was unreachable — you'd end up with 30 UNKNOWN-sector names in the
    shortlist thinking you had 3-per-sector diversification. Capping UNKNOWN
    keeps the shortlist honest even when sector data is missing.
    """
    if max_per_sector <= 0 or cand.empty:
        return cand, {}
    kept_idx, counts, dropped = [], {}, {}
    for i, row in cand.iterrows():
        sec = row.get("sector", "UNKNOWN") or "UNKNOWN"
        c = counts.get(sec, 0)
        if c < max_per_sector:
            counts[sec] = c + 1
            kept_idx.append(i)
        else:
            dropped[sec] = dropped.get(sec, 0) + 1
    return cand.loc[kept_idx], dropped


def load_universe():
    """Delegate to universe_loader.load_full_universe() — chained live NSE sources
    with 24h disk cache and graceful stale-cache fallback. Zero CSV maintenance."""
    bundle = _ul_load()
    buckets = bundle["buckets"] or BUILTIN_UNIVERSE
    meta = bundle["meta"]
    if bundle["meta"]["source"].startswith("built-in"):
        # Only show a warning on the absolute last-resort path; stale cache is fine silently.
        st.warning(
            f"⚠️ {meta['source']}. Live NSE and disk cache both failed. "
            f"If you just installed the app, install `curl_cffi` "
            f"(`pip install curl_cffi`) — it bypasses NSE's TLS bot-block. "
            f"Otherwise wait a few minutes and re-run: NSE 503s are usually transient."
        )
    elif "STALE" in meta["source"]:
        st.info(f"ℹ️ Using **stale disk cache** ({meta['cache_age_hours']}h old) — "
                f"NSE unreachable right now, but last-known-good universe is loaded so "
                f"you can still scan. Cache will refresh automatically next time NSE responds.")
    return buckets, meta["source"]


def to_yahoo(sym: str) -> str:
    sym = sym.strip().upper()
    return sym if sym.endswith((".NS", ".BO")) else sym + ".NS"


# ======================================================================================
#  PER-STOCK SCAN
# ======================================================================================
MIN_DAYS = 250
TARGET_YEARS = 10
RS_WINDOW = 63                       # ~3 months for relative-strength
BENCH_TICKERS = ["^CRSLDX", "^NSEI"] # Nifty 500 (broad), fallback Nifty 50
# Segment indices: your universe lives here, not in the IT mega-caps that can lift the headline.
# Several candidates each — Yahoo's coverage of Indian segment indices is inconsistent.
SEGMENT_TICKERS = {
    "MidCap":   ["^NSEMDCP50", "NIFTY_MIDCAP_100.NS", "^CNXMIDCAP"],
    "SmallCap": ["^CNXSC", "NIFTYSMLCAP250.NS", "^CNXSMCAP"],
}


# ---------------------------------------------------------------------------
# CACHE-BUSTER (C2 fix, Aug-2026)
# ---------------------------------------------------------------------------
# yfinance's `auto_adjust=True` rescales the ENTIRE history whenever a stock
# has a corporate action (split, bonus, spin-off). If we cache the pre-split
# series for 12h and Yahoo publishes the post-split scaling within that TTL,
# any downstream indicator (SMA200, ATR, %vs SMA) computed on the mixed-scale
# series produces false signals. The bug is silent — the app still runs.
#
# Fix: rotate the cache key every 4 hours. Any split takes effect within 4h.
# `_cache_bucket()` is an explicit fn arg passed at every call site so it is
# part of Streamlit's cache key (positional args are keys; kwargs prefixed
# with _ are excluded — we want it INCLUDED).
# ---------------------------------------------------------------------------
def _cache_bucket() -> str:
    """Rotates every 4 hours. Included in cache keys of fetch_one / fetch_index /
    fetch_segments so any split-adjust change from yfinance is picked up within
    a single 4h window, regardless of the 12h TTL."""
    now = dt.datetime.now()
    return now.strftime("%Y%m%d_") + f"{now.hour // 4:02d}"


@st.cache_data(show_spinner=False, ttl=60 * 60 * 12)
def _fetch_one_impl(ticker: str, start: dt.date, end: dt.date,
                    cache_bucket: str) -> pd.DataFrame:
    """The actual fetcher. `cache_bucket` is a versioning string that participates
    in the cache key so a new 4h window forces a re-download (see C2 fix note)."""
    t = yf.Ticker(ticker)
    df = t.history(start=start, end=end, interval="1d", auto_adjust=True)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df.dropna()


def fetch_one(ticker: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    """Public wrapper that injects the 4h cache_bucket into the cache key."""
    return _fetch_one_impl(ticker, start, end, _cache_bucket())


@st.cache_data(show_spinner=False, ttl=60 * 60 * 12)     # M9 FIX: unified to 12h (was 6h)
def _fetch_index_impl(start: dt.date, end: dt.date, cache_bucket: str):
    """Fetch a broad benchmark index (Nifty 500, fallback Nifty 50) for regime + RS.
    `cache_bucket` participates in the cache key (see C2 fix note above)."""
    if yf is None:
        return None, pd.DataFrame()
    for t in BENCH_TICKERS:
        try:
            df = yf.Ticker(t).history(start=start, end=end, interval="1d", auto_adjust=True)
            if df is not None and not df.empty:
                df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
                df.index = pd.to_datetime(df.index).tz_localize(None)
                return t, df.dropna()
        except Exception:
            continue
    return None, pd.DataFrame()


def fetch_index(start: dt.date, end: dt.date):
    return _fetch_index_impl(start, end, _cache_bucket())


@st.cache_data(show_spinner=False, ttl=60 * 60 * 12)     # M9 FIX: unified to 12h (was 6h)
def _fetch_segments_impl(start: dt.date, end: dt.date, cache_bucket: str) -> dict:
    """Fetch mid/small-cap segment indices. Returns {name: pct_vs_200dma} for those that resolve."""
    out = {}
    if yf is None:
        return out
    for seg, candidates in SEGMENT_TICKERS.items():
        for t in candidates:
            try:
                df = yf.Ticker(t).history(start=start, end=end, interval="1d", auto_adjust=True)
                if df is None or df.empty or len(df) < 210:
                    continue
                c = df["Close"].dropna()
                s200 = c.rolling(200).mean().iloc[-1]
                if not np.isfinite(s200):
                    continue
                out[seg] = {"ticker": t,
                            "pct_vs_200": round(float(c.iloc[-1] / s200 - 1) * 100, 2),
                            "above_200": bool(c.iloc[-1] > s200)}
                break
            except Exception:
                continue
    return out


def fetch_segments(start: dt.date, end: dt.date) -> dict:
    return _fetch_segments_impl(start, end, _cache_bucket())


def compute_breadth(rows: list) -> dict:
    """Advance/decline breadth computed from the scanned universe itself (no extra fetches).
    This is the piece that catches a narrow, breadth-negative day behind a green headline index."""
    ok = [r for r in rows if r.get("status") == "ok" and np.isfinite(r.get("day_chg_%", np.nan))]
    n = len(ok)
    if n == 0:
        return {"status": "UNKNOWN", "n": 0}
    adv = sum(1 for r in ok if r["day_chg_%"] > 0)
    dec = sum(1 for r in ok if r["day_chg_%"] < 0)
    above50 = sum(1 for r in ok if r.get("above_50dma"))
    ad_ratio = adv / max(dec, 1)
    pct_adv = 100 * adv / n
    pct_above50 = 100 * above50 / n
    # negative breadth: most names fell today, or most sit below their own 50-DMA
    if pct_adv >= 55 and pct_above50 >= 50:
        status = "POSITIVE"
    elif pct_adv < 40 or pct_above50 < 40:
        status = "NEGATIVE"
    else:
        status = "MIXED"
    return {"status": status, "n": n, "advancers": adv, "decliners": dec,
            "pct_advancers": round(pct_adv, 1), "pct_above_50dma": round(pct_above50, 1),
            "ad_ratio": round(ad_ratio, 2)}


def compute_regime(idx_df: pd.DataFrame) -> dict:
    """Trend/momentum of the broad benchmark (one input to the composite gate)."""
    if idx_df.empty or len(idx_df) < 210:
        return {"status": "UNKNOWN", "note": "index data unavailable",
                "idx_ret_window": 0.0, "index_ok": False}
    c = idx_df["Close"]
    s200 = c.rolling(200).mean().iloc[-1]
    last = float(c.iloc[-1])
    above200 = bool(last > s200) if np.isfinite(s200) else True
    pct_vs200 = (last / s200 - 1) * 100 if np.isfinite(s200) else np.nan
    roc10 = (c.iloc[-1] / c.iloc[-11] - 1) * 100 if len(c) > 11 else 0.0
    idx_ret_window = (c.iloc[-1] / c.iloc[-(RS_WINDOW + 1)] - 1) * 100 if len(c) > RS_WINDOW else 0.0
    if above200 and roc10 > -1.0:
        status = "RISK-ON"
    elif above200 or roc10 > -3.0:
        status = "NEUTRAL"
    else:
        status = "RISK-OFF"
    return {"status": status, "above_200": above200, "pct_vs_200": round(float(pct_vs200), 2),
            "roc10": round(float(roc10), 2), "idx_ret_window": float(idx_ret_window),
            "last": round(last, 2), "index_ok": True}


BREADTH_MIN_N_FOR_VETO = 100          # H8 statistical floor — see composite_gate docs


def composite_gate(regime: dict, segments: dict, breadth: dict) -> dict:
    """Combine index trend + segment trend + breadth into one verdict.

    Negative BREADTH can force RISK-OFF even when the headline index is green —
    the exact scenario where a basket of longs sinks behind a mega-cap-driven
    index. BUT only when breadth is measured on a statistically-significant
    sample.

    H8 FIX (Aug-2026): breadth is computed from the SCANNED tickers, which may
    be as few as 50 (LargeCap bucket). A 50-name advance/decline read is noise;
    treating it as a hard veto over-rejected on small-universe runs. Fix:
      * breadth participates in the score only when `n >= BREADTH_MIN_N_FOR_VETO`
      * the hard "breadth veto over green index" branch requires the same floor
      * when n is below the floor, breadth is displayed but marked "advisory"
        and doesn't influence the verdict
    """
    idx_state = regime.get("status", "UNKNOWN")
    br = breadth.get("status", "UNKNOWN")
    br_n = int(breadth.get("n", 0))
    br_significant = br_n >= BREADTH_MIN_N_FOR_VETO
    seg_below = [s for s, v in segments.items() if not v.get("above_200", True)]

    score = 0
    if idx_state == "RISK-ON":  score += 1
    elif idx_state == "RISK-OFF": score -= 1
    # H8: only score breadth when the sample is large enough to trust it
    if br_significant:
        if br == "POSITIVE": score += 1
        elif br == "NEGATIVE": score -= 1
    if seg_below: score -= 1                    # your universe's own segment is in a downtrend

    # H8: veto only when breadth is negative AND statistically significant
    if br_significant and br == "NEGATIVE" and idx_state != "RISK-ON":
        final = "RISK-OFF"
    elif score >= 2:
        final = "RISK-ON"
    elif score <= -1:
        final = "RISK-OFF"
    else:
        final = "NEUTRAL"

    br_label = br if br_significant else f"{br} (advisory, n={br_n} < {BREADTH_MIN_N_FOR_VETO})"
    reasons = [f"index {idx_state}", f"breadth {br_label}"]
    if seg_below:
        reasons.append(f"{'/'.join(seg_below)} below 200-DMA")
    elif segments:
        reasons.append("segments above 200-DMA")
    return {"final": final, "score": score, "reasons": reasons,
            "breadth_significant": br_significant,
            "breadth_veto": (br_significant and br == "NEGATIVE" and idx_state == "RISK-ON")}


def scan_one(ticker, start, end, strategy, p, bt_kwargs, idx_ret_window=0.0,
             sector_map=None, require_confirmation: bool = False,
             bench_close=None, block_risk_off: bool = False) -> dict:
    try:
        raw = fetch_one(ticker, start, end)
    except Exception as e:
        return {"ticker": ticker, "status": f"fetch error: {str(e)[:40]}"}
    if raw.empty:
        return {"ticker": ticker, "status": "no data"}
    if len(raw) < MIN_DAYS:
        return {"ticker": ticker, "status": f"insufficient data ({len(raw)}d) - skipped"}

    df = engine.compute_indicators(raw)
    df = engine.generate_signals(df, strategy, p,
                                  bench_close=bench_close,
                                  require_confirmation=require_confirmation,
                                  block_risk_off=block_risk_off)
    trades = engine.run_backtest(df, **bt_kwargs)
    stats = engine.summarize(trades)

    yrs = (raw.index[-1] - raw.index[0]).days / 365.25
    remark = "" if yrs >= TARGET_YEARS - 0.5 else \
             f"limited history: {yrs:.1f}y (<{TARGET_YEARS}y) - lower confidence"

    last = df.iloc[-1]
    signals_today = bool(last["signal"])

    # =====================================================================
    # CHANGE #2 (Aug-2026) — SCANNER-SIDE POST-STOP DEDUP
    # ---------------------------------------------------------------------
    # Mirror the engine's post_stop_cooldown_days behaviour at LIVE scan
    # time. `signals_today` above is computed from the RAW `signal` column
    # — it doesn't know about cooldown. Without this block, a stock that
    # stopped out 3 days ago would still surface in tonight's shortlist
    # even though the engine's cooldown filter has already excluded that
    # trade from the historical record.
    #
    # Logic: look at the FILTERED trade log the engine returned. If the
    # most recent STOP-exit is within `post_stop_cooldown_days` calendar
    # days of today's bar, downgrade signals_today to False and stash a
    # reason on the row so the UI can explain WHY.
    # =====================================================================
    cooldown_days = int(bt_kwargs.get("post_stop_cooldown_days", 0) or 0)
    cooldown_blocked = False
    cooldown_reason = ""
    if signals_today and cooldown_days > 0 and not trades.empty:
        today_ts = pd.Timestamp(df.index[-1])
        _t = trades.copy()
        _t["exit_dt"] = pd.to_datetime(_t["exit_date"])
        recent_stops = _t[(_t["outcome"] == "STOP")
                          & (_t["exit_dt"] <= today_ts)
                          & ((today_ts - _t["exit_dt"]).dt.days < cooldown_days)]
        if not recent_stops.empty:
            last_stop_dt = recent_stops["exit_dt"].max()
            days_since = int((today_ts - last_stop_dt).days)
            cooldown_blocked = True
            cooldown_reason = (f"post-stop cooldown: last STOP {days_since}d ago "
                               f"(cooldown = {cooldown_days}d)")
            signals_today = False    # downgrade — algo would be in cooldown live

    regime_today = (last.get("trade_type", "") or "UPTREND") if signals_today else ""

    # Legacy all-signals stats (kept for the "raw edge" columns in the results table).
    n = stats.get("trades", 0)
    exp = stats.get("expectancy_%", 0.0)
    winr = stats.get("profitable_%", 0.0)
    exp_day = stats.get("exp_per_day_%", 0.0)

    # =====================================================================
    # C5 FIX (Aug-2026) — ranking is now driven by SEQUENTIAL stats
    # ---------------------------------------------------------------------
    # `stats["expectancy_%"]` etc. are computed on the overlapping trade
    # pool — every historical signal, including signals that fired while a
    # prior trade was still open. That pool is heavily correlated and
    # inflates statistical significance. `seq_*` fields come from the
    # sequential (one-position-at-a-time) trade list — the trades you would
    # actually take with one pool of capital. Ranking should use these.
    # =====================================================================
    n_seq        = stats.get("seq_trades", 0)
    exp_seq      = stats.get("seq_expectancy_%", 0.0)
    winr_seq     = stats.get("seq_win_%", 0.0)
    exp_day_seq  = stats.get("seq_exp_per_day_%", 0.0)
    size_factor  = n_seq / (n_seq + 30.0)          # sample-size damping on SEQUENTIAL count
    confidence = round(max(exp_day_seq, 0) * (winr_seq / 100.0) * size_factor * 100, 2)

    # --- RELATIVE STRENGTH: stock's return minus the index's over the RS window ---
    c_ser = df["Close"]
    if len(c_ser) > RS_WINDOW:
        stock_ret_window = (c_ser.iloc[-1] / c_ser.iloc[-(RS_WINDOW + 1)] - 1) * 100
    else:
        stock_ret_window = (c_ser.iloc[-1] / c_ser.iloc[0] - 1) * 100
    rel_strength = round(float(stock_ret_window - idx_ret_window), 2)   # >0 = beating the market
    # blended rank: confidence tilted by relative strength (bounded ±50%)
    rs_norm = max(min(rel_strength / 30.0, 0.5), -0.5)
    rank_score = round(confidence * (1 + rs_norm), 2)

    # =====================================================================
    # CHANGE #6 (Aug-2026) — FRESHNESS × EXTENSION PENALTY
    # ---------------------------------------------------------------------
    # Soft-block ranking penalty. Complements Change #1's hard-block
    # cooldown: cooldown removes trades that would come right after a
    # stop; extension penalty demotes trades that arrive stretched or
    # stale even when cooldown doesn't fire. Belt-and-suspenders.
    #
    # Two multiplicative down-weights applied to rank_score:
    #   FRESHNESS      — signal fired today vs earlier (stale = 0.4x)
    #   EXTENSION_PEN  — RSI overbought / far above 20-DMA / at 52w-high /
    #                    riding upper Bollinger band. Each fires ~0.80-0.85x.
    #                    Stacked multiplicatively; a stock hitting all four
    #                    caps out at ~0.46x rank (heavy demotion).
    #
    # We keep the RAW score too (`rank_score_raw`) so the audit table can
    # show WHY a stock was demoted, and `ranking_penalty_reason` supplies
    # a human-readable string ("RSI 78, +9.4% vs 20DMA, at 52wH").
    # =====================================================================
    rank_score_raw = rank_score
    freshness = 1.0 if signals_today else 0.4
    ext_pen = 1.0
    penalty_bits = []
    def _f(colname, default):
        v = last.get(colname, default)
        return float(v) if pd.notna(v) else default
    rsi_now = _f("rsi14", 50.0)
    pct_20  = _f("pct_vs_sma20", 0.0)
    dist52  = _f("dist_52wH", -10.0)
    bb_pctb = _f("bb_pctB", 50.0)
    if rsi_now > 70:
        ext_pen *= 0.80; penalty_bits.append(f"RSI {rsi_now:.0f}")
    if pct_20 > 8:
        ext_pen *= 0.85; penalty_bits.append(f"+{pct_20:.1f}% vs 20DMA")
    if dist52 > -2:
        ext_pen *= 0.85; penalty_bits.append(f"{dist52:+.1f}% from 52wH")
    if bb_pctb > 90:
        ext_pen *= 0.85; penalty_bits.append(f"BB%B {bb_pctb:.0f}")
    if not signals_today:
        penalty_bits.append("stale (no signal today)")
    ranking_penalty_reason = " · ".join(penalty_bits) if penalty_bits else ""
    rank_score = round(rank_score * freshness * ext_pen, 2)

    # --- point 1: concrete stop-loss for a trade entered ~ at the last close ---
    entry_ref = float(last["Close"])
    atr_now = float(last["atr14"]) if np.isfinite(last["atr14"]) else 0.0
    stop_mult = bt_kwargs.get("stop_value", 2.0)
    max_stop_pct = bt_kwargs.get("max_stop_pct", 8.0) or 8.0
    tgt_pct = bt_kwargs.get("target_pct", 10.0)

    # --- LIMIT ENTRY: the price to actually place the order at (don't chase the open) ---
    entry_mode = bt_kwargs.get("entry_mode", "Market open")
    limit_pct = bt_kwargs.get("limit_pct", 0.0)
    if entry_mode == "Limit":
        limit_price = round(entry_ref * (1 - limit_pct / 100.0), 2)
        plan_entry = limit_price                # stop/target computed off the price you'd pay
    else:
        limit_price = np.nan
        plan_entry = entry_ref

    stop_price = plan_entry - stop_mult * atr_now
    floor = plan_entry * (1 - max_stop_pct / 100)
    stop_price = max(stop_price, floor)                       # respect max-loss cap
    stop_pct = round((stop_price / plan_entry - 1) * 100, 2)
    target_price = round(plan_entry * (1 + tgt_pct / 100), 2)

    # --- point 2: expected days to target, from historical winners ---
    med_days = stats.get("med_days_to_target", np.nan)
    n_win = stats.get("n_winners", 0)
    if np.isnan(med_days):
        days_to_target = "n/a"
    elif n_win < 5:
        days_to_target = f"{med_days:.0f}d ⚠ thin"
    else:
        days_to_target = f"{med_days:.0f}d"

    return {
        "ticker": ticker.replace(".NS", "").replace(".BO", ""), "yahoo": ticker, "status": "ok",
        "sector": (sector_map or {}).get(ticker.replace(".NS", "").replace(".BO", "").upper(), "UNKNOWN"),
        "signals_today": signals_today, "regime_today": regime_today,
        "cooldown_blocked": cooldown_blocked, "cooldown_reason": cooldown_reason,
        "rank_score_raw": rank_score_raw,               # Change #6: pre-penalty for audit
        "ranking_penalty_reason": ranking_penalty_reason,
        "bt_from": raw.index[0].date(), "bt_to": raw.index[-1].date(), "years": round(yrs, 1),
        "hist_trades": n, "win_%": winr, "expectancy_%": exp,
        "avg_win_%": stats.get("avg_win_%", 0.0), "avg_loss_%": stats.get("avg_loss_%", 0.0),
        "avg_days": stats.get("avg_days", 0.0), "confidence": confidence,
        "exp_per_day_%": exp_day, "cut_exits": stats.get("cut_exits", 0),
        # ---- C5 FIX: portfolio-realistic sequential-trade stats ----
        "seq_trades":         n_seq,
        "seq_win_%":          winr_seq,
        "seq_expectancy_%":   exp_seq,
        "seq_total_return_%": stats.get("seq_total_return_%", 0.0),
        "seq_exp_per_day_%":  exp_day_seq,
        "seq_profit_factor":  stats.get("seq_profit_factor", np.nan),
        "trail_exits": stats.get("trail_exits", 0),
        # --- new: profitability / risk / account-level metrics ---
        "total_return_sum_%": stats.get("total_return_sum_%", np.nan),
        "profit_factor": stats.get("profit_factor", np.nan),
        "reward_risk_ratio": stats.get("reward_risk_ratio", np.nan),
        "cagr_%": stats.get("cagr_%", np.nan),
        "max_drawdown_%": stats.get("max_drawdown_%", np.nan),
        "recovery_factor": stats.get("recovery_factor", np.nan),
        "max_consecutive_losses": stats.get("max_consecutive_losses", np.nan),
        "seq_trades": stats.get("seq_trades", np.nan),
        "rel_strength": rel_strength, "rank_score": rank_score,
        # --- breadth inputs (free: computed from data already fetched) ---
        "day_chg_%": round(float(c_ser.iloc[-1] / c_ser.iloc[-2] - 1) * 100, 2) if len(c_ser) > 1 else np.nan,
        "above_50dma": bool(last["Close"] > last["sma50"]) if np.isfinite(last["sma50"]) else False,
        "last_close": round(entry_ref, 2), "last_atr_pct": round(float(last["atr_pct"]), 2),
        # point 1 outputs
        "entry_ref": round(entry_ref, 2), "limit_price": limit_price, "plan_entry": round(plan_entry, 2),
        "stop_price": round(stop_price, 2), "stop_%": stop_pct,
        "target_price": target_price,
        # point 2 output
        "exp_days_to_target": days_to_target,
        # point 3 outputs (counts + %)
        "target_hits": stats.get("target_hits", 0), "target_%": stats.get("target_pct_of_all", 0.0),
        "stop_hits": stats.get("stop_hits", 0), "stop_hit_%": stats.get("stop_pct_of_all", 0.0),
         "trail_%": stats.get("trail_pct_of_all", 0.0),
        "time_exits": stats.get("time_exits", 0), "time_%": stats.get("time_pct_of_all", 0.0),
        "time_win": stats.get("time_win", 0), "time_loss": stats.get("time_loss", 0),
        "mom_exits": stats.get("mom_exits", 0), "mom_%": stats.get("mom_pct_of_all", 0.0),
        "decay_exits": stats.get("decay_exits", 0), "decay_%": stats.get("decay_pct_of_all", 0.0),
        "staircase_partials": stats.get("staircase_partials", 0),
        "remark": remark,
    }


# ======================================================================================
#  CHART (for click-to-view)
# ======================================================================================
def build_stock_chart(yahoo_ticker, start, end, strategy, p, bt_kwargs):
    """Recompute one stock and return a plotly price+trades figure + its trade log."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    raw = fetch_one(yahoo_ticker, start, end)
    if raw.empty:
        return None, None
    df = engine.compute_indicators(raw)
    df = engine.generate_signals(df, strategy, p)
    trades = engine.run_backtest(df, **bt_kwargs)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28],
                        vertical_spacing=0.05,
                        subplot_titles=("Close price with entries", "Cumulative net return (overlapping proxy)"))
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close",
                             line=dict(color="#334155", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["sma200"], name="200-DMA",
                             line=dict(color="#f59e0b", width=1, dash="dot")), row=1, col=1)
    colors = {"TARGET": "#16a34a", "TRAIL": "#86efac", "STOP": "#dc2626",
              "TIME": "#94a3b8", "MOMEXIT": "#a855f7", "DECAY": "#0891b2",
              "CUT": "#f97316"}
    if not trades.empty:
        for oc, cl in colors.items():
            sub = trades[trades["outcome"] == oc]
            if not sub.empty:
                fig.add_trace(go.Scatter(x=pd.to_datetime(sub["entry_date"]), y=sub["entry_price"],
                                         mode="markers", name=oc,
                                         marker=dict(color=cl, size=6, line=dict(width=0.4, color="white"))),
                              row=1, col=1)
        eq = trades.sort_values("entry_date").copy()
        eq["cum"] = eq["net_return_%"].cumsum()
        fig.add_trace(go.Scatter(x=pd.to_datetime(eq["entry_date"]), y=eq["cum"],
                                 name="Cumulative net %", line=dict(color="#2563eb", width=1.4)),
                      row=2, col=1)
    fig.update_layout(height=560, hovermode="x unified", legend_orientation="h",
                      margin=dict(t=40, b=10))
    return fig, trades


# ======================================================================================
#  UI
# ======================================================================================
def main():
    """Standalone entry-point — sets page config, then renders body().
    trading_suite.py imports body() directly to avoid a duplicate config call."""
    st.set_page_config(page_title="NSE Daily Swing Scanner", layout="wide")
    body()


def body():
    """All render logic, no set_page_config (safe to call inside a larger app)."""
    st.title("NSE Market-Wide Daily Swing Scanner")
    st.caption("Runs the swing engine across a universe of NSE stocks and shortlists names "
               "signalling today - candidates for a 15%+ move in 1-30 days. Educational tool, "
               "not investment advice. Run after market close.")
    st.caption(f"⚙️ Strategy engine loaded: **{ENGINE_FILE}** (newest `*screener*.py` in this folder)")

    universe, src = load_universe()

    with st.sidebar:
        st.header("1 - Universe")
        bucket = st.selectbox("Segment",
                              ["LargeCap", "MidCap", "SmallCap", "Nifty500", "AllNSE", "Enter manually"])
        if bucket == "Enter manually":
            txt = st.text_area("Tickers (comma/space separated)", "HUDCO, IRFC, RVNL, BEL")
            tickers = [t for t in txt.replace(",", " ").split()]
        else:
            tickers = universe.get(bucket, [])
        st.caption(f"Source: {src}. {len(tickers)} stocks in {bucket}.")
        if len(tickers) > 200:
            st.warning(f"{len(tickers)} stocks is a heavy run on free Yahoo data (expect several "
                       "minutes and some fetch failures). Consider running in batches via the limit below.")
        max_n = st.slider("Limit stocks this run", 5, max(5, len(tickers)), min(2000, len(tickers)),
                          help="Yahoo rate-limits large runs. Start small; raise once stable.")

        st.header("2 - Backtest window")
        yrs = st.slider("Years of history (target)", 3, 15, TARGET_YEARS,
                        help="Expert default 10y - spans multiple bull/bear cycles. Stocks with "
                             "less history use what's available and are flagged.")
        end = dt.date.today()
        start = end - dt.timedelta(days=int(yrs * 365.25) + 300)

        st.header("2b · Fundamentals gate (NO-TRADE filter)")
        # ==============================================================================
        # OPTION C (Aug-2026) — MOMENTUM-FRIENDLY PRESET
        # ------------------------------------------------------------------------------
        # A 15%/30-day swing strategy is a MOMENTUM play — fundamentals (annual /
        # quarterly, lagging) don't predict a 30-day price move. What matters for
        # short-term momentum risk is: (a) fraud/governance blowups (promoter pledge,
        # auditor issues), (b) imminent bankruptcy (negative EBIT + high debt).
        # Everything else (ROE ≥ 5, D/E ≤ 3, 12mo return floor, 200-DMA slope, growth,
        # valuation) HURTS the momentum strategy — we backtested on Nifty 100 and
        # found the strict gate was rejecting 50% of the universe, including stocks
        # that delivered dozens of 15%+ 30-day windows in the past year alone
        # (ADANIGREEN 82 windows, TRENT 51, TVSMOTOR 28, MARUTI 22, MAZDOCK 19, ...).
        # Preset checkbox = one-click switch to the momentum-safe config.
        # ==============================================================================
        momentum_preset = st.checkbox(
            "🚀 Momentum-friendly preset  (recommended for 15%/30-day swings)",
            value=True,
            help="ONE-CLICK CONFIG for the momentum strategy this app implements. When ON:\n\n"
                 "• Trend / Growth / Valuation / Ownership pillars → OFF (technical scan handles trend; "
                 "recovery stocks can rip; momentum can carry rich P/E).\n"
                 "• Quality pillar → ON with LOOSENED thresholds (D/E ≤ 5.0, interest cover ≥ 1.0). "
                 "Catches imminent-bankruptcy risk without rejecting high-D/E capital-intensive growth.\n"
                 "• Governance pillar → ON at strict defaults. Catches fraud/pledge blowup risk that "
                 "matters even for 30-day trades.\n\n"
                 "Turn OFF to configure each pillar and threshold manually. On a Nifty 100 test, "
                 "preset dropped rejection rate from 50% to ~12% — releasing dozens of high-momentum "
                 "large-cap trades that the strict gate was falsely rejecting.")
        apply_funda_gate = st.checkbox(
            "Apply fundamental gate before technical scan",
            value=True,
            help="Hard-rejects structurally broken / governance-risky stocks "
                 "before they even reach the technical backtest. Missing data "
                 "passes with a warning (unless strict mode).")
        with st.expander("Fundamentals — thresholds & pillars"):
            funda_valuation  = st.checkbox("Valuation pillar",  value=False,
                help="OFF by default — momentum swings can carry rich P/E.")
            funda_quality    = st.checkbox("Quality pillar",    value=True,
                help="Real 'broken business' filter — ROE, D/E, interest cover, current ratio.")
            funda_growth     = st.checkbox("Growth pillar",     value=False,
                help="OFF by default — recovery stories can show negative growth.")
            funda_governance = st.checkbox("Governance pillar", value=True,
                help="Promoter pledge, holding, auditor qualification. "
                     "Needs governance_overrides.csv for full effect.")
            funda_ownership  = st.checkbox("Ownership-flow pillar", value=False,
                help="FII/DII delta from override CSV.")
            funda_trend      = st.checkbox("🆕 Trend & liquidity pillar", value=True,
                help="NEW (Aug-2026) — rejects secular downtrenders and illiquid names "
                     "using stock's OWN 14-month price history. Directly targets the "
                     "5 chronic-loser stocks (RBLBANK, SAIL, INDUSINDBK, PHOENIXLTD, "
                     "IREDA) that lost money in every exit strategy.")
            funda_strict = st.checkbox("Strict mode (no data ⇒ reject)", value=False,
                help="If ON, stocks with missing fundamentals are rejected. "
                     "OFF (default) is safer — most Indian smallcaps have partial data.")
            st.markdown("**Thresholds** (defaults are lenient — this is a NO-TRADE filter, "
                        "not a stock picker)")
            roe_min      = st.slider("Quality: min ROE %",              -5.0, 20.0, 5.0, 0.5)
            roce_min     = st.slider("Quality: min ROCE % (non-fin only)", -5.0, 30.0, 5.0, 0.5)
            de_max       = st.slider("Quality: max D/E (non-fin only)",  0.5, 10.0, 3.0, 0.1)
            ic_min       = st.slider("Quality: min interest cover (×)",  0.5, 10.0, 1.5, 0.1)
            cr_min       = st.slider("Quality: min current ratio",       0.3,  3.0, 0.8, 0.05)
            evebitda_max = st.slider("Valuation: max EV/EBITDA (abs)",   5.0, 100.0, 30.0, 1.0)
            peg_max      = st.slider("Valuation: max PEG ratio",         0.5, 10.0, 3.0, 0.1)
            pat_decline  = st.slider("Growth: PAT YoY decline floor %", -80.0, 0.0, -25.0, 1.0)
            pledge_max   = st.slider("Governance: max promoter pledge %", 0.0, 100.0, 40.0, 5.0)
            phold_min    = st.slider("Governance: min promoter holding %", 0.0, 75.0, 15.0, 1.0)
            dii_min      = st.slider("Ownership: min DII delta QoQ (pp)", -10.0, 0.0, -3.0, 0.5)
            mf_min       = st.slider("Ownership: min MF delta QoQ (pp)",  -10.0, 0.0, -3.0, 0.5)
            # ---- Trend & liquidity thresholds (new) ----
            r12m_min     = st.slider("🆕 Trend: min 12-month return %",  -50.0, 20.0, -15.0, 1.0,
                help="Rejects secular downtrenders. Default −15%: 12-mo return "
                     "worse than −15% ⇒ NO-TRADE. Tighten to −5% for stricter "
                     "trend requirement.")
            slope_min    = st.slider("🆕 Trend: min 200-DMA slope %",   -20.0, 10.0, 0.0, 0.5,
                help="Slope of 200-DMA over last 60 sessions. Default 0% ⇒ "
                     "reject any stock whose long-term trend is turning down.")
            turn_min     = st.slider("🆕 Liquidity: min avg turnover (₹cr)", 0.0, 50.0, 5.0, 0.5,
                help="20-day avg ₹ turnover floor. Default ₹5 cr filters smallcap "
                     "traps you can't actually enter at size.")

        # ==============================================================================
        # SECTION 2c — NEWS & EVENT RISK  🆕 (Aug-2026)
        # ------------------------------------------------------------------------------
        # Tier 1 (event risk): pulls scheduled corporate events (results / board
        #   meetings / dividends / splits / bonuses / AGMs) from NSE's own API. If a
        #   stock has a scheduled event in the next N sessions → HARD BLOCK the trade.
        #   Prevents "buy today, gap 12% on results tomorrow" blowups.
        # Tier 2 (news sentiment): keyword-based sentiment on yfinance news + Google
        #   News RSS. Tilts the rank_score ±20% based on last-3-day headlines.
        #   Detects upgrades, order wins, downgrades, SEBI probes, resignations, etc.
        # ==============================================================================
        st.header("2c · News & Event risk  🆕")
        news_available = HAVE_NEWS
        if not news_available:
            st.caption("⚠️ News/event modules unavailable (missing dependency). Section disabled.")
            use_event_block = False
            use_news_tilt = False
            event_window = 5
        else:
            use_event_block = st.checkbox(
                "🚫 Block trades with a pending event (results / board meeting / dividend)",
                value=True,
                help="Fetches scheduled events from NSE's official announcements API for each "
                     "signalling stock. If an event falls within the window below, that stock is "
                     "removed from tonight's shortlist. Highest-ROI news factor — prevents "
                     "results-gap blowups where a 12% swing setup becomes a 15% overnight loss.")
            event_window = st.slider(
                "Block if event is within (trading sessions)", 1, 15, 5, 1,
                disabled=not use_event_block,
                help="5 sessions = ~7 calendar days. Tighter (3) allows more trades but less "
                     "safety margin; wider (10) blocks more.")
            use_news_tilt = st.checkbox(
                "📰 Apply news-sentiment tilt to rank score (±20%)",
                value=True,
                help="Fetches recent headlines (yfinance wire + Google News RSS), scores them "
                     "with an Indian-market keyword lexicon (upgrades, order wins, probes, "
                     "resignations, etc). Positive news lifts rank; negative dampens. Bounded "
                     "±20% so it never dominates the technical signal — just a tie-breaker.")
        strategy = st.selectbox("Strategy", ["PASS_combined", "PASS_recommended", "PASS_tight",
                                             "PASS_balanced", "PASS_reversal"], index=0)
        target_pct = st.number_input("Target (%)", 1.0, 100.0, 15.0, 0.5)
        cA, cB = st.columns(2)
        min_hold = cA.number_input("Min hold (d)", 1, 60, 1)
        max_hold = cB.number_input("Max hold (d)", 1, 120, 30)
        entry_choice = st.radio("Entry style",
                                ["Limit near signal close (recommended)", "Market at next open"], index=0,
                                help="Market buys whatever the open gives — a gap-up means a worse fill. "
                                     "Limit places a resting buy and skips the trade if price never returns.")
        if entry_choice.startswith("Limit"):
            entry_mode = "Limit"
            limit_pct = st.slider("Limit below signal close (%)", 0.0, 5.0, 0.5, 0.1,
                                  help="0 = order at the signal close. Higher = wait for a deeper "
                                       "pullback: better fills, but more signals never fill.")
            fill_days = st.number_input("Order valid for (sessions)", 1, 5, 1)
        else:
            entry_mode, limit_pct, fill_days = "Market open", 0.0, 1
        exit_mode = st.radio("Exit style", ["Trailing", "Fixed target"], index=1,
                             help="Trailing lets winners run past the target.")
        trail_mult = st.slider("Trailing x ATR", 0.5, 5.0, 2.0, 0.5) if exit_mode == "Trailing" else 2.0
        lock_pct = st.slider("Lock profit once objective hit (%)", 0.0, 30.0, 10.0, 0.5,
                             help="After +10% is touched, the stop never falls below this. "
                                  "Protects the objective while letting the trade run to 20-30%.") \
                   if exit_mode == "Trailing" else None
        cut_on = st.checkbox("Cut dead trades early (conviction exit)", value=False,
                             help="Still red after N days -> free the capital for the next signal.")
        cut_day = st.number_input("Cut on day", 1, 10, 2) if cut_on else None
        cut_threshold = st.slider("if return below (%)", -8.0, 2.0, 0.0, 0.5) if cut_on else 0.0

        st.header("4 - Risk")
        stop_anchor = st.radio("Stop anchoring", ["ATR distance", "Structure (swing low)"],
                               index=1,
                               help="Structure = below recent support; survives normal pullbacks "
                                    "in a channel. ATR = fixed volatility distance. "
                                    "**Default = Structure** (per Aug-2026 review — structure stops "
                                    "outperform ATR on this universe).")
        stop_anchor = "Structure" if stop_anchor.startswith("Structure") else "ATR"
        trail_anchor = st.radio("Trail anchoring", ["ATR distance", "Structure (rising swing low)"],
                                index=1,
                                help="Structure trailing: bigger winners per trade, longer holds. "
                                     "**Default = Structure**.")
        trail_anchor = "Structure" if trail_anchor.startswith("Structure") else "ATR"
        stop_value = st.slider("Stop (x ATR)", 0.5, 5.0, 2.0, 0.5)
        max_stop_pct = st.slider("Max loss cap (%)", 2.0, 20.0, 10.0, 0.5)
        max_atr_pct = st.slider("Skip if ATR% above", 3.0, 15.0, 8.0, 0.5)
        cost_pct = st.number_input("Round-trip cost (%)", 0.0, 5.0, 0.20, 0.05)
        apply_stcg = st.checkbox("Apply 20% STCG on gains", value=True,
                                 help="ON by default: ignoring tax overstates the edge.")

        # ==============================================================================
        # SECTION 4b — EXIT-STACK v2  (Layers A + B + C, refined defaults)
        # ------------------------------------------------------------------------------
        # v2 changes vs v1:
        #   A ratchet ladder — first rung raised from (peak 10%, floor 3%) to
        #     (peak 15%, floor 5%) to stop cutting choppy stocks like PAYTM/RBLBANK
        #     at breakeven on ordinary 10-12% pushes.
        #   C mom-exit arm threshold — dropped from 15% to 10% (v1 only fired on
        #     ~0.8% of trades; most rollovers happen below 15%).
        # All three default ON. Uncheck to A/B test against legacy.
        # ==============================================================================
        st.header("4b - Exit stack  🆕 A + B + C")
        st.caption("Stops good winners turning into small winners. Any layer fires "
                   "→ trade exits. **Refined defaults from v1 backtest analysis.**")
        use_ratchet = st.checkbox(
            "🔒 A · Ratcheting profit lock  (softer v2 ladder)",
            value=False,
            help="Peak crosses 15/25/40/60/85/120 % → floor moves to 5/12/25/40/60/85 %. "
                 "Softer than v1: first rung armed at +15% peak (was +10%), so ordinary "
                 "10-12% pushes without follow-through don't get cut at breakeven. "
                 "**Default OFF** — router chooses Fixed vs Trailing; layers add complexity "
                 "we'd rather A/B-test in explicitly."
        )
        use_shrink = st.checkbox(
            "📉 B · Shrinking trail multiplier",
            value=False,
            help="Trail width narrows as gain grows: 2.0×ATR below 10%, 1.5×ATR at "
                 "10–25%, 1.0×ATR at 25–50%, 0.75×ATR above 50%. Uses CURRENT-day ATR. "
                 "**Default OFF**."
        )
        use_momexit = st.checkbox(
            "⚡ C · Momentum-exhaustion exit  (arm at 10%, was 15%)",
            value=False,
            help="Fires on RSI rollover / MACD flip / 2 down closes on heavy volume / "
                 "bearish engulfing / 5-DMA break. Exit at next-day open (no look-ahead). "
                 "**Default OFF**."
        )
        mom_min_gain = st.slider(
            "C · Arm momentum exit only when up ≥ (%)", 5.0, 30.0, 10.0, 1.0,
            disabled=not use_momexit,
            help="Threshold below which momentum signals are ignored. Lowered from 15 "
                 "to 10 based on v1 results (only 0.8% of trades hit 15% before exit)."
        )

        # ==============================================================================
        # SECTION 4c — EXIT-STACK v2 EXTENSIONS  (Layers D + E)
        # ------------------------------------------------------------------------------
        # D (time-decay) — tightens trail on stalled trades; forced exit after N stalls.
        #     Safe complement to A/B/C: only affects trades that stopped making highs.
        # E (staircase)  — books fractional profits at fixed milestones. TRADEOFF: caps
        #     upside on runaway winners in exchange for guaranteed profit-taking on
        #     choppy ones. Default OFF because our v1 already showed A+B+C sometimes
        #     cuts winners too tight — this makes that direction more aggressive.
        # ==============================================================================
        st.header("4c - Exit stack extensions  🆕 D + E")
        st.caption("D safe to enable; E is a tradeoff — see help text.")
        use_decay = st.checkbox(
            "⏳ D · Time-decay tightening",
            value=False,
            help="If price hasn't made a new peak in N sessions, tighten the trail each "
                 "day. Forced exit at close after M stall sessions. Frees capital tied "
                 "up in trades that stopped rewarding you. **Default OFF**."
        )
        decay_after = st.slider(
            "D · Start tightening after (sessions without new high)", 3, 15, 5, 1,
            disabled=not use_decay,
            help="Trail begins tightening after this many sessions with no new peak."
        )
        decay_shrink = st.slider(
            "D · Tightening rate (% of stop-to-price gap per day)", 10.0, 60.0, 25.0, 5.0,
            disabled=not use_decay,
            help="Each stall session, stop rises this % of the gap between stop and close."
        )
        decay_exit = st.slider(
            "D · Force exit after (sessions without new high)", 5, 25, 10, 1,
            disabled=not use_decay,
            help="After this many stalled sessions, exit at close regardless of trail."
        )
        use_staircase = st.checkbox(
            "🪜 E · Staircase partial exits",
            value=False,
            help="Book 30% at +10%, another 30% at +20%, let 40% run with the trail. "
                 "GUARANTEES some profit on any decent move — but CAPS runaway winners "
                 "(e.g. HAL parabolic +60% peak: A+B+C returns +41%, +E drops to +24%). "
                 "Default OFF; enable if consistency matters more than home runs."
        )

        # ==============================================================================
        # SECTION 4f — POST-STOP COOLDOWN  🆕 (Aug-2026, Change #1)
        # ------------------------------------------------------------------------------
        # After a STOP outcome on a stock, block re-entries for N sessions.
        # Kills same-stock cluster losses (TITAN Jan-Apr 2024: 9 stops in a row;
        # ADANIENT Oct-Nov 2024: 4 stops averaging -12%). Signals typically
        # re-fire 2-5 days after a stop; in choppy/downtrending regimes each
        # re-entry stops out too, chaining 4-8 losing trades in 3-4 weeks.
        #
        # Evidence — 5-stock OOS demo (2024-01 .. 2025-06):
        #   cooldown=0  →  93 trades, 38.7% win, TOTAL  -5.5%
        #   cooldown=7  →  65 trades, 44.6% win, TOTAL +74.2%    (+79.7pp)
        # Every stock in the basket improved except TRENT (a runaway winner
        # where some re-entries were genuine — still positive).
        # ==============================================================================
        st.header("4f - Post-stop cooldown  🆕")
        cooldown_days = st.slider(
            "Block re-entries for N sessions after any STOP", 0, 30, 7, 1,
            help="After a stop-out on a stock, skip any new signal on THAT stock "
                 "for N calendar days. 0 = disabled (legacy behaviour). "
                 "7 (default) is the value that flipped a 5-stock 2024-2025 OOS "
                 "test from -5.5% to +74.2% total return. Ranges to try:\n"
                 "   3-5  : loose — catches obvious re-entries only\n"
                 "   7-10 : recommended — kills cluster losses without blocking most winners\n"
                 "  14-21 : strict — for choppy/high-vol universes\n"
                 "Cooldown fires ONLY on STOP outcomes; targets/trails/time exits do not trigger it."
        )
        st.caption("💡 On the 2024-2025 OOS test this single filter fixed the ADANIENT "
                   "(-52% → -5%) and TITAN (-43% → -9%) cluster disasters.")

        # ==============================================================================
        # SECTION 4e — HISTORICAL-TRACK-RECORD SELF-CHECK  🆕 (Aug-2026)
        # ------------------------------------------------------------------------------
        # Post-backtest filter: reject any stock where the algo's OWN historical
        # performance has been weak. Rationale — if the strategy has generated 400
        # trades on a stock and never made money, no future signal on that stock is
        # worth taking. Directly catches the 5 chronic losers even when they slip
        # past the fundamental gate. This is the "algo learns from its own mistakes"
        # loop we've been missing.
        # ==============================================================================
        st.header("4e - Historical self-check 🆕")
        st.caption("After the backtest runs, drop any stock where the algorithm's own "
                   "track record has been weak. Silver bullet against chronic losers "
                   "like RBLBANK / SAIL / INDUSINDBK that slipped past the fundamental gate.")
        use_selfcheck = st.checkbox(
            "🪞 Reject stocks where own algo history is weak",
            value=True,
            help="After the technical backtest, reject stocks whose historical win "
                 "rate OR total return falls below the floors set below. Uses the "
                 "algo's own track record as the arbiter — the strongest possible "
                 "filter against strategy-market mismatch."
        )
        with st.expander("Self-check thresholds"):
            selfcheck_min_win = st.slider("Min historical win rate %", 0.0, 60.0, 35.0, 1.0,
                disabled=not use_selfcheck,
                help="Stocks below this win rate on their own historical backtest "
                     "are dropped. Default 35% is conservative — the 5 chronic "
                     "losers all had win rates 25–33%.")
            selfcheck_min_total = st.slider("Min historical total return (sum) %",
                                             -500.0, 500.0, 0.0, 25.0,
                disabled=not use_selfcheck,
                help="Stocks whose cumulative net return sum is below this floor "
                     "are dropped. Default 0% — no track record of losing money "
                     "on aggregate.")
            selfcheck_min_trades = st.slider("Min historical SEQUENTIAL trades to apply filter",
                                              5, 100, 30, 1,
                disabled=not use_selfcheck,
                help="Only apply the filter if the stock has at least this many "
                     "historical SEQUENTIAL (non-overlapping) trades. Default 30 "
                     "(lowered from 50 in Aug-2026, M6 fix): after C5 switched the "
                     "self-check to sequential counts, most mid/small caps have "
                     "20-50 sequential trades over 10y — a 50 floor was silently "
                     "disabling the check on exactly the names it was designed to "
                     "catch.")

        # ==============================================================================
        # SECTION 4d — REGIME-AWARE EXIT ROUTER
        # ------------------------------------------------------------------------------
        # Decides per-trade whether to use trailing (with A/B/C) or fixed target at
        # entry, using 4 objective conditions on the entry bar. Trending trades keep
        # the fat right tail (parabolic runs); choppy trades get a hard +15% cap.
        # ==============================================================================
        st.header("4d - Regime-aware exit router  🆕")
        st.caption("Per-trade routing at entry: trending → trailing with A+B+C · "
                   "choppy/weak → fixed target. Solves the PAYTM-style regression "
                   "seen in A+B+C-only backtests without losing HAL/ADANIENT upside.")
        use_router = st.checkbox(
            "🧭 Route each trade by regime (recommended)",
            value=True,
            help="When ON, evaluates 4 conditions at each entry: (1) ADX ≥ threshold, "
                 "(2) 200-DMA rising, (3) price ≥ N% above 200-DMA, (4) 3-mo realized "
                 "vol > baseline. ALL 4 pass → trailing exit (A+B+C apply). Any fail "
                 "→ fixed target. When OFF, all trades use your global exit mode "
                 "(the current Section 3 setting)."
        )
        with st.expander("Router thresholds"):
            route_min_adx = st.slider("ADX(14) ≥", 10.0, 40.0, 20.0, 1.0,
                                       disabled=not use_router,
                                       help="Directional strength floor. Standard "
                                            "convention: ADX ≥ 20 = trending. Below = chop.")
            route_sma_slope_lb = st.slider("200-DMA rising over (sessions)", 5, 60, 20, 1,
                                            disabled=not use_router,
                                            help="How many sessions back to compare "
                                                 "current 200-DMA to. 20 sessions ≈ 1 "
                                                 "month — captures a genuine slope, not day-to-day noise.")
            route_min_dist_pct = st.slider("Price above 200-DMA by ≥ (%)", 0.0, 30.0, 15.0, 0.5,
                                            disabled=not use_router,
                                            help="Distance floor: stock must be clearly ABOVE "
                                                 "its long trend, not just touching it.")
            route_fixed_target_pct = st.slider("Fixed-target level when choppy (%)", 5.0, 30.0, 15.0, 0.5,
                                                disabled=not use_router,
                                                help="When routed to fixed-target, exit at "
                                                     "this gross profit. Defaults to your "
                                                     "primary +15% objective.")
            route_vol_lb = st.slider("Realized-vol window (sessions)", 20, 126, 63, 1,
                                      disabled=not use_router,
                                      help="Window for computing realized volatility. "
                                           "63 sessions ≈ 3 months (as originally specified).")
            route_vol_baseline_lb = st.slider("Vol baseline window (sessions)", 60, 504, 252, 1,
                                               disabled=not use_router,
                                               help="Look-back window for the vol baseline. "
                                                    "Current vol must EXCEED this rolling "
                                                    "median. 252 = 1 year.")

        st.header("7 - Market regime")
        use_gate = st.checkbox("Apply market-regime gate (ranking-level, RS filter)",
                               value=True,
                               help="On RISK-OFF days, show only stocks beating the market "
                                    "(positive relative strength) instead of a full basket of longs. "
                                    "SOFT filter — trims the shortlist AFTER signals fire.")
        # ---- CHANGE #3 (Aug-2026) — regime hard-block at signal generation ----
        # DEFAULT OFF based on Nifty 500 top-100 stress test:
        #   Baseline (7.5y, 6749 trades): +17609.51% total, 54.3% win
        #   Change #3 ON: +17041.36% total, 54.5% win  (285 trades blocked,
        #     blocked-set avg = +1.99% i.e. mostly winners, net cost -568pp)
        # The confirmation + cooldown + fundamentals stack already filters out
        # the weak RISK-OFF signals; the residual RISK-OFF signals on top-100
        # Nifty stocks tend to be dip-buys that recover. Toggle ON only for
        # weaker universes (AllNSE / small-caps) or if you want a stricter
        # regime discipline regardless of the return cost.
        block_risk_off = st.checkbox(
            "🚫 HARD-BLOCK signals on RISK-OFF bars (Change #3)", value=False,
            help="Aug-2026 experimental — when the benchmark (Nifty 500) is below its 200-DMA "
                 "AND 10-day ROC < -3%, force signal=False for ALL stocks on that day. Applied "
                 "INSIDE the engine so both the historical trade log AND today's live signals "
                 "respect the rule.\n\n"
                 "⚠️ Evidence on Nifty 500 top-100 (2018-2025): blocked trades were on average "
                 "**profitable** (+1.99%), so the filter COST ~3% of total return. Keep OFF for "
                 "Nifty 100/500 baskets. Consider ON for AllNSE / small-cap universes where "
                 "RISK-OFF signal quality is worse.")

        st.header("8 - Diversification")
        max_per_sector = st.slider("Max names per sector", 0, 10, 3,
                                   help="Caps how many stocks from one industry can enter the "
                                        "shortlist (highest-ranked kept). 0 = no cap. Prevents "
                                        "one sector's bad day from sinking the whole book.")

        with st.expander("Advanced filter thresholds"):
            p = {
                # Aug-2026 EVIDENCE-BASED CHANGE: default lowered 15 → 8.
                # Multi-cutoff walk-forward on Nifty 500 showed the 15% threshold
                # entered momentum stocks TOO LATE (already stretched, near local
                # tops). Lowering to 8% catches trends earlier when there's still
                # room to run. Combined with the "signal day confirmation" toggle
                # below, this took a 9% win-rate loser to a 50% win-rate winner.
                "regime": st.slider("Uptrend: % above 200-DMA", 0.0, 50.0, 8.0, 1.0,
                                    help="Was 15%; lowered to 8% based on walk-forward evidence"),
                "atr":    st.slider("Volatility floor: ATR%", 0.0, 10.0, 3.5, 0.5),
                "roc":    st.slider("Breakout ROC(10) >", 0.0, 15.0, 3.0, 0.5),
                "volr":   st.slider("Breakout volume ratio >", 0.5, 4.0, 1.2, 0.1),
                "rsi_os": st.slider("Reversal oversold RSI <", 10.0, 45.0, 30.0, 1.0),
            }
        # Aug-2026 EVIDENCE-BASED ADDITION — same-day confirmation filter.
        require_confirmation = st.checkbox(
            "🆕 Require signal-day confirmation (green close + volume)  🚀",
            value=True,
            help="EVIDENCE-BASED FILTER (Aug-2026): only take signals where the signal "
                 "day itself closes GREEN (close > open) AND on above-average volume "
                 "(vol > 20-day avg). Walk-forward on Nifty 500 showed this filter "
                 "converts a 9% win-rate loser (11 trades, avg −4.57%) into a 50% "
                 "win-rate winner (4 trades, avg +6.75%) by rejecting failed breakouts. "
                 "Cuts trade count ~65% — quality over quantity. Recommend leaving ON.")
        run = st.button("Scan market", type="primary", use_container_width=True)

    if not run and "scan" not in st.session_state:
        st.info("Pick a segment and click Scan market. Tip: for a nightly full run, schedule this "
                "after 4pm IST once you've confirmed a small run works.")
        return

    if run:
        bt_kwargs = dict(target_pct=target_pct, max_hold=int(max_hold),
                         min_hold=int(min_hold),          # M7 FIX — was silently ignored
                         stop_method="ATR",
                         stop_value=stop_value, cost_pct=cost_pct, apply_stcg=apply_stcg,
                         rev_target_pct=6.0, rev_stop_value=1.5,
                         exit_mode=("Trailing" if exit_mode == "Trailing" else "Fixed target"),
                         trail_mult=trail_mult, max_stop_pct=max_stop_pct, max_atr_pct=max_atr_pct,
                         entry_mode=entry_mode, limit_pct=limit_pct, fill_days=int(fill_days),
                         lock_pct=lock_pct, cut_day=(int(cut_day) if cut_day else None),
                         cut_threshold=cut_threshold, partial_frac=0.0, partial_atr=3.0,
                         stop_anchor=stop_anchor, trail_anchor=trail_anchor,
                         # ---- Section 4b: A + B + C ----
                         ratchet_lock=use_ratchet,
                         shrink_trail=use_shrink,
                         momentum_exit=use_momexit,
                         mom_exit_min_gain=mom_min_gain,
                         # ---- Section 4c: D + E ----
                         time_decay=use_decay,
                         decay_after_days=int(decay_after),
                         decay_shrink_pct=decay_shrink,
                         decay_exit_days=int(decay_exit),
                         staircase=use_staircase,
                         # ---- Section 4d: Regime-aware exit router ----
                         regime_route=use_router,
                         route_min_adx=route_min_adx,
                         route_sma_slope_lb=int(route_sma_slope_lb),
                         route_min_dist_pct=route_min_dist_pct,
                         route_vol_lb=int(route_vol_lb),
                         route_vol_baseline_lb=int(route_vol_baseline_lb),
                         route_fixed_target_pct=route_fixed_target_pct,
                         # ---- Section 4f: Post-stop cooldown (Change #1) ----
                         post_stop_cooldown_days=int(cooldown_days))

        # --- market regime + relative-strength benchmark (fetched once) ---
        bench_name, idx_df = fetch_index(start, end)
        regime = compute_regime(idx_df)
        if not regime.get("index_ok"):
            st.warning("⚠️ Benchmark index unavailable — relative strength falls back to ABSOLUTE "
                       "momentum (not market-relative), and the index leg of the gate is skipped. "
                       "Breadth is still computed from the scanned stocks.")
        idx_ret_window = regime.get("idx_ret_window", 0.0)
        segments = fetch_segments(start, end)
        sector_map = fetch_sector_map()
        if not sector_map:
            st.warning("⚠️ Sector data unavailable (NSE unreachable) — sector caps cannot be applied "
                       "this run. All stocks will show sector UNKNOWN.")

        # ================== FUNDAMENTAL NO-TRADE GATE ==================
        funda_results, funda_rejects_df = {}, pd.DataFrame()
        pre_gate_count = len(tickers[:max_n])
        if apply_funda_gate:
            if momentum_preset:
                # OPTION C — momentum-friendly preset OVERRIDES pillar toggles
                # and thresholds regardless of what the sliders show. See
                # explanation in the preset checkbox help text above.
                funda_cfg = {
                    **DEFAULT_FUNDA_CONFIG,
                    # Only quality (loose) + governance stay ON
                    "valuation_enabled":  False,
                    "quality_enabled":    True,
                    "growth_enabled":     False,
                    "governance_enabled": True,
                    "ownership_enabled":  False,
                    "trend_enabled":      False,
                    "strict_mode":        False,
                    # Quality — LOOSENED so momentum-friendly capital-intensive
                    # stocks (autos, utilities, infra) are not falsely rejected
                    "roe_min_%":            0.0,        # 0% floor — only reject actual loss-makers
                    "roce_min_%":           0.0,        # same
                    "debt_to_equity_max":   5.0,        # was 3.0 — ADANIGREEN/TVSMOTOR now pass
                    "interest_cover_min":   1.0,        # was 1.5 — INDIGO-level (0.7) still blocked
                    "current_ratio_min":    0.4,        # was 0.8 — telcos/subscriptions
                    # Governance — STRICT (real fraud/blowup protection)
                    "promoter_pledge_max_%":  40.0,     # unchanged
                    "promoter_holding_min_%": 15.0,     # unchanged
                    "flag_auditor_qualified": True,
                    "flag_rpt_concern":       True,
                }
                st.caption("🚀 **Momentum preset active** — quality (loose) + governance only. "
                           "D/E ≤ 5.0, interest cover ≥ 1.0, ROE / ROCE ≥ 0. "
                           "Uncheck the preset above to configure pillars manually.")
            else:
                funda_cfg = {
                    **DEFAULT_FUNDA_CONFIG,
                    "valuation_enabled":  funda_valuation,
                    "quality_enabled":    funda_quality,
                    "growth_enabled":     funda_growth,
                    "governance_enabled": funda_governance,
                    "ownership_enabled":  funda_ownership,
                    "trend_enabled":      funda_trend,
                    "strict_mode":        funda_strict,
                    # Quality
                    "roe_min_%":            roe_min,
                    "roce_min_%":           roce_min,
                    "debt_to_equity_max":   de_max,
                    "interest_cover_min":   ic_min,
                    "current_ratio_min":    cr_min,
                    # Valuation
                    "ev_ebitda_max":        evebitda_max,
                    "peg_max":              peg_max,
                    # Growth
                    "pat_yoy_decline_max_%": pat_decline,
                    # Governance
                    "promoter_pledge_max_%":  pledge_max,
                    "promoter_holding_min_%": phold_min,
                    # Ownership flow
                    "dii_delta_qoq_min_pp":   dii_min,
                    "mf_delta_qoq_min_pp":    mf_min,
                    # Trend & Liquidity (new)
                    "min_12m_return_%":       r12m_min,
                    "min_sma200_slope_%":     slope_min,
                    "min_avg_turnover_cr":    turn_min,
                }
            st.info("🧾 Running fundamentals no-trade screen "
                    "(24h cached — reruns are instant)...")
            tickers_yahoo = [to_yahoo(s) for s in tickers[:max_n]]
            f_prog = st.progress(0.0); f_stat = st.empty()

            def _fund_cb(k, n, sym):
                f_stat.write(f"Fund-check {sym.replace('.NS','')}  ({k+1}/{n})")
                f_prog.progress((k + 1) / n)

            funda_results, _sec_medians = fs_screen_universe(
                tickers_yahoo, sector_map, funda_cfg, _fund_cb
            )
            f_stat.empty(); f_prog.empty()

            passing_bare = {t for t, r in funda_results.items()
                            if r["status"] in ("pass", "pass_no_data")}
            filtered = [s for s in tickers[:max_n]
                        if s.upper().replace(".NS", "").replace(".BO", "") in passing_bare]
            summ = fs_summarize(funda_results)
            st.success(
                f"✅ Fundamentals gate: **{summ['pass']} passed**, "
                f"**{summ['reject']} rejected**, "
                f"{summ['no_data']} no-data (passed), "
                f"{summ['warn_only']} passed with warnings. "
                f"Technical scan now runs on {len(filtered)} names "
                f"(from {pre_gate_count})."
            )
            run_list = filtered
            funda_rejects_df = fs_rejects_df(funda_results)
        else:
            run_list = tickers[:max_n]
        # =============== END FUNDAMENTAL NO-TRADE GATE =================

        rows, prog, status = [], st.progress(0.0), st.empty()
        for k, sym in enumerate(run_list):
            status.write(f"Scanning {sym}  ({k+1}/{len(run_list)}) ...")
            rows.append(scan_one(to_yahoo(sym), start, end, strategy, p, bt_kwargs,
                                 idx_ret_window, sector_map,
                                 require_confirmation=require_confirmation,
                                 bench_close=(idx_df["Close"] if not idx_df.empty else None),
                                 block_risk_off=block_risk_off))
            prog.progress((k + 1) / len(run_list))
            time.sleep(0.05)
        status.empty(); prog.empty()

        # ================== NEWS & EVENT-RISK PASS  🆕 (Aug-2026) =========
        # Fetch news/events for EVERY fundamentally-passing stock, not just
        # those firing a technical signal today. Rationale: news can be a
        # LEADING indicator — a stock with major positive news today may set
        # up for a technical signal tomorrow or day-after. Missing that
        # means the "News-driven opportunities" section can't surface real
        # catalyst plays that the technical filter hasn't yet caught up to.
        # Runtime cost: adds ~1-2 min per 100-stock scan (cached 60 min so
        # subsequent runs within the hour are instant).
        if (use_event_block or use_news_tilt) and HAVE_NEWS:
            signal_rows = [r for r in rows if r.get("status") == "ok"]
            if signal_rows:
                n_prog = st.progress(0.0); n_stat = st.empty()
                for k, r in enumerate(signal_rows):
                    tk_bare = r["ticker"]
                    n_stat.write(f"News/event check: {tk_bare}  ({k+1}/{len(signal_rows)})")
                    # Event risk (NSE announcements)
                    if use_event_block:
                        try:
                            ev = _event_risk(tk_bare, next_sessions=int(event_window))
                        except Exception:
                            ev = {"blocked": False, "type": None, "days_until": None,
                                  "subject": None}
                        r["event_blocked"]     = bool(ev.get("blocked"))
                        r["event_type"]        = ev.get("type")
                        r["event_days_until"]  = ev.get("days_until")
                        r["event_subject"]     = ev.get("subject")
                    else:
                        r["event_blocked"] = False
                        r["event_type"] = None
                        r["event_days_until"] = None
                        r["event_subject"] = None
                    # News sentiment
                    if use_news_tilt:
                        try:
                            n = _news_score(r["yahoo"])
                        except Exception:
                            n = {"score": 0.0, "n_articles": 0, "top_headline": None,
                                 "matched_terms": []}
                        r["news_score"]     = float(n.get("score", 0.0))
                        r["news_n"]         = int(n.get("n_articles", 0))
                        r["news_top"]       = n.get("top_headline")
                        r["news_matched"]   = ",".join(n.get("matched_terms", [])[:5])
                        # Rank blending — news tilts ±20%.
                        tilt = max(min(r["news_score"], 1.0), -1.0) * 0.20
                        r["rank_score"] = round(r["rank_score"] * (1.0 + tilt), 2)
                    else:
                        r["news_score"] = 0.0; r["news_n"] = 0
                        r["news_top"] = None; r["news_matched"] = ""
                    n_prog.progress((k + 1) / len(signal_rows))
                n_stat.empty(); n_prog.empty()
                blocked_ct = sum(1 for r in signal_rows if r.get("event_blocked"))
                if blocked_ct:
                    st.warning(f"🚫 **{blocked_ct} signalling stock(s) BLOCKED** by upcoming events "
                               f"(within {event_window} sessions). See 'Event-risk blocked' section below.")

        # Defaults for stocks that didn't signal (so column always exists)
        for r in rows:
            r.setdefault("event_blocked", False)
            r.setdefault("event_type", None)
            r.setdefault("event_days_until", None)
            r.setdefault("event_subject", None)
            r.setdefault("news_score", 0.0)
            r.setdefault("news_n", 0)
            r.setdefault("news_top", None)
            r.setdefault("news_matched", "")

        # breadth from the scanned universe, then the composite verdict
        breadth = compute_breadth(rows)
        gate = composite_gate(regime, segments, breadth)

        # persist so row-clicks (which rerun the script) don't trigger a re-scan
        st.session_state["scan"] = {
            "res": pd.DataFrame(rows), "start": start, "end": end, "strategy": strategy,
            "p": p, "bt_kwargs": bt_kwargs, "yrs": yrs, "exit_mode": exit_mode,
            "trail_mult": trail_mult, "target_pct": target_pct, "min_hold": min_hold,
            "max_hold": max_hold, "stop_value": stop_value, "max_stop_pct": max_stop_pct,
            "max_atr_pct": max_atr_pct, "cost_pct": cost_pct, "apply_stcg": apply_stcg,
            "regime": regime, "bench_name": bench_name or "index n/a", "use_gate": use_gate,
            "segments": segments, "breadth": breadth, "gate": gate,
            "max_per_sector": max_per_sector,
            "entry_mode": entry_mode, "limit_pct": limit_pct, "fill_days": fill_days,
            "funda_results": funda_results,
            "funda_rejects_df": funda_rejects_df,
            # ---- Section 4e: historical self-check filter ----
            "use_selfcheck":        use_selfcheck,
            "selfcheck_min_win":    selfcheck_min_win,
            "selfcheck_min_total":  selfcheck_min_total,
            "selfcheck_min_trades": selfcheck_min_trades,
        }

    render_results()


def render_stock_backtest(row, S, key_prefix=""):
    """Full drill-down for one analyzed stock: chart + every backtest trade with all
    technical parameters at entry. `row` is a scan_one result row (needs 'yahoo'/'ticker')."""
    tick = row["ticker"]
    st.markdown(f"### 📈 {tick}  —  price, entries & outcomes")
    cc = st.columns(5)
    cc[0].metric("Last close", f"₹{row.get('entry_ref', row.get('last_close', '—'))}")
    cc[1].metric("Objective", f"₹{row.get('target_price', '—')}")
    cc[2].metric("Stop-loss", f"₹{row.get('stop_price', '—')}",
                 f"{row.get('stop_%', '')}%")
    cc[3].metric("Exp. days→objective", row.get("exp_days_to_target", "—"))
    cc[4].metric("Exp/DAY", f"{row.get('exp_per_day_%', 0)}%")

    with st.spinner(f"Building {tick} backtest…"):
        fig, trades_one = build_stock_chart(row["yahoo"], S["start"], S["end"],
                                            S["strategy"], S["p"], S["bt_kwargs"])
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}fig_{tick}")
        st.caption("Green = objective hit · light green = trailing profit · red = stop · grey = time exit. "
                   "Lower panel = cumulative net % of all historical trades.")

    if trades_one is None or trades_one.empty:
        st.info("No historical trades for this stock under the current settings.")
        return

    t = trades_one.copy()
    st.markdown(f"#### 🧾 Every backtest trade for {tick}  ({len(t)} trades)")
    stats_one = engine.summarize(t)

    def _fmt1(v, dash="—"):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "∞" if v == np.inf else dash
        return f"{v:.2f}" if isinstance(v, float) else f"{v}"

    wins = (t["net_return_%"] > 0).sum()
    q = st.columns(6)
    q[0].metric("① Trades", len(t))
    q[1].metric("② Win rate", f"{100*wins/len(t):.0f}%")
    q[2].metric("⑭ Hit objective", f"{100*(t['outcome']=='TARGET').mean():.0f}%")
    q[3].metric("③ Avg net / trade", f"{t['net_return_%'].mean():+.2f}%")
    q[4].metric("Best / Worst", f"{t['net_return_%'].max():+.0f}% / {t['net_return_%'].min():+.0f}%")
    q[5].metric("⑫ Avg hold", f"{t['days_held'].mean():.1f}d")

    q2 = st.columns(6)
    q2[0].metric("④ Total return (sum)", f'{stats_one.get("total_return_sum_%", 0):+.2f}%')
    q2[1].metric("⑤ CAGR", f'{stats_one.get("cagr_%", 0):+.2f}%' if "cagr_%" in stats_one else "—")
    q2[2].metric("⑥ Max drawdown", f'{stats_one.get("max_drawdown_%", 0):.2f}%' if "max_drawdown_%" in stats_one else "—")
    q2[3].metric("⑦ Profit factor", _fmt1(stats_one.get("profit_factor")))
    q2[4].metric("⑧ Recovery factor", _fmt1(stats_one.get("recovery_factor")))
    q2[5].metric("⑬ Max consec. losses", stats_one.get("max_consecutive_losses", "—"))

    q3 = st.columns(6)
    q3[0].metric("⑨ Avg winner", f'{stats_one.get("avg_win_%", 0):+.2f}%')
    q3[1].metric("⑩ Avg loser", f'{stats_one.get("avg_loss_%", 0):+.2f}%')
    q3[2].metric("⑪ Reward/Risk", _fmt1(stats_one.get("reward_risk_ratio")))
    q3[3].metric("⑮ Stop %", f'{stats_one.get("stop_pct_of_all", 0)}%')
    q3[4].metric("⑯ Trail %", f'{stats_one.get("trail_pct_of_all", 0)}%')
    q3[5].metric("⑰ Time exit %", f'{stats_one.get("time_pct_of_all", 0)}%')
    st.caption("① Total trades ② Win rate ③ Avg return/trade ④ Total return (sum) ⑤ CAGR "
               "⑥ Max drawdown ⑦ Profit factor ⑧ Recovery factor ⑨⑩ Avg winner/loser "
               "⑪ Reward/Risk ⑫ Avg holding days ⑬ Max consecutive losses ⑭ Target hit % "
               "⑮ Stop-loss % ⑯ Trailing-stop % ⑰ Time exit %. ⑤⑥⑧⑬ use a sequential "
               "(one-trade-at-a-time) equity curve — see the main table caption for why.")

    # ---- Regime router usage summary (only if router populated the audit column) ----
    if "exit_route" in t.columns and (t["exit_route"] != "").any():
        tr = t[t["exit_route"] == "trailing"]
        fx = t[t["exit_route"] == "fixed"]
        total = len(t)
        r1 = st.columns(4)
        r1[0].metric("🧭 Router: trailing #",
                     f'{len(tr)} ({100*len(tr)/total:.0f}%)',
                     help="Trades that met all 4 trending conditions at entry "
                          "and used trailing exit with A+B+C protection.")
        r1[1].metric("🧭 Router: fixed-target #",
                     f'{len(fx)} ({100*len(fx)/total:.0f}%)',
                     help="Trades that failed at least one condition at entry "
                          "and were routed to fixed target.")
        _mean = lambda d: (d["net_return_%"].mean() if len(d) else float("nan"))
        _win  = lambda d: (100*(d["net_return_%"]>0).mean() if len(d) else float("nan"))
        r1[2].metric("Trailing avg / win%",
                     f'{_mean(tr):+.2f}% / {_win(tr):.0f}%' if len(tr) else "—")
        r1[3].metric("Fixed avg / win%",
                     f'{_mean(fx):+.2f}% / {_win(fx):.0f}%' if len(fx) else "—")
        st.caption("**Regime router**: each trade sees only the exit style the entry "
                   "regime called for. Compare the two right cells — if trailing is "
                   "outperforming fixed on the same stock, trending trades really are "
                   "worth the give-back risk; if fixed is winning, the current router "
                   "thresholds may be letting through too many marginal trends.")

    if "equity_curve" in stats_one and not stats_one["equity_curve"].empty:
        import plotly.graph_objects as go
        ec = stats_one["equity_curve"].copy()
        ec["growth_%"] = (ec["equity_mult"] - 1) * 100
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(x=pd.to_datetime(ec["exit_date"]), y=ec["growth_%"],
                                    name="Cumulative growth %", line=dict(color="#16a34a", width=1.4),
                                    fill="tozeroy", fillcolor="rgba(22,163,74,0.08)"))
        fig_eq.update_layout(height=220, margin=dict(t=10, b=10), showlegend=False,
                             title=dict(text="Sequential equity curve (one-trade-at-a-time)", font=dict(size=12)))
        st.plotly_chart(fig_eq, use_container_width=True, key=f"{key_prefix}eq_{tick}")
        if stats_one.get("years_sequenced", 99) < 1:
            st.caption("⚠️ Under 1 year of sequential trading history for this stock — CAGR is "
                       "annualised from a short window and should be treated as illustrative only.")

    facts = ["signal_date", "entry_date", "exit_date", "days_held", "outcome",
             "exit_route", "route_reason",
             "trade_type", "signal_close", "limit_price", "entry_price",
             "target_price", "stop_price", "exit_price",
             "gross_return_%", "net_return_%", "hit_target", "partial_taken", "peak_gain_%"]
    tech = [c for c in ["pct_vs_sma200", "pct_vs_sma20", "rsi14", "roc10", "atr_pct",
                        "vol_ratio", "macd_hist", "adx14", "bb_pctB", "dist_52wH",
                        "obv_slope10"] if c in t.columns]
    ordered = [c for c in facts if c in t.columns] + tech
    ordered += [c for c in t.columns if c not in ordered]
    t = t[ordered].sort_values("entry_date", ascending=False).reset_index(drop=True)
    st.dataframe(t, use_container_width=True, height=380, key=f"{key_prefix}tbl_{tick}")
    st.caption("Trade facts first (dates, entry/limit/objective/stop/exit, return, holding days, "
               "outcome), then the **technical parameters at entry** — trend (%vs 200/20-DMA, ADX), "
               "momentum (RSI, ROC, MACD), volatility (ATR%), volume ratio, Bollinger %B, distance "
               "from 52-week high, OBV slope. Exactly what the signal saw on the day it fired.")
    st.download_button(f"⬇️ Download {tick} full backtest", t.to_csv(index=False).encode(),
                       file_name=f"{tick}_backtest_trades.csv", mime="text/csv",
                       key=f"{key_prefix}dl_{tick}")
    with st.expander("What each technical column means"):
        st.markdown(
            "- **pct_vs_sma200 / sma20** — % above the 200- and 20-day moving average\n"
            "- **rsi14** — momentum oscillator (<30 oversold, >70 overbought)\n"
            "- **roc10** — 10-day rate of change\n"
            "- **macd_hist** — MACD histogram (momentum turning)\n"
            "- **adx14** — trend strength (>~25 = real trend)\n"
            "- **atr_pct** — volatility as % of price\n"
            "- **vol_ratio** — volume ÷ 20-day average\n"
            "- **bb_pctB** — position in Bollinger band (0 lower, 100 upper)\n"
            "- **dist_52wH** — % below the 52-week high\n"
            "- **obv_slope10** — accumulation vs distribution"
        )


def render_results():
    S = st.session_state.get("scan")
    if not S:
        return
    res = S["res"]
    ok = res[res["status"] == "ok"].copy()
    bad = res[res["status"] != "ok"].copy()

    # ================ HISTORICAL SELF-CHECK FILTER (Section 4e) ================
    # Reject stocks where the algo's OWN historical performance has been weak.
    # This catches chronic losers even when they slipped past the fundamental
    # gate. Applied AFTER scan_one populates hist_trades / win_% / total return
    # for every stock, BEFORE the results are displayed in any table.
    # Rejected stocks are moved to `selfcheck_rejects` (shown separately below
    # the main results so the user can see WHY the algo dropped them).
    selfcheck_rejects = pd.DataFrame()
    if S.get("use_selfcheck", False) and not ok.empty:
        min_win    = S.get("selfcheck_min_win", 35.0)
        min_total  = S.get("selfcheck_min_total", 0.0)
        min_trades = S.get("selfcheck_min_trades", 50)

        # Only apply the filter to stocks with enough SEQUENTIAL trades to be
        # statistically meaningful. Stocks below `min_trades` pass automatically
        # (small-sample protection — don't drop a stock that lost 1 of 2 trades).
        # C5 FIX (Aug-2026): switched to `seq_*` fields — the overlapping pool
        # over-counts (400 raw signals often boil down to 40 non-overlapping
        # trades). Using seq_trades makes the min_trades floor honest.
        n_trades  = pd.to_numeric(ok.get("seq_trades", 0), errors="coerce").fillna(0)
        win_rate  = pd.to_numeric(ok.get("seq_win_%", 0), errors="coerce").fillna(0)
        tot_ret   = pd.to_numeric(ok.get("seq_total_return_%", 0), errors="coerce").fillna(0)

        # A stock is chronically bad if BOTH win rate AND total return are weak
        # (either one alone can happen by chance in a strong-trending stock).
        chronic_mask = ((n_trades >= min_trades) &
                        (win_rate < min_win) &
                        (tot_ret  < min_total))
        selfcheck_rejects = ok[chronic_mask].copy()
        ok = ok[~chronic_mask].copy()

        if not selfcheck_rejects.empty:
            selfcheck_rejects["_selfcheck_reason"] = (
                "win " + win_rate[chronic_mask].round(1).astype(str) + "% < " +
                f"{min_win:.0f}% AND total " +
                tot_ret[chronic_mask].round(0).astype(int).astype(str) +
                f"pp < {min_total:+.0f}pp over " +
                n_trades[chronic_mask].astype(int).astype(str) + " trades"
            )

    # ---------- market-regime banner ----------
    regime = S.get("regime", {"status": "UNKNOWN"})
    use_gate = S.get("use_gate", True)
    bench = S.get("bench_name", "index")
    segments = S.get("segments", {})
    breadth = S.get("breadth", {"status": "UNKNOWN"})
    gate = S.get("gate", {"final": regime.get("status", "UNKNOWN"), "reasons": []})
    rstat = gate.get("final", "UNKNOWN")          # composite verdict drives the gate

    seg_txt = " · ".join(f"{s} {v['pct_vs_200']:+.1f}% vs 200-DMA" for s, v in segments.items()) \
              or "segment indices unavailable"
    br_txt = (f"breadth {breadth.get('status')} "
              f"({breadth.get('advancers','?')} adv / {breadth.get('decliners','?')} dec, "
              f"{breadth.get('pct_above_50dma','?')}% above 50-DMA)")
    idx_txt = (f"{bench} {regime.get('pct_vs_200','?')}% vs 200-DMA, 10d {regime.get('roc10','?')}%"
               if regime.get("index_ok") else f"{bench} unavailable")

    if rstat == "RISK-ON":
        st.success(f"🟢 Market: **RISK-ON** — {idx_txt} · {br_txt} · {seg_txt}. "
                   "Long setups favoured; full shortlist shown.")
    elif rstat == "NEUTRAL":
        st.warning(f"🟡 Market: **NEUTRAL** — {idx_txt} · {br_txt} · {seg_txt}. "
                   + ("Gate ON: list trimmed to relative-strength leaders (RS > 0)."
                      if use_gate else "Gate OFF: full list shown."))
    elif rstat == "RISK-OFF":
        veto = gate.get("breadth_veto")
        st.error(f"🔴 Market: **RISK-OFF** — {idx_txt} · {br_txt} · {seg_txt}. "
                 + ("⚠️ **Breadth veto**: the headline index is green but the broad market is falling — "
                    "exactly the trap that sinks a basket of longs. " if veto else "")
                 + ("Gate ON: only stocks beating the market (RS > 0) are shown."
                    if use_gate else "Gate OFF: full basket of longs shown (higher risk)."))
    else:
        st.info("⚪ Market state unknown (index + breadth unavailable) — gate not applied this run.")
    if gate.get("reasons"):
        st.caption("Gate inputs → " + " | ".join(gate["reasons"]))

    # ---------- point 5: how prioritisation works ----------
    with st.expander("❓ How are stocks prioritised?  (ranking logic)", expanded=False):
        st.markdown(
            "Candidates are ranked by a **blended score = confidence × relative-strength tilt**:\n\n"
            "**confidence = max(expectancy, 0) × win-rate × sample-size-damping × 10** — the historical edge.\n\n"
            "**relative strength (RS%)** = the stock's return minus the index's over ~3 months. "
            "RS > 0 means it's *beating the market*. The blend nudges market-beating stocks up and "
            "laggards down (bounded ±50%), so on weak days the leaders rise to the top.\n\n"
            "**Market-regime gate (composite):** three inputs decide RISK-ON / NEUTRAL / RISK-OFF — "
            "(1) the broad index vs its 200-DMA and 10-day momentum, (2) the **mid/small-cap segment "
            "indices** vs their own 200-DMA (your universe lives there, not in the IT mega-caps that "
            "can lift the headline), and (3) **advance/decline breadth** counted across the stocks you "
            "just scanned. Negative breadth can **veto a green index** — the exact trap where a basket "
            "of longs sinks while the Nifty prints positive. On RISK-OFF *and* NEUTRAL the gate keeps "
            "only positive-RS names. Sample-size damping still discounts thin, lucky histories.\n\n"
            "**Sector-exposure cap:** after ranking and gating, at most *N* names per industry survive "
            "(the highest-ranked ones). A shortlist of 98 stocks that are really 4–5 correlated sector "
            "bets offers *fake* diversification — one sector's bad day sinks the whole book. The cap "
            "converts that into real diversification. Sectors come from NSE's own Industry classification."
        )

    if ok.empty:
        st.warning("No stocks scanned successfully. Re-run (Yahoo can be patchy).")
    else:
        # ======= TABLE 1: TONIGHT'S INVESTMENT ANALYSIS (action) =======
        st.subheader("🎯 Tonight's Investment Analysis  —  what to do if you buy tomorrow")
        cand = ok[ok["signals_today"]].copy()

        # NEWS/EVENT hard-block — remove stocks with a scheduled event in the
        # blocking window. Split them into `event_blocked_df` for display below.
        event_blocked_df = pd.DataFrame()
        if "event_blocked" in cand.columns and cand["event_blocked"].any():
            event_blocked_df = cand[cand["event_blocked"]].copy()
            cand = cand[~cand["event_blocked"]].copy()

        # apply composite gate: on RISK-OFF and NEUTRAL, keep only market-beating (positive RS) names
        gate_note = ""
        if use_gate and rstat in ("RISK-OFF", "NEUTRAL") and not cand.empty:
            before = len(cand)
            cand = cand[cand["rel_strength"] > 0]
            gate_note = (f"{rstat} gate: removed {before - len(cand)} laggard(s), "
                         f"kept {len(cand)} name(s) beating the market.")
        cand = cand.sort_values("rank_score", ascending=False).reset_index(drop=True)

        # --- SECTOR-EXPOSURE CAP: keep at most N per industry (best-ranked survive) ---
        max_per_sector = S.get("max_per_sector", 3)
        pre_cap = cand.copy()
        sector_note, dropped = "", {}
        if max_per_sector > 0 and not cand.empty:
            cand, dropped = apply_sector_caps(cand, max_per_sector)
            cand = cand.reset_index(drop=True)
            if dropped:
                det = ", ".join(f"{s} (−{n})" for s, n in sorted(dropped.items(), key=lambda x: -x[1]))
                sector_note = (f"Sector cap ({max_per_sector}/sector): trimmed "
                               f"{sum(dropped.values())} correlated name(s) → {det}")

        if cand.empty:
            st.info("No qualifying candidate after the regime gate. On a risk-off day with no "
                    "market-beating setups, standing aside is the correct output — check tomorrow.")
        else:
            if gate_note:
                st.caption("🔴 " + gate_note)
            if sector_note:
                st.caption("🧩 " + sector_note)
            # concentration before vs after (shows the fake-diversification problem plainly)
            if max_per_sector > 0 and not pre_cap.empty and "sector" in pre_cap.columns:
                top_pre = pre_cap["sector"].value_counts()
                if len(top_pre) and top_pre.iloc[0] > max_per_sector:
                    st.caption(f"Concentration: before cap, **{top_pre.index[0]}** alone held "
                               f"{top_pre.iloc[0]} of {len(pre_cap)} names "
                               f"({100*top_pre.iloc[0]/len(pre_cap):.0f}%). "
                               f"After cap: {len(cand)} names across {cand['sector'].nunique()} sector(s).")
            # Include news_score column when the news tilt is populated
            has_news = ("news_score" in cand.columns) and (cand["news_score"].abs().sum() > 0
                                                            or (cand.get("news_n", pd.Series([0]*len(cand))).sum() > 0))
            # Change #6 audit column — only show when at least one candidate has a penalty
            has_penalty = ("ranking_penalty_reason" in cand.columns) and \
                          cand["ranking_penalty_reason"].astype(str).str.len().gt(0).any()
            base_cols = ["ticker", "sector", "regime_today", "rank_score", "confidence", "rel_strength"]
            if has_news:
                base_cols += ["news_score"]
            if has_penalty:
                base_cols += ["ranking_penalty_reason"]
            base_cols += ["entry_ref", "plan_entry", "target_price", "stop_price", "stop_%",
                          "exp_days_to_target", "last_atr_pct", "remark"]
            inv = cand[base_cols].copy()
            _em = S.get("entry_mode", "Market open")
            _entry_label = "BUY limit ₹" if _em == "Limit" else "Entry (open)"
            new_cols = ["Stock", "Sector", "Signal", "Rank", "Conf(/day)", "RS%"]
            if has_news:
                new_cols += ["News"]
            if has_penalty:
                new_cols += ["Rank penalty"]
            new_cols += ["Last close", _entry_label, "Objective ₹", "Stop ₹", "Stop %",
                         "Exp. days→objective", "ATR%", "Remark"]
            inv.columns = new_cols
            if S.get("entry_mode") == "Limit":
                st.caption(f"📥 **Place a BUY LIMIT at the 'BUY limit ₹' price** "
                           f"({S.get('limit_pct',0)}% below the signal close), valid "
                           f"{S.get('fill_days',1)} session(s). If it never fills, **skip the trade** — "
                           f"do not chase the open. Target/Stop are computed off that limit price.")
            else:
                st.caption("⚠️ Market-at-open: you accept whatever the open gives, including gap-ups. "
                           "Switch to Limit entry to avoid chasing.")
            st.caption("Ranked by blended score (confidence × relative-strength). RS% > 0 = beating the "
                       "market. Click a row for the chart. Stop ₹ respects your max-loss cap.")
            sel = st.dataframe(inv, use_container_width=True, height=340, hide_index=True,
                               on_select="rerun", selection_mode="single-row",
                               key="inv_table")
            st.download_button("⬇️ Download tonight's analysis", inv.to_csv(index=False).encode(),
                               file_name=f"investment_analysis_{dt.date.today()}.csv", mime="text/csv")

            # ---------- point 4: click a row -> chart ----------
            picked = None
            if sel and sel.get("selection", {}).get("rows"):
                picked = cand.iloc[sel["selection"]["rows"][0]]
            if picked is not None:
                render_stock_backtest(picked, S, key_prefix="inv_")

        # ======= TABLE 2: BACKTEST TRACK RECORD (evidence) =======
        st.subheader("📊 Backtest Track Record  —  historical proof behind each stock")
        bt = ok.sort_values("rank_score", ascending=False).reset_index(drop=True)
        rec = bt[["ticker", "signals_today", "rank_score", "exp_per_day_%", "rel_strength", "hist_trades", "win_%",
                  "target_hits", "target_%", "trail_exits", "trail_%", "stop_hits", "stop_hit_%",
                  "time_exits", "time_%", "time_win", "time_loss",
                  "mom_exits", "mom_%", "decay_exits", "decay_%", "staircase_partials",
                  "expectancy_%", "avg_win_%", "avg_loss_%", "reward_risk_ratio", "avg_days",
                  "total_return_sum_%", "cagr_%", "max_drawdown_%", "profit_factor",
                  "recovery_factor", "max_consecutive_losses", "seq_trades",
                  "bt_from", "bt_to", "years", "remark"]].copy()
        rec.columns = ["Stock", "Signals today", "Rank", "Exp/DAY%", "RS%", "Trades", "Win%",
                       "Target #", "Target %", "Trail #", "Trail %", "Stop #", "Stop %",
                       "Time #", "Time %", "Time-win", "Time-loss",
                       "MomExit #", "MomExit %", "Decay #", "Decay %", "Staircase #",
                       "Expectancy%", "Avg win%", "Avg loss%", "R:R", "Avg days",
                       "Total return (sum)%", "CAGR%", "Max DD%", "Profit factor",
                       "Recovery factor", "Max consec. losses", "Seq. trades",
                       "BT from", "BT to", "Years", "Remark"]
        st.caption("Point 3 breakdown: Win% / Target = trades that truly hit 10%. "
                   "Trail = profitable trailing exits below 10%. Stop = losses. "
                   "Time = held to max days (split into win/loss). Profit trades = Target + Trail + Time-win.")
        st.caption("**Win% / Target% / Trail% / Stop% / Time% / Profit factor / R:R / Avg win-loss** are "
                   "computed on every historical signal (all trades) for statistical power. "
                   "**CAGR / Max DD / Recovery factor / Max consec. losses** are computed on a "
                   "*sequential*, one-position-at-a-time equity curve (**Seq. trades** column) — "
                   "because compounding needs a single capital timeline and raw signals can overlap. "
                   "**Total return (sum)%** is the simple (non-compounded) sum of every trade's return — "
                   "a quick scorecard, not an account balance.")
        st.caption("**Click any row to open that stock's full backtest** — every trade, entry/exit/"
                   "stop prices, returns, holding days, and the technical parameters at entry.")
        sel_rec = st.dataframe(rec, use_container_width=True, height=340, hide_index=True,
                               on_select="rerun", selection_mode="single-row", key="rec_table")
        if sel_rec and sel_rec.get("selection", {}).get("rows"):
            picked_rec = bt.iloc[sel_rec["selection"]["rows"][0]]
            render_stock_backtest(picked_rec, S, key_prefix="rec_")
        st.download_button("⬇️ Download backtest track record", rec.to_csv(index=False).encode(),
                           file_name=f"backtest_record_{dt.date.today()}.csv", mime="text/csv")

        c1, c2, c3 = st.columns(3)
        c1.metric("Stocks scanned OK", len(ok))
        c2.metric("Signalling today", int(ok["signals_today"].sum()))
        c3.metric("Avg expectancy (segment)", f'{ok["expectancy_%"].mean():.2f}%')

    with st.expander("📐 What the scan uses (parameters, indicators, risk)"):
        _bk = S.get("bt_kwargs", {})
        _stack_bits = []
        if _bk.get("ratchet_lock"):
            _stack_bits.append("A ratchet-lock (v2 softer)")
        if _bk.get("shrink_trail"):
            _stack_bits.append("B shrink-trail")
        if _bk.get("momentum_exit"):
            _stack_bits.append(f"C momentum-exit (arm ≥ {_bk.get('mom_exit_min_gain', 10):.0f}%)")
        if _bk.get("time_decay"):
            _stack_bits.append(f"D time-decay (tight@{_bk.get('decay_after_days', 5)}d, exit@{_bk.get('decay_exit_days', 10)}d)")
        if _bk.get("staircase"):
            _stack_bits.append("E staircase (30% at +10%, 30% at +20%)")
        _stack_txt = " · ".join(_stack_bits) if _stack_bits else "LEGACY only (no exit-stack layers)"
        st.markdown(
            f"**Strategy:** `{S['strategy']}` - {engine.STRATEGY_HELP[S['strategy']]}\n\n"
            f"**Backtest window:** target {S['yrs']}y (per-stock actual range in the tables).\n\n"
            f"**Exit:** {S['exit_mode']} (trail {S['trail_mult']}x ATR) - Target {S['target_pct']}% - "
            f"Hold {S['min_hold']}-{S['max_hold']}d.\n\n"
            f"**Exit stack:** {_stack_txt}\n\n"
            f"**Risk:** stop {S['stop_value']}x ATR, max-loss cap {S['max_stop_pct']}%, "
            f"skip if ATR% > {S['max_atr_pct']}, cost {S['cost_pct']}%"
            f"{', 20% STCG' if S['apply_stcg'] else ''}.\n\n"
            "**Indicators per stock (no look-ahead):** MAs 5/20/50/200, RSI(14), MACD(12,26,9), ATR%, "
            "Bollinger %B & bandwidth, Stochastics, OBV & slope, ADX/+-DI, volume ratio & surge, "
            "20-day breakout, 52-week distance, candle body/gap."
        )

    # ==================================================================
    # 📰  NEWS IMPACT DISPLAY (Aug-2026 — user-requested)
    # ------------------------------------------------------------------
    # Show news for ALL fundamentally-passing stocks, not just those
    # firing a technical signal today. News can be a LEADING indicator —
    # a stock with strong positive news may set up a signal tomorrow,
    # or a stock with negative news may pre-warn a stop-out risk we'd
    # otherwise walk into.
    # ==================================================================
    if "news_score" in res.columns:
        # Universe with any news activity (n > 0 OR non-zero score)
        with_news = res[(res["status"] == "ok")
                        & ((res["news_n"].fillna(0) > 0)
                           | (res["news_score"].fillna(0) != 0))].copy()

        # (A) News-driven WATCHLIST — no signal today but material news
        #     Positive news (>= +0.15) → possible early setup; catalyst plays.
        #     Negative news (<= -0.15) → avoid re-entering these until price stabilises.
        news_watchlist = with_news[(~with_news["signals_today"].fillna(False))
                                    & (with_news["news_score"].abs() >= 0.15)].copy()
        news_watchlist = news_watchlist.sort_values("news_score", key=lambda s: s.abs(), ascending=False)
        if not news_watchlist.empty:
            n_pos = int((news_watchlist["news_score"] > 0).sum())
            n_neg = int((news_watchlist["news_score"] < 0).sum())
            with st.expander(f"📰 News-driven WATCHLIST — {len(news_watchlist)} "
                             f"stocks with material news but NO technical signal today "
                             f"({n_pos} positive / {n_neg} negative)"):
                st.caption("Stocks that passed the fundamental gate but did NOT fire a "
                           "technical signal today, yet have material news activity in the "
                           "last 5 days. **Positive news** (+ score) → possible upcoming "
                           "setup as buyers respond; monitor over next 2-3 sessions. "
                           "**Negative news** (− score) → avoid re-entering these until "
                           "sentiment stabilises. News does NOT bypass the technical "
                           "signal requirement — this section is informational.")
                nw_view = news_watchlist[["ticker", "sector", "news_score", "news_n",
                                           "news_top", "news_matched"]].copy()
                nw_view.columns = ["Stock", "Sector", "News", "# articles",
                                    "Top headline", "Keywords matched"]
                st.dataframe(nw_view, use_container_width=True, hide_index=True, height=280)
                st.download_button(
                    "⬇️ Download news watchlist",
                    nw_view.to_csv(index=False).encode(),
                    file_name=f"news_watchlist_{dt.date.today()}.csv",
                    mime="text/csv",
                )

        # (B) FULL news audit — every scanned stock's news score, expander closed by default
        if not with_news.empty:
            n_all = len(with_news)
            n_material = int((with_news["news_score"].abs() >= 0.15).sum())
            with st.expander(f"📊 Full news audit — {n_all} scanned stocks with any news activity "
                             f"({n_material} material)"):
                st.caption("Every fundamentally-passing stock's news score in the last 5 days. "
                           "Sorted by |news score|. Zero-score rows are stocks where headlines "
                           "existed but none matched the sentiment lexicon (neutral coverage).")
                full_view = with_news[["ticker", "sector", "signals_today",
                                        "news_score", "news_n", "news_top", "news_matched"]].copy()
                full_view["|news|"] = full_view["news_score"].abs()
                full_view = full_view.sort_values("|news|", ascending=False).drop(columns=["|news|"])
                full_view.columns = ["Stock", "Sector", "Signals today?", "News",
                                     "# articles", "Top headline", "Keywords matched"]
                st.dataframe(full_view, use_container_width=True, hide_index=True, height=350)
                st.download_button(
                    "⬇️ Download full news audit",
                    full_view.to_csv(index=False).encode(),
                    file_name=f"news_audit_{dt.date.today()}.csv",
                    mime="text/csv",
                )

    # ---- Event-risk blocked names (news pillar) ----
    # Rebuild from `res` — every stock row carries `event_blocked` bool.
    if "event_blocked" in res.columns:
        ev_blocked = res[(res["status"] == "ok")
                         & (res.get("signals_today", False))
                         & (res["event_blocked"] == True)].copy()
    else:
        ev_blocked = pd.DataFrame()
    if not ev_blocked.empty:
        with st.expander(f"🚫 {len(ev_blocked)} signalling stock(s) BLOCKED by upcoming event risk"):
            st.caption("These stocks fired a technical signal today, BUT have a scheduled "
                       "corporate event (results / board meeting / dividend ex-date / split / "
                       "AGM) within the blocking window. Buying today would carry event-gap "
                       "risk — a single overnight news release can move the stock 5–20%, "
                       "invalidating any technical setup. Wait until AFTER the event, then "
                       "re-evaluate the technical signal on the post-event price.")
            ev_view = ev_blocked[["ticker", "sector", "regime_today", "rank_score",
                                  "event_type", "event_days_until", "event_subject"]].copy()
            ev_view.columns = ["Stock", "Sector", "Signal", "Rank",
                               "Event type", "Days until", "Subject"]
            st.dataframe(ev_view, use_container_width=True, hide_index=True)
            st.download_button(
                "⬇️ Download event-blocked list",
                ev_view.to_csv(index=False).encode(),
                file_name=f"event_blocked_{dt.date.today()}.csv",
                mime="text/csv",
            )

    # ---- Cooldown-blocked names (Change #2) ----
    # Stocks whose signal fired today but were blocked by post-stop cooldown:
    # the same stock stopped out within the last N sessions, so the algo would
    # be sitting in cooldown live. Shown separately so the user knows the algo
    # SAW the signal but declined to act on it (and why).
    if "cooldown_blocked" in res.columns:
        cd_blocked = res[(res["status"] == "ok") & (res["cooldown_blocked"] == True)].copy()
        if not cd_blocked.empty:
            with st.expander(f"❄️ {len(cd_blocked)} stock(s) blocked by post-stop cooldown "
                             f"(recent STOP still inside cooldown window)"):
                st.caption("These stocks fired a technical signal today, BUT the algorithm "
                           "stopped out on this same stock within the cooldown window (see "
                           "Section 4f). The engine already excludes these re-entries from the "
                           "historical trade log; the scanner now honours the same rule for "
                           "live signals. Historical evidence (ADANIENT Oct-Nov 2024): the "
                           "first STOP loses ~9% but the 3 blocked re-entries would have added "
                           "another −41% if taken.")
                cd_view = cd_blocked[["ticker", "sector", "cooldown_reason"]].rename(
                    columns={"ticker": "Stock", "sector": "Sector",
                             "cooldown_reason": "Why blocked"})
                st.dataframe(cd_view, use_container_width=True, hide_index=True)
                st.download_button(
                    "⬇️ Download cooldown-blocked list",
                    cd_view.to_csv(index=False).encode(),
                    file_name=f"cooldown_blocked_{dt.date.today()}.csv",
                    mime="text/csv",
                )

    # ---- Fundamentals-rejected names ----
    fr_df = S.get("funda_rejects_df")
    if fr_df is not None and not fr_df.empty:
        with st.expander(f"🚫 {len(fr_df)} stocks rejected by fundamentals gate "
                         f"(never reached the technical scan)"):
            st.caption("These would have been in tonight's universe but failed the "
                       "no-trade filter — usually one of: sub-5% ROE, D/E > 3, "
                       "interest cover < 1.5, promoter pledge > 40%, or auditor "
                       "qualification. Populate governance_overrides.csv to enable "
                       "the pledge / auditor checks. **NEW Aug-2026**: also catches "
                       "secular downtrenders (12-mo return < −15%) and illiquid "
                       "names (turnover < ₹5 cr).")
            st.dataframe(fr_df, use_container_width=True, hide_index=True, height=300)
            st.download_button(
                "⬇️ Download rejected list",
                fr_df.to_csv(index=False).encode(),
                file_name=f"fundamentals_rejected_{dt.date.today()}.csv",
                mime="text/csv",
            )

    # ---- Self-check rejected names (Section 4e post-backtest filter) ----
    if not selfcheck_rejects.empty:
        with st.expander(f"🪞 {len(selfcheck_rejects)} stocks rejected by historical "
                         f"self-check (algo's own weak track record)"):
            st.caption("These stocks passed the fundamentals gate BUT the algorithm "
                       "has itself lost money on them historically. The self-check "
                       "uses the strategy's OWN backtest record as the arbiter — if "
                       "we haven't made money on this stock across hundreds of past "
                       "trades, we shouldn't take tonight's signal either.")
            display_cols = ["ticker", "hist_trades", "win_%", "total_return_sum_%",
                            "cagr_%", "profit_factor", "_selfcheck_reason"]
            display_cols = [c for c in display_cols if c in selfcheck_rejects.columns]
            _view = selfcheck_rejects[display_cols].rename(columns={
                "hist_trades": "Trades", "win_%": "Win %",
                "total_return_sum_%": "Total return (sum)%",
                "cagr_%": "CAGR %", "profit_factor": "Profit factor",
                "_selfcheck_reason": "Why rejected",
            })
            st.dataframe(_view, use_container_width=True, hide_index=True, height=200)
            st.download_button(
                "⬇️ Download self-check rejections",
                selfcheck_rejects.to_csv(index=False).encode(),
                file_name=f"selfcheck_rejected_{dt.date.today()}.csv",
                mime="text/csv",
            )

    if not bad.empty:
        with st.expander(f"⚠️ {len(bad)} stocks skipped or failed"):
            st.dataframe(bad[["ticker", "status"]].reset_index(drop=True),
                         use_container_width=True, hide_index=True)
            st.caption("'insufficient data' = recent listing below the 1-year bar. "
                       "'fetch error/no data' = Yahoo hiccup or wrong symbol; re-run to retry.")

    st.divider()
    st.caption("Signals are historical-edge candidates, not guarantees. ~1/3 of trades hit target and "
               "some lose; size positions and honour stops. Yahoo data can be delayed/patchy. "
               "Educational tool - not investment advice.")


if __name__ == "__main__":
    main()