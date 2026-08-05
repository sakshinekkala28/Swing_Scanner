"""
Position Monitor (v1, Aug-2026)
================================
Daily position-management dashboard — the missing "SELL SIDE" of the swing
trading loop.

While `swing_scanner_app.py` finds NEW setups (buy side), this app looks at
YOUR EXISTING HOLDINGS and asks, for each one:
   * Should I hold?
   * Should I raise my stop?
   * Should I take some off?
   * Should I get out entirely?
   * Should I add more?

Uses the same engine (swing_screener_app.py), same news scorer, same event
fetcher as the scanner — but with position-management decision logic instead
of new-signal generation.

USAGE
-----
    streamlit run monitor_app.py

INPUT
-----
    positions.csv (next to this file) — one row per open long position.
    See the header comment inside positions.csv for the exact schema and how
    to populate it from your scanner output.

DECISION LOGIC (evaluated top-down; first match wins for the ACTION):
    ┌─────────────────────────────────────────────────────────────────────┐
    │ TIER 1 — URGENT EXIT (no override, protects capital)               │
    │   1a  Stop-loss hit                                                 │
    │   1b  Scheduled event (results/AGM/split/etc.) within 2 sessions   │
    │   1c  Severe negative news (score < -0.5 AND >= 2 articles)         │
    │                                                                     │
    │ TIER 2 — EXIT (trend broken)                                        │
    │   2a  >= 2 technical breakdown signals                              │
    │                                                                     │
    │ TIER 3 — REDUCE (book half)                                         │
    │   3a  Moderate negative news (score < -0.3, >= 2 articles)          │
    │   3b  Exactly 1 breakdown signal AND position in loss               │
    │                                                                     │
    │ TIER 4 — HOLD  (position stays; refinements applied below)          │
    │   4a  Ratchet stop suggestion (see ladder)                          │
    │   4b  Add-on signal (only for winners in a favourable regime)       │
    │                                                                     │
    │ TIER 5 — HOLD (default, TA intact)                                  │
    └─────────────────────────────────────────────────────────────────────┘

RATCHET LADDER (as-you-earn stop tightening):
    +5%  gain → raise stop to break-even
    +10% gain → raise stop to +3%
    +20% gain → raise stop to +12%
    +30% gain → raise stop to +20%
    +40% gain → raise stop to +28%
    +50% gain → raise stop to +37%
    +75% gain → raise stop to +60%
   +100% gain → raise stop to +80%
"""
import os
import io
import time
import importlib.util
import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st

try:
    import yfinance as yf
except Exception:
    yf = None


# ======================================================================================
#  ENGINE LOADER — reuse the swing engine
# ======================================================================================
_HERE = os.path.dirname(os.path.abspath(__file__))
_ENGINE_PATH = os.path.join(_HERE, "swing_screener_app.py")
_spec = importlib.util.spec_from_file_location("engine", _ENGINE_PATH)
engine = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(engine)

# Optional modules — news + events + sector
try:
    from news_sentiment import fetch_news_score as _news_score
    HAVE_NEWS = True
except Exception:
    HAVE_NEWS = False

try:
    from nse_events import event_risk as _event_risk
    HAVE_EVENTS = True
except Exception:
    HAVE_EVENTS = False

try:
    from universe_loader import load_full_universe as _ul_load
    HAVE_UNIVERSE = True
except Exception:
    HAVE_UNIVERSE = False


POSITIONS_CSV = os.path.join(_HERE, "positions.csv")
BENCH_TICKERS = ["^CRSLDX", "^NSEI"]


# ======================================================================================
#  DATA HELPERS
# ======================================================================================
def _to_yahoo(sym: str) -> str:
    s = str(sym).strip().upper()
    return s if s.endswith((".NS", ".BO")) else s + ".NS"


@st.cache_data(ttl=60 * 60, show_spinner=False)
def _fetch_stock(ticker_yahoo: str, days: int = 400) -> pd.DataFrame:
    if yf is None:
        return pd.DataFrame()
    end = dt.date.today() + dt.timedelta(days=1)
    start = end - dt.timedelta(days=days)
    try:
        df = yf.Ticker(ticker_yahoo).history(start=start, end=end,
                                              interval="1d", auto_adjust=True)
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df.dropna()


@st.cache_data(ttl=60 * 60, show_spinner=False)
def _fetch_bench(days: int = 400):
    if yf is None:
        return None, pd.DataFrame()
    end = dt.date.today() + dt.timedelta(days=1)
    start = end - dt.timedelta(days=days)
    for t in BENCH_TICKERS:
        try:
            df = yf.Ticker(t).history(start=start, end=end,
                                       interval="1d", auto_adjust=True)
            if df is not None and not df.empty:
                df = df[["Open", "High", "Low", "Close"]].copy()
                df.index = pd.to_datetime(df.index).tz_localize(None)
                return t, df.dropna()
        except Exception:
            continue
    return None, pd.DataFrame()


def _regime_from_bench(bench_df: pd.DataFrame) -> str:
    if bench_df.empty or len(bench_df) < 210:
        return "UNKNOWN"
    c = bench_df["Close"]
    s200 = float(c.rolling(200).mean().iloc[-1])
    last = float(c.iloc[-1])
    above = last > s200
    roc10 = (c.iloc[-1] / c.iloc[-11] - 1) * 100 if len(c) > 11 else 0.0
    if above and roc10 > -1.0: return "RISK-ON"
    if above or roc10 > -3.0:  return "NEUTRAL"
    return "RISK-OFF"


# ======================================================================================
#  POSITIONS CSV LOADER
# ======================================================================================
# What's TRULY required vs OPTIONAL after v2:
#   REQUIRED : ticker, buy_date, buy_price, quantity
#   OPTIONAL : stop_loss (auto-derived from 2xATR if blank),
#              target     (advisory only — if blank, we still work),
#              signal_date (defaults to buy_date - 1),
#              sector     (auto-filled from universe loader),
#              notes      (free text)
REQUIRED_COLS = ("ticker", "buy_date", "buy_price", "quantity")
OPTIONAL_COLS = ("stop_loss", "target", "signal_date", "sector", "notes")

import re as _re
_TICKER_ALLOWED = _re.compile(r"[^A-Z0-9&\-]")


def _clean_ticker(raw: str) -> str:
    """Strip Unicode artefacts / stray whitespace from a ticker.
    NSE symbols use A-Z, 0-9, '&', '-'. Everything else gets stripped.
    e.g. 'SAREGAMA�' -> 'SAREGAMA'; 'M&M' -> 'M&M' unchanged."""
    if not isinstance(raw, str):
        return ""
    s = raw.strip().upper()
    s = _TICKER_ALLOWED.sub("", s)
    return s


def _parse_flex_date(v):
    """Try DD-MM-YYYY first (user's format), then fall back to pandas default.
    Returns a `date` or None."""
    if pd.isna(v) or str(v).strip() in ("", "nan", "None"):
        return None
    s = str(v).strip()
    # Try DD-MM-YYYY explicit
    for fmt in ("%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d", "%Y/%m/%d"):
        try:
            return dt.datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    # Last-resort pandas parser (dayfirst=True to prefer 03-08-2026 = 3 Aug not 8 Mar)
    try:
        return pd.to_datetime(s, dayfirst=True, errors="coerce").date()
    except Exception:
        return None


def load_positions(path: str = POSITIONS_CSV) -> tuple:
    """Return (df, errors). Positions with missing OPTIONAL fields still load
    — those fields get auto-derived downstream. Only rows missing REQUIRED
    fields (ticker, buy_date, buy_price, quantity) are dropped."""
    if not os.path.exists(path):
        return pd.DataFrame(columns=list(REQUIRED_COLS) + list(OPTIONAL_COLS)), \
               [f"File not found: {path}"]
    # Read with encoding fallback — user files sometimes contain non-UTF8
    # bytes (Windows-1252 quotes/dashes, Unicode replacement chars from
    # copy/paste). Try encodings in order; last resort uses error replacement.
    df = None
    read_err = None
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            df = pd.read_csv(path, comment="#", skip_blank_lines=True, encoding=enc)
            break
        except Exception as e:
            read_err = e
            continue
    if df is None:
        # Absolute fallback: read as bytes, drop non-decodable characters
        try:
            with open(path, "rb") as f:
                raw_bytes = f.read()
            text = raw_bytes.decode("utf-8", errors="replace")
            df = pd.read_csv(io.StringIO(text), comment="#", skip_blank_lines=True)
        except Exception as e:
            return pd.DataFrame(), [f"Read error: {read_err} / {e}"]
    df.columns = [c.strip().lower() for c in df.columns]

    errors = []
    for c in REQUIRED_COLS:
        if c not in df.columns:
            errors.append(f"Missing required column: {c}")
    if errors:
        return pd.DataFrame(), errors

    # Fill any missing optional columns as empty
    for c in OPTIONAL_COLS:
        if c not in df.columns:
            df[c] = np.nan

    # SANITIZE tickers (strip Unicode replacement chars, whitespace, non-symbol chars)
    original_tickers = df["ticker"].astype(str).copy()
    df["ticker"] = original_tickers.apply(_clean_ticker)
    cleaned_map = {orig: new for orig, new in zip(original_tickers, df["ticker"])
                   if orig.strip() != new and new}
    if cleaned_map:
        errors.append("Cleaned corrupted ticker(s): "
                      + ", ".join(f"'{o.strip()}' -> '{n}'" for o, n in cleaned_map.items()))

    # Drop blank tickers and obvious example rows
    df = df[df["ticker"].str.len() > 0]
    df = df[~df["ticker"].str.contains("EXAMPLE", na=False)]
    if df.empty:
        return df, ["No real positions found. Edit positions.csv and re-run."]

    # Coerce numerics
    for c in ("buy_price", "quantity", "stop_loss", "target"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Flexible date parsing (DD-MM-YYYY, YYYY-MM-DD, both slashes and dashes)
    df["buy_date"]    = df["buy_date"].apply(_parse_flex_date)
    df["signal_date"] = df["signal_date"].apply(_parse_flex_date)

    # Fill signal_date default = buy_date - 1 (day before fill)
    def _sig_default(row):
        if row["signal_date"] is not None:
            return row["signal_date"]
        if row["buy_date"] is not None:
            return row["buy_date"] - dt.timedelta(days=1)
        return None
    df["signal_date"] = df.apply(_sig_default, axis=1)

    # Sector / notes: leave NaN → treated as blank downstream
    for c in ("sector", "notes"):
        df[c] = df[c].fillna("").astype(str)

    # ROW-LEVEL VALIDATION: only REQUIRED fields must be present
    bad_mask = (df["buy_price"].isna() | df["quantity"].isna() | df["buy_date"].isna()
                | (df["ticker"].str.len() == 0))
    bad = df[bad_mask]
    if not bad.empty:
        errors.append(f"{len(bad)} row(s) dropped for missing REQUIRED fields: "
                      f"{list(bad['ticker'])}")
    df = df[~bad_mask].reset_index(drop=True)

    # Flag rows that will need auto-derivation (informational, not an error)
    n_missing_stop   = int(df["stop_loss"].isna().sum())
    n_missing_sector = int((df["sector"].str.strip() == "").sum())
    n_missing_target = int(df["target"].isna().sum())
    if n_missing_stop:
        errors.append(f"ℹ️ {n_missing_stop} row(s) have no stop_loss "
                      f"— will be auto-derived from 2×ATR at load time.")
    if n_missing_sector:
        errors.append(f"ℹ️ {n_missing_sector} row(s) have no sector — will auto-fill from NSE map.")
    if n_missing_target:
        errors.append(f"ℹ️ {n_missing_target} row(s) have no target — advisory field, safe to omit.")

    return df, errors


# ======================================================================================
#  DECISION LOGIC
# ======================================================================================
# Ratchet ladder: (min_gain_pct, floor_pct_of_entry)
RATCHET_LADDER = [
    (5.0,    0.0),
    (10.0,   3.0),
    (20.0,  12.0),
    (30.0,  20.0),
    (40.0,  28.0),
    (50.0,  37.0),
    (75.0,  60.0),
    (100.0, 80.0),
]


def _ratchet_stop(entry_price: float, pnl_pct: float, current_stop: float) -> float:
    """Return the highest floor from the ladder that pnl_pct qualifies for,
    strictly greater than current_stop. Returns None if no raise applies."""
    best = None
    for peak, floor in RATCHET_LADDER:
        if pnl_pct >= peak:
            candidate = entry_price * (1 + floor / 100)
            if candidate > current_stop:
                best = candidate
    return round(best, 2) if best is not None else None


def _derive_stop_loss(buy_price: float, ta_snapshot: dict,
                       max_pct: float = 10.0, atr_mult: float = 2.0) -> tuple:
    """When user leaves stop_loss blank, compute a sensible default matching
    the scanner's convention: entry - 2×ATR, capped at max_pct (10%) loss.

    Returns (stop_price, method_str) where method_str explains the derivation.
    Never returns None — always produces a stop (10% flat if ATR unavailable).
    """
    atr_pct = ta_snapshot.get("atr_pct", np.nan)
    if pd.notna(atr_pct) and atr_pct > 0:
        # ATR-based stop
        atr_abs = buy_price * atr_pct / 100.0
        stop = buy_price - atr_mult * atr_abs
        floor = buy_price * (1 - max_pct / 100.0)
        stop = max(stop, floor)
        method = (f"2×ATR ({atr_pct:.1f}% ATR → {atr_mult*atr_pct:.1f}% risk), "
                  f"capped at {max_pct:.0f}%")
        return round(stop, 2), method
    # Fallback: flat 10% stop
    return round(buy_price * (1 - max_pct / 100.0), 2), \
           f"{max_pct:.0f}% flat (ATR unavailable)"


def _ta_snapshot(df_ind: pd.DataFrame) -> dict:
    """Extract a compact TA snapshot at the last bar."""
    if df_ind is None or df_ind.empty:
        return {}
    last = df_ind.iloc[-1]
    prev = df_ind.iloc[-2] if len(df_ind) > 1 else last
    # 10-day RSI max for "momentum peaked then broke down" check
    rsi_max_10d = float(df_ind["rsi14"].tail(10).max()) if "rsi14" in df_ind else np.nan
    return {
        "close":         float(last["Close"]),
        "pct_vs_sma20":  float(last.get("pct_vs_sma20", np.nan)),
        "pct_vs_sma50":  float(last.get("pct_vs_sma50", np.nan)),
        "pct_vs_sma200": float(last.get("pct_vs_sma200", np.nan)),
        "rsi14":         float(last.get("rsi14", np.nan)),
        "rsi_max_10d":   rsi_max_10d,
        "macd_hist":     float(last.get("macd_hist", np.nan)),
        "macd_hist_prev":float(prev.get("macd_hist", np.nan)),
        "atr_pct":       float(last.get("atr_pct", np.nan)),
        "vol_ratio":     float(last.get("vol_ratio", np.nan)),
        "dist_52wH":     float(last.get("dist_52wH", np.nan)),
        "adx14":         float(last.get("adx14", np.nan)),
        "signal_today":  bool(last.get("signal", False)),
    }


def _mk_check(category: str, name: str, value, verdict: str, note: str = "") -> dict:
    """Uniform check record — used for the audit table in the UI."""
    return {"category": category, "name": name, "value": value,
            "verdict": verdict, "note": note}


def decide(position: pd.Series, ta: dict, news: dict, events: dict,
           regime: str) -> dict:
    """Central decision engine. Runs a full battery of TA + News + Event
    checks, records EVERY check with pass/warn/fail badge, then maps the
    verdicts to a final action following a strict priority tree.

    Returns
    -------
    dict with:
        action        : "EXIT" | "REDUCE" | "HOLD" | "NO_DATA"
        urgency       : "URGENT" | "normal"
        pnl_pct, pnl_abs, days_held
        new_stop      : ratchet stop suggestion (or None)
        add_qty       : + integer for ADD, - integer for partial-book,
                        0 for no size change
        reasons       : [str] — high-level bullet list (shown in summary)
        narrative     : [str] — full multi-line story (shown in drill-down)
        checks        : [{category, name, value, verdict, note}] — audit
        score         : {ta, news, event, regime, ratchet, total}
    """
    entry = float(position["buy_price"])
    qty   = float(position["quantity"])
    stop  = float(position["stop_loss"])
    price = ta.get("close", entry)
    pnl_pct = (price / entry - 1) * 100 if np.isfinite(price) else 0.0
    pnl_abs = (price - entry) * qty     if np.isfinite(price) else 0.0
    days_held = (dt.date.today() - position["buy_date"]).days if position["buy_date"] else 0

    checks   = []
    narrative = []
    ta_score = news_score = event_score = 0

    # ---------------- 1. STOP-LOSS CHECK ----------------
    stop_hit = np.isfinite(price) and price <= stop
    checks.append(_mk_check(
        "stop", "Price vs stop-loss",
        f"₹{price:.2f} vs ₹{stop:.2f}",
        "FAIL" if stop_hit else "PASS",
        "STOP TOUCHED — capital preservation trigger" if stop_hit else
            f"cushion = ₹{price-stop:.2f} ({100*(price-stop)/price:.1f}%)"
    ))

    # ---------------- 2. EVENT CHECK ----------------
    ev_type = events.get("type")
    ev_days = events.get("days_until")
    ev_urgent = events.get("blocked") and ev_days is not None and ev_days <= 2
    if ev_type:
        checks.append(_mk_check(
            "event", f"Upcoming {ev_type}",
            f"{ev_days}d away",
            "FAIL" if ev_urgent else ("WARN" if (ev_days or 99) <= 5 else "INFO"),
            (events.get("subject") or "")[:120]
        ))
        if ev_urgent:
            event_score -= 10
        elif (ev_days or 99) <= 5:
            event_score -= 2
    else:
        checks.append(_mk_check("event", "Upcoming event risk",
                                 "none in next 5 sessions", "PASS", ""))

    # ---------------- 3. NEWS CHECKS ----------------
    ns = float(news.get("score", 0.0))
    nn = int(news.get("n_articles", 0))
    if nn == 0:
        checks.append(_mk_check("news", "News sentiment (5d)",
                                 "no articles", "INFO",
                                 "no news coverage found in last 5 sessions"))
    else:
        # Categorize
        if ns <= -0.5 and nn >= 2:
            label, verdict = "SEVERE NEGATIVE", "FAIL"
            news_score -= 10
        elif ns <= -0.3 and nn >= 2:
            label, verdict = "moderate negative", "WARN"
            news_score -= 4
        elif ns <= -0.15:
            label, verdict = "mildly negative", "WARN"
            news_score -= 1
        elif ns >= 0.5 and nn >= 2:
            label, verdict = "STRONG POSITIVE", "PASS"
            news_score += 4
        elif ns >= 0.15:
            label, verdict = "mildly positive", "PASS"
            news_score += 2
        else:
            label, verdict = "neutral", "INFO"
        top = (news.get("top_headline") or "")[:120]
        checks.append(_mk_check(
            "news", "News sentiment (5d)",
            f"{ns:+.2f} ({nn} articles) — {label}",
            verdict,
            top and f'top: "{top}"' or ""
        ))

    # ---------------- 4. TA CHECKS (each individually recorded) ----------------
    def _has(k): return k in ta and np.isfinite(ta.get(k, np.nan))

    # RSI
    if _has("rsi14"):
        rsi = ta["rsi14"]
        if rsi > 70:
            v = "WARN"; ta_score -= 1
            note = "overbought — susceptible to pullback"
        elif rsi < 40 and _has("rsi_max_10d") and ta["rsi_max_10d"] > 60:
            v = "FAIL"; ta_score -= 3
            note = f"momentum broke — peaked at {ta['rsi_max_10d']:.0f}"
        elif rsi < 30:
            v = "WARN"; ta_score -= 1
            note = "oversold — could be exhaustion or capitulation"
        else:
            v = "PASS"; ta_score += 1
            note = "in healthy range"
        checks.append(_mk_check("ta", "RSI(14)", f"{rsi:.1f}", v, note))

    # MACD histogram — 2 consecutive negative
    if _has("macd_hist") and _has("macd_hist_prev"):
        m, mp = ta["macd_hist"], ta["macd_hist_prev"]
        if m < 0 and mp < 0:
            v = "FAIL"; ta_score -= 3
            note = "negative 2 sessions — trend momentum broken"
        elif m < 0:
            v = "WARN"; ta_score -= 1
            note = "just flipped negative — 1st bar"
        else:
            v = "PASS"; ta_score += 1
            note = "positive — momentum with the trend"
        checks.append(_mk_check("ta", "MACD histogram",
                                 f"{m:+.2f} (prev {mp:+.2f})", v, note))

    # Price vs 20-DMA on volume
    if _has("pct_vs_sma20") and _has("vol_ratio"):
        p20, vr = ta["pct_vs_sma20"], ta["vol_ratio"]
        if p20 < -1 and vr > 1.2:
            v = "FAIL"; ta_score -= 3
            note = f"broke 20-DMA on heavy {vr:.1f}× volume — real distribution"
        elif p20 < -1:
            v = "WARN"; ta_score -= 1
            note = "below 20-DMA but light volume — may recover"
        elif p20 > 0:
            v = "PASS"; ta_score += 1
            note = "above short trend — continuation intact"
        else:
            v = "WARN"; ta_score += 0
            note = "under 20-DMA but not yet decisive"
        checks.append(_mk_check("ta", "Price vs 20-DMA",
                                 f"{p20:+.2f}% (vol {vr:.2f}×)", v, note))

    # Price vs 50-DMA
    if _has("pct_vs_sma50"):
        p50 = ta["pct_vs_sma50"]
        if p50 < 0 and pnl_pct < 0:
            v = "FAIL"; ta_score -= 2
            note = "below 50-DMA AND in loss — trend breakdown confirmed"
        elif p50 < 0:
            v = "WARN"; ta_score -= 1
            note = "below 50-DMA but still net-up — trend weakening"
        else:
            v = "PASS"; ta_score += 1
            note = "above intermediate trend"
        checks.append(_mk_check("ta", "Price vs 50-DMA",
                                 f"{p50:+.2f}%", v, note))

    # Price vs 200-DMA (long-trend context)
    if _has("pct_vs_sma200"):
        p200 = ta["pct_vs_sma200"]
        if p200 < 0:
            v = "WARN"; ta_score -= 1
            note = "below 200-DMA — long-term trend is DOWN"
        elif p200 > 15:
            v = "WARN"
            note = "extended > 15% above 200-DMA — mean-reversion risk"
        else:
            v = "PASS"; ta_score += 1
            note = "in healthy uptrend zone"
        checks.append(_mk_check("ta", "Price vs 200-DMA",
                                 f"{p200:+.2f}%", v, note))

    # ADX (trend strength)
    if _has("adx14"):
        adx = ta["adx14"]
        if adx >= 25:
            v = "PASS"; ta_score += 1
            note = "strong directional trend"
        elif adx >= 20:
            v = "INFO"
            note = "moderate trend"
        else:
            v = "WARN"
            note = "weak/no trend (choppy)"
        checks.append(_mk_check("ta", "ADX(14)", f"{adx:.1f}", v, note))

    # ATR% (volatility state)
    if _has("atr_pct"):
        atrp = ta["atr_pct"]
        if atrp > 7:
            v = "WARN"
            note = "very volatile — expect wide swings"
        elif atrp < 1.5:
            v = "INFO"
            note = "low volatility — quiet regime"
        else:
            v = "PASS"
            note = "typical volatility"
        checks.append(_mk_check("ta", "ATR%", f"{atrp:.2f}%", v, note))

    # ---------------- 5. REGIME + SIGNAL-TODAY (context) ----------------
    reg_verdict = {"RISK-ON": "PASS", "NEUTRAL": "INFO",
                   "RISK-OFF": "WARN", "UNKNOWN": "INFO"}.get(regime, "INFO")
    reg_score = {"RISK-ON": +2, "NEUTRAL": 0, "RISK-OFF": -2, "UNKNOWN": 0}[regime]
    checks.append(_mk_check("regime", "Market regime (bench)",
                             regime, reg_verdict,
                             {"RISK-ON": "supportive backdrop for longs",
                              "NEUTRAL": "mixed — take only leaders",
                              "RISK-OFF": "unfriendly — avoid new adds",
                              "UNKNOWN": "no benchmark data"}[regime]))

    if ta.get("signal_today"):
        checks.append(_mk_check("signal", "Scanner signal today?",
                                 "YES", "PASS",
                                 "engine's PASS_combined rule fires on today's bar"))
    else:
        checks.append(_mk_check("signal", "Scanner signal today?",
                                 "no", "INFO", ""))

    # ---------------- COMPUTE BREAKDOWN COUNT (for tier logic) ----------------
    breakdowns = [c for c in checks if c["category"] == "ta" and c["verdict"] == "FAIL"]

    # ================ DECISION PRIORITY TREE ================
    # ---------- TIER 1 : URGENT EXIT ----------
    if stop_hit:
        action, urgency = "EXIT", "URGENT"
        narrative.append(
            f"🛑 STOP-LOSS TOUCHED. Price ₹{price:.2f} closed at or below "
            f"your stop at ₹{stop:.2f}. Capital-preservation discipline: "
            f"exit at the next open regardless of other signals. Loss on this "
            f"trade: {pnl_pct:+.1f}% (₹{pnl_abs:+.0f})."
        )
        return dict(action=action, urgency=urgency, pnl_pct=pnl_pct, pnl_abs=pnl_abs,
                    days_held=days_held, new_stop=None, add_qty=0,
                    reasons=[narrative[0]],
                    narrative=narrative, checks=checks,
                    score=dict(ta=ta_score, news=news_score,
                               event=event_score, regime=reg_score,
                               ratchet=0, total=ta_score+news_score+event_score+reg_score))

    if ev_urgent:
        action, urgency = "EXIT", "URGENT"
        narrative.append(
            f"⚠️ IMMINENT EVENT ({ev_type} in {ev_days}d). Corporate actions "
            f"like results/AGM/split routinely produce 5-20% overnight gaps. "
            f"Exit before the event and re-evaluate on the post-event tape."
        )
        return dict(action=action, urgency=urgency, pnl_pct=pnl_pct, pnl_abs=pnl_abs,
                    days_held=days_held, new_stop=None, add_qty=0,
                    reasons=[narrative[0]],
                    narrative=narrative, checks=checks,
                    score=dict(ta=ta_score, news=news_score,
                               event=event_score, regime=reg_score,
                               ratchet=0, total=ta_score+news_score+event_score+reg_score))

    if ns <= -0.5 and nn >= 2:
        action, urgency = "EXIT", "URGENT"
        top = (news.get("top_headline") or "")[:120]
        narrative.append(
            f"📰 SEVERE NEGATIVE NEWS. Sentiment score {ns:+.2f} across "
            f"{nn} articles. Headlines of this severity (SEBI probe / auditor "
            f"resignation / large default / fraud allegation) historically "
            f"deliver -15 to -40% moves. Exit even if technicals still look OK."
        )
        if top: narrative.append(f'Top headline: "{top}"')
        return dict(action=action, urgency=urgency, pnl_pct=pnl_pct, pnl_abs=pnl_abs,
                    days_held=days_held, new_stop=None, add_qty=0,
                    reasons=[narrative[0]],
                    narrative=narrative, checks=checks,
                    score=dict(ta=ta_score, news=news_score,
                               event=event_score, regime=reg_score,
                               ratchet=0, total=ta_score+news_score+event_score+reg_score))

    # ---------- TIER 2 : EXIT (≥ 2 TA breakdowns) ----------
    if len(breakdowns) >= 2:
        action, urgency = "EXIT", "normal"
        detail = "; ".join(f"{c['name']} — {c['note']}" for c in breakdowns)
        narrative.append(
            f"🔻 MULTIPLE TA BREAKDOWNS ({len(breakdowns)}). The trend that "
            f"justified this position is broken. Exit at open."
        )
        narrative.append(f"Broken checks → {detail}")
        return dict(action=action, urgency=urgency, pnl_pct=pnl_pct, pnl_abs=pnl_abs,
                    days_held=days_held, new_stop=None, add_qty=0,
                    reasons=[narrative[0]],
                    narrative=narrative, checks=checks,
                    score=dict(ta=ta_score, news=news_score,
                               event=event_score, regime=reg_score,
                               ratchet=0, total=ta_score+news_score+event_score+reg_score))

    # ---------- TIER 3 : REDUCE (book half) ----------
    if -0.3 >= ns > -0.5 and nn >= 2:
        top = (news.get("top_headline") or "")[:120]
        narrative.append(
            f"📰 MODERATE NEGATIVE NEWS (score {ns:+.2f}, {nn} articles). "
            f"Not severe enough to force exit, but material enough to "
            f"reduce exposure. Book roughly half; reassess in 2-3 sessions."
        )
        if top: narrative.append(f'Top: "{top}"')
        return dict(action="REDUCE", urgency="normal", pnl_pct=pnl_pct, pnl_abs=pnl_abs,
                    days_held=days_held, new_stop=None,
                    add_qty=-int(qty // 2),
                    reasons=[narrative[0]],
                    narrative=narrative, checks=checks,
                    score=dict(ta=ta_score, news=news_score,
                               event=event_score, regime=reg_score,
                               ratchet=0, total=ta_score+news_score+event_score+reg_score))

    if len(breakdowns) == 1 and pnl_pct < 0:
        bd = breakdowns[0]
        narrative.append(
            f"⚠️ WARNING SIGNAL WHILE IN LOSS ({pnl_pct:+.1f}%). "
            f"{bd['name']} — {bd['note']}. One broken check plus red ink is "
            f"enough to trim exposure; book half and let the rest ride."
        )
        return dict(action="REDUCE", urgency="normal", pnl_pct=pnl_pct, pnl_abs=pnl_abs,
                    days_held=days_held, new_stop=None,
                    add_qty=-int(qty // 2),
                    reasons=[narrative[0]],
                    narrative=narrative, checks=checks,
                    score=dict(ta=ta_score, news=news_score,
                               event=event_score, regime=reg_score,
                               ratchet=0, total=ta_score+news_score+event_score+reg_score))

    # ---------- TIER 4 : HOLD (with ratchet + optional ADD) ----------
    reasons = []
    add_qty = 0

    new_stop = _ratchet_stop(entry, pnl_pct, stop)
    ratchet_score = 0
    if new_stop is not None:
        floor_pct = (new_stop / entry - 1) * 100
        reasons.append(f"🔒 Raise stop-loss to ₹{new_stop:.2f} "
                       f"(was ₹{stop:.2f}) — protects {floor_pct:+.1f}% floor")
        narrative.append(
            f"🔒 RATCHET LADDER: with {pnl_pct:+.1f}% gain, the ladder qualifies "
            f"you for a new stop floor at +{floor_pct:.1f}% of entry. "
            f"Raising the stop from ₹{stop:.2f} to ₹{new_stop:.2f} locks in "
            f"₹{(new_stop-entry)*qty:+.0f} of your open profit while still "
            f"letting the position run further."
        )
        ratchet_score = 2

    # Add signal — profit + fresh signal + friendly regime + no negative news
    add_ok = (pnl_pct >= 5 and ta.get("signal_today")
              and regime in ("RISK-ON", "NEUTRAL")
              and ns >= -0.1)
    if add_ok:
        add_qty = max(1, int(qty * 0.5))
        reasons.append(f"➕ Fresh signal on a winner (+{pnl_pct:.1f}%) — "
                       f"consider adding {add_qty} shares (~50% of {int(qty)})")
        narrative.append(
            f"➕ SCALE-IN OPPORTUNITY: position is up {pnl_pct:+.1f}% AND the "
            f"scanner's PASS_combined rule is firing on today's bar AND market "
            f"regime is {regime}. Consider adding ~50% ({add_qty} shares) of "
            f"your original size. This is scale-in, not averaging down — "
            f"never fires on losing positions."
        )

    # Positive news support (informational, positive tilt)
    if ns >= 0.3 and nn >= 2:
        top = (news.get("top_headline") or "")[:120]
        reasons.append(f"📰 Positive news support (score {ns:+.2f}, {nn} articles)")
        narrative.append(
            f"📰 POSITIVE NEWS BACKS THE THESIS: {nn} articles with net "
            f"sentiment {ns:+.2f}. Reinforces the HOLD."
        )
        if top: narrative.append(f'Top: "{top}"')

    # 1 breakdown while in profit — WATCH
    if len(breakdowns) == 1 and pnl_pct >= 0:
        bd = breakdowns[0]
        reasons.append(f"⚠️ Watch: {bd['name']} — {bd['note']}")
        narrative.append(
            f"⚠️ ONE BROKEN TA CHECK ({bd['name']}). Not enough for exit on "
            f"its own since you're still in profit, but if a second check "
            f"joins it, action will escalate to EXIT next run."
        )

    # Default reasoning when everything is clean
    if not reasons:
        rsi = ta.get("rsi14", np.nan)
        p200 = ta.get("pct_vs_sma200", np.nan)
        rsi_txt = f"RSI {rsi:.0f}" if np.isfinite(rsi) else "TA neutral"
        p200_txt = f"{p200:+.1f}% vs 200-DMA" if np.isfinite(p200) else ""
        reasons.append(f"📈 Trend intact ({rsi_txt}, {p200_txt}, {pnl_pct:+.1f}% P&L) "
                       f"— continue holding")
        narrative.append(
            f"📈 ALL CLEAR — no breakdown signals, no material negative news, "
            f"no imminent events. Continue holding. Current: {pnl_pct:+.1f}% "
            f"P&L over {days_held} days."
        )

    total_score = ta_score + news_score + event_score + reg_score + ratchet_score

    return dict(action="HOLD", urgency="normal", pnl_pct=pnl_pct, pnl_abs=pnl_abs,
                days_held=days_held, new_stop=new_stop, add_qty=add_qty,
                reasons=reasons,
                narrative=narrative, checks=checks,
                score=dict(ta=ta_score, news=news_score, event=event_score,
                           regime=reg_score, ratchet=ratchet_score, total=total_score))


# ======================================================================================
#  ANALYSIS DRIVER (per position)
# ======================================================================================
def analyze_position(position: pd.Series, bench_close: pd.Series,
                      regime: str, use_news: bool, use_events: bool,
                      sector_map: dict) -> dict:
    """Full analysis pipeline for one position."""
    ticker_bare = str(position["ticker"]).upper()
    ty = _to_yahoo(ticker_bare)

    # Auto-fill sector if blank (from NSE map)
    user_sector = str(position.get("sector") or "").strip()
    if user_sector:
        sector = user_sector
        sector_source = "user"
    else:
        sector = sector_map.get(ticker_bare, "-") or "-"
        sector_source = "auto (NSE map)" if sector != "-" else "unknown"

    raw = _fetch_stock(ty)
    result = {
        "ticker":    ticker_bare,
        "yahoo":     ty,
        "buy_date":  position["buy_date"],
        "buy_price": position["buy_price"],
        "quantity":  position["quantity"],
        "stop_loss": position.get("stop_loss"),          # may be NaN — derived below
        "target":    position.get("target"),
        "sector":    sector,
        "sector_source": sector_source,
        "notes":     position.get("notes", ""),
    }
    if raw.empty or len(raw) < 220:
        result.update({"action": "NO_DATA", "urgency": "normal",
                       "reasons": [f"insufficient data ({len(raw)} bars) — "
                                    f"ticker may be delisted or Yahoo-unavailable"],
                       "current_price": np.nan, "pnl_pct": np.nan, "pnl_abs": np.nan,
                       "new_stop": None, "add_qty": 0, "days_held": None,
                       "ta": {}, "news": {}, "events": {},
                       "checks": [], "narrative": [],
                       "stop_source": "n/a"})
        return result

    # Compute indicators + signal (uses SAME engine as scanner)
    df_ind = engine.compute_indicators(raw)
    df_ind = engine.generate_signals(df_ind, "PASS_combined",
                                      {"regime": 8.0, "atr": 3.5, "roc": 3.0,
                                       "volr": 1.2, "rsi_os": 30.0},
                                      bench_close=bench_close,
                                      require_confirmation=True,
                                      block_risk_off=False)
    ta = _ta_snapshot(df_ind)

    # --- AUTO-DERIVE stop_loss if user left it blank ---
    user_stop = position.get("stop_loss")
    if pd.isna(user_stop) or user_stop in (None, 0):
        derived_stop, stop_method = _derive_stop_loss(float(position["buy_price"]), ta)
        result["stop_loss"] = derived_stop
        result["stop_source"] = f"auto: {stop_method}"
        # Rewrite the position series so decide() sees the derived stop
        position = position.copy()
        position["stop_loss"] = derived_stop
    else:
        result["stop_source"] = "user"

    # News
    news = {"score": 0.0, "n_articles": 0, "top_headline": None,
            "matched_terms": [], "all_headlines": []}
    if use_news and HAVE_NEWS:
        try:
            news = _news_score(ty)
        except Exception:
            pass

    # Events
    events = {"blocked": False, "type": None, "days_until": None,
              "subject": None, "all_upcoming": []}
    if use_events and HAVE_EVENTS:
        try:
            events = _event_risk(ticker_bare, next_sessions=5)
        except Exception:
            pass

    decision = decide(position, ta, news, events, regime)
    result.update(decision)
    result["current_price"] = ta.get("close", np.nan)
    result["ta"] = ta
    result["news"] = news
    result["events"] = events
    return result


# ======================================================================================
#  RENDER
# ======================================================================================
ACTION_STYLE = {
    "EXIT":     ("🔴", "#dc2626"),
    "REDUCE":   ("🟠", "#f97316"),
    "HOLD":     ("🟢", "#16a34a"),
    "NO_DATA":  ("⚪", "#64748b"),
}


def _style_action(a: str) -> str:
    emo, _ = ACTION_STYLE.get(a, ("•", "#374151"))
    return f"{emo} {a}"


def main():
    """Standalone entry-point — sets page config, then renders body()."""
    st.set_page_config(page_title="Position Monitor", layout="wide")
    body()


def body():
    """All render logic, no set_page_config (safe to call inside a larger app)."""
    st.title("📊 Position Monitor — daily hold / exit / add decisions")
    st.caption("Reads your open positions from **positions.csv** and applies "
               "position-management decision logic. Run after market close. "
               "Uses the same engine as the scanner + live news + upcoming events.")

    with st.sidebar:
        st.header("Data sources")
        use_news   = st.checkbox("Use news sentiment (yfinance + Google News)",
                                  value=HAVE_NEWS, disabled=not HAVE_NEWS,
                                  help="Adds Tier-1 (severe) and Tier-3 (moderate) "
                                       "negative-news exits.")
        use_events = st.checkbox("Use NSE upcoming events",
                                  value=HAVE_EVENTS, disabled=not HAVE_EVENTS,
                                  help="Adds Tier-1 forced-exit when a scheduled "
                                       "event (results/AGM/split/dividend) is within "
                                       "2 sessions — prevents overnight gap-risk.")
        st.divider()
        st.header("File")
        st.code(POSITIONS_CSV, language="text")
        st.caption("Edit that file to add / update / remove positions. "
                   "Then click **Refresh** below.")
        if st.button("🔄 Refresh (re-fetch prices & news)", type="secondary",
                     use_container_width=True):
            _fetch_stock.clear()
            _fetch_bench.clear()
            st.rerun()
        st.divider()
        st.header("Ratchet ladder")
        rows = "\n".join([f"| +{p:.0f}% gain | stop → +{f:.0f}% |"
                          for p, f in RATCHET_LADDER])
        st.markdown("| P&L | New stop floor |\n|---|---|\n" + rows)

    # ---- Load positions ----
    positions, errors = load_positions(POSITIONS_CSV)
    if errors:
        for e in errors:
            st.warning(e)
    if positions.empty:
        st.info("No open positions to analyze. Edit **positions.csv** and click Refresh.")
        with st.expander("Show positions.csv template"):
            try:
                with open(POSITIONS_CSV, "r", encoding="utf-8") as f:
                    st.code(f.read(), language="csv")
            except Exception as ex:
                st.error(f"Can't read template: {ex}")
        return

    st.caption(f"Loaded **{len(positions)}** open positions.")

    # ---- Sector map + benchmark + regime ----
    sector_map = {}
    if HAVE_UNIVERSE:
        try:
            sector_map = _ul_load().get("sector_map", {})
        except Exception:
            sector_map = {}
    with st.spinner("Fetching benchmark index for regime..."):
        bench_name, bench_df = _fetch_bench()
    regime = _regime_from_bench(bench_df)
    bench_close = bench_df["Close"] if not bench_df.empty else None

    regime_emoji = {"RISK-ON": "🟢", "NEUTRAL": "🟡", "RISK-OFF": "🔴",
                    "UNKNOWN": "⚪"}
    st.info(f"{regime_emoji.get(regime, '⚪')} **Market regime: {regime}** "
            f"(benchmark: {bench_name or 'unavailable'})")

    # ---- Analyze each position ----
    results = []
    prog = st.progress(0.0); status = st.empty()
    for k, (_, pos) in enumerate(positions.iterrows(), 1):
        status.write(f"Analyzing {pos['ticker']} ({k}/{len(positions)})...")
        r = analyze_position(pos, bench_close, regime, use_news, use_events, sector_map)
        results.append(r)
        prog.progress(k / len(positions))
        time.sleep(0.1)
    status.empty(); prog.empty()

    # ---- Summary table ----
    st.subheader("🎯 Tonight's action list")
    tbl = pd.DataFrame([{
        "Stock":    r["ticker"],
        "Sector":   r.get("sector", "-") or "-",
        "Action":   _style_action(r["action"]),
        "Urgency":  r.get("urgency", "normal"),
        "Days":     r.get("days_held", "-"),
        "Buy ₹":    round(r["buy_price"], 2),
        "Now ₹":    round(r.get("current_price", 0), 2) if pd.notna(r.get("current_price", np.nan)) else "-",
        "P&L %":    f"{r.get('pnl_pct', 0):+.1f}%" if pd.notna(r.get('pnl_pct', np.nan)) else "-",
        "P&L ₹":    f"{r.get('pnl_abs', 0):+.0f}" if pd.notna(r.get('pnl_abs', np.nan)) else "-",
        "Stop ₹":   round(r["stop_loss"], 2),
        "New stop ₹": (round(r["new_stop"], 2) if r.get("new_stop") else "—"),
        "Add qty":  (r.get("add_qty", 0) if r.get("add_qty", 0) != 0 else "—"),
        "Reason (short)": r["reasons"][0] if r.get("reasons") else "",
    } for r in results])

    # Sort urgent first, then by action severity, then by |P&L|
    action_order = {"EXIT": 0, "REDUCE": 1, "HOLD": 2, "NO_DATA": 3}
    urgency_order = {"URGENT": 0, "normal": 1}
    tbl["_a"] = tbl["Action"].str.extract(r"(\w+)$")[0].map(action_order).fillna(9)
    tbl["_u"] = tbl["Urgency"].map(urgency_order).fillna(9)
    tbl = tbl.sort_values(["_u", "_a", "Stock"]).drop(columns=["_a", "_u"]).reset_index(drop=True)
    st.dataframe(tbl, use_container_width=True, hide_index=True, height=min(60 + 35*len(tbl), 500))

    # Aggregate P&L
    c1, c2, c3, c4 = st.columns(4)
    total_pnl = sum(r.get("pnl_abs", 0) for r in results if pd.notna(r.get("pnl_abs", np.nan)))
    total_cap = sum(r["buy_price"] * r["quantity"] for r in results)
    total_now = sum((r.get("current_price", r["buy_price"]) or r["buy_price"]) * r["quantity"] for r in results)
    n_exit = sum(1 for r in results if r["action"] == "EXIT")
    n_reduce = sum(1 for r in results if r["action"] == "REDUCE")
    n_hold_add = sum(1 for r in results if r["action"] == "HOLD" and r.get("add_qty", 0) > 0)
    c1.metric("Portfolio P&L", f"₹{total_pnl:+,.0f}",
              f"{100*total_pnl/total_cap:+.2f}%" if total_cap > 0 else "-")
    c2.metric("Capital deployed", f"₹{total_cap:,.0f}")
    c3.metric("Current value", f"₹{total_now:,.0f}")
    c4.metric("Actions", f"{n_exit} exit · {n_reduce} reduce · {n_hold_add} add-signal")

    st.download_button("⬇️ Download action list",
                        tbl.to_csv(index=False).encode(),
                        file_name=f"monitor_actions_{dt.date.today()}.csv",
                        mime="text/csv")

    # ---- Per-stock drill-down (full analysis) ----
    st.subheader("🔎 Per-stock analysis  —  every check that ran")
    st.caption("Each stock's card shows: the full TA check-list (with pass/warn/fail badges), "
                "news headlines with scores, upcoming events, ratchet-ladder computation, "
                "and a step-by-step narrative of why the algo landed on its recommendation.")

    _V_ICON = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌", "INFO": "ℹ️"}

    for r in results:
        emo, colour = ACTION_STYLE.get(r["action"], ("•", "#374151"))
        pnl = r.get("pnl_pct", 0) or 0
        header = (f"{emo} **{r['ticker']}** ({r.get('sector','-')})  ·  "
                  f"{r['action']}  ·  P&L {pnl:+.1f}%  ·  {r.get('days_held','-')}d held")
        with st.expander(header):
            # ---- Top-level position metrics ----
            cA, cB, cC, cD = st.columns(4)
            cA.metric("Buy price", f"₹{r['buy_price']:.2f}",
                      f"{r.get('days_held','-')}d ago")
            cB.metric("Current", f"₹{r.get('current_price', 0):.2f}" if pd.notna(r.get('current_price', np.nan)) else "n/a",
                      f"{pnl:+.1f}%  (₹{r.get('pnl_abs', 0):+,.0f})")
            stop_delta = (f"→ ₹{r['new_stop']:.2f} (raise)"
                          if r.get("new_stop") else "unchanged")
            cC.metric("Stop-loss", f"₹{r['stop_loss']:.2f}", stop_delta,
                      help=r.get("stop_source", ""))
            tgt = r.get("target")
            tgt_txt = f"₹{tgt:.2f}" if (tgt and pd.notna(tgt)) else "advisory / none"
            cD.metric("Target (advisory)", tgt_txt)

            # ---- Provenance line ----
            src_bits = []
            if r.get("stop_source"): src_bits.append(f"stop: {r['stop_source']}")
            if r.get("sector_source"): src_bits.append(f"sector: {r['sector_source']}")
            if src_bits:
                st.caption("Data source — " + " · ".join(src_bits))

            # ---- Score contributions ----
            score = r.get("score") or {}
            if score:
                total = score.get("total", 0)
                score_txt = (f"**Composite score = {total:+d}**  ·  "
                             f"TA {score.get('ta',0):+d}  ·  "
                             f"News {score.get('news',0):+d}  ·  "
                             f"Events {score.get('event',0):+d}  ·  "
                             f"Regime {score.get('regime',0):+d}  ·  "
                             f"Ratchet {score.get('ratchet',0):+d}")
                st.markdown(score_txt)

            # ---- Narrative (multi-line story) ----
            narrative = r.get("narrative") or []
            if narrative:
                st.markdown("**Full reasoning:**")
                for line in narrative:
                    st.markdown(f"> {line}")

            # ---- Checklist table ----
            checks = r.get("checks") or []
            if checks:
                st.markdown("**Detailed check-list**")
                check_rows = []
                for c in checks:
                    check_rows.append({
                        "": _V_ICON.get(c["verdict"], "•"),
                        "Category": c["category"].upper(),
                        "Check": c["name"],
                        "Value": str(c["value"]),
                        "Verdict": c["verdict"],
                        "Note": c["note"],
                    })
                cdf = pd.DataFrame(check_rows)
                st.dataframe(cdf, use_container_width=True, hide_index=True,
                              height=min(50 + 32*len(cdf), 400))

            # ---- News detail ----
            news = r.get("news") or {}
            all_h = news.get("all_headlines") or []
            if all_h:
                st.markdown(f"**📰 Headlines analysed** "
                             f"(net score {news.get('score', 0):+.2f} across {len(all_h)})")
                head_rows = []
                for h in all_h[:15]:
                    d = h.get("date")
                    d_str = d.strftime("%Y-%m-%d") if d and hasattr(d, "strftime") else "-"
                    head_rows.append({
                        "Date":   d_str,
                        "Score":  f"{h.get('score', 0):+.2f}",
                        "Source": h.get("source", "-"),
                        "Headline": (h.get("title") or "")[:180],
                    })
                st.dataframe(pd.DataFrame(head_rows), use_container_width=True, hide_index=True,
                              height=min(50 + 32*len(head_rows), 400))
            elif news and news.get("n_articles", 0) > 0:
                st.markdown(f"**News:** score **{news.get('score', 0):+.2f}** "
                             f"({news.get('n_articles', 0)} articles)")
                if news.get("top_headline"):
                    st.caption(f"Top: \"{news['top_headline']}\"")
            else:
                st.caption("📰 No news articles found in the last 5 sessions.")

            # ---- Events detail ----
            events = r.get("events") or {}
            all_ev = events.get("all_upcoming") or []
            if all_ev:
                st.markdown(f"**📅 Upcoming corporate events (next 30 days)**")
                ev_rows = [{
                    "Date":   str(e.get("date", "-")),
                    "Type":   e.get("type", "-"),
                    "Subject": (e.get("subject") or "")[:200],
                } for e in all_ev]
                st.dataframe(pd.DataFrame(ev_rows), use_container_width=True, hide_index=True,
                              height=min(50 + 32*len(ev_rows), 250))
            elif events and events.get("type"):
                st.markdown(f"**Upcoming event:** {events['type']} in "
                             f"{events.get('days_until', '?')}d")
                if events.get("subject"):
                    st.caption(events["subject"][:200])
            else:
                st.caption("📅 No corporate events scheduled in the next 5 sessions.")

            # ---- Ratchet ladder — highlight which rung applied ----
            if r.get("new_stop"):
                pnl_here = pnl
                st.markdown("**🔒 Ratchet ladder (which rung fired)**")
                lad_rows = []
                for peak, floor in RATCHET_LADDER:
                    fires = pnl_here >= peak
                    lad_rows.append({
                        " ":     "🔥" if fires else "",
                        "Gain ≥": f"+{peak:.0f}%",
                        "New stop floor": f"+{floor:.0f}% of entry",
                        "Absolute stop": f"₹{r['buy_price'] * (1 + floor/100):.2f}",
                    })
                st.dataframe(pd.DataFrame(lad_rows), use_container_width=True, hide_index=True,
                              height=min(50 + 32*len(lad_rows), 350))
                st.caption(f"Current P&L {pnl_here:+.1f}% qualifies for the highest 🔥 rung. "
                            f"Update `stop_loss` in positions.csv to ₹{r['new_stop']:.2f} to lock in.")

            if r.get("notes"):
                st.caption(f"Your notes: {r['notes']}")


if __name__ == "__main__":
    main()
