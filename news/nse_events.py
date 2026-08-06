"""
nse_events.py
=============
NSE corporate-events fetcher for the swing scanner's news pillar.

Pulls upcoming corporate events (board meetings for results, dividend ex-dates,
splits, bonuses, AGMs) directly from NSE's public APIs. Detects whether a stock
has a scheduled event in the next N trading sessions — if so, the scanner blocks
the trade to prevent event-driven blowups (results-gap risk).

Uses the same curl_cffi impersonated session as governance_fetcher — reuses its
session helpers, no extra dependencies.

Endpoints used (both public, no auth):
  /api/corporate-announcements  — board meetings, results notices, AGM intimations
  /api/corporates-corporateActions — dividends, splits, bonuses (with ex-dates)

Public API:
    fetch_upcoming_events(ticker) -> list[dict]
    event_risk(ticker, next_sessions=5) -> dict
"""

import io
import json
import time
import datetime as dt
import re

try:
    from curl_cffi import requests as _curl
    HAVE_CURL_CFFI = True
except Exception:
    HAVE_CURL_CFFI = False

try:
    import streamlit as st
except Exception:
    st = None

# Reuse the NSE session primer from governance_fetcher (already primes cookies etc.)
try:
    from governance_fetcher import _make_session as _sess, _prime_nse as _prime
    HAVE_HELPERS = True
except Exception:
    HAVE_HELPERS = False


# ======================================================================================
#  Session singleton (lazy-init, primed once)
# ======================================================================================
_session = None
_primed = False


def _get_session():
    global _session, _primed
    if not HAVE_HELPERS or not HAVE_CURL_CFFI:
        return None
    if _session is None:
        _session = _sess()
    if _session is not None and not _primed:
        _prime(_session)
        _primed = True
    return _session


# ======================================================================================
#  EVENT-KEYWORD CLASSIFICATION
# ======================================================================================
# Priority order matters — a "board meeting for results" is a HIGHER-risk event than
# a plain "board meeting" or "dividend record date". The classifier picks the FIRST
# match, so keep the highest-impact keywords at the top.
_EVENT_PATTERNS = [
    ("RESULTS",    re.compile(r"\b(quarterly results|financial results|q[1-4] fy|q[1-4] results|annual results|earnings)\b", re.I)),
    ("BOARD_MTG",  re.compile(r"\b(board meeting|meeting of the board|board of directors)\b", re.I)),
    ("AGM",        re.compile(r"\b(annual general meeting|AGM|EGM|extra.?ordinary general meeting)\b", re.I)),
    ("DIVIDEND",   re.compile(r"\b(dividend|interim dividend|final dividend|record date.*dividend)\b", re.I)),
    ("SPLIT",      re.compile(r"\b(stock split|share split|face value split|sub.?division)\b", re.I)),
    ("BONUS",      re.compile(r"\b(bonus (issue|share)|bonus of|bonus 1:)\b", re.I)),
    ("BUYBACK",    re.compile(r"\bbuy.?back\b", re.I)),
    ("RIGHTS",     re.compile(r"\brights (issue|entitlement)\b", re.I)),
    ("SCHEME",     re.compile(r"\b(scheme of arrangement|amalgamation|demerger|merger)\b", re.I)),
]

# Which event types are HARD-BLOCK (skip the trade) vs SOFT-WARN (still trade,
# but flag). Results and board meetings can produce 5-20% gap-moves overnight —
# hard block. Corporate actions (dividend / split / bonus) also cause ex-date
# price adjustments — hard block within 3 sessions.
_HARD_BLOCK_TYPES = {"RESULTS", "BOARD_MTG", "AGM", "DIVIDEND", "SPLIT", "BONUS", "BUYBACK", "RIGHTS", "SCHEME"}


def _classify(text: str) -> str:
    """Classify an announcement text → event type ('OTHER' if no keyword match)."""
    if not text:
        return "OTHER"
    for tag, rx in _EVENT_PATTERNS:
        if rx.search(text):
            return tag
    return "OTHER"


def _parse_nse_date(s: str) -> dt.date:
    """Parse '25-Jul-2026' or '25-Jul-2026 14:37:56' → date. None on failure."""
    if not s or s in ("-", "--"):
        return None
    try:
        return dt.datetime.strptime(s.split()[0], "%d-%b-%Y").date()
    except Exception:
        return None


# ======================================================================================
#  FETCHERS
# ======================================================================================
def _fetch_json(url: str) -> object:
    """GET url, return parsed JSON or None on any failure."""
    s = _get_session()
    if s is None:
        return None
    try:
        r = s.get(url, timeout=15)
    except Exception:
        return None
    if r.status_code != 200:
        return None
    ct = r.headers.get("content-type", "").lower()
    if "json" not in ct:
        return None
    try:
        return r.json()
    except Exception:
        try:
            return json.loads(r.text)
        except Exception:
            return None


def _fetch_announcements(ticker: str, lookback_days: int = 14) -> list:
    """Fetch corporate announcements for `ticker` over the last `lookback_days`.
    Returns list of {date, subject, type} sorted newest-first."""
    to_dt = dt.date.today()
    from_dt = to_dt - dt.timedelta(days=lookback_days)
    from_s = from_dt.strftime("%d-%m-%Y")
    to_s = to_dt.strftime("%d-%m-%Y")
    url = (f"https://www.nseindia.com/api/corporate-announcements?"
           f"index=equities&symbol={ticker}&from_date={from_s}&to_date={to_s}")
    data = _fetch_json(url)
    if not isinstance(data, list):
        return []
    out = []
    for row in data:
        d = _parse_nse_date(row.get("an_dt", ""))
        if d is None:
            continue
        subject = str(row.get("desc") or "") + " | " + str(row.get("attchmntText") or "")
        etype = _classify(subject)
        out.append({"date": d, "subject": subject.strip()[:200], "type": etype})
    return sorted(out, key=lambda r: r["date"], reverse=True)


def _fetch_corporate_actions(ticker: str) -> list:
    """Fetch upcoming and recent corporate actions (dividend/split/bonus).
    Returns list of {date, subject, type} — 'date' is the ex-date."""
    url = f"https://www.nseindia.com/api/corporates-corporateActions?index=equities&symbol={ticker}"
    data = _fetch_json(url)
    if not isinstance(data, list):
        return []
    out = []
    for row in data:
        ex = _parse_nse_date(row.get("exDate", "")) or _parse_nse_date(row.get("recDate", ""))
        if ex is None:
            continue
        subject = str(row.get("subject", "")).strip()
        etype = _classify(subject)
        out.append({"date": ex, "subject": subject[:200], "type": etype})
    return sorted(out, key=lambda r: r["date"])


# ======================================================================================
#  PUBLIC API
# ======================================================================================
def fetch_upcoming_events(ticker: str, lookback_days: int = 14,
                          lookahead_days: int = 30) -> list:
    """Combined feed: announcements + corporate actions, deduplicated by date + type.

    Args:
        ticker: bare NSE symbol (no .NS suffix)
        lookback_days: how far back to look at announcements (default 14)
        lookahead_days: how far forward to keep (default 30)

    Returns list of {date, subject, type} sorted by date ascending.
    """
    today = dt.date.today()
    max_date = today + dt.timedelta(days=lookahead_days)

    ann = _fetch_announcements(ticker, lookback_days=lookback_days)
    ca = _fetch_corporate_actions(ticker)

    all_events = ann + ca
    # Keep only events with date in range [today - lookback, today + lookahead]
    min_date = today - dt.timedelta(days=lookback_days)
    filtered = [e for e in all_events if min_date <= e["date"] <= max_date]

    # Deduplicate by (date, type)
    seen = set()
    unique = []
    for e in filtered:
        k = (e["date"], e["type"])
        if k in seen:
            continue
        seen.add(k)
        unique.append(e)
    return sorted(unique, key=lambda r: r["date"])


def _cached_events(ticker: str) -> list:
    """Internal 30-min cached wrapper. Streamlit's @cache_data used if available;
    else no cache (still one lookup per scanner run)."""
    return fetch_upcoming_events(ticker)


if st is not None:
    _cached_events = st.cache_data(ttl=30 * 60, show_spinner=False)(fetch_upcoming_events)


def event_risk(ticker: str, next_sessions: int = 5) -> dict:
    """Assess event risk for `ticker` for the next N trading sessions.

    Trading-session approximation: business days × 1.4 (adjust for holidays).
    A 5-session window ≈ 7 calendar days.

    Returns:
        {"blocked": bool,           # True if a HARD-BLOCK-TYPE event in window
         "type": str or None,       # RESULTS / BOARD_MTG / DIVIDEND / etc.
         "days_until": int or None, # calendar days until the earliest event
         "subject": str or None,    # first-few-words of the announcement
         "all_upcoming": list}      # everything in the window (for display)
    """
    try:
        events = _cached_events(ticker)
    except Exception:
        events = []
    today = dt.date.today()
    window_end = today + dt.timedelta(days=int(next_sessions * 1.4) + 1)
    upcoming = [e for e in events if today <= e["date"] <= window_end]
    if not upcoming:
        return {"blocked": False, "type": None, "days_until": None,
                "subject": None, "all_upcoming": []}
    upcoming.sort(key=lambda r: r["date"])
    hard = [e for e in upcoming if e["type"] in _HARD_BLOCK_TYPES]
    if hard:
        first = hard[0]
        return {"blocked": True, "type": first["type"],
                "days_until": (first["date"] - today).days,
                "subject": first["subject"][:120],
                "all_upcoming": upcoming}
    first = upcoming[0]
    return {"blocked": False, "type": first["type"],
            "days_until": (first["date"] - today).days,
            "subject": first["subject"][:120],
            "all_upcoming": upcoming}
