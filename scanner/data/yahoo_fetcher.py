"""
Yahoo Finance data access layer.

Responsibilities
----------------
- Fetch OHLCV data
- Fetch benchmark index
- Fetch sector indices
- Cache downloaded data
- Retry failed requests
"""

from __future__ import annotations


import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from scanner.utils.constants import (
    BENCH_TICKERS,
    SEGMENT_TICKERS,
)


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
