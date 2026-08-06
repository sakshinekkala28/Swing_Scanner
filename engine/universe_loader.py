"""
universe_loader.py
==================
Robust, self-healing loader for the FULL NSE equity universe.
Zero manual CSV maintenance — pulls everything live from NSE.

THE PROBLEM THIS SOLVES
-----------------------
NSE's archive endpoints are aggressively bot-filtered. Python's default
`requests` library uses a TLS fingerprint (JA3 hash) that NSE's WAF instantly
identifies as non-browser traffic and returns HTTP 503 — even with perfect
browser-mimicking headers. The block is at the TLS-handshake level, not the
HTTP-header level, which is why sending more headers doesn't help.

THE SOLUTION
------------
curl_cffi with `impersonate="chrome124"` produces the *exact* TLS handshake
bytes a real Chrome browser produces. NSE's WAF cannot distinguish it from
a browser and lets it through.

COVERAGE
--------
 * EQUITY_L.csv     -> all ~2000 NSE-listed equities  (AllNSE bucket)
 * ind_nifty100     -> 100 large-caps                 (LargeCap bucket)
 * ind_niftymidcap150 -> 150 mid-caps                 (MidCap bucket)
 * ind_niftysmallcap250 -> 250 small-caps             (SmallCap bucket)
 * ind_nifty500     -> 500 broad-market names         (Nifty500 bucket)
 * sector_map       -> {symbol: industry} from the four index CSVs (~750 names)

FALLBACK CHAIN (each attempted in order until one works)
--------------------------------------------------------
 1. Fresh local disk cache  (24h TTL)      — instant
 2. curl_cffi on nsearchives.nseindia.com  — primary live source
 3. curl_cffi on archives.nseindia.com     — alternate NSE host
 4. nsepython library (if installed)       — active third-party data package
 5. plain `requests`                       — occasionally works
 6. STALE local disk cache (however old)   — better than nothing
 7. Built-in ~130 hardcoded stocks         — absolute last resort

The disk cache is NEVER deleted on failure — if all live sources fail, we
still return the last successful pull, however old. A daily-run trading
system must not go dark because NSE had a bad afternoon.

USAGE
-----
    from universe_loader import load_full_universe
    bundle = load_full_universe()
    buckets    = bundle["buckets"]        # {LargeCap, MidCap, SmallCap, Nifty500, AllNSE}
    sector_map = bundle["sector_map"]     # {SYMBOL: Industry}
    meta       = bundle["meta"]           # {source, cache_age_hours, notes}
"""

import io
import json
import os
import time

import pandas as pd
import streamlit as st

# --- Chrome TLS impersonation via curl_cffi (the ONE dependency that matters) ---
try:
    from curl_cffi import requests as curl_requests
    HAVE_CURL_CFFI = True
except Exception:
    HAVE_CURL_CFFI = False

# --- Optional secondary source ---
try:
    import nsepython
    HAVE_NSEPYTHON = True
except Exception:
    HAVE_NSEPYTHON = False

# --- Last-resort plain requests ---
try:
    import requests
except Exception:
    requests = None


# ======================================================================================
#  CONFIG
# ======================================================================================
_HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(_HERE, ".universe_cache.json")
CACHE_TTL_SEC = 24 * 60 * 60          # refresh cache every 24h

# NSE serves the same files from two hostnames. Try both.
NSE_HOSTS = ["nsearchives.nseindia.com", "archives.nseindia.com"]

INDEX_CSVS = {                          # {bucket_name: path_relative_to_host}
    "LargeCap":  "content/indices/ind_nifty100list.csv",
    "MidCap":    "content/indices/ind_niftymidcap150list.csv",
    "SmallCap":  "content/indices/ind_niftysmallcap250list.csv",
    "Nifty500":  "content/indices/ind_nifty500list.csv",
}
EQUITY_L_PATH = "content/equities/EQUITY_L.csv"

# Built-in last-resort universe (mirrors the one in swing_scanner_app.py).
_BUILTIN = {
    "LargeCap": [
        "RELIANCE","TCS","HDFCBANK","ICICIBANK","INFY","HINDUNILVR","ITC","SBIN",
        "BHARTIARTL","KOTAKBANK","LT","AXISBANK","BAJFINANCE","ASIANPAINT","MARUTI",
        "SUNPHARMA","TITAN","ULTRACEMCO","WIPRO","NESTLEIND","ONGC","NTPC","POWERGRID",
        "M&M","TATAMOTORS","TATASTEEL","JSWSTEEL","ADANIENT","ADANIPORTS","COALINDIA",
        "HCLTECH","BAJAJFINSV","TECHM","GRASIM","HINDALCO","DRREDDY","CIPLA","BPCL",
        "BRITANNIA","EICHERMOT","DIVISLAB","HEROMOTOCO","INDUSINDBK","APOLLOHOSP",
        "TATACONSUM","BAJAJ-AUTO","SBILIFE","HDFCLIFE","LTIM","SHRIRAMFIN",
    ],
    "MidCap": [
        "HUDCO","IRFC","RVNL","BEL","BHEL","IOC","GAIL","PFC","RECLTD","IRCTC",
        "ABCAPITAL","ASHOKLEY","AUROPHARMA","BANKBARODA","CANBK","CGPOWER","CONCOR",
        "COFORGE","CUMMINSIND","DLF","GODREJPROP","HAVELLS","INDHOTEL","JUBLFOOD",
        "LICHSGFIN","LUPIN","MRF","NMDC","OBEROIRLTY","PAGEIND","PERSISTENT",
        "PIIND","POLYCAB","SAIL","SUZLON","TATAPOWER","TORNTPHARM","TRENT","VBL",
        "YESBANK","IDFCFIRSTB","PNB","UNIONBANK","MAXHEALTH","LODHA","HINDZINC",
    ],
    "SmallCap": [
        "IREDA","MAZDOCK","COCHINSHIP","GRSE","HAL","BDL","MIDHANI","RITES",
        "IRCON","NBCC","ENGINERSIN","HFCL","GMRINFRA","JWL","KALYANKJIL","KAYNES",
        "TATATECH","ZOMATO","NYKAA","PAYTM","POLICYBZR","DELHIVERY","MAPMYINDIA",
        "IEX","CDSL","BSE","ANGELONE","CAMS","KFINTECH","MCX","INTELLECT",
        "TANLA","ROUTE","HAPPSTMNDS","LATENTVIEW","SONACOMS","OLECTRA",
    ],
}


# ======================================================================================
#  FETCH PRIMITIVES
# ======================================================================================
def _fetch_curl_cffi(url: str, timeout: int = 25) -> str:
    """Fetch a URL with curl_cffi impersonating real Chrome (TLS + HTTP/2 fingerprint).
    Beats NSE's WAF because the JA3 hash matches actual Chrome traffic. Raises on fail."""
    if not HAVE_CURL_CFFI:
        raise RuntimeError("curl_cffi not installed — pip install curl_cffi")
    s = curl_requests.Session(impersonate="chrome124")
    # Prime cookies exactly the way a real browser lands on the archives page.
    try:
        s.get("https://www.nseindia.com", timeout=10)
        time.sleep(0.3)
    except Exception:
        pass
    r = s.get(url, timeout=timeout)
    r.raise_for_status()
    if not r.text or r.text.strip().lower().startswith("<!doctype html"):
        raise RuntimeError("got HTML instead of CSV (blocked)")
    return r.text


def _fetch_plain(url: str, timeout: int = 25) -> str:
    """Last-resort plain `requests` fetch. Usually blocked by NSE but occasionally works."""
    if requests is None:
        raise RuntimeError("requests not installed")
    s = requests.Session()
    s.headers.update({
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"),
        "Accept": "text/csv,*/*;q=0.8",
        "Referer": "https://www.nseindia.com/",
    })
    try:
        s.get("https://www.nseindia.com", timeout=10)
    except Exception:
        pass
    r = s.get(url, timeout=timeout)
    r.raise_for_status()
    if r.text.strip().lower().startswith("<!doctype html"):
        raise RuntimeError("got HTML instead of CSV (blocked)")
    return r.text


def _try_all_ways(rel_path: str) -> str:
    """Try each NSE host with each fetcher until one returns real CSV."""
    last_err = None
    for host in NSE_HOSTS:
        url = f"https://{host}/{rel_path}"
        for fetcher in (_fetch_curl_cffi, _fetch_plain):
            try:
                return fetcher(url)
            except Exception as e:
                last_err = e
                continue
    raise last_err or RuntimeError("all hosts × all fetchers failed")


# ======================================================================================
#  PARSERS
# ======================================================================================
def _parse_index_csv(csv_text: str) -> tuple:
    """Parse an index-constituent CSV. Returns (symbols_list, {symbol: industry})."""
    df = pd.read_csv(io.StringIO(csv_text))
    df.columns = [c.strip() for c in df.columns]
    sym_col = "Symbol" if "Symbol" in df.columns else df.columns[2]
    symbols = sorted({str(x).strip().upper() for x in df[sym_col] if str(x).strip()})
    sector = {}
    if "Industry" in df.columns:
        for s, ind in zip(df[sym_col], df["Industry"]):
            s, ind = str(s).strip().upper(), str(ind).strip()
            if s and ind and ind.lower() != "nan":
                sector[s] = ind
    return symbols, sector


def _parse_equity_l(csv_text: str) -> list:
    """Parse EQUITY_L.csv → the FULL NSE listing (~2000 equities).

    Keeps only 'EQ' and 'BE' series (main-board tradable equities); intentionally
    drops:
      * 'T'   — trade-for-trade (surveillance-flagged, mandatory-delivery, no
                intraday leverage; usually unsafe for swing strategies)
      * 'SM'  — SME Emerge board (different lot sizes, thinner liquidity)
      * 'IL', 'ILT'  — illiquid segment
      * 'ST', 'GS'   — govt securities / debt

    M10 NOTE (Aug-2026): the "AllNSE" label is thus really "AllNSE-swing-safe".
    Excluding the risky series is deliberate — if you want the T/SM stocks in
    your universe, add them to your override CSV manually."""
    df = pd.read_csv(io.StringIO(csv_text))
    df.columns = [c.strip() for c in df.columns]
    if "SYMBOL" not in df.columns:
        raise ValueError("EQUITY_L.csv missing SYMBOL column")
    if "SERIES" in df.columns:
        df = df[df["SERIES"].astype(str).str.strip().isin(["EQ", "BE"])]
    return sorted({str(x).strip().upper() for x in df["SYMBOL"] if str(x).strip()})


# ======================================================================================
#  DISK CACHE
# ======================================================================================
def _read_cache() -> dict:
    if not os.path.exists(CACHE_FILE):
        return {}
    try:
        with open(CACHE_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _write_cache(payload: dict):
    """Save payload with timestamp. Never raises."""
    payload = dict(payload)
    payload["_ts"] = time.time()
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f)
    except Exception:
        pass


def _cache_age_hours(cache: dict) -> float:
    ts = cache.get("_ts", 0)
    return (time.time() - ts) / 3600.0 if ts else float("inf")


# ======================================================================================
#  THE MAIN LOADER  (this is what the scanner calls)
# ======================================================================================
@st.cache_data(ttl=60 * 60, show_spinner=False)   # 1h in-memory cache on top of disk cache
def load_full_universe(force_refresh: bool = False) -> dict:
    """Return {buckets, sector_map, meta}. Never raises — worst case returns
    the built-in ~130-stock fallback with a warning in meta['notes']."""
    cache = _read_cache()
    age_h = _cache_age_hours(cache)

    # Fresh cache — return without hitting the network at all.
    if (not force_refresh
            and cache
            and cache.get("buckets")
            and age_h * 3600 < CACHE_TTL_SEC):
        return {
            "buckets":    cache["buckets"],
            "sector_map": cache.get("sector_map", {}),
            "meta": {
                "source": "local disk cache (fresh)",
                "cache_age_hours": round(age_h, 1),
                "notes": [f"Cached {age_h:.1f}h ago — next refresh in "
                         f"{max(0, 24 - age_h):.1f}h"],
                "have_curl_cffi": HAVE_CURL_CFFI,
            },
        }

    # Attempt live refresh.
    live, notes = _attempt_live_refresh()

    if live and live["buckets"].get("AllNSE"):
        _write_cache(live)
        return {
            "buckets":    live["buckets"],
            "sector_map": live["sector_map"],
            "meta": {
                "source": "live NSE (fresh pull)",
                "cache_age_hours": 0.0,
                "notes": notes,
                "have_curl_cffi": HAVE_CURL_CFFI,
            },
        }

    # Live failed. Fall back to STALE cache if we have one — better than nothing.
    if cache and cache.get("buckets"):
        return {
            "buckets":    cache["buckets"],
            "sector_map": cache.get("sector_map", {}),
            "meta": {
                "source": f"local disk cache (STALE, {age_h:.0f}h old — NSE unreachable)",
                "cache_age_hours": round(age_h, 1),
                "notes": ["Live refresh failed — using last-known-good"] + notes,
                "have_curl_cffi": HAVE_CURL_CFFI,
            },
        }

    # Absolute last resort: hardcoded fallback.
    fallback = dict(_BUILTIN)
    fallback["Nifty500"] = sorted(set().union(*[set(v) for v in fallback.values()]))
    fallback["AllNSE"] = fallback["Nifty500"]
    return {
        "buckets": fallback,
        "sector_map": {},
        "meta": {
            "source": "built-in ~130 stocks (LAST RESORT)",
            "cache_age_hours": None,
            "notes": ["No cache, live refresh failed"] + notes,
            "have_curl_cffi": HAVE_CURL_CFFI,
        },
    }


def _attempt_live_refresh() -> tuple:
    """Build a fresh universe from live NSE. Returns (result_or_None, notes_list)."""
    notes = []
    buckets = {}
    sector_map = {}

    # 1) Fetch the 4 index CSVs — populates cap buckets AND sector_map (Industry column)
    for bucket, path in INDEX_CSVS.items():
        try:
            text = _try_all_ways(path)
            syms, sec = _parse_index_csv(text)
            buckets[bucket] = syms
            for k, v in sec.items():
                sector_map.setdefault(k, v)  # first hit wins (LargeCap > MidCap > ... > Nifty500)
            notes.append(f"✅ {bucket}: {len(syms)} stocks")
        except Exception as e:
            notes.append(f"❌ {bucket}: {str(e)[:60]}")

    # 2) Fetch EQUITY_L.csv — the master list of ALL NSE-listed equities (~2000)
    try:
        text = _try_all_ways(EQUITY_L_PATH)
        all_syms = _parse_equity_l(text)
        buckets["AllNSE"] = all_syms
        notes.append(f"✅ AllNSE (EQUITY_L): {len(all_syms)} stocks")
    except Exception as e:
        notes.append(f"❌ AllNSE (EQUITY_L): {str(e)[:60]}")
        if buckets:                        # substitute with union of index CSVs
            union = sorted(set().union(*[set(v) for v in buckets.values()]))
            buckets["AllNSE"] = union
            notes.append(f"↪ AllNSE substituted with union of index CSVs ({len(union)})")

    # 3) If literally nothing worked, try nsepython as a last live fallback
    if not buckets and HAVE_NSEPYTHON:
        try:
            all_syms = list(nsepython.nse_eq_symbols())
            buckets["AllNSE"] = sorted({str(s).strip().upper() for s in all_syms})
            notes.append(f"✅ nsepython AllNSE: {len(buckets['AllNSE'])} stocks")
        except Exception as e:
            notes.append(f"❌ nsepython: {str(e)[:60]}")

    if not buckets:
        return None, notes
    return {"buckets": buckets, "sector_map": sector_map}, notes


# ======================================================================================
#  Convenience wrapper for legacy scanner code
# ======================================================================================
def get_buckets_and_sectors() -> tuple:
    """Convenience: returns (buckets_dict, sector_map_dict, meta_dict).
    Drop-in replacement for `fetch_nse_universe() + fetch_sector_map()`."""
    bundle = load_full_universe()
    return bundle["buckets"], bundle["sector_map"], bundle["meta"]
