"""
news_sentiment.py
=================
Free news + keyword-based sentiment scorer for the swing scanner.

Sources (both free, no auth):
  1. yfinance Ticker.news       — Reuters/Bloomberg wire, ~10 recent items
  2. Google News RSS            — very broad, pulls from 100+ Indian outlets

Sentiment engine: hand-curated Indian-market keyword lexicon. Higher precision
than off-the-shelf financial sentiment models on Indian news specifically
(picks up things like "SEBI probe", "auditor resigns", "wins order" which are
strong signals for INR-swing outcomes but poorly weighted by generic models).

Score returned in [-1, +1]:
   > +0.3  strong positive news (upgrade, buyback, big order, expansion)
   -0.3 to +0.3  neutral / mixed / no news
   < -0.3  strong negative news (downgrade, probe, resignation, penalty)

Cached 60 min — news moves fast but our scanner runs after market close, so
60min lets us re-scan within the same session without re-fetching everything.

Public API:
    fetch_news_score(ticker_yahoo) -> dict
"""

import datetime as dt
import re
import urllib.parse
import urllib.request

try:
    import streamlit as st
except Exception:
    st = None

try:
    import yfinance as yf
except Exception:
    yf = None


# ======================================================================================
#  KEYWORD LEXICON — Indian-market tuned
# ======================================================================================
# Each keyword weighted by its typical price impact when it appears in a headline.
# Weights sum to ±5 for the strongest signals so a single big-impact word can
# drive the whole score.
_POSITIVE = {
    # Broker actions
    r"\bupgrade[sd]?\b": 3, r"\btarget (raised?|hiked?|increased?)\b": 3,
    r"\b(buy call|buy rating)\b": 2, r"\boutperform\b": 2, r"\boverweight\b": 2,
    # Results / operations
    r"\bbeats? (estimates?|expectations?|forecasts?|view|street|analysts?)\b": 4,
    r"\btops? (estimates?|expectations?|forecasts?|view|street)\b": 3,
    r"\b(strong|solid|robust) (quarter|q[1-4]|results|earnings|show|performance)\b": 3,
    r"\bprofit (surge|jump|rise|growth|rises?|jumps?|climbs?)\b": 3,
    r"\brecord (high|profit|revenue|quarter)\b": 3,
    r"\b(revenue|profit) (up|jumps?|rises?|surges?) \d+\s*%\b": 3,
    # Deals / orders
    r"\bwins? (order|contract|deal|tender)\b": 4,
    r"\bbags? (order|contract)\b": 4,
    r"\bawarded (contract|order)\b": 4,
    r"\b(joint venture|strategic partnership|acquisition)\b": 2,
    # Capital returns
    r"\bbuyback\b": 3, r"\bbonus (issue|share)\b": 2,
    r"\bdividend (hike|increased?|higher)\b": 2,
    # Expansion / momentum
    r"\bexpansion\b": 1, r"\bnew (plant|facility|capacity)\b": 2,
    r"\blaunch(es|ed)?\b": 1,
    r"\b(all.?time|52.?week) high\b": 2,
    r"\bhits? upper circuit\b": 3, r"\bsurge[sd]?\b": 2, r"\bralli(es|ed)\b": 1,
}

_NEGATIVE = {
    # Broker actions
    r"\bdowngrade[sd]?\b": -3, r"\btarget (cut|lowered?|reduced?)\b": -3,
    r"\b(sell call|sell rating)\b": -2, r"\bunderperform\b": -2,
    # Regulatory / legal (BIG signals — most predictive of crash)
    r"\bSEBI (probe|order|penalty|action|investigation)\b": -5,
    r"\b(income tax|IT department) (raid|search|notice)\b": -5,
    r"\b(ED|CBI|SFIO|CCI) (probe|raid|search|investigation)\b": -5,
    r"\bpenalty\b": -2, r"\bfined?\b": -2,
    r"\bshow.?cause notice\b": -3,
    # Management issues
    r"\b(resigns?|resignation)\b": -3, r"\b(quit[st]?|steps? down|exit[ed]?)\b": -2,
    r"\barrest(ed)?\b": -5, r"\bhospitali[sz]ed\b": -3,
    r"\bauditor (resigns?|qualifies?|adverse|disclaimer)\b": -5,
    # Financial distress
    r"\bloss (widens?|deepens?)\b": -3,
    r"\bmisses? (estimates?|expectations?|forecasts?)\b": -3,
    r"\b(weak|disappointing) (quarter|q[1-4]|results)\b": -3,
    r"\b(profit|earnings) (falls?|declines?|drops?|plunges?)\b": -3,
    r"\bdefault(s|ed)?\b": -4, r"\binsolvency\b": -4, r"\bbankruptcy\b": -5,
    r"\b(NPA|non.?performing)\b": -2, r"\brestructure[dr]?\b": -2,
    # Sentiment
    r"\btumble[sd]?\b": -2, r"\bplunge[sd]?\b": -2, r"\bslump[sd]?\b": -2,
    r"\bhits? lower circuit\b": -3, r"\bcrash(es|ed)?\b": -3,
    # Deal breakdowns
    r"\b(deal|merger|acquisition) (fails?|falls? through|terminated?)\b": -3,
    r"\bfraud\b": -5, r"\bscandal\b": -4,
}


def _score_headline(text: str) -> tuple:
    """Return (raw_score, matched_terms_list) for a single headline."""
    if not text:
        return 0.0, []
    matched = []
    score = 0.0
    for pat, w in _POSITIVE.items():
        if re.search(pat, text, re.IGNORECASE):
            score += w
            matched.append(re.search(pat, text, re.IGNORECASE).group(0).lower())
    for pat, w in _NEGATIVE.items():
        if re.search(pat, text, re.IGNORECASE):
            score += w
            matched.append(re.search(pat, text, re.IGNORECASE).group(0).lower())
    return score, matched


# ======================================================================================
#  NEWS FETCHERS
# ======================================================================================
def _fetch_yfinance_news(ticker_yahoo: str) -> list:
    """Fetch news from yfinance. yfinance's news schema changed in 2024 — the
    payload is now nested under 'content'. Handles both old and new shapes."""
    if yf is None:
        return []
    try:
        news = yf.Ticker(ticker_yahoo).news
    except Exception:
        return []
    if not news:
        return []
    out = []
    for a in news:
        c = a.get("content", a)               # new schema wraps under 'content'
        title = c.get("title") or a.get("title") or ""
        # timestamp: content.pubDate (ISO string) or old providerPublishTime (epoch)
        ts_raw = c.get("pubDate") or a.get("providerPublishTime") or ""
        pub_dt = None
        if isinstance(ts_raw, str) and ts_raw:
            try:
                pub_dt = dt.datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
            except Exception:
                pass
        elif isinstance(ts_raw, (int, float)):
            try:
                pub_dt = dt.datetime.fromtimestamp(float(ts_raw))
            except Exception:
                pass
        if title:
            out.append({"title": title.strip(), "date": pub_dt, "source": "yfinance"})
    return out


def _fetch_google_news(query: str, max_items: int = 10, days: int = 3) -> list:
    """Fetch Google News RSS for a query. No dependencies — parses XML with regex.

    IMPORTANT (Aug-2026 freshness fix): Google News RSS defaults to sorting by
    RELEVANCE, which surfaces stale-but-topical articles (median age 44 days
    in a 10-stock audit). We force date-restriction using Google's advanced
    search operators:
      * `when:Nd`  → only results from the last N days (Google's RSS-native)
      * `+news`    → prioritise news over aggregator noise

    Note: Google may still return items older than `days` (its filter is
    approximate); the caller re-applies a hard cutoff for safety.
    """
    q_with_time = f"{query} when:{int(days)}d"
    q = urllib.parse.quote(q_with_time)
    url = f"https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en"
    try:
        req = urllib.request.Request(url, headers={"User-Agent":
            "Mozilla/5.0 (Windows NT 10.0) AppleWebKit/537.36"})
        raw = urllib.request.urlopen(req, timeout=15).read().decode("utf-8", errors="replace")
    except Exception:
        return []
    # Extract each <item> block, then title + pubDate within
    items = re.findall(r"<item>(.*?)</item>", raw, re.DOTALL)
    out = []
    for block in items[:max_items]:
        tm = re.search(r"<title>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</title>", block, re.DOTALL)
        pm = re.search(r"<pubDate>(.*?)</pubDate>", block)
        if not tm:
            continue
        title = tm.group(1).strip()
        pub_dt = None
        if pm:
            try:
                pub_dt = dt.datetime.strptime(pm.group(1)[:25], "%a, %d %b %Y %H:%M:%S")
            except Exception:
                pass
        out.append({"title": title, "date": pub_dt, "source": "google"})
    return out


# ======================================================================================
#  PUBLIC API
# ======================================================================================
def fetch_news_score(ticker_yahoo: str, company_name: str = None,
                     lookback_days: int = 3) -> dict:
    """Fetch news for a ticker and compute a sentiment score.

    Args:
        ticker_yahoo: e.g. "RELIANCE.NS"
        company_name: optional — improves Google News query. If None, uses bare ticker.
        lookback_days: only headlines within this many days count for the score

    Returns:
        {"score": float in [-1, +1],
         "n_articles": int,
         "top_headline": str,   # most-impact headline in the window
         "top_impact": float,   # its individual score
         "matched_terms": list, # keywords that fired
         "sources": {"yfinance": n, "google": n},
         "all_headlines": list}  # for display (title, date, source, score)
    """
    bare = ticker_yahoo.replace(".NS", "").replace(".BO", "").upper()
    query = f"{bare} stock NSE" if not company_name else f"{company_name} stock NSE"

    yf_news = _fetch_yfinance_news(ticker_yahoo)
    # Pass lookback_days to Google — it filters at query time (fresher results)
    gn_news = _fetch_google_news(query, days=lookback_days)

    # RELEVANCE FILTER (Aug-2026): Google News's `when:3d` operator gives us
    # fresh results but the freshness comes at a relevance cost — many recent
    # items are generic aggregator pages ("HCL Tech Share Price Today") that
    # happen to match search terms. Filter: an article is relevant only if
    # its TITLE contains the bare ticker OR the company name (case-insensitive).
    # yfinance results are already ticker-targeted so we don't filter them.
    _keywords = [bare.lower()]
    if company_name:
        # Add each word of the company name (excluding common noise like Ltd, Company)
        _stop = {"ltd", "ltd.", "limited", "company", "co.", "co", "corp",
                 "corporation", "the", "and", "of", "in", "&"}
        for w in company_name.split():
            wl = w.lower().strip(",.()")
            if wl and wl not in _stop and len(wl) >= 3:
                _keywords.append(wl)
    def _is_relevant(title: str) -> bool:
        if not title:
            return False
        tl = title.lower()
        return any(k in tl for k in _keywords)
    gn_news = [it for it in gn_news if _is_relevant(it["title"])]

    cutoff = dt.datetime.now() - dt.timedelta(days=lookback_days)
    all_items = []
    for it in (yf_news + gn_news):
        d = it.get("date")
        # If we can't tell how old it is, include it (be permissive)
        if d is not None and d.replace(tzinfo=None) < cutoff:
            continue
        score, matched = _score_headline(it["title"])
        all_items.append({**it, "score": score, "matched": matched})

    if not all_items:
        return {"score": 0.0, "n_articles": 0, "top_headline": None,
                "top_impact": 0.0, "matched_terms": [],
                "sources": {"yfinance": 0, "google": 0},
                "all_headlines": []}

    # Aggregate: take the average absolute impact, weighted by recency.
    # Normalise to [-1, +1] using a soft compression (tanh-like).
    raw_scores = [it["score"] for it in all_items]
    net_raw = sum(raw_scores) / len(raw_scores)      # mean, not sum — insensitive to N
    # Compress to [-1, +1]. A headline with |score|=5 gives ~ ±0.76 after tanh.
    import math
    net = math.tanh(net_raw / 4.0)

    # Top headline = the one with the largest absolute individual score
    top = max(all_items, key=lambda x: abs(x["score"]))
    matched_terms = sorted({m for it in all_items for m in it["matched"]})

    return {
        "score": round(net, 3),
        "n_articles": len(all_items),
        "top_headline": top["title"],
        "top_impact": top["score"],
        "matched_terms": matched_terms,
        "sources": {"yfinance": sum(1 for it in all_items if it["source"] == "yfinance"),
                    "google":   sum(1 for it in all_items if it["source"] == "google")},
        "all_headlines": [
            {"title": it["title"], "date": it["date"], "source": it["source"],
             "score": it["score"]} for it in all_items
        ],
    }


# Streamlit cache wrapper (60 min TTL)
if st is not None:
    fetch_news_score = st.cache_data(ttl=60 * 60, show_spinner=False)(fetch_news_score)
