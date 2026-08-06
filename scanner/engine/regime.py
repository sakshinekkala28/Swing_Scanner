"""
scanner.engine.regime
=====================

Market regime engine.

Responsibilities
----------------
✓ Breadth analysis
✓ Market regime
✓ Composite gate
✓ Market health
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from scanner.utils.constants import (
    BREADTH_MIN_N_FOR_VETO,
    RS_WINDOW,
)

logger = logging.getLogger(__name__)

Stats = dict[str, Any]
DataFrame = pd.DataFrame

__version__ = "1.0.0"


# =============================================================================
# Breadth
# =============================================================================

def compute_breadth(
    rows: list[Stats],
) -> Stats:
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



# =============================================================================
# Market Regime
# =============================================================================

def compute_regime(
    benchmark: DataFrame,
) -> Stats:
    """Trend/momentum of the broad benchmark (one input to the composite gate)."""
    if benchmark.empty or len(benchmark) < 210:
        return {"status": "UNKNOWN", "note": "index data unavailable",
                "idx_ret_window": 0.0, "index_ok": False}
    c = benchmark["Close"]
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



# =============================================================================
# Composite Gate
# =============================================================================

def composite_gate(
    regime: Stats,
    segments: Stats,
    breadth: Stats,
) -> Stats:
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




__all__ = [
    "compute_breadth",
    "compute_regime",
    "composite_gate",
]