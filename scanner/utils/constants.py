"""
scanner.utils.constants
=======================

Project-wide constants used across the Swing Scanner.

Modules
-------
- scanner.data
- scanner.engine
- scanner.orchestrator
- app
"""

from __future__ import annotations

# =============================================================================
# DATA
# =============================================================================

MIN_DAYS = 250
TARGET_YEARS = 10
RS_WINDOW = 63  # ~3 months


# =============================================================================
# BENCHMARKS
# =============================================================================

BENCH_TICKERS = [
    "^CRSLDX",  # Nifty 500
    "^NSEI",    # Nifty 50 (fallback)
]


# =============================================================================
# SEGMENT INDICES
# =============================================================================

SEGMENT_TICKERS = {
    "MidCap": [
        "^NSEMDCP50",
        "NIFTY_MIDCAP_100.NS",
        "^CNXMIDCAP",
    ],
    "SmallCap": [
        "^CNXSC",
        "NIFTYSMLCAP250.NS",
        "^CNXSMCAP",
    ],
}


# =============================================================================
# MARKET REGIME
# =============================================================================

BREADTH_MIN_N_FOR_VETO = 100


# =============================================================================
# CACHE
# =============================================================================

CACHE_TTL_SECONDS = 60 * 60 * 12
CACHE_BUCKET_HOURS = 4


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "MIN_DAYS",
    "TARGET_YEARS",
    "RS_WINDOW",
    "BENCH_TICKERS",
    "SEGMENT_TICKERS",
    "BREADTH_MIN_N_FOR_VETO",
    "CACHE_TTL_SECONDS",
    "CACHE_BUCKET_HOURS",
]