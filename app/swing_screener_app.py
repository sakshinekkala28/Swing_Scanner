"""
NSE Swing-Trade Screener & Backtester
=====================================
An interactive Streamlit app that reproduces the full analysis pipeline built in our
research chat: it pulls a stock from Yahoo Finance, computes ~38 technical indicators
(no look-ahead), applies one of three filter strategies (PASS_recommended / PASS_tight /
PASS_balanced), and backtests a swing strategy using the triple-barrier method
(take-profit / stop-loss / time-stop).

Run with:  streamlit run swing_screener_app.py
"""

import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

try:
    import yfinance as yf
except Exception:
    yf = None


# ======================================================================================
#  1. DATA
# ======================================================================================
@st.cache_data(show_spinner=False)
def fetch_data(ticker: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    """Download daily OHLCV from Yahoo Finance (auto-adjusted for splits/bonus)."""
    if yf is None:
        raise RuntimeError("yfinance is not installed.")
    t = yf.Ticker(ticker)
    df = t.history(start=start, end=end, interval="1d", auto_adjust=True)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.dropna()
    return df


# ======================================================================================
#  2. INDICATORS  (identical logic to the research pipeline; strictly no look-ahead)
# ======================================================================================
def _ema(s, n):  return s.ewm(span=n, adjust=False).mean()
def _sma(s, n):  return s.rolling(n).mean()
def _rma(s, n):  return s.ewm(alpha=1 / n, adjust=False).mean()   # Wilder's smoothing


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    o, h, l, c, v = df["Open"], df["High"], df["Low"], df["Close"], df["Volume"]

    # --- Trend / moving averages ---
    for n in (5, 10, 20, 50, 200):
        df[f"sma{n}"] = _sma(c, n)
    df["ema20"], df["ema50"] = _ema(c, 20), _ema(c, 50)
    df["pct_vs_sma20"]  = (c / df["sma20"]  - 1) * 100
    df["pct_vs_sma50"]  = (c / df["sma50"]  - 1) * 100
    df["pct_vs_sma200"] = (c / df["sma200"] - 1) * 100
    df["ma_aligned_up"] = ((df["sma5"] > df["sma20"]) & (df["sma20"] > df["sma50"])).astype(int)

    # --- RSI(14) ---
    delta = c.diff()
    gain = _rma(delta.clip(lower=0), 14)
    loss = _rma(-delta.clip(upper=0), 14)
    rs = gain / loss.replace(0, np.nan)
    df["rsi14"] = 100 - 100 / (1 + rs)

    # --- Rate of change ---
    df["roc5"]  = (c / c.shift(5)  - 1) * 100
    df["roc10"] = (c / c.shift(10) - 1) * 100

    # --- MACD(12,26,9) ---
    macd = _ema(c, 12) - _ema(c, 26)
    sig = _ema(macd, 9)
    df["macd"], df["macd_signal"], df["macd_hist"] = macd, sig, macd - sig
    df["macd_bull"] = (macd > sig).astype(int)

    # --- ATR(14) & ATR% ---
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    df["atr14"] = _rma(tr, 14)
    df["atr_pct"] = df["atr14"] / c * 100

    # --- Bollinger(20,2) ---
    mid = _sma(c, 20); sd = c.rolling(20).std()
    upper, lower = mid + 2 * sd, mid - 2 * sd
    df["bb_pctB"] = (c - lower) / (upper - lower) * 100
    df["bb_bandwidth"] = (upper - lower) / mid * 100

    # --- Stochastic(14,3) ---
    ll, hh = l.rolling(14).min(), h.rolling(14).max()
    k = (c - ll) / (hh - ll) * 100
    df["stoch_k"], df["stoch_d"] = k, k.rolling(3).mean()

    # --- Volume ---
    df["vol_sma20"] = _sma(v, 20)
    df["vol_ratio"] = v / df["vol_sma20"]
    df["vol_surge"] = (df["vol_ratio"] >= 1.5).astype(int)
    obv = (np.sign(c.diff()).fillna(0) * v).cumsum()
    df["obv"] = obv
    df["obv_slope10"] = obv.diff(10)

    # --- Breakout / range position ---
    df["high20"] = h.rolling(20).max().shift(1)
    df["breakout20"] = (c > df["high20"]).astype(int)
    hi52 = h.rolling(252).max(); lo52 = l.rolling(252).min()
    df["dist_52wH"] = (c / hi52 - 1) * 100
    df["dist_52wL"] = (c / lo52 - 1) * 100
    rng14 = h.rolling(14).max() - l.rolling(14).min()
    df["pos_in_range14"] = (c - l.rolling(14).min()) / rng14 * 100

    # --- Candle / price action ---
    df["gap_up"] = (o - c.shift()) / c.shift() * 100
    df["body_pct"] = (c - o) / o * 100
    df["up_day"] = (c > c.shift()).astype(int)

    # --- ADX(14) ---
    up = h.diff(); dn = -l.diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    atr = _rma(tr, 14)
    plus_di = 100 * _rma(pd.Series(plus_dm, index=c.index), 14) / atr
    minus_di = 100 * _rma(pd.Series(minus_dm, index=c.index), 14) / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df["adx14"], df["plus_di"], df["minus_di"] = _rma(dx, 14), plus_di, minus_di

    return df


# ======================================================================================
#  3. SIGNALS  (the three strategies from our analysis)
# ======================================================================================
def generate_signals(df: pd.DataFrame, strategy: str, p: dict,
                     bench_close: pd.Series = None, require_rs: bool = False,
                     rs_window: int = 63,
                     require_confirmation: bool = False,
                     block_risk_off: bool = False) -> pd.DataFrame:
    """Aug-2026 EVIDENCE-BASED addition: `require_confirmation` — when True,
    signal is valid only if the SIGNAL DAY closes GREEN (close > open) AND
    volume > 20-day average. This is a same-day-confirmation filter that
    doesn't require waiting a day (unlike walk-forward's next-day follow-
    through). Available at scan time in the live scanner. Multi-cutoff walk-
    forward showed this filter converts a 9% win-rate loser into a 50%
    win-rate winner by rejecting failed breakouts."""
    """If `bench_close` is supplied, a rolling relative-strength series is computed
    (stock return minus index return over `rs_window`). With require_rs=True, signals
    only fire when the stock is OUTPERFORMING the index — i.e. it can rise even while
    the market falls. This makes the RS filter *backtestable*, not just a live-scan tilt."""
    df = df.copy()
    c, o = df["Close"], df["Open"]

    # --- rolling relative strength vs the benchmark (no look-ahead: all trailing) ---
    if bench_close is not None:
        b = bench_close.reindex(df.index).ffill()
        stock_ret = c / c.shift(rs_window) - 1
        bench_ret = b / b.shift(rs_window) - 1
        df["rs_roll"] = (stock_ret - bench_ret) * 100
    else:
        df["rs_roll"] = np.nan
    regime   = df["pct_vs_sma200"] > p["regime"]          # strong uptrend
    obv_rise = df["obv_slope10"] > 0                       # accumulation
    vol_ok   = df["atr_pct"] > p["atr"]                    # enough volatility

    A = regime & obv_rise                                  # momentum-continuation branch
    B = (df["pct_vs_sma20"] > 0) & (df["roc10"] > p["roc"]) & (df["vol_ratio"] > p["volr"])  # breakout branch

    # ---- COUNTER-TREND / REVERSAL branch (fires BELOW the 200-DMA) ----
    # Never a naked "buy the dip": every condition demands PROOF the fall is pausing.
    rsi = df["rsi14"]
    downtrend    = c < df["sma50"]                         # genuinely weak / below 50-DMA
    was_oversold = rsi.rolling(5).min() < p["rsi_os"]      # knife was falling (deeply oversold)
    rsi_turning  = rsi > rsi.shift(1)                      # momentum ticking back up
    reversal_bar = (c > c.shift(1)) & (c > o)             # a bullish reversal candle
    volp         = df["vol_ratio"] > 1.0                  # buyers stepping in
    bounce = downtrend & was_oversold & rsi_turning & reversal_bar & volp
    # confirmed turn: reclaiming the 50-DMA from below, with momentum + MACD flip
    reclaim = ((c > df["sma50"]) & (c.shift(1) <= df["sma50"].shift(1))
               & (rsi > 50) & (df["macd_hist"] > 0))
    R = bounce | reclaim

    # ---- COMBINED: regime switch on the 200-DMA ----
    # Above 200-DMA -> trend logic (A). Below -> reversal logic (R). Mutually exclusive.
    above_200 = df["pct_vs_sma200"] > 0
    C = (above_200 & A) | (~above_200 & R)

    # tag every bar with which regime's logic is firing (used by the combined backtest)
    ttype = pd.Series("", index=df.index)
    ttype[A.fillna(False)] = "UPTREND"
    ttype[(R.fillna(False)) & (~A.fillna(False))] = "DOWNTREND"
    df["trade_type"] = ttype

    df["branch_momentum"] = A.astype(int)
    df["branch_breakout"] = B.astype(int)
    df["branch_reversal"] = R.astype(int)
    passes = {
        "PASS_recommended": A,
        "PASS_tight":       A & vol_ok,
        "PASS_balanced":    A | B,
        "PASS_reversal":    R,
        "PASS_combined":    C,
    }
    sig = passes[strategy].fillna(False)
    if require_rs and bench_close is not None:
        sig = sig & (df["rs_roll"] > 0)        # only take longs that are beating the market
    # NEW (Aug-2026): same-day confirmation — signal day must close GREEN with
    # above-average volume. Filters "failed breakouts" that produce most stops.
    if require_confirmation:
        vol_avg20 = df["Volume"].rolling(20).mean().shift(1)
        strong_day = ((df["Close"] > df["Open"])                # green candle
                      & (df["Close"] > df["Close"].shift(1))    # up-day vs yesterday
                      & (df["Volume"] > vol_avg20))              # participation
        sig = sig & strong_day.fillna(False)
    # =====================================================================
    # CHANGE #3 (Aug-2026) — REGIME HARD-BLOCK AT SIGNAL LEVEL
    # ---------------------------------------------------------------------
    # Previously the market-regime gate only trimmed the RANKED shortlist
    # in the scanner ("keep only RS>0 names on RISK-OFF days"). Historical
    # signals still went into the trade log during RISK-OFF regimes,
    # polluting per-stock stats and self-check scores with trades the algo
    # would never actually take live if a regime gate were enforced.
    #
    # This block computes per-bar regime from `bench_close` (no look-ahead:
    # only trailing 200-DMA and 10d ROC) and MASKS signal=False on any bar
    # where the benchmark is RISK-OFF. Same definition as
    # forward_validate_app.regime_at_cutoff:
    #     RISK-ON   : bench > 200-DMA AND 10d ROC > -1.0%
    #     NEUTRAL   : bench > 200-DMA OR 10d ROC > -3.0%
    #     RISK-OFF  : else  (below 200-DMA AND falling)
    #
    # Evidence — forward_validate_app already treats RISK-OFF as a hard
    # block; Change #3 ports the same logic into the live engine so the
    # backtest, scanner and forward-validator all agree.
    # =====================================================================
    if block_risk_off and bench_close is not None:
        b = bench_close.reindex(df.index).ffill()
        bench_sma200 = b.rolling(200).mean()
        bench_above  = b > bench_sma200
        bench_roc10  = (b / b.shift(10) - 1) * 100
        # RISK-OFF = below 200-DMA AND 10d ROC <= -3%
        # (matches forward_validate_app.regime_at_cutoff — everything not
        # RISK-ON or NEUTRAL falls into RISK-OFF)
        is_neutral_or_on = bench_above | (bench_roc10 > -3.0)
        risk_off = ~is_neutral_or_on.fillna(False)
        # Bars where we don't yet have 200 sessions of bench history:
        # treat as UNKNOWN → do NOT block (be permissive at series head).
        risk_off = risk_off & bench_sma200.notna()
        sig = sig & (~risk_off)
        df["regime_risk_off"] = risk_off       # audit column
    else:
        df["regime_risk_off"] = False
    df["signal"] = sig.fillna(False)
    return df


# ======================================================================================
#  4. BACKTEST  (triple-barrier: take-profit / stop-loss / time-stop; overlapping trades)
# ======================================================================================
SNAP_COLS = ["pct_vs_sma200", "pct_vs_sma20", "rsi14", "roc10", "atr_pct",
             "vol_ratio", "macd_hist", "adx14", "bb_pctB", "dist_52wH", "obv_slope10"]


#  Default ladders / schedules for the layered exit stack.
#  All are module-level so the scanner UI (or a future config file) can
#  swap them in without touching engine code.
#
#  RATCHET_LADDER  (Layer A): (peak_gain_%, floor_%) pairs.
#  Once peak crosses `peak_gain_%`, stop floor rises to entry*(1+floor_%/100).
#
#  --- TIGHT CONSTANT-GIVEBACK LADDER (per user proposal Aug-2026) ---
#  Give-back held constant at ~5pp regardless of peak size:
#      peak +20% → floor +15%   (give-back = 5pp)
#      peak +40% → floor +35%   (give-back = 5pp)
#      peak +75% → floor +70%   (give-back = 5pp)
#      peak +100% → floor +95%  (give-back = 5pp)
#
#  Rationale: on the HAL/RBLBANK/SAIL/PAYTM/ADANIENT benchmark, this
#  ladder simulates ~2.3× the improvement of the older wider ladder
#  because it closes the peak-vs-exit gap that historically ate the
#  bulk of profitable moves. Trade-off: tighter floors will exit some
#  trades early on normal 3-5% intraday wobbles that would have
#  recovered — real backtests will show smaller improvement than the
#  optimistic simulation, but direction is strongly positive.
#
#  Alternative wider ladders retained for A/B testing:
#    WIDE_LADDER  = [(10,3),(20,10),(30,18),(50,33),(75,52),(100,72)]
#    SOFTER_V2    = [(15,5),(25,12),(40,25),(60,40),(85,60),(120,85)]
DEFAULT_RATCHET_LADDER = [
    ( 10.0,   5.0),
    ( 20.0,  15.0),
    ( 30.0,  25.0),
    ( 40.0,  35.0),
    ( 50.0,  45.0),
    ( 60.0,  55.0),
    ( 75.0,  70.0),
    (100.0,  95.0),
]

#  SHRINK_SCHEDULE  (Layer B): (min_gain_%, atr_mult) pairs.
#  Trail width narrows as gain accumulates, using CURRENT-day ATR.
DEFAULT_SHRINK_SCHEDULE = [
    (0.0,   2.0),
    (10.0,  1.5),
    (25.0,  1.0),
    (50.0,  0.75),
]

#  STAIRCASE  (Layer E): (gain_%, fraction_of_position_to_sell) tuples.
#  Every level books a partial slice off the table when reached (limit-fill at
#  the level price, or gap-open if the market opens above). The remaining
#  fraction rides with the rest of the exit stack (stop/trail/target/A/B/C/D).
#  Default: book 30% at +10%, another 30% at +20%, let 40% run.
DEFAULT_STAIRCASE = [
    (10.0, 0.30),
    (20.0, 0.30),
]


def run_backtest(df: pd.DataFrame, target_pct: float, max_hold: int,
                 stop_method: str, stop_value: float, cost_pct: float = 0.20,
                 apply_stcg: bool = True, rev_target_pct: float = None,
                 rev_stop_value: float = None, exit_mode: str = "Trailing",
                 min_hold: int = 1,           # M7 FIX (Aug-2026) — was ignored
                 trail_mult: float = 2.0, max_stop_pct: float = 5.0,
                 max_atr_pct: float = None, entry_mode: str = "Market open",
                 limit_pct: float = 0.0, fill_days: int = 1,
                 lock_pct: float = None, cut_day: int = None,
                 cut_threshold: float = 0.0, partial_frac: float = 0.0,
                 partial_atr: float = 1.5, stop_anchor: str = "ATR",
                 trail_anchor: str = "ATR",
                 # ---- CHANGE #4 (Aug-2026) — INITIAL STRUCTURE-STOP WINDOW ----
                 # Sessions of price history used to compute the initial swing-low
                 # when stop_anchor="Structure". Old hard-coded value was 10;
                 # exposed as a param so we can A/B test. Default kept at 10
                 # for backward compatibility — Change #4 candidate is 5.
                 initial_stop_lookback: int = 10,
                 # ---- LAYER A: RATCHETING PROFIT LOCK ------------------------
                 ratchet_lock: bool = False,
                 ratchet_ladder: list = None,
                 # ---- LAYER B: SHRINKING TRAIL MULTIPLIER --------------------
                 shrink_trail: bool = False,
                 shrink_schedule: list = None,
                 # ---- LAYER C: MOMENTUM-EXHAUSTION EXIT ----------------------
                 # NB: default arm threshold lowered from 15.0 -> 10.0 based on
                 # first A/B/C backtest: at 15% the layer only fired 0.8% of
                 # trades, meaning most peaks & rollovers happened below that
                 # cutoff and went uncaught.
                 momentum_exit: bool = False,
                 mom_exit_min_gain: float = 10.0,
                 # ---- LAYER D: TIME-DECAY TIGHTENING -------------------------
                 # After N sessions with no new peak, aggressively shrink the
                 # trail toward current close (raises stop). After M sessions,
                 # exit at close regardless — the pattern has clearly stalled.
                 time_decay: bool = False,
                 decay_after_days: int = 5,
                 decay_shrink_pct: float = 25.0,
                 decay_exit_days: int = 10,
                 # ---- LAYER E: STAIRCASE PARTIAL EXITS -----------------------
                 # Book fractional profits at fixed milestones. gross_return_%
                 # becomes the position-weighted blend of realised partials and
                 # the final exit fill on the remaining fraction. Cost model
                 # adds ~cost_pct/2 per partial (extra sell leg).
                 staircase: bool = False,
                 staircase_levels: list = None,
                 # ---- REGIME-AWARE EXIT ROUTER (per-trade routing) -----------
                 # When enabled, evaluates at each entry:
                 #   (1) ADX(14)                 >= route_min_adx        (trending)
                 #   (2) SMA200 rising over last route_sma_slope_lb sessions
                 #   (3) pct_vs_sma200           >= route_min_dist_pct   (above 200-DMA)
                 #   (4) 3-month realized vol    >  baseline vol
                 # ALL 4 pass  → use trailing exit (with A/B/C if enabled)
                 # ANY fails   → use fixed target at route_fixed_target_pct
                 # `sector_vol_median` (annualized %) is an OPTIONAL
                 # cross-sectional override provided by the scanner. If None,
                 # the engine falls back to comparing against the stock's own
                 # rolling median vol over `route_vol_baseline_lb` sessions.
                 regime_route: bool = False,
                 route_min_adx: float = 20.0,
                 route_sma_slope_lb: int = 20,
                 route_min_dist_pct: float = 15.0,
                 route_vol_lb: int = 63,
                 route_vol_baseline_lb: int = 252,
                 route_fixed_target_pct: float = 15.0,
                 sector_vol_median: float = None,
                 # ---- POST-STOP COOLDOWN (Aug-2026) ---------------------------
                 # Block new entries for N sessions after any STOP outcome on
                 # this instrument. Evidence: 9-stock OOS test 2024-2025 showed
                 # cooldown=7 alone takes aggregate return from -66% to +93%,
                 # win-rate 38% -> 47%. Kills same-stock cluster losses in
                 # choppy / down-trending regimes (TITAN Jan-Apr 2024, ADANIENT
                 # Oct-Nov 2024, TCS Mar-Apr 2024). Set to 0 to disable.
                 post_stop_cooldown_days: int = 7) -> pd.DataFrame:
    o = df["Open"].values; h = df["High"].values
    l = df["Low"].values; c = df["Close"].values
    v = df["Volume"].values if "Volume" in df.columns else np.zeros(len(df))
    atr = df["atr14"].values
    atrp = df["atr_pct"].values
    sig = df["signal"].values
    ttype = df["trade_type"].values if "trade_type" in df.columns else np.array([""] * len(df))
    idx = df.index
    n = len(df)
    snaps = {col: df[col].values for col in SNAP_COLS}

    # ---- M1 FIX: 20-day rolling average volume for limit-fill realism ----
    # A limit order isn't guaranteed to fill just because SOMEONE printed at
    # the price. On thin days (volume < 50% of 20-day avg) our order may
    # simply not have been part of the trade. Reject fills on such days.
    vol_avg20 = (df["Volume"].rolling(20).mean().shift(1).values
                 if "Volume" in df.columns else np.full(n, np.nan))
    MIN_FILL_VOL_RATIO = 0.5              # today's vol must be >= 50% of 20-day avg

    # ---- Extra arrays for Layer C (momentum-exhaustion detector) ----
    # All computed no-look-ahead in compute_indicators(); the loop just needs
    # fast numpy access. NaN fallback so partial data can't crash the run.
    def _col_or_nan(name):
        return df[name].values if name in df.columns else np.full(n, np.nan)
    rsi_arr    = _col_or_nan("rsi14")
    mhist_arr  = _col_or_nan("macd_hist")
    vratio_arr = _col_or_nan("vol_ratio")
    sma5_arr   = _col_or_nan("sma5")

    # ---- Extra arrays for REGIME-AWARE EXIT ROUTER ----
    # All computed no-look-ahead. Router evaluates only at entry_idx.
    adx_arr    = _col_or_nan("adx14")
    sma200_arr = _col_or_nan("sma200")
    pctv200_arr= _col_or_nan("pct_vs_sma200")

    # Rolling annualized realized vol from log-returns. Computed once for the
    # whole df so router evaluation is O(1) per trade.
    if regime_route:
        _closes = pd.Series(c)
        _logret = np.log(_closes / _closes.shift(1))
        _rvol   = _logret.rolling(int(route_vol_lb)).std() * np.sqrt(252) * 100.0
        rvol_arr = _rvol.values                                    # annualized %
        # Baseline: rolling median of realized-vol over `route_vol_baseline_lb`
        # sessions ENDING at the entry bar. This is the self-comparison fallback
        # used only when sector_vol_median is not provided.
        _rvol_base = pd.Series(rvol_arr).rolling(int(route_vol_baseline_lb)).median()
        rvol_base_arr = _rvol_base.values
    else:
        rvol_arr = np.full(n, np.nan)
        rvol_base_arr = np.full(n, np.nan)

    # ---- Resolve defaults for ladders / schedules (None -> module-level) ----
    if ratchet_lock and (ratchet_ladder is None or len(ratchet_ladder) == 0):
        ratchet_ladder = DEFAULT_RATCHET_LADDER
    if shrink_trail and (shrink_schedule is None or len(shrink_schedule) == 0):
        shrink_schedule = DEFAULT_SHRINK_SCHEDULE
    if staircase and (staircase_levels is None or len(staircase_levels) == 0):
        staircase_levels = DEFAULT_STAIRCASE
    # Sort staircase ascending by gain-level so we can walk it in order.
    if staircase and staircase_levels:
        staircase_levels = sorted(staircase_levels, key=lambda x: x[0])

    trades = []

    for i in range(n - 1):
        if not sig[i]:
            continue
        # VOLATILITY CEILING: skip entries when the stock is too wild to stop-loss safely
        if max_atr_pct is not None and np.isfinite(atrp[i]) and atrp[i] > max_atr_pct:
            continue
        # ---------------- ENTRY ----------------
        # Market open  : buy at next day's open, whatever the gap (old behaviour).
        # Limit        : place a resting buy at signal_close * (1 - limit_pct/100).
        #                Fill only if price trades down to it within `fill_days` sessions.
        #                A gap-DOWN opening below the limit fills at the open (better price).
        #                If it never trades there, the order expires -> NO TRADE (chase avoided).
        signal_close = c[i]
        if entry_mode == "Market open":
            entry_idx = i + 1
            if entry_idx >= n:
                break
            entry = o[entry_idx]
            limit_price = np.nan
        else:
            limit_price = signal_close * (1 - limit_pct / 100.0)
            entry_idx, entry = None, None
            for d in range(i + 1, min(i + 1 + max(int(fill_days), 1), n)):
                # M1 FIX (Aug-2026): volume-aware fill realism.
                # If this day traded on less than 50% of 20-day average volume,
                # assume our limit order was NOT part of the (thin) fills that
                # printed at the price. Skip the day. On illiquid smallcaps this
                # rules out "phantom fills" from a single lot trading at 3:15pm.
                vratio = (v[d] / vol_avg20[d]) if (np.isfinite(vol_avg20[d])
                                                    and vol_avg20[d] > 0) else 1.0
                if vratio < MIN_FILL_VOL_RATIO:
                    continue
                if o[d] <= limit_price:          # gapped below the limit -> fill at open
                    entry_idx, entry = d, o[d]
                    break
                if l[d] <= limit_price:          # traded down through the limit -> fill at limit
                    entry_idx, entry = d, limit_price
                    break
            if entry_idx is None:                # never filled: skip, don't chase
                continue
        if entry is None or not np.isfinite(entry) or entry <= 0:
            continue

        # ================================================================
        # REGIME-AWARE EXIT ROUTER — decide per-trade which exit style
        # ----------------------------------------------------------------
        # Evaluated at the SIGNAL BAR `i` (not the fill bar entry_idx) so
        # the routing decision uses the SAME data that generated the signal.
        # Rationale: with Limit entries, entry_idx can be up to `fill_days`
        # sessions AFTER `i`; the regime may have flipped between signal and
        # fill. Aligning the router to the signal keeps signal-generation and
        # exit-mode-selection based on ONE consistent snapshot of state.
        # (Bug fix C3 — was `e = entry_idx` before Aug-2026.)
        # No look-ahead: all indicators used are computed from bars <= i.
        # When regime_route=False (default), behaves exactly as before.
        # ================================================================
        this_exit_mode = exit_mode         # default: user's global choice
        this_target_pct = target_pct       # default: user's global target
        exit_route_tag = ""                # audit column ("" | "trailing" | "fixed")
        route_reason = ""                  # human-readable why (for the trade record)

        if regime_route:
            e = i                          # C3 FIX: evaluate at signal bar, not fill bar
            # (1) ADX(14) >= route_min_adx
            adx_val = adx_arr[e] if e < len(adx_arr) else np.nan
            cond_adx = np.isfinite(adx_val) and adx_val >= route_min_adx

            # (2) SMA200 rising over `route_sma_slope_lb` sessions
            lb = e - int(route_sma_slope_lb)
            s_now = sma200_arr[e] if e < len(sma200_arr) else np.nan
            s_old = sma200_arr[lb] if lb >= 0 else np.nan
            cond_slope = np.isfinite(s_now) and np.isfinite(s_old) and s_now > s_old

            # (3) Price >= route_min_dist_pct above SMA200
            dist_val = pctv200_arr[e] if e < len(pctv200_arr) else np.nan
            cond_dist = np.isfinite(dist_val) and dist_val >= route_min_dist_pct

            # (4) Realized vol vs baseline. Prefer cross-sectional sector median
            #     if the scanner provided one; else fall back to self-median.
            cur_vol = rvol_arr[e] if e < len(rvol_arr) else np.nan
            if sector_vol_median is not None and np.isfinite(sector_vol_median):
                baseline_vol = float(sector_vol_median)
                vol_source = "sector"
            else:
                baseline_vol = rvol_base_arr[e] if e < len(rvol_base_arr) else np.nan
                vol_source = "self"
            cond_vol = (np.isfinite(cur_vol) and np.isfinite(baseline_vol)
                        and cur_vol > baseline_vol)

            is_trending = cond_adx and cond_slope and cond_dist and cond_vol
            if is_trending:
                this_exit_mode = "Trailing"
                this_target_pct = target_pct
                exit_route_tag = "trailing"
                route_reason = "trending"
            else:
                this_exit_mode = "Fixed target"
                this_target_pct = route_fixed_target_pct
                exit_route_tag = "fixed"
                # Compact reason: 4-char failure map (a=adx,s=slope,d=dist,v=vol)
                fails = []
                if not cond_adx:   fails.append(f"adx={adx_val:.0f}")
                if not cond_slope: fails.append("sma200_flat")
                if not cond_dist:  fails.append(f"dist={dist_val:.1f}")
                if not cond_vol:   fails.append(f"vol_{vol_source}")
                route_reason = "chop:" + "|".join(fails) if fails else "chop"

        # regime-aware exits: downtrend/reversal trades use tighter target & stop if provided
        is_rev = (ttype[i] == "DOWNTREND")
        tp = rev_target_pct if (is_rev and rev_target_pct is not None) else this_target_pct
        sv = rev_stop_value if (is_rev and rev_stop_value is not None) else stop_value

        a = atr[i]
        target = entry * (1 + tp / 100)
        if stop_anchor == "Structure":
            # STRUCTURE stop: just below the lowest low of the prior 10 sessions (recent
            # support / channel floor), with a small ATR buffer so ordinary wicks don't tag it.
            if not np.isfinite(a) or a <= 0:
                continue
            swing = np.nanmin(l[max(0, i - (initial_stop_lookback - 1)):i + 1])
            init_stop = swing - 0.25 * a
        elif stop_method == "ATR":
            if not np.isfinite(a) or a <= 0:
                continue
            init_stop = entry - sv * a
        else:                                  # fixed %
            init_stop = entry * (1 - sv / 100)

        # MAX-STOP CAP: never risk more than max_stop_pct from entry, whatever ATR says
        if max_stop_pct is not None:
            floor_stop = entry * (1 - max_stop_pct / 100)
            init_stop = max(init_stop, floor_stop)   # pull a too-wide stop closer
        atr_dist = entry - init_stop           # trailing distance = actual risk distance

        last = min(entry_idx + max_hold, n - 1)
        stop = init_stop
        peak = entry
        hit_target = False          # did price actually reach the 10% objective during the hold?
        outcome, exit_price, exit_idx = None, None, None

        # --- DYNAMIC partial-profit level: volatility-scaled, NOT a fixed % ---
        # A calm stock books its partial at a smaller move than a wild one, because for a calm
        # stock a 1.5-ATR move is already a meaningful advance.
        # NB: legacy single-partial is DISABLED when Layer E staircase is active — Layer E
        # supersedes it with a multi-level version.
        _use_legacy_partial = (partial_frac > 0 and not staircase)
        partial_level = entry + partial_atr * a if (_use_legacy_partial and np.isfinite(a) and a > 0) else np.inf
        partial_taken, partial_ret = False, 0.0

        # --- LAYER D state: session-counter since the last new peak.
        # Reset to 0 whenever h[d] makes a new peak; increment otherwise.
        # Triggers trail-tightening at decay_after_days, forced close at decay_exit_days.
        days_since_high = 0

        # --- LAYER E state: per-level "taken" flags + running blended-return accumulator.
        # `realized_partials_gain` = weighted sum of (fraction × gain%) already booked.
        # `remaining_frac` = 1.0 minus every partial fraction taken so far. When this hits
        # 0 the trade is fully out; when the final exit fires it settles the remainder.
        if staircase and staircase_levels:
            n_levels = len(staircase_levels)
            staircase_taken = [False] * n_levels
            realized_partials_gain = 0.0
            remaining_frac = 1.0
            partials_taken_count = 0
        else:
            staircase_taken, realized_partials_gain = None, 0.0
            remaining_frac = 1.0
            partials_taken_count = 0

        for d in range(entry_idx, last + 1):
            day_k = d - entry_idx

            # M7 FIX (Aug-2026): honour min_hold — skip ALL exit checks (stop,
            # target, staircase, momentum-exit, decay, cut) until we have held
            # the position for the minimum number of sessions the user asked
            # for. Threshold is `day_k < min_hold` so that min_hold=3 yields
            # days_held >= 3 (see days_held formula further below).
            # Peak / days_since_high tracking below still updates so trailing
            # state is correct when exits become active. min_hold=1 (default)
            # skips only day_k=0 (the entry day) — you always hold overnight
            # at minimum, which is the natural reading of "min hold 1 day".
            if day_k < min_hold:
                if h[d] > peak:
                    peak = h[d]; days_since_high = 0
                else:
                    days_since_high += 1
                continue

            # --- STOP first, with GAP-AWARE fill ---
            if l[d] <= stop:
                fill = o[d] if o[d] < stop else stop      # gapped past the stop -> fill at open
                outcome, exit_price, exit_idx = "STOP", fill, d
                break

            # ================================================================
            # LAYER E — STAIRCASE PARTIAL EXITS
            # ----------------------------------------------------------------
            # Book fractional profits when price crosses each pre-set gain
            # milestone. Uses GAP-AWARE fill: if the market opens above the
            # level, fill at open (better than the level); else fill at level.
            # After partial, the remaining fraction stays exposed to the rest
            # of the exit stack (stop / trail / A / B / C / D).
            # ================================================================
            if staircase and staircase_levels and remaining_frac > 1e-9:
                for si, (lvl_gain, lvl_frac) in enumerate(staircase_levels):
                    if staircase_taken[si]:
                        continue
                    lvl_price = entry * (1 + lvl_gain / 100)
                    if h[d] >= lvl_price:
                        # GAP-AWARE: if today opened above the level, fill at open
                        fill_price = o[d] if (np.isfinite(o[d]) and o[d] > lvl_price) else lvl_price
                        take_frac = min(lvl_frac, remaining_frac)
                        gain_pct = (fill_price / entry - 1) * 100
                        realized_partials_gain += take_frac * gain_pct
                        remaining_frac -= take_frac
                        staircase_taken[si] = True
                        partials_taken_count += 1
                        if remaining_frac <= 1e-9:
                            # Fully out via staircase — book the last fill as "TARGET"
                            outcome, exit_price, exit_idx = "TARGET", fill_price, d
                            break
                if outcome is not None:
                    break

            if this_exit_mode == "Fixed target":
                if h[d] >= target:                        # classic barrier: caps upside
                    fill = o[d] if o[d] > target else target
                    outcome, exit_price, exit_idx = "TARGET", fill, d
                    break
            else:
                # --- TRAILING: the target is a MINIMUM objective, never a ceiling ---
                if (not partial_taken) and h[d] >= partial_level:
                    partial_taken = True
                    partial_ret = (partial_level / entry - 1) * 100   # book `partial_frac` here

                if h[d] >= target and not hit_target:
                    hit_target = True
                    # PROFIT LOCK (legacy): once objective is reached, never give it all back.
                    if lock_pct is not None:
                        stop = max(stop, entry * (1 + lock_pct / 100))

                # ============================================================
                # LAYER A — RATCHETING PROFIT LOCK  (refined ladder)
                # ------------------------------------------------------------
                # Softer first-rung than v1 (peak +15% → floor +5%, was +10% →
                # +3%) so ordinary 10-12% pushes without follow-through don't
                # get chopped at breakeven. Rungs still keep ~65-70% of the
                # highest one crossed. Checked BEFORE updating `peak` so
                # today's new high can already raise the floor.
                # ============================================================
                if ratchet_lock and ratchet_ladder:
                    best_seen_price = max(peak, h[d]) if np.isfinite(h[d]) else peak
                    best_seen_gain = (best_seen_price / entry - 1) * 100
                    for peak_thr, floor_pct in ratchet_ladder:
                        if best_seen_gain >= peak_thr:
                            floor_price = entry * (1 + floor_pct / 100)
                            if floor_price > stop:
                                stop = floor_price

                if h[d] > peak:
                    peak = h[d]
                    # LAYER D: reset the stall-counter when a new peak is made.
                    days_since_high = 0
                    # ========================================================
                    # LAYER B — SHRINKING TRAIL MULTIPLIER
                    # --------------------------------------------------------
                    # Multiplier chosen by CURRENT peak gain, trail off
                    # CURRENT-day ATR. With trail_anchor="Structure", combine
                    # by taking the tighter (higher) of ATR-trail and
                    # structure-trail so profit protection wins over channel
                    # breathing room once deeply green.
                    # ========================================================
                    if shrink_trail and shrink_schedule:
                        cur_gain = (peak / entry - 1) * 100
                        mult = trail_mult
                        for gain_thr, m in shrink_schedule:
                            if cur_gain >= gain_thr:
                                mult = m
                        cur_atr = atr[d] if np.isfinite(atr[d]) and atr[d] > 0 else a
                        atr_trail = peak - mult * cur_atr
                        if trail_anchor == "Structure":
                            sw = np.nanmin(l[max(0, d - 4):d + 1])
                            buf = 0.25 * cur_atr
                            struct_trail = sw - buf
                            stop = max(stop, atr_trail, struct_trail)
                        else:
                            stop = max(stop, atr_trail)
                    else:
                        # C4 FIX (Aug-2026): use TODAY's ATR for trailing distance,
                        # not the frozen entry-day atr_dist. Rationale — a 30-day
                        # trade in a stock that quieted down keeps a stale wide
                        # stop; a stock that got choppier keeps a stale tight
                        # stop. Refreshing daily makes trailing adapt to current
                        # volatility. Falls back to entry-day `a` if today's ATR
                        # is NaN (early bars only).
                        cur_atr = atr[d] if np.isfinite(atr[d]) and atr[d] > 0 else a
                        if trail_anchor == "Structure":
                            sw = np.nanmin(l[max(0, d - 4):d + 1])
                            buf = 0.25 * cur_atr
                            stop = max(stop, sw - buf)
                        else:
                            stop = max(stop, peak - trail_mult * cur_atr)
                else:
                    # No new peak today — bump the stall counter for Layer D.
                    days_since_high += 1

            # ================================================================
            # LAYER D — TIME-DECAY TIGHTENING
            # ----------------------------------------------------------------
            # If the trade hasn't made a new peak in `decay_after_days`, close
            # part of the stop-to-price gap each day. After `decay_exit_days`
            # of stall, exit at today's close — the pattern has stalled and is
            # tying up capital. Only meaningful in trailing mode.
            # ================================================================
            if (time_decay and this_exit_mode != "Fixed target"
                    and days_since_high >= decay_after_days
                    and np.isfinite(c[d])):
                if days_since_high >= decay_exit_days:
                    outcome, exit_price, exit_idx = "DECAY", c[d], d
                    break
                gap = c[d] - stop
                if gap > 0:
                    new_stop = stop + gap * (decay_shrink_pct / 100.0)
                    if new_stop > stop:
                        stop = new_stop

            # ================================================================
            # LAYER C — MOMENTUM-EXHAUSTION EXIT  (refined arm-threshold)
            # ----------------------------------------------------------------
            # Arm threshold DROPPED from 15% to 10% (default) — the v1 default
            # only fired 0.8% of trades because most rollovers happen at more
            # modest gains. Five detectors, any one fires → exit at NEXT-day
            # open. No look-ahead (signal today, execute tomorrow).
            #   C1 RSI rollover        : peaked > 75 last 5 bars, closes < 70
            #   C2 MACD flip           : histogram negative two days running
            #   C3 heavy-vol pullback  : two down closes on vol ≥ 1.5x avg
            #   C4 bearish engulfing   : today's red body covers prior green
            #   C5 5-DMA break         : close below 5-DMA when up ≥ 20%
            # ================================================================
            if (momentum_exit and this_exit_mode != "Fixed target"
                    and np.isfinite(c[d]) and d + 1 <= last):
                close_gain = (c[d] / entry - 1) * 100
                if close_gain >= mom_exit_min_gain:
                    exhausted, mom_reason = False, ""

                    # C1: RSI peaked > 75 in last 5 bars and now closes < 70
                    rsi_now = rsi_arr[d]
                    if not exhausted and np.isfinite(rsi_now):
                        lb0 = max(0, d - 4)
                        rsi_slice = rsi_arr[lb0:d + 1]
                        rsi_max = np.nanmax(rsi_slice) if rsi_slice.size else np.nan
                        if np.isfinite(rsi_max) and rsi_max >= 75 and rsi_now < 70:
                            exhausted, mom_reason = True, "RSI rollover"

                    # C2: MACD histogram negative two consecutive days
                    if not exhausted and d >= 1:
                        mh_t, mh_y = mhist_arr[d], mhist_arr[d - 1]
                        if np.isfinite(mh_t) and np.isfinite(mh_y) and mh_t < 0 and mh_y < 0:
                            exhausted, mom_reason = True, "MACD flip"

                    # C3: Two down closes on heavy volume (>= 1.5x avg)
                    if not exhausted and d >= 2:
                        if c[d] < c[d - 1] and c[d - 1] < c[d - 2]:
                            vr = vratio_arr[d]
                            if np.isfinite(vr) and vr >= 1.5:
                                exhausted, mom_reason = True, "heavy-vol pullback"

                    # C4: Bearish engulfing at/near the peak
                    if not exhausted and d >= 1:
                        body_t = c[d] - o[d]
                        body_y = c[d - 1] - o[d - 1]
                        if (body_t < 0 and body_y > 0
                                and o[d] >= c[d - 1] and c[d] <= o[d - 1]):
                            exhausted, mom_reason = True, "bearish engulfing"

                    # C5: Close below 5-DMA when trade is strongly up (>= 20%)
                    if not exhausted and close_gain >= 20.0:
                        s5 = sma5_arr[d]
                        if np.isfinite(s5) and c[d] < s5:
                            exhausted, mom_reason = True, "5-DMA break"

                    if exhausted:
                        nxt = d + 1
                        exit_idx = nxt
                        exit_price = o[nxt] if np.isfinite(o[nxt]) else c[d]
                        outcome = "MOMEXIT"
                        break

            # --- CONVICTION EXIT: if it isn't working early, free the capital ---
            # Only applies before the objective is reached and before a partial is booked.
            if (cut_day is not None and day_k == cut_day and not hit_target and not partial_taken
                    and (c[d] / entry - 1) * 100 < cut_threshold):
                outcome, exit_price, exit_idx = "CUT", c[d], d
                break

        if outcome is None:
            outcome, exit_price, exit_idx = "TIME", c[last], last

        # Relabel so TARGET always means "reached the objective", in both modes.
        # MOMEXIT / DECAY / CUT are explicit outcomes and preserved as-is.
        if this_exit_mode != "Fixed target" and outcome in ("STOP", "TIME"):
            if hit_target and exit_price >= entry:
                outcome = "TARGET"
            elif exit_price > entry:
                outcome = "TRAIL"

        # ---- P&L: blend staircase partials, legacy partial, and final exit ----
        final_ret = (exit_price / entry - 1) * 100
        if staircase and partials_taken_count > 0:
            # Layer E blend: realized_partials_gain is already the fraction-weighted
            # sum of booked gains; remaining_frac carries the final exit return.
            gross = realized_partials_gain + remaining_frac * final_ret
            # Extra sell-leg cost per partial (≈ half a round-trip each).
            total_cost = cost_pct + partials_taken_count * (cost_pct / 2.0)
        elif partial_taken:
            gross = partial_frac * partial_ret + (1 - partial_frac) * final_ret
            total_cost = cost_pct
        else:
            gross = final_ret
            total_cost = cost_pct
        net = gross - total_cost
        # H4 FIX (Aug-2026): STCG is NO LONGER applied per-trade here. The
        # per-trade `net *= 0.80` was wrong because India STCG is netted
        # against LOSSES within the same fiscal year — applying it per
        # winner overstated tax drag on winning strategies. Tax is now
        # applied at the fiscal-year net level in `_apply_stcg_yearly`,
        # called once after the loop below when apply_stcg=True.

        row = {
            "signal_date": idx[i].date(),
            "signal_close": round(float(signal_close), 2),
            "limit_price": round(float(limit_price), 2) if np.isfinite(limit_price) else np.nan,
            "entry_date":  idx[entry_idx].date(),
            "entry_price": round(entry, 2),
            "fill_edge_%": round((signal_close / entry - 1) * 100, 2),  # +ve = bought below signal close
            "target_price": round(target, 2),
            "init_stop_price": round(init_stop, 2),   # H5: initial risk anchor for R-multiple
            "stop_price":  round(stop, 2),
            "exit_date":   idx[exit_idx].date(),
            "exit_price":  round(exit_price, 2),
            "days_held":   max(exit_idx - entry_idx, 1),
            "outcome":     outcome,
            "hit_target":  bool(hit_target),
            "partial_taken": bool(partial_taken),
            "partial_at_%": round(partial_ret, 2) if partial_taken else np.nan,
            "partials_taken_n": int(partials_taken_count),        # NEW: staircase count
            "remaining_frac":  round(float(remaining_frac), 3),    # NEW: what fraction saw the final exit
            "days_since_high": int(days_since_high),               # NEW: stall length at exit
            "peak_gain_%": round((peak / entry - 1) * 100, 2),
            "trade_type":  ttype[i] if ttype[i] else "UPTREND",
            "gross_return_%": round(gross, 2),
            "net_return_%":   round(net, 2),
            "exit_route":  exit_route_tag,       # "" | "trailing" | "fixed"  (router audit)
            "route_reason": route_reason,        # human-readable why (fails compactly listed)
        }
        for col in SNAP_COLS:
            row[col] = round(float(snaps[col][i]), 2) if np.isfinite(snaps[col][i]) else np.nan
        trades.append(row)

    trades_df = pd.DataFrame(trades)

    # POST-STOP COOLDOWN — remove re-entries within N sessions of a stop
    # on the same instrument. Applied BEFORE STCG so the tax base excludes
    # dropped trades. See _apply_post_stop_cooldown docstring for evidence.
    if post_stop_cooldown_days > 0 and not trades_df.empty:
        trades_df = _apply_post_stop_cooldown(trades_df, days=post_stop_cooldown_days)

    # H4 FIX (Aug-2026): apply STCG at fiscal-year net level (see helper docs).
    if apply_stcg and not trades_df.empty:
        trades_df = _apply_stcg_yearly(trades_df, stcg_rate=0.20)

    return trades_df


# =========================================================================
# POST-STOP COOLDOWN HELPER  (Aug-2026)
# -------------------------------------------------------------------------
# After a STOP outcome on this instrument, block any new entry whose
# entry_date falls within `days` calendar days of the stopped trade's
# EXIT date. Prevents same-stock cluster losses: signals typically re-fire
# 2-5 days after a stop when a bounce reclaims the trigger, but in choppy
# or down-trending regimes those re-entries stop out too, chaining 4-8
# losing trades in 3-4 weeks (TITAN Jan-Apr 2024, ADANIENT Oct-Nov 2024).
#
# Evidence (9-stock OOS 2024-01 .. 2025-06, scanner defaults):
#   Baseline (no cooldown):   161 trades, 38% win, -0.4% avg, total -66%
#   Cooldown = 7 sessions:    117 trades, 47% win, +0.8% avg, total +93%
# Improvement in every stock except TRENT (a runaway winner that
# occasionally re-fires cleanly during momentum runs).
#
# Cooldown does NOT trigger on TARGET / TRAIL / TIME / TRAIL exits — only
# STOP. A profitable exit is not evidence the pattern has failed.
# =========================================================================
def _apply_post_stop_cooldown(trades: pd.DataFrame, days: int = 7) -> pd.DataFrame:
    if trades.empty or days <= 0:
        return trades
    t = trades.sort_values("entry_date").reset_index(drop=True)
    last_stop_exit = None
    keep_idx = []
    for i, row in t.iterrows():
        entry_ts = pd.Timestamp(row["entry_date"])
        if last_stop_exit is not None and (entry_ts - last_stop_exit).days < days:
            continue                              # inside cooldown window — skip
        keep_idx.append(i)
        if row["outcome"] == "STOP":
            last_stop_exit = pd.Timestamp(row["exit_date"])
    return t.loc[keep_idx].reset_index(drop=True)


# =========================================================================
# H4 FIX HELPER — India STCG at fiscal-year net gain, distributed pro-rata
# -------------------------------------------------------------------------
# India STCG on listed-equity short-term gains is 20% (raised from 15% in
# the July-2024 budget). Critically, it is charged on ANNUAL NET short-term
# gains — losses within the same FY offset winners before tax is computed.
# The prior implementation applied 20% to EVERY winning trade in isolation,
# which materially overstated tax drag on strategies that mix winners and
# losers throughout the year (basically all trend-following systems).
#
# This helper walks the trade log by Indian Fiscal Year (Apr 1 - Mar 31),
# computes each FY's net %-return sum, and only when that net is positive
# subtracts 20% of the FY net — distributed pro-rata across the WINNERS of
# that FY so per-trade net_return_% stays interpretable in reporting.
# Losing trades are untouched (a loss has no STCG anyway).
# =========================================================================
def _apply_stcg_yearly(trades: pd.DataFrame, stcg_rate: float = 0.20) -> pd.DataFrame:
    if trades.empty or "exit_date" not in trades.columns:
        return trades
    t = trades.copy()
    ex = pd.to_datetime(t["exit_date"])
    # Indian FY: Apr Y - Mar Y+1  →  fy = calendar year, minus 1 if before April
    fy = ex.dt.year.where(ex.dt.month >= 4, ex.dt.year - 1)
    t["_fy"] = fy.values
    for fy_val, sub_idx in t.groupby("_fy").groups.items():
        fy_net = float(t.loc[sub_idx, "net_return_%"].sum())
        if fy_net <= 0:
            continue                              # no STCG payable this FY
        tax_pp = fy_net * stcg_rate               # total tax in %-points for this FY
        winners_mask = (t.index.isin(sub_idx)) & (t["net_return_%"] > 0)
        winners_sum = float(t.loc[winners_mask, "net_return_%"].sum())
        if winners_sum <= 0:
            continue
        pro_rata = t.loc[winners_mask, "net_return_%"] / winners_sum
        t.loc[winners_mask, "net_return_%"] = (
            t.loc[winners_mask, "net_return_%"] - pro_rata * tax_pp
        ).round(4)
    return t.drop(columns=["_fy"])


def build_sequential_trades(trades: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct a ONE-POSITION-AT-A-TIME trade sequence from the full (overlapping)
    trade list that `run_backtest` produces.

    `run_backtest` fires a fresh signal on every qualifying bar, so it is normal for
    several trades to have overlapping entry/exit windows for the same stock — that's
    correct for measuring "how often did this *pattern* work", but it makes compounding
    undefined (two trades can't both claim 100% of the same capital at once).

    Any metric that needs a single capital timeline — CAGR, drawdown, recovery factor,
    consecutive-loss streaks — is computed on THIS sequential list instead: sort by
    entry date, take a trade, then ignore any further signal until that trade has
    exited (capital freed). This mirrors how you'd actually trade one stock with one
    pool of capital.
    """
    if trades.empty:
        return trades
    t = trades.sort_values("entry_date").reset_index(drop=True)
    keep, last_exit = [], None
    for i, row in t.iterrows():
        ed = pd.Timestamp(row["entry_date"])
        if last_exit is not None and ed < last_exit:
            continue                       # capital still deployed in the prior trade
        keep.append(i)
        last_exit = pd.Timestamp(row["exit_date"])
    return t.loc[keep].reset_index(drop=True)


def compute_account_metrics(trades: pd.DataFrame) -> dict:
    """Account-level metrics that require a single compounding capital base:
    CAGR, Max Drawdown, Recovery Factor, Max Consecutive Losses.
    Computed on the sequential (non-overlapping) trade list — see
    `build_sequential_trades`. Returns {} if there's nothing to sequence."""
    seq = build_sequential_trades(trades)
    if seq.empty:
        return {}
    seq = seq.copy()
    seq["equity_mult"] = (1 + seq["net_return_%"] / 100).cumprod()
    eq = seq["equity_mult"].values
    running_peak = np.maximum.accumulate(eq)
    dd_series = (eq - running_peak) / running_peak * 100
    max_dd = float(dd_series.min()) if len(dd_series) else 0.0     # negative, e.g. -18.4

    start = pd.Timestamp(seq["entry_date"].iloc[0])
    end = pd.Timestamp(seq["exit_date"].iloc[-1])
    years_raw = (end - start).days / 365.25
    # H6 FIX (Aug-2026): CAGR needs a MEANINGFUL time base. Previously the
    # denominator was floored at 1/365.25 (one day), which produced absurd
    # 4- and 5-digit CAGR numbers on stocks with only a few weeks of
    # sequential history (e.g. newly-listed IPOs). If we have less than
    # 6 months of sequential trading history, CAGR is not statistically
    # meaningful — report NaN so the UI shows "n/a" instead of a nonsense
    # figure that misleads the ranking.
    if years_raw < 0.5:
        cagr = np.nan
        years = years_raw
    else:
        years = years_raw
        final_mult_local = float(eq[-1])
        # (cagr computed below using `years` and final_mult)
    final_mult = float(eq[-1])
    net_profit_seq_pct = (final_mult - 1) * 100
    if years_raw >= 0.5:
        cagr = (final_mult ** (1 / years) - 1) * 100 if final_mult > 0 else -100.0
    # else cagr remains np.nan (set above)
    # H2 NOTE (Aug-2026): This is compounded-total-return-% divided by
    # peak-relative-max-DD-%. It is a common industry variant (sometimes called
    # MAR ratio) and NOT a bug — despite the two components measuring on
    # different bases (initial-equity vs running-peak), both are dimensionless
    # ratios and the resulting number matches how most trading platforms
    # (TradingView, MT4/5) report Recovery Factor. Strict Van-Tharp version
    # would use max-DD as % of INITIAL equity — that variant is available if
    # requested, but not the default here.
    recovery_factor = (net_profit_seq_pct / abs(max_dd)) if max_dd < -1e-9 else np.nan

    # --- max consecutive losses, in chronological (sequential) trading order ---
    # H1 FIX (Aug-2026): only STRICT losses (< 0) count as losses. Breakeven
    # trades (net_return_% == 0) previously counted as losses, inflating the
    # "max consec losses" figure users key off for position-sizing.
    is_loss = (seq["net_return_%"] < 0).astype(int).values
    max_run = cur = 0
    for v in is_loss:
        cur = cur + 1 if v else 0
        max_run = max(max_run, cur)

    return {
        "seq_trades": len(seq),
        "cagr_%": (round(cagr, 2) if np.isfinite(cagr) else np.nan),
        "max_drawdown_%": round(max_dd, 2),
        "net_profit_seq_%": round(net_profit_seq_pct, 2),
        "recovery_factor": round(recovery_factor, 2) if np.isfinite(recovery_factor) else np.nan,
        "max_consecutive_losses": int(max_run),
        "years_sequenced": round(years, 2),
        "equity_curve": seq[["entry_date", "exit_date", "net_return_%", "equity_mult"]],
    }


def summarize(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {}
    n = len(trades)
    tgt = int((trades["outcome"] == "TARGET").sum())
    stp = int((trades["outcome"] == "STOP").sum())
    tim = int((trades["outcome"] == "TIME").sum())
    trl = int((trades["outcome"] == "TRAIL").sum())      # trailing profit, below 10% target
    cut = int((trades["outcome"] == "CUT").sum())        # conviction exit (dead trade, freed capital)
    mom = int((trades["outcome"] == "MOMEXIT").sum())    # Layer-C momentum-exhaustion exit
    dcy = int((trades["outcome"] == "DECAY").sum())      # Layer-D time-decay forced exit
    wins = int((trades["net_return_%"] > 0).sum())
    losers = n - wins
    time_df = trades[trades["outcome"] == "TIME"]
    time_win = int((time_df["net_return_%"] > 0).sum())
    time_loss = int((time_df["net_return_%"] <= 0).sum())
    win_trades = trades[trades["outcome"] == "TARGET"]
    med_days_to_target = float(win_trades["days_held"].median()) if len(win_trades) else np.nan

    avg_win = round(trades.loc[trades["net_return_%"] > 0, "net_return_%"].mean(), 2) if wins else 0.0
    avg_loss = round(trades.loc[trades["net_return_%"] <= 0, "net_return_%"].mean(), 2) if losers else 0.0

    # --- Profit factor: gross profit / gross loss, in % points (all historical signal
    # instances, since this is about the edge quality of the pattern, not capital timing) ---
    # H3 NOTE (Aug-2026): sum-of-%-returns is the STANDARD profit-factor formula
    # when position sizes per trade are equal (the normalisation cancels). This
    # matches how MetaTrader/AmiBroker/TradingView compute PF. Not a bug —
    # verified against industry references. See also seq_profit_factor below,
    # which applies the same math on the sequential (non-overlapping) list.
    gross_profit = float(trades.loc[trades["net_return_%"] > 0, "net_return_%"].sum())
    gross_loss = float(abs(trades.loc[trades["net_return_%"] <= 0, "net_return_%"].sum()))
    if gross_loss > 1e-9:
        profit_factor = round(gross_profit / gross_loss, 2)
    else:
        profit_factor = np.inf if gross_profit > 0 else np.nan

    reward_risk = round(abs(avg_win / avg_loss), 2) if avg_loss not in (0, 0.0) else np.nan

    # ---- H5 FIX (Aug-2026): true R-multiple expectancy ----
    # Per-trade initial risk = (entry - INIT_STOP) / entry, in %. NOTE: uses
    # `init_stop_price` (recorded once at entry), NOT `stop_price` (the final
    # trailing stop — a subtle bug in the first cut of H5 that made all
    # R-multiples look implausibly small on trailing exits).
    # Expectancy in R-units = mean(net_return_% / initial_risk_%) — return
    # per unit of capital risked. A stock with 0.4R expectancy makes 40% of
    # your risked amount on average per trade; strategies with the same
    # %-expectancy can have very different R-multiples (a 1% edge risking 2%
    # is 0.5R; the same 1% edge risking 5% is 0.2R). R-multiples let you
    # compare edges across stocks with different stop widths.
    _stop_col = "init_stop_price" if "init_stop_price" in trades.columns else "stop_price"
    r_pct = ((trades["entry_price"] - trades[_stop_col]) / trades["entry_price"] * 100)
    r_pct = r_pct.where(r_pct > 0, np.nan)
    r_mult = trades["net_return_%"] / r_pct
    exp_R = round(float(r_mult.mean()), 3) if r_mult.notna().any() else np.nan
    median_R = round(float(r_mult.median()), 3) if r_mult.notna().any() else np.nan

    out = {
        "trades": n,
        "target_hits": tgt,
        "stop_hits": stp,
        "time_exits": tim,
        "trail_exits": trl,
        "time_win": time_win,
        "time_loss": time_loss,
        # NOTE: hit_rate_% == target_pct_of_all by definition (both = 100*tgt/n).
        # `hit_rate_%` kept as a semantically-clear alias for older callers.
        "hit_rate_%": round(100 * tgt / n, 1),
        "target_pct_of_all": round(100 * tgt / n, 1),
        "trail_pct_of_all": round(100 * trl / n, 1),
        "stop_pct_of_all": round(100 * stp / n, 1),
        "time_pct_of_all": round(100 * tim / n, 1),
        "profitable_%": round(100 * wins / n, 1),
        # H5 FIX: `avg_net_%` REMOVED — it was a duplicate of `expectancy_%`.
        # Downstream code (scanner/screener UIs) references `expectancy_%`.
        "expectancy_%": round(trades["net_return_%"].mean(), 2),
        "expectancy_R": exp_R,                    # true R-multiple expectancy
        "median_R":     median_R,                 # more robust to outliers
        "avg_win_%": avg_win,
        "avg_loss_%": avg_loss,
        "avg_days": round(trades["days_held"].mean(), 1),
        "med_days_to_target": med_days_to_target,
        "n_winners": len(win_trades),
        "cut_exits": cut,
        "cut_pct_of_all": round(100 * cut / n, 1),
        "mom_exits": mom,
        "mom_pct_of_all": round(100 * mom / n, 1),
        "decay_exits": dcy,
        "decay_pct_of_all": round(100 * dcy / n, 1),
        "staircase_partials": int(trades["partials_taken_n"].sum())
                              if "partials_taken_n" in trades else 0,
        # --- EXIT ROUTER activity (populated only when regime_route=True) ---
        # `n_routed_*` = trades sent to that exit style. `avg_*_route` = per-trade
        # avg net return within each route. Split lets us see WHERE the edge is:
        # if trailing-route trades average +5% and fixed-route trades average +2%,
        # the router is directing high-alpha stocks correctly.
        "n_routed_trailing":
            int((trades["exit_route"] == "trailing").sum()) if "exit_route" in trades else 0,
        "n_routed_fixed":
            int((trades["exit_route"] == "fixed").sum()) if "exit_route" in trades else 0,
        "pct_routed_trailing": (
            round(100 * (trades["exit_route"] == "trailing").sum() / n, 1)
            if "exit_route" in trades else 0.0),
        "avg_net_trailing_route": (
            round(trades.loc[trades["exit_route"] == "trailing", "net_return_%"].mean(), 2)
            if "exit_route" in trades and (trades["exit_route"] == "trailing").any() else np.nan),
        "avg_net_fixed_route": (
            round(trades.loc[trades["exit_route"] == "fixed", "net_return_%"].mean(), 2)
            if "exit_route" in trades and (trades["exit_route"] == "fixed").any() else np.nan),
        # --- EXPECTANCY PER DAY: return earned per day of capital tied up (the metric that
        # matters when you rotate capital between stocks rather than buy-and-hold). ---
        "exp_per_day_%": round(trades["net_return_%"].mean() / max(trades["days_held"].mean(), 1e-9), 3),
        "partials_taken": int(trades.get("partial_taken", pd.Series(dtype=bool)).sum())
                          if "partial_taken" in trades else 0,
        "avg_peak_gain_%": round(trades["peak_gain_%"].mean(), 2) if "peak_gain_%" in trades else np.nan,
        # ---- new: profitability / risk metrics ----
        "gross_profit_%": round(gross_profit, 2),
        "gross_loss_%": round(gross_loss, 2),
        "profit_factor": profit_factor,
        "reward_risk_ratio": reward_risk,
        "total_return_sum_%": round(float(trades["net_return_%"].sum()), 2),
    }

    # ---- account-level metrics (CAGR, Max DD, Recovery Factor, consecutive losses) ----
    out.update(compute_account_metrics(trades))

    # =========================================================================
    # C5 FIX (Aug-2026) — SEQUENTIAL (portfolio-realistic) STATS
    # -------------------------------------------------------------------------
    # The `trades` DataFrame contains every historical signal, including
    # signals that fired while an earlier trade was still open. These are
    # highly correlated near-duplicates. Win-rate / expectancy / profit-factor
    # / R:R computed on this pool overstate statistical significance and bias
    # the confidence score toward stocks whose pattern fires OFTEN (not
    # necessarily INDEPENDENTLY).
    #
    # Fix: compute the same core stats on the SEQUENTIAL (one-position-at-a-time)
    # trade list — this is what you would actually achieve in a real book, and
    # what the ranking + self-check should use.
    #
    # Prefix: `seq_` on every field. Original fields kept unchanged so nothing
    # else in the codebase breaks; new columns are additive.
    # =========================================================================
    seq_df = build_sequential_trades(trades)
    n_seq = len(seq_df)
    if n_seq > 0:
        wins_seq = int((seq_df["net_return_%"] > 0).sum())
        gp_seq = float(seq_df.loc[seq_df["net_return_%"] > 0, "net_return_%"].sum())
        gl_seq = float(abs(seq_df.loc[seq_df["net_return_%"] <= 0, "net_return_%"].sum()))
        pf_seq = round(gp_seq / gl_seq, 2) if gl_seq > 1e-9 else (np.inf if gp_seq > 0 else np.nan)
        avg_win_seq  = round(seq_df.loc[seq_df["net_return_%"] > 0, "net_return_%"].mean(), 2) if wins_seq else 0.0
        avg_loss_seq = round(seq_df.loc[seq_df["net_return_%"] <= 0, "net_return_%"].mean(), 2) if (n_seq - wins_seq) else 0.0
        rr_seq = round(abs(avg_win_seq / avg_loss_seq), 2) if avg_loss_seq not in (0, 0.0) else np.nan
        exp_seq = round(float(seq_df["net_return_%"].mean()), 2)
        avg_days_seq = round(float(seq_df["days_held"].mean()), 1)
        # H5 FIX: sequential R-multiple expectancy (uses INIT stop)
        _sc = "init_stop_price" if "init_stop_price" in seq_df.columns else "stop_price"
        r_pct_seq = ((seq_df["entry_price"] - seq_df[_sc]) / seq_df["entry_price"] * 100)
        r_pct_seq = r_pct_seq.where(r_pct_seq > 0, np.nan)
        r_mult_seq = seq_df["net_return_%"] / r_pct_seq
        exp_R_seq = round(float(r_mult_seq.mean()), 3) if r_mult_seq.notna().any() else np.nan
        out.update({
            "seq_trades":            n_seq,
            "seq_win_%":             round(100.0 * wins_seq / n_seq, 1),
            "seq_expectancy_%":      exp_seq,
            "seq_expectancy_R":      exp_R_seq,
            "seq_total_return_%":    round(float(seq_df["net_return_%"].sum()), 2),
            "seq_avg_win_%":         avg_win_seq,
            "seq_avg_loss_%":        avg_loss_seq,
            "seq_reward_risk":       rr_seq,
            "seq_profit_factor":     pf_seq,
            "seq_avg_days":          avg_days_seq,
            "seq_exp_per_day_%":     round(exp_seq / max(avg_days_seq, 1e-9), 3),
        })
    else:
        out.update({
            "seq_trades": 0, "seq_win_%": 0.0, "seq_expectancy_%": 0.0,
            "seq_expectancy_R": np.nan,
            "seq_total_return_%": 0.0, "seq_avg_win_%": 0.0, "seq_avg_loss_%": 0.0,
            "seq_reward_risk": np.nan, "seq_profit_factor": np.nan,
            "seq_avg_days": 0.0, "seq_exp_per_day_%": 0.0,
        })
    return out


# ======================================================================================
#  5. STREAMLIT UI
# ======================================================================================
STRATEGY_HELP = {
    "PASS_recommended": "Core rule: strong uptrend (price >X% above 200-DMA) AND OBV rising. "
                        "Best balance of hit-rate and coverage.",
    "PASS_tight":       "Strictest: uptrend AND OBV rising AND enough volatility (ATR%). "
                        "Fewer, higher-conviction trades.",
    "PASS_balanced":    "Widest net: momentum-continuation OR early breakout-thrust branch. "
                        "Catches more winners at a slightly lower hit-rate.",
    "PASS_reversal":    "COUNTER-TREND (below the 200-DMA): buys oversold bounces and 50-DMA "
                        "reclaims — only after a confirmed turn. Rare, higher-risk, for downtrends.",
    "PASS_combined":    "ADAPTIVE: uses the 200-DMA as a regime switch — trend logic when the stock "
                        "is above it, reversal logic when below. Trades in BOTH up and down trends.",
}


def main():
    st.set_page_config(page_title="NSE Swing Screener & Backtester", layout="wide")
    st.title("📈 NSE Swing-Trade Screener & Backtester")
    st.caption("Triple-barrier swing backtest with the technical filters from our research. "
               "Educational tool — not investment advice.")

    # ---------------- Sidebar controls ----------------
    with st.sidebar:
        st.header("1 · Instrument & data")
        ticker = st.text_input("Yahoo ticker", value="HUDCO.NS",
                               help="Use .NS for NSE, .BO for BSE (e.g. HUDCO.NS).")
        today = dt.date.today()
        default_start = dt.date(2017, 6, 1)
        col1, col2 = st.columns(2)
        start = col1.date_input("Start date", value=default_start,
                                min_value=dt.date(2000, 1, 1), max_value=today)
        end = col2.date_input("End date", value=today,
                              min_value=dt.date(2000, 1, 1), max_value=today)
        st.info("**How much data?** Use **5+ years, ideally full history**. A range covering "
                "only a bull market (e.g. 2020–2024) flatters results — include down/choppy "
                "years (2018–19, 2022) so the edge is genuinely stress-tested.")

        st.header("2 · Strategy")
        strategy = st.selectbox("Filter logic",
                                ["PASS_recommended", "PASS_tight", "PASS_balanced",
                                 "PASS_reversal", "PASS_combined"])
        st.caption(STRATEGY_HELP[strategy])
        if strategy == "PASS_reversal":
            st.warning("Counter-trend mode buys weakness. It fires rarely and carries higher risk "
                       "(catching falling knives). Use small size, tighter targets, and treat it "
                       "as a secondary strategy — validate across many stocks first.")
        if strategy == "PASS_combined":
            st.info("Uptrend trades use the main target/stop below. Downtrend (reversal) trades "
                    "use their own tighter target/stop, since bounces are smaller than trends.")

        st.header("3 · Trade rules")
        entry_choice = st.radio("Entry style",
                                ["Limit near signal close (recommended)", "Market at next open"],
                                help="Market buys whatever the open gives you — a gap-up means a worse "
                                     "fill. Limit places a resting buy and simply skips the trade if "
                                     "price never comes back to it.")
        if entry_choice.startswith("Limit"):
            entry_mode = "Limit"
            limit_pct = st.slider("Limit below signal close (%)", 0.0, 5.0, 0.5, 0.1,
                                  help="0 = buy at the signal close price. Higher = wait for a deeper "
                                       "pullback (better fills, but more trades never fill).")
            fill_days = st.number_input("Order valid for (sessions)", 1, 5, 1,
                                        help="How many days the limit order rests before expiring.")
        else:
            entry_mode, limit_pct, fill_days = "Market open", 0.0, 1
        exit_mode = st.radio("Exit style",
                             ["Trailing stop (let winners run)", "Fixed target"], index=0,
                             help="Fixed target caps gains at your target. Trailing stop removes "
                                  "the ceiling and exits only on a pullback — so a big move is captured.")
        target_pct = st.number_input("Minimum return objective (%)", 1.0, 100.0, 10.0, 0.5,
                                     help="In Trailing mode this is a MINIMUM, not a cap: once reached, "
                                          "the trade keeps running while the trend holds. In Fixed-target "
                                          "mode it caps the gain.")
        trail_mult, lock_pct = 2.0, None
        if exit_mode.startswith("Trailing"):
            trail_mult = st.slider("Trailing distance (× ATR)", 0.5, 5.0, 2.0, 0.5,
                                   help="Stop trails this far below the highest price reached. "
                                        "Wider = lets it breathe and run further.")
            lock_on = st.checkbox("Lock in profit once objective reached", value=True,
                                  help="Once the stock touches +10%, raise the stop so the trade can "
                                       "never end below the locked level. Protects the objective while "
                                       "still letting it run to 20-30%.")
            lock_pct = st.slider("Lock profit at (%)", 0.0, 30.0, 10.0, 0.5,
                                 help="Floor under a winner after it hits the objective.") if lock_on else None
            exit_mode = "Trailing"
        else:
            exit_mode = "Fixed target"
        cA, cB = st.columns(2)
        min_hold = cA.number_input("Min holding (days)", 1, 60, 7,
                                   help="Advisory. Stop-loss is always active; "
                                        "the max value below is the hard time-stop.")
        max_hold = cB.number_input("Max holding (days)", 1, 120, 10)

        st.header("3b · Conviction exit & partial profit")
        cut_on = st.checkbox("Cut dead trades early", value=False,
                             help="If the trade is still red after N days it rarely reaches the "
                                  "objective — free the capital for the next signal.")
        if cut_on:
            cut_day = st.number_input("Cut if still below threshold on day", 1, 10, 2)
            cut_threshold = st.slider("…and return is below (%)", -8.0, 2.0, 0.0, 0.5)
        else:
            cut_day, cut_threshold = None, 0.0
        partial_on = st.checkbox("Take partial profit (volatility-scaled)", value=False,
                                 help="Books part of the position at entry + k×ATR — so a calm stock "
                                      "banks at a smaller move than a wild one. Not a fixed %.")
        if partial_on:
            st.caption("⚠️ On HUDCO this REDUCED return per day at every setting — it caps winners. "
                       "Test on your own basket before enabling.")
            partial_frac = st.slider("Fraction to book", 0.0, 0.9, 0.3, 0.1)
            partial_atr = st.slider("Book at entry + (× ATR)", 0.5, 4.0, 3.0, 0.25)
        else:
            partial_frac, partial_atr = 0.0, 1.5

        st.header("4 · Stop-loss")
        # NOTE: defaults changed to Structure (Aug-2026) to match the scanner
        # (C1 fix) — the scanner's per-stock drill-down and this single-stock
        # tool now produce IDENTICAL trades for the same ticker + settings.
        stop_anchor = st.radio("Initial stop anchoring",
                               ["ATR distance", "Structure (below 10-day swing low)"],
                               index=1,
                               help="ATR = fixed volatility distance from entry. Structure = just "
                                    "below recent support, so a normal pullback inside a rising "
                                    "channel doesn't stop you out. **Default = Structure**.")
        stop_anchor = "Structure" if stop_anchor.startswith("Structure") else "ATR"
        trail_anchor = st.radio("Trailing anchoring",
                                ["ATR distance", "Structure (below rising 5-day swing low)"],
                                index=1,
                                help="Structure trailing exits only when support actually breaks — "
                                     "bigger winners per trade, longer holds. **Default = Structure**.")
        trail_anchor = "Structure" if trail_anchor.startswith("Structure") else "ATR"
        stop_method = st.selectbox("Stop type", ["ATR (volatility-based)", "Fixed %"])
        if stop_method.startswith("ATR"):
            stop_value = st.slider("ATR multiple", 0.5, 5.0, 2.0, 0.5,
                                   help="Stop = entry − (multiple × ATR14). 2× is a typical swing setting.")
            stop_method = "ATR"
        else:
            stop_value = st.slider("Stop-loss (%)", 1.0, 20.0, 5.0, 0.5)
            stop_method = "FIXED"
        max_stop_pct = st.slider("Max loss cap (%)", 2.0, 20.0, 5.0, 0.5,
                                 help="Hard ceiling on risk: an ATR stop wider than this is pulled "
                                      "in. Caps your worst-case loss per trade.")
        max_atr_pct = st.slider("Skip if ATR% above", 3.0, 15.0, 8.0, 0.5,
                                help="Volatility ceiling: don't enter stocks so wild that a safe "
                                     "stop is impossible. Set high to disable.")

        st.header("5 · Costs")
        cost_pct = st.number_input("Round-trip cost (%)", 0.0, 5.0, 0.20, 0.05,
                                   help="Brokerage + STT + slippage per trade, applied to each trade.")
        apply_stcg = st.checkbox("Apply 20% STCG on gains (approx.)", value=True,
                                 help="ON by default: a backtest that ignores tax overstates the edge.")

        rev_target_pct, rev_stop_value = None, None
        if strategy in ("PASS_combined", "PASS_reversal"):
            st.header("4b · Reversal-trade risk")
            rev_target_pct = st.number_input("Reversal target (%)", 1.0, 100.0, 6.0, 0.5,
                                             help="Smaller target for counter-trend bounces.")
            rev_stop_value = st.slider("Reversal stop (× ATR)", 0.5, 5.0, 1.5, 0.5,
                                       help="Tighter stop for counter-trend trades.")
            if stop_method == "FIXED":
                rev_stop_value = st.slider("Reversal stop (%)", 1.0, 20.0, 4.0, 0.5,
                                           key="rev_fixed")

        # ==============================================================================
        # SECTION 4b/4c/4d — EXIT-STACK + REGIME ROUTER
        # ------------------------------------------------------------------------------
        # C1 FIX (Aug-2026): the standalone screener previously called run_backtest
        # with NO exit-stack layers and NO regime router, so it silently disagreed
        # with the scanner's per-stock drill-down. This section adds the same UI
        # blocks as the scanner (sections 4b, 4c, 4d), with the same defaults:
        #   * A / B / C / D off,   E off
        #   * Regime router ON     (evaluated at signal bar — C3 fix)
        # Users who want to tune the exit stack on a single stock can now do so here,
        # and the scanner's drill-down of the same stock will produce identical trades.
        # ==============================================================================
        st.header("7 · Exit stack  A + B + C")
        st.caption("Optional layers that tighten profits on winners. All default OFF — "
                   "the regime router (below) will pick trailing vs fixed target per trade.")
        use_ratchet = st.checkbox(
            "🔒 A · Ratcheting profit lock", value=False,
            help="Peak crosses 15/25/40/60/85/120 % → floor moves to 5/12/25/40/60/85 %."
        )
        use_shrink = st.checkbox(
            "📉 B · Shrinking trail multiplier", value=False,
            help="Trail width narrows as gain grows (uses CURRENT-day ATR)."
        )
        use_momexit = st.checkbox(
            "⚡ C · Momentum-exhaustion exit", value=False,
            help="Fires on RSI rollover / MACD flip / heavy-vol pullback / bearish engulfing / 5-DMA break."
        )
        mom_min_gain = st.slider(
            "C · Arm momentum exit only when up ≥ (%)", 5.0, 30.0, 10.0, 1.0,
            disabled=not use_momexit)

        st.header("7b · Exit stack extensions  D + E")
        use_decay = st.checkbox(
            "⏳ D · Time-decay tightening", value=False,
            help="Tighten trail after N sessions with no new peak; force exit after M sessions.")
        decay_after = st.slider(
            "D · Start tightening after (sessions)", 3, 15, 5, 1, disabled=not use_decay)
        decay_shrink = st.slider(
            "D · Tightening rate (% of stop-to-price gap per day)", 10.0, 60.0, 25.0, 5.0,
            disabled=not use_decay)
        decay_exit = st.slider(
            "D · Force exit after (sessions)", 5, 25, 10, 1, disabled=not use_decay)
        use_staircase = st.checkbox(
            "🪜 E · Staircase partial exits", value=False,
            help="Book 30% at +10%, 30% at +20%, rest runs. GUARANTEES some profit but CAPS runaway winners.")

        st.header("7c · Regime-aware exit router")
        st.caption("Per-trade routing at the SIGNAL BAR (C3 fix — was fill bar): trending → trailing "
                   "with A+B+C · choppy/weak → fixed target.")
        use_router = st.checkbox(
            "🧭 Route each trade by regime (recommended)", value=True,
            help="Evaluates 4 conditions at each signal: (1) ADX ≥ threshold, (2) 200-DMA rising, "
                 "(3) price ≥ N% above 200-DMA, (4) realized vol > baseline. ALL 4 pass → trailing "
                 "(A+B+C apply). Any fail → fixed target.")
        with st.expander("Router thresholds"):
            route_min_adx = st.slider("ADX(14) ≥", 10.0, 40.0, 20.0, 1.0, disabled=not use_router)
            route_sma_slope_lb = st.slider("200-DMA rising over (sessions)", 5, 60, 20, 1, disabled=not use_router)
            route_min_dist_pct = st.slider("Price above 200-DMA by ≥ (%)", 0.0, 30.0, 15.0, 0.5, disabled=not use_router)
            route_fixed_target_pct = st.slider("Fixed-target level when choppy (%)", 5.0, 30.0, 15.0, 0.5, disabled=not use_router)
            route_vol_lb = st.slider("Realized-vol window (sessions)", 20, 126, 63, 1, disabled=not use_router)
            route_vol_baseline_lb = st.slider("Vol baseline window (sessions)", 60, 504, 252, 1, disabled=not use_router)

        st.header("6 · Filter thresholds")
        with st.expander("Advanced (defaults are HUDCO-derived)"):
            p = {
                "regime": st.slider("Uptrend: % above 200-DMA", 0.0, 50.0, 15.0, 1.0),
                "atr":    st.slider("Volatility floor: ATR%", 0.0, 10.0, 3.5, 0.5),
                "roc":    st.slider("Breakout branch: ROC(10) >", 0.0, 15.0, 3.0, 0.5),
                "volr":   st.slider("Breakout branch: volume ratio >", 0.5, 4.0, 1.2, 0.1),
                "rsi_os": st.slider("Reversal branch: oversold RSI <", 10.0, 45.0, 30.0, 1.0),
            }
        run = st.button("🚀 Run backtest", type="primary", use_container_width=True)

    if not run:
        st.write("Set your parameters in the sidebar and click **Run backtest**.")
        st.stop()

    # ---------------- Pipeline ----------------
    with st.spinner(f"Fetching {ticker} …"):
        try:
            raw = fetch_data(ticker, start, end)
        except Exception as e:
            st.error(f"Data fetch failed: {e}")
            st.stop()
    if raw.empty:
        st.error("No data returned. Check the ticker (NSE needs the .NS suffix) and date range.")
        st.stop()

    df = compute_indicators(raw)
    df = generate_signals(df, strategy, p)
    trades = run_backtest(df, target_pct, int(max_hold), stop_method,
                          stop_value, cost_pct, apply_stcg,
                          rev_target_pct=rev_target_pct, rev_stop_value=rev_stop_value,
                          exit_mode=exit_mode, trail_mult=trail_mult,
                          min_hold=int(min_hold),         # M7 FIX — now honoured
                          max_stop_pct=max_stop_pct, max_atr_pct=max_atr_pct,
                          entry_mode=entry_mode, limit_pct=limit_pct, fill_days=int(fill_days),
                          lock_pct=lock_pct, cut_day=(int(cut_day) if cut_day else None),
                          cut_threshold=cut_threshold, partial_frac=partial_frac,
                          partial_atr=partial_atr, stop_anchor=stop_anchor,
                          trail_anchor=trail_anchor,
                          # ---- C1 FIX: pass exit-stack + router so this
                          # standalone tool produces the SAME trades as the
                          # scanner's per-stock drill-down for the same ticker.
                          # Section 4b: A + B + C
                          ratchet_lock=use_ratchet,
                          shrink_trail=use_shrink,
                          momentum_exit=use_momexit,
                          mom_exit_min_gain=mom_min_gain,
                          # Section 4c: D + E
                          time_decay=use_decay,
                          decay_after_days=int(decay_after),
                          decay_shrink_pct=decay_shrink,
                          decay_exit_days=int(decay_exit),
                          staircase=use_staircase,
                          # Section 4d: Regime-aware exit router
                          regime_route=use_router,
                          route_min_adx=route_min_adx,
                          route_sma_slope_lb=int(route_sma_slope_lb),
                          route_min_dist_pct=route_min_dist_pct,
                          route_vol_lb=int(route_vol_lb),
                          route_vol_baseline_lb=int(route_vol_baseline_lb),
                          route_fixed_target_pct=route_fixed_target_pct)
    n_signals = int(df["signal"].fillna(False).sum())
    if entry_mode == "Limit" and n_signals:
        fill_rate = 100.0 * len(trades) / n_signals
        st.info(f"📥 **Limit entry:** {len(trades)} of {n_signals} signals filled "
                f"({fill_rate:.0f}%). The unfilled ones ran away without you — that is the cost of "
                f"not chasing. Average fill edge: "
                f"{trades['fill_edge_%'].mean():+.2f}% vs the signal close."
                if len(trades) else
                f"📥 **Limit entry:** 0 of {n_signals} signals filled — try a smaller limit % "
                f"or more valid sessions.")
    stats = summarize(trades)

    st.success(f"Loaded {len(df)} trading days for {ticker} "
               f"({df.index[0].date()} → {df.index[-1].date()}).")

    if trades.empty:
        st.warning("No signals fired for this strategy over this window. "
                   "Try PASS_balanced or a longer date range.")
        st.stop()

    # ---------------- Headline metrics ----------------
    st.subheader("Results")

    def _fmt(v, suffix="", dp=2, dash="—"):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "∞" if v == np.inf else dash
        return f"{v:.{dp}f}{suffix}" if isinstance(v, float) else f"{v}{suffix}"

    # --- Row 1: sample size, win rate, avg return / trade, overall performance ---
    m = st.columns(4)
    m[0].metric("① Total trades", stats["trades"],
                help="Sample size. <30 trades = treat the edge as indicative, not proven.")
    m[1].metric("② Win rate", f'{stats["profitable_%"]}%',
               help="Share of trades closing with a positive net return.")
    m[2].metric("③ Avg return / trade", f'{stats["expectancy_%"]:+.2f}%',
               f'{stats.get("exp_per_day_%", 0)}% per day',
               help="Expectancy — the single most important profitability number. "
                    "Positive means the system has an edge after costs.")
    m[3].metric("④ Total return (sum)", f'{stats["total_return_sum_%"]:+.2f}%',
               help="Simple sum of every trade's net return — a quick read on overall "
                    "strategy performance on this stock. NOT compounded (trades overlap), "
                    "so treat it as a scorecard total, not an account balance. For a "
                    "compounded figure use CAGR / Total return (compounded) below.")

    # --- Row 2: annualised return, drawdown, profit factor, recovery factor ---
    m2 = st.columns(4)
    seq_n = stats.get("seq_trades")
    if seq_n:
        m2[0].metric("⑤ CAGR", f'{stats.get("cagr_%", 0):+.2f}%',
                    help=f"Annualised return, compounded on a sequential (one-trade-at-a-time) "
                         f"equity curve of {seq_n} non-overlapping trades over "
                         f"{stats.get('years_sequenced','?')}y. This is the account-level "
                         f"return figure, distinct from the per-trade average above.")
        m2[1].metric("⑥ Max drawdown", f'{stats.get("max_drawdown_%", 0):.2f}%',
                    help="Largest peak-to-trough decline in the sequential equity curve "
                         "before it recovered to a new high.")
    else:
        m2[0].metric("⑤ CAGR", "—")
        m2[1].metric("⑥ Max drawdown", "—")
    m2[2].metric("⑦ Profit factor", _fmt(stats.get("profit_factor")),
               help="Gross profit ÷ gross loss (both in % points, summed across all trades). "
                    ">1 = profitable system. >1.5 is generally considered robust.")
    m2[3].metric("⑧ Recovery factor", _fmt(stats.get("recovery_factor")),
               help="Net profit ÷ max drawdown on the sequential equity curve — how many "
                    "times over the strategy recovered its worst drawdown.")

    # --- Row 3: win/loss shape, reward:risk, holding period, consecutive losses ---
    m3 = st.columns(4)
    m3[0].metric("⑨⑩ Avg winner / loser", f'{stats["avg_win_%"]:+.2f}% / {stats["avg_loss_%"]:+.2f}%',
               help="Average % return of winning trades vs. average % return of losing trades.")
    m3[1].metric("⑪ Reward/Risk ratio", _fmt(stats.get("reward_risk_ratio")),
               help="Average winner ÷ average loser. A ratio of 2 means winners are twice "
                    "the size of losers on average — lets the system stay profitable even "
                    "below a 50% win rate.")
    m3[2].metric("⑫ Avg holding days", f'{stats["avg_days"]} days',
               help="Typical trade duration — how long capital stays tied up per trade.")
    m3[3].metric("⑬ Max consecutive losses", stats.get("max_consecutive_losses", "—"),
               help="Longest losing streak in chronological (sequential) trading order. "
                    "The number of losses in a row you must be psychologically prepared for.")

    # --- Row 4: exit-reason mix ---
    m4 = st.columns(4)
    m4[0].metric("⑭ Target hit %", f'{stats["target_pct_of_all"]}%',
               help="Share of trades that exited because they reached the profit target.")
    m4[1].metric("⑮ Stop-loss %", f'{stats["stop_pct_of_all"]}%',
               help="Share of trades that exited via the stop-loss.")
    m4[2].metric("⑯ Trailing-stop %", f'{stats["trail_pct_of_all"]}%',
               help="Share of trades that exited profitably via the trailing stop, below the "
                    "fixed target (only applies when Exit style = Trailing).")
    m4[3].metric("⑰ Time exit %", f'{stats["time_pct_of_all"]}%',
               help="Share of trades that exited only because the max holding period was reached.")

    st.caption("**Hit-rate** = share of trades that reached your target. **Profitable %** also "
               "counts time-exits and trailing-stop exits that closed green. **Expectancy** is "
               "the average net return per trade — the number that decides if the system makes "
               "money. Circled numbers ①–⑰ correspond to the 17 requested metrics.")
    st.caption("⚠️ **Metric families:** ①②③⑨⑩⑪⑭⑮⑯⑰ and ⑦ (profit factor) use **every** historical "
               "signal instance — best for judging the raw edge with maximum statistical power. "
               "⑤⑥⑧⑬ use the **sequential, one-position-at-a-time** trade list (see expander below) "
               "because CAGR/drawdown/recovery/streaks only make sense on a single capital timeline, "
               "and this strategy's raw signals can overlap.")

    with st.expander("Why two different trade counts? (sequential vs. all-signals)"):
        st.markdown(
            f"- **All-signal trades ({stats['trades']}):** every time the pattern fired historically, "
            "even if a prior trade in the same stock was still open. Used for win rate, profit "
            "factor, average winner/loser, reward:risk, and exit-mix % — these describe the "
            "**quality of the edge**, and more instances = more statistical power.\n"
            f"- **Sequential trades ({seq_n or 0}):** capital is only redeployed after the previous "
            "trade closes — one position at a time in this stock. Used for CAGR, max drawdown, "
            "recovery factor, and max consecutive losses, since compounding math needs a single "
            "capital timeline. This count is naturally lower than the all-signal count."
        )

    # ---- regime split (only meaningful for combined / mixed strategies) ----
    if trades["trade_type"].nunique() > 1:
        st.markdown("**Performance by regime** — how the uptrend and downtrend legs each contribute:")
        rows = []
        for rt, sub in trades.groupby("trade_type"):
            wins = (sub["net_return_%"] > 0).mean() * 100
            rows.append({
                "regime": rt, "trades": len(sub),
                "target hits": int((sub["outcome"] == "TARGET").sum()),
                "stop hits": int((sub["outcome"] == "STOP").sum()),
                "profitable %": round(wins, 1),
                "expectancy %": round(sub["net_return_%"].mean(), 2),
                "avg days": round(sub["days_held"].mean(), 1),
            })
        st.dataframe(pd.DataFrame(rows).set_index("regime"), use_container_width=True)
        st.caption("If the DOWNTREND row's expectancy is negative for a given stock, the reversal "
                   "leg is hurting you on that name — favour the trend leg there.")

    # ---------------- Sequential equity curve (CAGR / Max Drawdown visualised) ----------------
    if "equity_curve" in stats and not stats["equity_curve"].empty:
        st.subheader("Sequential equity curve — CAGR & Max Drawdown")
        ec = stats["equity_curve"].copy()
        ec["growth_%"] = (ec["equity_mult"] - 1) * 100
        running_peak = ec["equity_mult"].cummax()
        ec["drawdown_%"] = (ec["equity_mult"] / running_peak - 1) * 100
        fig_eq = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35],
                               vertical_spacing=0.05,
                               subplot_titles=("Compounded growth (one-trade-at-a-time, ₹1 → ₹X)",
                                               "Drawdown from equity peak"))
        fig_eq.add_trace(go.Scatter(x=pd.to_datetime(ec["exit_date"]), y=ec["growth_%"],
                                    name="Cumulative growth %", line=dict(color="#16a34a", width=1.6),
                                    fill="tozeroy", fillcolor="rgba(22,163,74,0.08)"), row=1, col=1)
        fig_eq.add_trace(go.Scatter(x=pd.to_datetime(ec["exit_date"]), y=ec["drawdown_%"],
                                    name="Drawdown %", line=dict(color="#dc2626", width=1.4),
                                    fill="tozeroy", fillcolor="rgba(220,38,38,0.12)"), row=2, col=1)
        fig_eq.update_layout(height=420, hovermode="x unified", legend_orientation="h",
                             margin=dict(t=40, b=10), showlegend=False)
        st.plotly_chart(fig_eq, use_container_width=True)
        st.caption(f"Built from **{stats.get('seq_trades','?')} sequential (non-overlapping) trades** "
                   f"over **{stats.get('years_sequenced','?')} years** → "
                   f"**CAGR {stats.get('cagr_%',0):+.2f}%**, "
                   f"**Max drawdown {stats.get('max_drawdown_%',0):.2f}%**, "
                   f"**Recovery factor {_fmt(stats.get('recovery_factor'))}**. "
                   "One position at a time in this stock — not a multi-stock portfolio curve.")
        if stats.get("years_sequenced", 99) < 1:
            st.caption("⚠️ Under 1 year of sequential trading history — annualising to CAGR "
                       "amplifies noise heavily here. Treat CAGR as illustrative only until "
                       "more sequential trades accumulate.")

    # ---------------- Price chart with trade outcomes ----------------
    st.subheader("Price & trade outcomes")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
                        vertical_spacing=0.04,
                        subplot_titles=("Close price with entries", "Cumulative net return (overlapping proxy)"))
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close",
                             line=dict(color="#334155", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["sma200"], name="200-DMA",
                             line=dict(color="#f59e0b", width=1, dash="dot")), row=1, col=1)
    colors = {"TARGET": "#16a34a", "TRAIL": "#86efac", "STOP": "#dc2626", "TIME": "#94a3b8"}
    for oc, cl in colors.items():
        sub = trades[trades["outcome"] == oc]
        if not sub.empty:
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(sub["entry_date"]), y=sub["entry_price"],
                mode="markers", name=oc,
                marker=dict(color=cl, size=7, line=dict(width=0.5, color="white"))),
                row=1, col=1)
    eq = trades.sort_values("entry_date").copy()
    eq["cum"] = eq["net_return_%"].cumsum()
    fig.add_trace(go.Scatter(x=pd.to_datetime(eq["entry_date"]), y=eq["cum"],
                             name="Cumulative net %", line=dict(color="#2563eb", width=1.5)),
                  row=2, col=1)
    fig.update_layout(height=650, hovermode="x unified", legend_orientation="h",
                      margin=dict(t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Green = target hit, red = stop hit, grey = time exit. The lower panel sums "
               "individual trade returns in entry order — a rough performance proxy, not a "
               "compounded single-capital equity curve (signals overlap).")

    # ---------------- Which indicators separated winners from losers ----------------
    st.subheader("Technical parameters used & their discriminating power")
    st.markdown(
        "**Indicator suite computed each day (no look-ahead):** moving averages 5/20/50/200, "
        "RSI(14), MACD(12,26,9), ATR% , Bollinger %B & bandwidth, Stochastics, OBV & OBV-slope, "
        "ADX/+DI/−DI, volume ratio & surge, 20-day breakout, 52-week distance, candle body/gap.\n\n"
        f"**Active filter — `{strategy}`:** {STRATEGY_HELP[strategy]}"
    )
    wins = trades[trades["outcome"] == "TARGET"]
    losses = trades[trades["outcome"] == "STOP"]
    if not wins.empty and not losses.empty:
        comp = pd.DataFrame({
            "target-hit avg": wins[SNAP_COLS].mean().round(2),
            "stop-hit avg":   losses[SNAP_COLS].mean().round(2),
        })
        comp["separation"] = ((comp["target-hit avg"] - comp["stop-hit avg"]) /
                              trades[SNAP_COLS].std().replace(0, np.nan)).round(2)
        comp = comp.reindex(comp["separation"].abs().sort_values(ascending=False).index)
        st.dataframe(comp, use_container_width=True)
        st.caption("Entry-day indicator averages for winning vs stopped-out trades. "
                   "|separation| above ~0.5 means the indicator meaningfully tilts the odds.")

    # ---------------- Trade log ----------------
    st.subheader("Trade log")
    st.dataframe(trades, use_container_width=True, height=350)
    st.download_button("⬇️ Download trades CSV",
                       trades.to_csv(index=False).encode(),
                       file_name=f"{ticker}_backtest_trades.csv", mime="text/csv")

    st.divider()
    st.caption("⚠️ Single-stock, in-sample results flatter reality. Validate across many stocks "
               "and out-of-sample before trusting any threshold. This is an educational backtest, "
               "not investment advice.")


if __name__ == "__main__":
    main()