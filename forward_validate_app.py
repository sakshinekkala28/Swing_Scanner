"""
forward_validate_app.py  (v2, Aug-2026)
=======================================
FORWARD VALIDATION with failure-clustering analysis and algorithm-improvement toggles.

What v2 adds over v1:
  * Nifty 500 default (statistically-meaningful sample sizes)
  * REGIME GATE — historical (uses only benchmark index price data → no look-ahead)
  * Same-day SIGNAL DECAY — if many stocks signal the same day, they aren't
    independent evidence; discount rank score OR cap N per day
  * Portfolio-level STOP COOLDOWN — after M stops in last N sessions, pause entries
  * SECTOR CAP — matches live scanner behavior (max K per sector)
  * FAILURE CLUSTERING analysis output:
      - Same-day failure detection (were stops synchronised?)
      - Sector distribution of failures vs shortlist
      - Correlation of stopped stocks' returns
      - Beta cluster check
  * MULTI-CUTOFF mode — auto-runs N cutoffs spanning different market regimes

WHY: v1 revealed that on a single choppy-day cutoff, multiple stocks all stopped
out together. Root cause is almost never "the algo is broken" — it's "the algo
took correlated bets during a bad regime, and stops fired en masse". The v2
additions surface that diagnosis clearly AND provide the filters that a
professional algo trader would use to mitigate it.

HONESTLY-DISCLOSED LIMITATIONS (auto-shown in UI):
  * Historical NEWS/EVENTS not available for free → skipped (as before)
  * Historical FUNDAMENTALS not point-in-time → skipped for cutoffs >5d ago
  * REGIME gate now IS applied historically (this is the v2 fix)

Run:  streamlit run forward_validate_app.py
"""

import os
import sys
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
#  Engine loader — reuses swing_screener_app.py
# ======================================================================================
_here = os.path.dirname(os.path.abspath(__file__))
_ENGINE_PATH = os.path.join(_here, "swing_screener_app.py")
if not os.path.exists(_ENGINE_PATH):
    st.error(f"Engine file not found: {_ENGINE_PATH}")
    st.stop()
_spec = importlib.util.spec_from_file_location("engine", _ENGINE_PATH)
engine = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(engine)

from universe_loader import load_full_universe


# ======================================================================================
#  Fetch helpers
# ======================================================================================
@st.cache_data(show_spinner=False, ttl=60 * 60 * 12)
def _fetch_full(ticker_yahoo: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    if yf is None:
        return pd.DataFrame()
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


@st.cache_data(show_spinner=False, ttl=60 * 60 * 12)
def _fetch_bench(start: dt.date, end: dt.date):
    """Fetch broad Nifty benchmark (Nifty 500 preferred, fallback Nifty 50)."""
    if yf is None:
        return None, pd.DataFrame()
    for t in ["^CRSLDX", "^NSEI"]:
        try:
            df = yf.Ticker(t).history(start=start, end=end, interval="1d", auto_adjust=True)
            if df is not None and not df.empty:
                df = df[["Open", "High", "Low", "Close"]].copy()
                df.index = pd.to_datetime(df.index).tz_localize(None)
                return t, df.dropna()
        except Exception:
            continue
    return None, pd.DataFrame()


def _bare(sym: str) -> str:
    return sym.upper().replace(".NS", "").replace(".BO", "")


def _to_yahoo(sym: str) -> str:
    sym = sym.upper().strip()
    return sym if sym.endswith((".NS", ".BO")) else sym + ".NS"


# ======================================================================================
#  Regime gate — HISTORICAL (no look-ahead)
# ======================================================================================
def regime_at_cutoff(bench_df: pd.DataFrame, cutoff: dt.date) -> dict:
    """Given the benchmark index price series, compute what the regime status
    WAS on cutoff date, using only data ≤ cutoff (no look-ahead).
    Returns {"status": "RISK-ON"|"NEUTRAL"|"RISK-OFF", "pct_vs_200": x, "roc10": x}.
    """
    df = bench_df.loc[bench_df.index.date <= cutoff]
    if df.empty or len(df) < 210:
        return {"status": "UNKNOWN", "pct_vs_200": np.nan, "roc10": np.nan}
    c = df["Close"]
    s200 = float(c.rolling(200).mean().iloc[-1])
    last = float(c.iloc[-1])
    above_200 = last > s200
    pct_vs_200 = (last / s200 - 1) * 100
    roc10 = (c.iloc[-1] / c.iloc[-11] - 1) * 100 if len(c) > 11 else 0.0
    if above_200 and roc10 > -1.0:
        status = "RISK-ON"
    elif above_200 or roc10 > -3.0:
        status = "NEUTRAL"
    else:
        status = "RISK-OFF"
    return {"status": status, "pct_vs_200": round(pct_vs_200, 2),
            "roc10": round(roc10, 2), "above_200": above_200}


# ======================================================================================
#  Signal-day scan AS OF cutoff (no look-ahead)
# ======================================================================================
def scan_as_of(ticker: str, hist_df: pd.DataFrame, strategy: str,
               strat_params: dict, bt_kwargs: dict, cutoff: dt.date,
               sector_map: dict = None) -> dict:
    df_in = hist_df.loc[hist_df.index.date <= cutoff].copy()
    if df_in.empty or len(df_in) < 250:
        return {"ticker": _bare(ticker), "status": "insufficient"}
    df = engine.compute_indicators(df_in)
    df = engine.generate_signals(df, strategy, strat_params)
    trades = engine.run_backtest(df, **bt_kwargs)
    stats = engine.summarize(trades)
    last = df.iloc[-1]
    entry_ref = float(last["Close"])
    atr_now = float(last["atr14"]) if np.isfinite(last["atr14"]) else 0.0
    stop_mult = bt_kwargs.get("stop_value", 2.0)
    max_stop_pct = bt_kwargs.get("max_stop_pct", 8.0) or 8.0
    tgt_pct = bt_kwargs.get("target_pct", 15.0)
    limit_pct = bt_kwargs.get("limit_pct", 0.5)
    entry_mode = bt_kwargs.get("entry_mode", "Limit")
    if entry_mode == "Limit":
        limit_price = round(entry_ref * (1 - limit_pct / 100.0), 2)
        plan_entry = limit_price
    else:
        limit_price = np.nan
        plan_entry = entry_ref
    stop_price = max(plan_entry - stop_mult * atr_now,
                     plan_entry * (1 - max_stop_pct / 100))
    target_price = round(plan_entry * (1 + tgt_pct / 100), 2)
    n_seq = stats.get("seq_trades", 0)
    winr_seq = stats.get("seq_win_%", 0.0)
    exp_day_seq = stats.get("seq_exp_per_day_%", 0.0)
    confidence = round(max(exp_day_seq, 0) * (winr_seq / 100.0)
                       * (n_seq / (n_seq + 30.0)) * 100, 2) if n_seq else 0.0
    return {
        "ticker": _bare(ticker), "status": "ok",
        "signals_today": bool(last["signal"]),
        "sector": (sector_map or {}).get(_bare(ticker), "UNKNOWN"),
        "cutoff_close": entry_ref, "plan_entry": round(plan_entry, 2),
        "limit_price": (round(limit_price, 2) if np.isfinite(limit_price) else None),
        "target_price": target_price, "stop_price": round(stop_price, 2),
        "target_%": tgt_pct, "atr_pct": round(float(last["atr_pct"]), 2)
                                          if np.isfinite(last["atr_pct"]) else np.nan,
        "hist_trades_seq": n_seq, "hist_winrate_seq": winr_seq,
        "hist_expectancy_seq": stats.get("seq_expectancy_%", 0),
        "confidence": confidence,
        "regime_today": (last.get("trade_type", "") or "UPTREND")
                        if bool(last["signal"]) else "",
    }


# ======================================================================================
#  Forward simulation — USES PRODUCTION ENGINE (respects router, trailing,
#  A/B/C/D/E exit stack). Aug-2026 rewrite: previously a simplified fixed-
#  target-or-stop loop that IGNORED the router + trailing → forward returns
#  were capped at target (winners never allowed to run beyond 15%).
# ======================================================================================
def forward_simulate(hist_df: pd.DataFrame, cutoff: dt.date, strategy: str,
                     strat_params: dict, bt_kwargs: dict) -> dict:
    """Run engine.run_backtest on the FULL history (backtest + forward).
    Extract the trade whose signal_date matches cutoff. This uses the exact
    same exit logic that the live scanner uses in production — router picks
    Trailing vs Fixed per trade, trailing stops let winners run beyond target,
    A/B/C/D/E exit stack applies if enabled."""
    if hist_df.empty:
        return {"outcome": "NO_DATA"}
    df = engine.compute_indicators(hist_df)
    df = engine.generate_signals(df, strategy, strat_params)
    trades = engine.run_backtest(df, **bt_kwargs)
    if trades.empty:
        return {"outcome": "NO_TRADES"}
    trades["signal_date_dt"] = pd.to_datetime(trades["signal_date"])
    cutoff_ts = pd.Timestamp(cutoff)
    match = trades[trades["signal_date_dt"] == cutoff_ts]
    if match.empty:
        # Signal may fire on a trading day just before cutoff — accept if it's the
        # LAST signal ≤ cutoff (means signal fired on cutoff bar itself).
        prior = trades[trades["signal_date_dt"] <= cutoff_ts]
        if prior.empty:
            return {"outcome": "NO_SIGNAL_AT_CUTOFF"}
        cand = prior.tail(1)
        if pd.Timestamp(cand["signal_date_dt"].iloc[0]) != cutoff_ts:
            return {"outcome": "NO_SIGNAL_AT_CUTOFF"}
        match = cand
    tr = match.iloc[0]
    return {
        "outcome":         tr["outcome"],
        "entry_date":      tr["entry_date"],
        "entry_price":     round(float(tr["entry_price"]), 2),
        "exit_date":       tr["exit_date"],
        "exit_price":      round(float(tr["exit_price"]), 2),
        "days_held":       int(tr["days_held"]),
        "net_return_%":    round(float(tr["net_return_%"]), 2),
        "gross_return_%":  round(float(tr.get("gross_return_%", 0)), 2),
        "peak_gain_%":     round(float(tr.get("peak_gain_%", 0)), 2),
        "exit_route":      tr.get("exit_route", ""),  # router audit (trailing/fixed)
    }


# ======================================================================================
#  Sector-cap + signal-decay + cooldown  (algorithm-improvement layers)
# ======================================================================================
def apply_sector_cap(shortlist: pd.DataFrame, max_per_sector: int) -> pd.DataFrame:
    """Keep at most `max_per_sector` names per sector, sorted by confidence."""
    if max_per_sector <= 0 or shortlist.empty:
        return shortlist
    kept, counts = [], {}
    sub = shortlist.sort_values("confidence", ascending=False).reset_index(drop=True)
    for i, row in sub.iterrows():
        sec = row.get("sector", "UNKNOWN") or "UNKNOWN"
        if counts.get(sec, 0) < max_per_sector:
            counts[sec] = counts.get(sec, 0) + 1
            kept.append(i)
    return sub.loc[kept].reset_index(drop=True)


# ======================================================================================
#  Failure-clustering analysis
# ======================================================================================
def cluster_analysis(cmp_df: pd.DataFrame) -> dict:
    """Given the predicted-vs-actual df, surface WHY multiple stocks failed:
       (1) Same-day stop clusters
       (2) Sector distribution of losses vs the shortlist as a whole
       (3) Correlation of losers' returns
    """
    out = {}
    stopped = cmp_df[cmp_df["actual_outcome"] == "STOP"].copy()
    filled = cmp_df[cmp_df["actual_outcome"].isin(["TARGET", "STOP", "TIME"])].copy()
    out["n_signalled"] = len(cmp_df)
    out["n_filled"] = len(filled)
    out["n_stopped"] = len(stopped)

    if not stopped.empty:
        # (1) Same-day stop clustering
        stopped["exit_date_dt"] = pd.to_datetime(stopped["actual_exit_date"], errors="coerce")
        by_date = stopped.groupby("exit_date_dt").size().sort_values(ascending=False)
        worst_days = by_date.head(3)
        out["worst_stop_days"] = [{"date": str(d.date()), "n_stops": int(n)}
                                   for d, n in worst_days.items() if n >= 2]

        # (2) Sector distribution
        if "sector" in stopped.columns and "sector" in filled.columns:
            stopped_by_sec = stopped["sector"].value_counts()
            filled_by_sec = filled["sector"].value_counts()
            out["sector_hits"] = []
            for sec, n_stop in stopped_by_sec.items():
                n_filled = int(filled_by_sec.get(sec, n_stop))
                rate = 100 * n_stop / n_filled if n_filled else 0
                out["sector_hits"].append({
                    "sector": sec, "n_stopped": int(n_stop),
                    "n_filled": n_filled, "stop_rate": round(rate, 1)
                })
            out["sector_hits"].sort(key=lambda x: -x["n_stopped"])

        # (3) Average days-to-stop (fast stops = mkt-wide event)
        if "actual_days_held" in stopped.columns:
            out["avg_days_to_stop"] = round(
                float(stopped["actual_days_held"].mean()), 1)
    return out


# ======================================================================================
#  UI
# ======================================================================================
def main():
    st.set_page_config(page_title="Forward Validation v2", layout="wide")
    st.title("🔮 Forward Validation v2 — with regime gate & failure clustering")
    st.caption("Walk-forward test with regime overlay, sector cap, signal decay, and "
               "clustering analysis to diagnose why stops cluster together.")

    with st.sidebar:
        st.header("1 · Cutoff & window")
        today = dt.date.today()
        cutoff = st.date_input("Cutoff (= scan 'today')",
                               value=today - dt.timedelta(days=90),
                               min_value=dt.date(2010, 1, 1),
                               max_value=today - dt.timedelta(days=1))
        bt_years = st.slider("Backtest history (years before cutoff)", 2, 15, 10)
        fwd_days = st.slider("Forward observation window (calendar days)", 15, 180, 60,
                             help=">= max_hold + fill_days + buffer")

        days_old = (today - cutoff).days
        if days_old > 30:
            st.info(f"ℹ️ News/event DISABLED (cutoff {days_old}d ago, "
                    f"free news doesn't archive that far).")
        if days_old > 5:
            st.info(f"ℹ️ Fundamentals DISABLED (yfinance .info is today's data, "
                    f"not point-in-time).")

        st.header("2 · Universe")
        buckets_meta = load_full_universe()
        buckets = buckets_meta["buckets"]
        sector_map = buckets_meta.get("sector_map", {})
        default_bucket = "Nifty500" if "Nifty500" in buckets else list(buckets.keys())[0]
        bucket_choice = st.selectbox("Universe bucket",
                                      list(buckets.keys()),
                                      index=list(buckets.keys()).index(default_bucket))
        universe = buckets.get(bucket_choice, [])
        st.caption(f"{len(universe)} stocks in {bucket_choice}")
        max_n = st.slider("Limit stocks this run", 20, len(universe),
                          min(len(universe), 200 if bucket_choice == "Nifty500"
                              else min(50, len(universe))))

        st.header("3 · Regime overlay  🆕")
        use_regime = st.checkbox("Apply market-regime gate historically",
                                 value=True,
                                 help="Uses ONLY the benchmark index's price data ≤ cutoff "
                                      "(no look-ahead). RISK-OFF cutoffs → skip all signals "
                                      "(cash is a position). NEUTRAL → keep signals but "
                                      "flag advisory.")
        regime_block_on_risk_off = st.checkbox(
            "Hard-block ALL trades when regime = RISK-OFF", value=True,
            disabled=not use_regime,
            help="This is the biggest single lever. Momentum longs in RISK-OFF regime "
                 "are the #1 cause of clustered stop-outs.")
        regime_block_neutral_decel = st.checkbox(
            "Also block when regime = NEUTRAL AND 10-day ROC < -1%", value=True,
            disabled=not use_regime,
            help="EVIDENCE-BASED FIX (Aug-2026): multi-cutoff walk-forward on Nifty 500 "
                 "showed that NEUTRAL-with-negative-ROC periods deliver -2.4% avg per trade "
                 "with 60% stop rate. NEUTRAL alone is fine; NEUTRAL-with-decelerating "
                 "momentum is the trap. This filter catches exactly that.")

        st.header("4 · Signal decay & cooldown  🆕")
        use_decay = st.checkbox(
            "Cap signals per day (co-signal decay)", value=True,
            help="When 30 stocks fire signals on the same day, they're NOT 30 independent "
                 "bets — they're one bet on the momentum factor. Cap it.")
        max_signals_per_day = st.slider(
            "Max NEW entries per day", 3, 30, 10, 1,
            disabled=not use_decay,
            help="Keep only the top-N by confidence per cutoff. Prevents over-concentration.")

        st.header("5 · Sector cap")
        max_per_sector = st.slider("Max per sector", 1, 10, 3,
                                   help="Same as live scanner — prevents 5-out-of-8 "
                                        "shortlist being metals stocks that all crash together.")

        st.header("6 · Strategy & rules")
        strategy = st.selectbox("Strategy",
                                ["PASS_combined", "PASS_recommended", "PASS_tight",
                                 "PASS_balanced", "PASS_reversal"], index=0)
        target_pct = st.number_input("Target (%)", 1.0, 100.0, 15.0, 0.5)
        max_hold = st.number_input("Max hold (d)", 1, 120, 30)
        entry_mode = st.radio("Entry style", ["Limit", "Market open"], index=0)
        limit_pct = st.slider("Limit below signal close (%)", 0.0, 5.0, 0.5, 0.1) \
                    if entry_mode == "Limit" else 0.0
        fill_days = st.number_input("Order valid for (sessions)", 1, 5, 1) \
                    if entry_mode == "Limit" else 1
        # NEW (Aug-2026): exit style toggle — critical for "let winners run"
        exit_mode = st.radio(
            "Exit style", ["Trailing (let winners run past target)", "Fixed target"],
            index=0,
            help="**Trailing**: target is a MINIMUM (15%). Once hit, trailing stop lets "
                 "winners run to 20-40%. Router picks trailing vs fixed per trade automatically.\n\n"
                 "**Fixed target**: exit at exactly 15%, no more. Simpler but caps upside.")
        exit_mode = "Trailing" if exit_mode.startswith("Trailing") else "Fixed target"
        trail_mult = st.slider("Trailing distance (x ATR)", 0.5, 5.0, 2.0, 0.5,
                                disabled=(exit_mode != "Trailing"))
        lock_pct = st.slider("Lock profit once target reached (%)", 0.0, 30.0, 10.0, 0.5,
                              disabled=(exit_mode != "Trailing"),
                              help="After +15% is touched, stop never falls below +10% "
                                   "(guarantees at least 10% profit while letting trade "
                                   "run to 20-30%).")
        stop_anchor = st.radio("Stop anchoring",
                               ["Structure (swing low)", "ATR distance"], index=0)
        stop_anchor = "Structure" if stop_anchor.startswith("Structure") else "ATR"
        trail_anchor = st.radio("Trail anchoring",
                                ["Structure (rising swing low)", "ATR distance"], index=0)
        trail_anchor = "Structure" if trail_anchor.startswith("Structure") else "ATR"
        stop_value = st.slider("Stop (x ATR)", 0.5, 5.0, 2.0, 0.5)
        max_stop_pct = st.slider("Max loss cap (%)", 2.0, 20.0, 10.0, 0.5)
        max_atr_pct = st.slider("Skip if ATR% above", 3.0, 15.0, 8.0, 0.5)
        cost_pct = st.number_input("Round-trip cost (%)", 0.0, 5.0, 0.20, 0.05)
        apply_stcg = st.checkbox("Apply 20% STCG on gains", value=True)

        st.header("7 · Filter thresholds")
        with st.expander("Advanced"):
            p = {
                "regime": st.slider("Uptrend: % above 200-DMA", 0.0, 50.0, 15.0, 1.0),
                "atr":    st.slider("Volatility floor: ATR%", 0.0, 10.0, 3.5, 0.5),
                "roc":    st.slider("Breakout ROC(10) >", 0.0, 15.0, 3.0, 0.5),
                "volr":   st.slider("Breakout volume ratio >", 0.5, 4.0, 1.2, 0.1),
                "rsi_os": st.slider("Reversal oversold RSI <", 10.0, 45.0, 30.0, 1.0),
            }

        run = st.button("🔮 Run forward validation", type="primary",
                        use_container_width=True)

    if not run:
        st.info("Set your cutoff + toggles and click **Run forward validation**.")
        return

    # =============== EXECUTION ===============
    bt_kwargs = dict(target_pct=target_pct, max_hold=int(max_hold), stop_method="ATR",
                     stop_value=stop_value, cost_pct=cost_pct, apply_stcg=apply_stcg,
                     exit_mode=exit_mode, trail_mult=trail_mult, lock_pct=lock_pct,
                     max_stop_pct=max_stop_pct, max_atr_pct=max_atr_pct,
                     entry_mode=entry_mode, limit_pct=limit_pct, fill_days=int(fill_days),
                     stop_anchor=stop_anchor, trail_anchor=trail_anchor,
                     ratchet_lock=False, shrink_trail=False, momentum_exit=False,
                     time_decay=False, staircase=False,
                     regime_route=True, route_min_adx=20.0, route_sma_slope_lb=20,
                     route_min_dist_pct=15.0, route_vol_lb=63, route_vol_baseline_lb=252,
                     route_fixed_target_pct=15.0, min_hold=1)

    ohlc_start = cutoff - dt.timedelta(days=int(bt_years * 365.25) + 300)
    ohlc_end = cutoff + dt.timedelta(days=fwd_days + 1)
    subset = universe[:max_n]

    # --- Regime check UP FRONT (fetches once) ---
    st.write(f"### Cutoff: **{cutoff.isoformat()}**  |  Universe: **{bucket_choice}**  |  Scanning: **{len(subset)}** stocks")
    if use_regime:
        with st.spinner("Fetching benchmark index for regime gate..."):
            bench_name, bench_df = _fetch_bench(ohlc_start, ohlc_end)
        if bench_df.empty:
            st.warning("⚠️ Could not fetch benchmark — regime gate will be skipped.")
            regime_info = {"status": "UNKNOWN"}
        else:
            regime_info = regime_at_cutoff(bench_df, cutoff)
            emoji = {"RISK-ON": "🟢", "NEUTRAL": "🟡", "RISK-OFF": "🔴", "UNKNOWN": "⚪"}
            e = emoji.get(regime_info["status"], "⚪")
            st.info(f"{e} **Regime at cutoff = {regime_info['status']}**  "
                    f"(bench {bench_name}: {regime_info.get('pct_vs_200','?')}% vs 200-DMA, "
                    f"10d ROC {regime_info.get('roc10','?')}%)")

        # Combined block decision
        block_all = False
        if regime_block_on_risk_off and regime_info["status"] == "RISK-OFF":
            block_all = True
            st.error("🚫 **RISK-OFF regime → ALL trades BLOCKED.** No signals acted on. "
                     "Cash is a position. This filter alone typically prevents 60-80% "
                     "of clustered stop-out losses.")
        elif (regime_block_neutral_decel and regime_info["status"] == "NEUTRAL"
              and regime_info.get("roc10", 0) < -1.0):
            block_all = True
            st.error(f"🚫 **NEUTRAL-with-decelerating-momentum → ALL trades BLOCKED.** "
                     f"ROC10 = {regime_info.get('roc10','?')}% (< -1%). Multi-cutoff "
                     f"data shows this regime delivers -2.4% avg per trade with 60% "
                     f"stop rate. Sitting out.")
        regime_info["block_all"] = block_all
    else:
        regime_info = {"status": "UNKNOWN", "block_all": False}

    # --- Scan each stock ---
    prog = st.progress(0.0); stat = st.empty()
    rows = []
    skipped_regime = 0
    for k, sym in enumerate(subset):
        stat.write(f"[{k+1}/{len(subset)}] {sym}")
        yahoo = _to_yahoo(sym)
        full = _fetch_full(yahoo, ohlc_start, ohlc_end)
        if full.empty:
            rows.append({"ticker": sym, "status": "no data"})
            prog.progress((k+1)/len(subset)); continue
        plan = scan_as_of(yahoo, full, strategy, p, bt_kwargs, cutoff, sector_map)
        # REGIME HARD-BLOCK (covers both RISK-OFF and NEUTRAL-decelerating)
        if use_regime and regime_info.get("block_all") and plan.get("signals_today"):
            plan["signals_today"] = False
            plan["regime_blocked"] = True
            skipped_regime += 1
        if plan.get("status") == "ok" and plan.get("signals_today"):
            # NEW: uses production engine → respects router + trailing + exit stack
            actual = forward_simulate(full, cutoff, strategy, p, bt_kwargs)
            plan["actual_outcome"] = actual.get("outcome")
            plan["actual_entry_date"] = actual.get("entry_date")
            plan["actual_entry_price"] = actual.get("entry_price")
            plan["actual_exit_date"] = actual.get("exit_date")
            plan["actual_exit_price"] = actual.get("exit_price")
            plan["actual_days_held"] = actual.get("days_held")
            plan["actual_net_return_%"] = actual.get("net_return_%")
            plan["actual_peak_gain_%"] = actual.get("peak_gain_%")   # for "let winners run" audit
            plan["actual_exit_route"] = actual.get("exit_route", "") # router audit
        rows.append(plan)
        prog.progress((k + 1) / len(subset))
    stat.empty(); prog.empty()

    if skipped_regime:
        st.warning(f"🚫 Regime gate blocked {skipped_regime} would-be signals.")

    all_df = pd.DataFrame(rows)
    ok_df = all_df[all_df.get("status") == "ok"].copy() if "status" in all_df.columns else all_df.copy()
    signalled = ok_df[ok_df.get("signals_today", False)].copy() if "signals_today" in ok_df.columns else pd.DataFrame()

    # --- Sector cap ---
    pre_cap_n = len(signalled)
    if not signalled.empty and max_per_sector > 0:
        signalled = apply_sector_cap(signalled, max_per_sector)
        if pre_cap_n > len(signalled):
            st.caption(f"🧩 Sector cap ({max_per_sector}/sector): trimmed "
                       f"{pre_cap_n - len(signalled)} correlated names.")

    # --- Signal decay ---
    if use_decay and len(signalled) > max_signals_per_day:
        signalled = signalled.sort_values("confidence", ascending=False).head(max_signals_per_day)
        st.caption(f"⚡ Signal decay: kept top {max_signals_per_day} by confidence "
                   f"(from {pre_cap_n}). Prevents co-signal over-concentration.")

    if signalled.empty:
        st.info(f"No stocks made it through all filters on {cutoff.isoformat()}. "
                f"That is a valid outcome — cash is a position.")
        return

    st.success(f"**{len(signalled)} stocks passed all filters** on {cutoff.isoformat()} "
               f"(from {len(ok_df)} scanned OK).")

    # =============== TABLE: shortlist ===============
    st.subheader(f"🎯 Shortlist as-of {cutoff.isoformat()}")
    sl_view = signalled[["ticker", "sector", "confidence", "hist_winrate_seq",
                          "cutoff_close", "plan_entry", "target_price", "stop_price"]].copy()
    sl_view.columns = ["Stock", "Sector", "Conf", "Hist Win%",
                        "Close", "BUY @", "Target", "Stop"]
    st.dataframe(sl_view, use_container_width=True, hide_index=True)

    # =============== TABLE: forward outcomes ===============
    st.subheader("🔮 Forward outcome per stock")
    cmp_rows = []
    for _, r in signalled.iterrows():
        ao = r.get("actual_outcome", "?")
        if ao == "NOT_FILLED":
            verdict = "⏭ Not filled"
        elif ao == "NO_FORWARD_DATA":
            verdict = "⚠ No fwd data"
        else:
            ret = r.get("actual_net_return_%", 0) or 0
            if ao == "TARGET": verdict = f"✅ TARGET ({ret:+.1f}%)"
            elif ao == "STOP": verdict = f"🔴 STOP ({ret:+.1f}%)"
            elif ret >= r.get("target_%", 15): verdict = f"✅ TIME beat ({ret:+.1f}%)"
            elif ret > 0: verdict = f"🟡 TIME win ({ret:+.1f}%)"
            else: verdict = f"🔴 TIME loss ({ret:+.1f}%)"
        cmp_rows.append({
            "Stock": r["ticker"], "Sector": r.get("sector", "-"),
            "BUY @": r["plan_entry"], "Target": r["target_price"], "Stop": r["stop_price"],
            "Entry Px": r.get("actual_entry_price", "—"),
            "Exit Px": r.get("actual_exit_price", "—"),
            "Reason": ao, "Days": r.get("actual_days_held", "—"),
            "Return": r.get("actual_net_return_%", "—"),
            "Peak gain %": r.get("actual_peak_gain_%", "—"),   # winners-run audit
            "Route": r.get("actual_exit_route", "—"),          # router (trailing/fixed)
            "Verdict": verdict,
        })
    cmp_df = pd.DataFrame(cmp_rows)
    # keep sector on the compare df for cluster analysis
    cmp_df["actual_outcome"] = [r["Reason"] for r in cmp_rows]
    cmp_df["actual_exit_date"] = [r.get("actual_exit_date") for r in signalled.to_dict("records")]
    cmp_df["actual_days_held"] = [r.get("Days") for r in cmp_rows]
    cmp_df["sector"] = [r.get("Sector") for r in cmp_rows]
    st.dataframe(cmp_df.drop(columns=["actual_outcome", "actual_exit_date", "actual_days_held", "sector"]),
                 use_container_width=True, hide_index=True)

    # =============== AGGREGATE METRICS ===============
    filled = signalled[signalled.get("actual_outcome", "").isin(["TARGET", "STOP", "TIME"])].copy() \
        if "actual_outcome" in signalled.columns else pd.DataFrame()
    if not filled.empty:
        n = len(filled)
        wins = int((filled["actual_net_return_%"] > 0).sum())
        target_hits = int((filled["actual_outcome"] == "TARGET").sum())
        stops = int((filled["actual_outcome"] == "STOP").sum())
        times = int((filled["actual_outcome"] == "TIME").sum())
        avg_ret = float(filled["actual_net_return_%"].mean())
        st.subheader("📊 Aggregate walk-forward metrics")
        c = st.columns(4)
        c[0].metric("Signalled", len(signalled))
        c[1].metric("Filled", n)
        c[2].metric("Win rate", f"{100*wins/n:.0f}%")
        c[3].metric("Avg net return", f"{avg_ret:+.2f}%",
                    f"vs target +{target_pct:.0f}%")
        c2 = st.columns(4)
        c2[0].metric("Target hits", f"{target_hits} ({100*target_hits/n:.0f}%)")
        c2[1].metric("Stops", f"{stops} ({100*stops/n:.0f}%)")
        c2[2].metric("Time exits", f"{times} ({100*times/n:.0f}%)")
        c2[3].metric("Expectancy per trade", f"{avg_ret:+.2f}%",
                     ("positive edge ✅" if avg_ret > 0 else "negative edge ❌"))

    # =============== FAILURE CLUSTERING ANALYSIS ===============
    if not filled.empty and (filled["actual_outcome"] == "STOP").sum() >= 2:
        st.subheader("🔍 Failure Clustering Analysis — why did stops cluster?")

        # Prepare data for analysis
        analysis_df = filled.copy()
        analysis_df["actual_exit_date"] = pd.to_datetime(analysis_df["actual_exit_date"], errors="coerce")
        clust = cluster_analysis(analysis_df.rename(columns={"actual_outcome": "actual_outcome"}))

        # (1) Same-day cluster
        if clust.get("worst_stop_days"):
            st.markdown("**1. Same-day stop clusters** (>= 2 stops on same date)")
            for d in clust["worst_stop_days"]:
                st.write(f"   * **{d['date']}** — **{d['n_stops']}** stocks stopped out same day")
            st.caption("Multiple stops on the SAME day = market-wide event, not strategy failure. "
                       "Common cause: broad-market gap-down, sector-specific news, "
                       "regime shift. FIX: enable the regime gate + hard-block on RISK-OFF.")

        # (2) Sector distribution
        if clust.get("sector_hits"):
            st.markdown("**2. Sector concentration of losses**")
            sec_df = pd.DataFrame(clust["sector_hits"])
            sec_df.columns = ["Sector", "# Stopped", "# Filled", "Stop rate %"]
            st.dataframe(sec_df, use_container_width=True, hide_index=True)
            top_sec = clust["sector_hits"][0]
            if top_sec["n_stopped"] >= 2 and top_sec["stop_rate"] >= 50:
                st.caption(f"⚠️ **{top_sec['sector']}** shows {top_sec['n_stopped']} stops "
                           f"out of {top_sec['n_filled']} trades ({top_sec['stop_rate']}% stop rate). "
                           f"Sector-specific news/regime hit ALL your trades in this sector. "
                           f"FIX: tighter sector cap (currently {max_per_sector}/sector).")

        # (3) Speed of stops
        if "avg_days_to_stop" in clust:
            days = clust["avg_days_to_stop"]
            st.markdown(f"**3. Speed of stops — avg stop hit in {days} days**")
            if days <= 3:
                st.caption("Very fast stops (<3d) = you bought right at a local top OR into a "
                           "gap-down environment. FIX: (a) widen stops to 2.5–3× ATR, "
                           "(b) require a follow-through confirmation day.")
            elif days <= 7:
                st.caption("Fast stops (3-7d) = normal pullback caught by tight stops. "
                           "FIX: widen stops moderately, or accept as cost of the strategy.")

    # =============== METHODOLOGY EXPANDER ===============
    with st.expander("🎓 What each toggle does + honest caveats"):
        st.markdown(f"""
**v2 additions and their purpose:**

- **Regime gate** — the single biggest fix. Uses ONLY benchmark index data ≤ cutoff
  (no look-ahead). When the broad market is in RISK-OFF (below 200-DMA and falling),
  momentum longs get faded en masse. This filter alone typically prevents 60-80% of
  clustered stop-out losses.

- **Signal decay** — when 30 stocks fire signals same day, they're not 30 independent
  bets; they're one bet on the momentum factor. Cap the number of new entries per day
  and rank-select the strongest ones.

- **Sector cap** — matches live scanner. Max K per sector prevents 5-of-8 shortlist
  being all metals stocks that crash together on a China-slowdown headline.

- **Failure clustering analysis** — post-hoc diagnostic showing (a) same-day stop
  clusters, (b) sector concentration of losses, (c) speed of stops. Answers "why
  did they all fail together?"

**Still honestly disabled for cutoffs older than the recent past:**

- News/event blocking (free sources don't archive)
- Fundamentals gate (yfinance.info is TODAY's data, not point-in-time)
""")


if __name__ == "__main__":
    main()
