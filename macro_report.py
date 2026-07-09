# -*- coding: utf-8 -*-
# macro_report.py
"""
Macro Report - Output & Historical Analysis
────────────────────────────────────────────
Print functions and historical analysis for MacroAnalyzer.

Usage:
    from macro_analyzer import MacroAnalyzer
    from macro_report   import print_macro_snapshot, print_macro_history, print_crisis_analysis

    ma = MacroAnalyzer()
    ma.load(start="2000-01-01")

    # Current snapshot
    snap = ma.get_snapshot()
    print_macro_snapshot(snap)

    # Historical crisis analysis
    print_crisis_analysis(ma)

    # Phase distribution
    print_phase_stats(ma)

    # Asset performance per cycle phase (backtesting)
    print_phase_performance(ma)
"""

import pandas as pd
import numpy as np
from typing import Optional
from macro_analyzer import MacroAnalyzer, MacroSnapshot


# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

PHASE_EMOJIS = {
    "early_expansion":   "🟢",
    "late_expansion":    "🟡",
    "early_contraction": "🟠",
    "late_contraction":  "🔴",
    "unknown":           "⚪",
}

RISK_EMOJIS = {
    "risk_on":  "🟢",
    "neutral":  "🟡",
    "risk_off": "🔴",
}

INFLATION_EMOJIS = {
    "low":         "🔵",
    "rising":      "🟠",
    "high":        "🔴",
    "falling":     "🟢",
    "stagflation": "🔴",
}

DIRECTION_EMOJIS = {
    "bullish": "↑",
    "bearish": "↓",
    "neutral": "→",
}

# Known historical crises / events for comparison
HISTORICAL_EVENTS = {
    "2008 GFC":            ("2007-10-01", "2009-03-01"),
    "Euro crisis":         ("2011-07-01", "2012-07-01"),
    "China shock":         ("2015-08-01", "2016-02-01"),
    "2018 Q4 selloff":     ("2018-10-01", "2018-12-31"),
    "COVID crash":         ("2020-02-01", "2020-04-01"),
    "2022 bear market":    ("2022-01-01", "2022-10-01"),
}


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def _fmt_pct(val, decimals=1) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val*100:+.{decimals}f}%" if abs(val) < 10 else f"{val:+.{decimals}f}%"


def _fmt_float(val, decimals=2) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val:.{decimals}f}"


def _fmt_zscore(z) -> str:
    if z is None or (isinstance(z, float) and np.isnan(z)):
        return "N/A"
    bar = "█" * min(int(abs(z)), 5)
    sign = "+" if z > 0 else ""
    return f"{sign}{z:.2f}sd {bar}"


def _fmt_percentile(p) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "N/A"
    marker = "▼" if p < 20 else ("▲" if p > 80 else "►")
    return f"{p:.0f}th {marker}"


def _phase_label(phase: str) -> str:
    labels = {
        "early_expansion":   "Early Expansion",
        "late_expansion":    "Late Expansion",
        "early_contraction": "Early Contraction",
        "late_contraction":  "Late Contraction",
        "unknown":           "Unknown",
    }
    return labels.get(phase, phase)


# ─────────────────────────────────────────────────────────────
# MAIN SNAPSHOT REPORT
# ─────────────────────────────────────────────────────────────

def print_macro_snapshot(snap: MacroSnapshot) -> None:
    """
    Prints full macro snapshot for a given date.
    """
    W = 65

    # ── Header ───────────────────────────────────────────────
    print("\n" + "═" * W)
    print(f"  MACRO ENVIRONMENT - {snap.date.strftime('%d/%m/%Y')}")
    print("═" * W)

    phase_e = PHASE_EMOJIS.get(snap.cycle_phase, "⚪")
    risk_e  = RISK_EMOJIS.get(snap.risk_mode, "🟡")
    inf_e   = INFLATION_EMOJIS.get(snap.inflation_env, "🟡")

    print(f"\n  {phase_e} Cycle Phase:    {_phase_label(snap.cycle_phase)}")
    print(f"  {risk_e} Risk Mode:      {snap.risk_mode.replace('_', ' ').upper()}")
    print(f"  {inf_e} Inflation Env:  {snap.inflation_env.upper()}")
    print(f"  💵 Dollar Trend:   {snap.dollar_trend.upper()}")
    print(f"  📊 Macro Score:    {snap.macro_score:.1f} / 10")

    # ── Key Ratios ────────────────────────────────────────────
    print(f"\n  {'─'*60}")
    print(f"  KEY RATIOS")
    print(f"  {'─'*60}")

    def _ratio_line(label, val, extra=""):
        v = _fmt_float(val, 4) if val and abs(val) < 1 else _fmt_float(val, 2)
        print(f"  {label:<28} {v:>10}  {extra}")

    _ratio_line("Copper / Gold",    snap.copper_gold)
    _ratio_line("Gold / Silver",    snap.gold_silver)
    _ratio_line("Gold / Oil",       snap.gold_oil)
    _ratio_line("Yield Curve (10Y-3M)", snap.yield_curve,
                "⚠️  INVERTED" if snap.yield_curve and snap.yield_curve < 0 else "")
    _ratio_line("Credit (HYG/LQD)", snap.credit_spread)
    _ratio_line("VIX",              snap.vix)
    _ratio_line("DXY ROC (20d)",    snap.dxy_roc,
                _fmt_pct(snap.dxy_roc) if snap.dxy_roc else "")

    # ── Earnings Yield Gap ───────────────────────────────────
    if snap.eyg is not None:
        eyg_pct = snap.eyg * 100
        ey_pct  = (snap.earnings_yield or 0) * 100
        sig     = (snap.eyg_signal or "neutral")
        sig_map = {
            "bullish":    ("🟢", "BULLISH EQUITY REGIME",
                           "Equities cheap -- investors well compensated"),
            "neutral":    ("🟡", "NEUTRAL / STABLE",
                           "Valuation equilibrium -- orderly market"),
            "compressed": ("🟠", "COMPRESSED",
                           "Valuations compressed -- increasing risk"),
            "high_risk":  ("🔴", "HIGH RISK",
                           "Bonds more attractive -- potential crisis signal"),
            "crisis":     ("🔴", "CRISIS SIGNAL",
                           "Negative EYG -- bonds dominating equities"),
        }
        emoji, label, interp = sig_map.get(sig, ("🟡", sig.upper(), ""))
        print(f"\n  {'─'*60}")
        print(f"  EARNINGS YIELD GAP (EYG)")
        print(f"  {'─'*60}")
        print(f"  {emoji} EYG: {eyg_pct:+.2f}%  |  {label}")
        y10_display = getattr(snap, "yield_10y_abs", None) or snap.yield_curve or 0
        print(f"  SPY P/E: {snap.spy_pe:.1f}x  |  "
              f"Earnings Yield: {ey_pct:.2f}%  |  "
              f"10Y Yield (abs): {y10_display:.2f}%")
        print(f"  Interpretation: {interp}")
        print(f"  Zones: >3% Bullish | 1.5-2.5% Neutral | <0.5% Compressed | <0% Crisis")

    # ── ISM Manufacturing ────────────────────────────────────
    if snap.ism is not None:
        ism_emoji = ("🟢" if snap.ism > 55 else
                     "🟡" if snap.ism > 50 else
                     "🟠" if snap.ism > 45 else "🔴")
        sig = (snap.ism_signal or "").replace("_", " ").upper()
        print(f"\n  {'─'*60}")
        print(f"  ISM MANUFACTURING (FRED)")
        print(f"  {'─'*60}")
        print(f"  {ism_emoji} PMI: {snap.ism:.1f}  |  Signal: {sig}")
        print(f"  Threshold: >55 Strong | >50 Expanding | <50 Contracting | <45 Weak")

    # ── TED Spread / FRA-OIS ──────────────────────────────────
    ted_val = snap.ted_spread or snap.fra_ois_spread
    if ted_val is not None:
        ted_emoji = ("🟢" if ted_val < 0.5  else
                     "🟡" if ted_val < 1.0  else
                     "🟠" if ted_val < 2.0  else "🔴")
        label = "TED Spread" if snap.ted_spread else "FRA-OIS Spread"
        sig   = (snap.ted_signal or "").upper()
        print(f"\n  {'─'*60}")
        print(f"  BANK SYSTEMIC RISK ({label})")
        print(f"  {'─'*60}")
        print(f"  {ted_emoji} {label}: {ted_val:.2f}%  |  Signal: {sig}")
        print(f"  Threshold: <0.50% Normal | <1.0% Watch | <2.0% Elevated | >2.0% Crisis")

    # ── Real Yields (DFII10 / TIP proxy) ─────────────────────
    ry_val = getattr(snap, "real_yield_10y", None)
    ry_z   = getattr(snap, "real_yield_zscore_100d", None)
    if ry_val is not None:
        ry_emoji = ("🔴" if ry_z and ry_z > 1.5 else
                    "🟢" if ry_z and ry_z < -1.5 else "🟡")
        ry_sig   = ("TIGHTENING" if ry_z and ry_z > 1.5 else
                    "EASING"     if ry_z and ry_z < -1.5 else "NEUTRAL")
        print(f"\n  {'─'*60}")
        print(f"  REAL YIELDS (DFII10 - FRED)")
        print(f"  {'─'*60}")
        print(f"  {ry_emoji} 10Y Real Yield: {ry_val:.2f}%  |  "
              f"Z-score (100d): {(ry_z or 0):+.2f}  |  Signal: {ry_sig}")
        print(f"  Dynamic Discount Factor: Z > 1.5 = liquidity tightening")

    # ── Probit Recession Probability ─────────────────────────
    if snap.probit_recession_prob is not None:
        prob  = snap.probit_recession_prob
        sig   = (snap.probit_signal or "unknown").upper()
        emoji = ("🟢" if prob < 15 else
                 "🟡" if prob < 30 else
                 "🟠" if prob < 50 else "🔴")
        print(f"\n  {'─'*60}")
        print(f"  RECESSION PROBABILITY (Estrella & Mishkin)")
        print(f"  {'─'*60}")
        print(f"  {emoji} P(recession 12m): {prob:.1f}%  |  Signal: {sig}")
        bar_len = int(prob / 5)
        bar = "█" * bar_len + "-" * (20 - bar_len)
        print(f"  [{bar}] {prob:.1f}%")
        print(f"  Thresholds: <15% Low | 15-30% Elevated | 30-50% High | >50% Critical")

    # ── Signal Breakdown ──────────────────────────────────────
    print(f"\n  {'─'*60}")
    print(f"  SIGNAL BREAKDOWN")
    print(f"  {'─'*60}")
    print(f"  {'Signal':<24} {'Dir':>4}  {'Z-score':>10}  {'Percentile':>12}  Note")
    print(f"  {'-'*60}")

    for sig_name, sig in snap.signals.items():
        dir_e  = DIRECTION_EMOJIS.get(sig.direction, "→")
        z_str  = _fmt_zscore(sig.zscore)
        p_str  = _fmt_percentile(sig.percentile)
        print(f"  {sig.name:<24} {dir_e:>4}  {z_str:>10}  {p_str:>12}  {sig.label}")

    # ── Asset Rotation ────────────────────────────────────────
    print(f"\n  {'─'*60}")
    print(f"  ASSET ROTATION - {_phase_label(snap.cycle_phase)}")
    print(f"  {'─'*60}")

    print(f"\n  ✅ Favor sectors:")
    for s in snap.favor_sectors:
        print(f"     • {s}")

    print(f"\n  ❌ Avoid sectors:")
    for s in snap.avoid_sectors:
        print(f"     • {s}")

    print(f"\n  📦 Suggested Allocation:")
    for asset, weight in snap.asset_allocation.items():
        bar = "█" * int(weight * 20)
        print(f"     {asset:<12} {weight*100:>5.1f}%  {bar}")

    # ── Trading Integration Note ──────────────────────────────
    print(f"\n  {'─'*60}")
    print(f"  TRADING INTEGRATION HINTS")
    print(f"  {'─'*60}")
    _print_trading_hints(snap)

    print("\n" + "═" * W + "\n")


def _print_trading_hints(snap: MacroSnapshot) -> None:
    """Prints hints for the trading notebook."""

    hints = []

    if snap.cycle_phase == "early_contraction":
        hints.append("⚠️  Increased selectivity — only high composite score (>7.5)")
        hints.append("⚠️  Avoid cyclicals — Financials, Industrials")

    if snap.cycle_phase == "late_contraction":
        hints.append("🛑 Defensive mode — reduce equity exposure")
        hints.append("🛑 Check regime_detector: likely Bear mode")

    if snap.yield_curve and snap.yield_curve < 0:
        hints.append(f"⚠️  Yield curve inverted ({snap.yield_curve:.2f}%) — elevated recession risk")

    if snap.vix and snap.vix > VIX_HIGH:
        hints.append(f"⚠️  VIX={snap.vix:.0f} - volatility elevated, tight stops")

    if snap.risk_mode == "risk_on":
        hints.append("✅ Risk-on environment — favors growth/momentum strategies")

    if snap.inflation_env in ("rising", "high"):
        hints.append("📈 Inflationary env — Energy/Materials/Gold historically outperform")
        hints.append("📈 Growth/Tech under pressure from rising real yields")

    if snap.dollar_trend == "strong":
        hints.append("💵 Strong DXY — pressure on commodities and international stocks")

    if snap.dollar_trend == "weak":
        hints.append("💵 Weak DXY — tailwind for commodities, gold, EM")

    if not hints:
        hints.append("✅ Macro environment favorable — no notable warnings")

    for h in hints:
        print(f"  {h}")


# Import here to avoid circular imports
try:
    from macro_analyzer import VIX_HIGH
except (ModuleNotFoundError, ImportError):
    VIX_HIGH = 25


# ─────────────────────────────────────────────────────────────
# ASSET PERCENTILES (10Y rolling)
# ─────────────────────────────────────────────────────────────

def print_asset_percentiles(ma: "MacroAnalyzer") -> None:
    """
    Prints 10Y rolling percentiles for all assets and signals.
    Shows where each asset/ratio stands vs its 10-year history.
    Called separately from print_macro_snapshot:
        print_macro_snapshot(snap)
        print_asset_percentiles(ma)
    """
    inds = ma.get_indicators()
    if inds is None or inds.empty:
        print("  ⚠️  No indicators available.")
        return

    last = inds.iloc[-1]
    W    = 65

    def _pct_bar(p) -> str:
        """Visual percentile bar."""
        if p is None or (hasattr(p, '__class__') and p.__class__.__name__ == 'float' and p != p):
            return "N/A"
        p = float(p)
        filled = int(p / 5)   # 20 segments for 0-100
        bar    = "█" * filled + "-" * (20 - filled)
        return f"{bar} {p:>5.1f}th"

    def _emoji(p, invert=False) -> str:
        """🟢 high, 🔴 low — or reversed if invert=True."""
        if p is None: return "⚪"
        p = float(p)
        high = p > 75
        low  = p < 25
        if invert:
            high, low = low, high
        return "🟢" if high else ("🔴" if low else "🟡")

    print("\n" + "═" * W)
    print(f"  ASSET PERCENTILES — 10Y ROLLING WINDOW")
    print(f"  {ma.get_history().index[-1].strftime('%d/%m/%Y')}")
    print("═" * W)
    print(f"  {'Asset / Signal':<26} {'Value':>10}  {'Percentile (10Y)':>8}")
    print(f"  {'─'*60}")

    # ── Commodities (raw price) ──────────────────────────────
    print(f"\n  {'COMMODITIES':}")
    rows_c = [
        ("Gold (GC=F spot)",   "gold_pct",      last.get("dxy"),        False, "gold"),
        ("Silver (SI=F spot)", "silver_pct",    None,                   False, "silver"),
        ("Copper (HG=F)",      "copper_pct",    None,                   False, "copper"),
        ("Oil (CL=F)",         "oil_pct",       None,                   False, "oil"),
    ]
    # Raw price percentiles — computed inline if not in indicators
    price_pcts = {
        "gold_pct":   last.get("gold_pct"),
        "silver_pct": None,
        "copper_pct": None,
        "oil_pct":    None,
    }
    # Fallback: compute from prices if available
    try:
        from macro_analyzer import _rolling_percentile, PERCENTILE_WINDOW
        p = ma._prices
        # silver_spot (SI=F) for correct percentile, fallback to SLV
        _silv_sym = "silver_spot" if "silver_spot" in p else "silver"
        for key, sym in [("silver_pct", _silv_sym),("copper_pct","copper"),("oil_pct","oil")]:
            if sym in p:
                price_pcts[key] = _rolling_percentile(p[sym], PERCENTILE_WINDOW).iloc[-1]
    except Exception:
        pass

    # Raw prices from ma._prices (not in indicators DataFrame)
    p = ma._prices
    # Spot prices for displayed values: GC=F/SI=F fallback to GLD/SLV
    _gp = p.get("gold_spot",   p.get("gold"))
    _sp = p.get("silver_spot", p.get("silver"))
    commodity_rows = [
        ("Gold",   price_pcts.get("gold_pct"),   _gp.iloc[-1] if _gp is not None else None, False),
        ("Silver", price_pcts.get("silver_pct"), _sp.iloc[-1] if _sp is not None else None, False),
        ("Copper", price_pcts.get("copper_pct"), p["copper"].iloc[-1] if "copper" in p else None, False),
        ("Oil",    price_pcts.get("oil_pct"),    p["oil"].iloc[-1]    if "oil"    in p else None, False),
    ]
    for label, pct, val, inv in commodity_rows:
        e   = _emoji(pct, inv)
        v   = f"{val:.2f}" if val is not None else "—"
        bar = _pct_bar(pct)
        print(f"  {e} {label:<24} {v:>10}  {bar}")

    # ── Ratios ───────────────────────────────────────────────
    print(f"\n  {'RATIOS & SIGNALS':}")
    ratio_rows = [
        ("Copper/Gold",        last.get("copper_gold_pct"),  last.get("copper_gold"),       False),
        ("Gold/Silver",        last.get("gold_silver_pct"),  last.get("gold_silver"),        False),  # low pct = risk appetite
        ("Gold/Oil",           last.get("gold_oil_pct"),     last.get("gold_oil"),           False),  # high pct = deflationary stress
        ("Yield Curve (10Y-3M)",last.get("yield_curve_pct"), last.get("yield_curve"),        False),
        ("Credit Spreads",     last.get("credit_spread_pct"),last.get("credit_spread_ratio"),False),
        ("Real Yields",
         last.get("real_yield_10y_pct") or last.get("tip_roc_pct"),
         last.get("real_yield_10y") or last.get("tips_roc_60"),
         True),
        ("EYG",                last.get("eyg_pct"),          last.get("eyg"),                False),
    ]
    for label, pct, val, inv in ratio_rows:
        e   = _emoji(pct, inv)
        v   = f"{val:.4f}" if val is not None and abs(float(val)) < 10 else \
              f"{val:.2f}" if val is not None else "—"
        bar = _pct_bar(pct)
        print(f"  {e} {label:<24} {v:>10}  {bar}")

    # ── Market Indicators ────────────────────────────────────
    print(f"\n  {'MARKET INDICATORS':}")
    market_rows = [
        ("VIX",                last.get("vix_pct"),          last.get("vix"),                True),
        ("DXY",                last.get("dxy_pct"),          last.get("dxy"),                False),
        ("TED Spread",         last.get("ted_spread_pct"),   last.get("ted_spread"),         True),
        ("SPY",                None,                         last.get("spy"),                False),
    ]
    # SPY price + percentile from ma._prices
    try:
        from macro_analyzer import _rolling_percentile, PERCENTILE_WINDOW
        p = ma._prices
        spy_price = p["spy"].iloc[-1] if "spy" in p else None
        spy_pct = _rolling_percentile(p["spy"], PERCENTILE_WINDOW).iloc[-1] if "spy" in p else None
        market_rows[3] = ("SPY", spy_pct, spy_price, False)
    except Exception:
        pass

    for label, pct, val, inv in market_rows:
        e   = _emoji(pct, inv)
        v   = f"{val:.2f}" if val is not None else "—"
        bar = _pct_bar(pct)
        print(f"  {e} {label:<24} {v:>10}  {bar}")

    # ── Inflation Classifier Inputs ──────────────────────────
    print(f"\n  {'INFLATION CLASSIFIER INPUTS':}")
    inf_rows = [
        ("Gold (price pct)",   price_pcts.get("gold_pct"),   (_gp.iloc[-1] if _gp is not None else None), False),
        ("Gold ROC 60d",       None,                          last.get("gold_roc_60"), False),
        ("Gold/Oil z-score",   None,                          last.get("gold_oil_zscore"), True),
        ("Cu/Gold z-score",    None,                          last.get("copper_gold_zscore"), False),
        ("10Y Yield ROC 60d",  None,                          last.get("yield_10y_roc_60"), False),
    ]
    for label, pct, val, inv in inf_rows:
        e   = _emoji(pct, inv) if pct is not None else "  "
        v   = f"{val:.4f}" if val is not None else "—"
        bar = _pct_bar(pct) if pct is not None else "(z-score / ROC — no percentile)"
        print(f"  {e} {label:<24} {v:>10}  {bar}")

    print("\n" + "═" * W + "\n")


# ─────────────────────────────────────────────────────────────
# HISTORICAL PHASE STATS
# ─────────────────────────────────────────────────────────────

def print_phase_stats(ma: MacroAnalyzer) -> None:
    """
    Prints phase distribution across full history.
    """
    history = ma.get_history()
    if history.empty:
        print("No data available.")
        return

    W = 65
    print(f"\n{'═'*W}")
    print(f"  HISTORICAL PHASE DISTRIBUTION")
    print(f"  {history.index[0].strftime('%Y-%m-%d')} -> {history.index[-1].strftime('%Y-%m-%d')}")
    print(f"  Total: {len(history)} trading days")
    print(f"{'═'*W}\n")

    phases = ["early_expansion", "late_expansion", "early_contraction", "late_contraction"]

    for phase in phases:
        count = (history["cycle_phase"] == phase).sum()
        pct   = count / len(history) * 100
        emoji = PHASE_EMOJIS.get(phase, "⚪")
        bar   = "█" * int(pct / 3)
        print(f"  {emoji} {_phase_label(phase):<22} {count:>5} days  {pct:>5.1f}%  {bar}")

    print(f"\n  RISK MODE")
    for mode in ["risk_on", "neutral", "risk_off"]:
        count = (history["risk_mode"] == mode).sum()
        pct   = count / len(history) * 100
        emoji = RISK_EMOJIS.get(mode, "🟡")
        bar   = "█" * int(pct / 3)
        print(f"  {emoji} {mode.replace('_', ' '):<22} {count:>5} days  {pct:>5.1f}%  {bar}")

    print(f"\n  INFLATION ENV")
    for env in ["low", "rising", "high", "falling"]:
        count = (history["inflation_env"] == env).sum()
        pct   = count / len(history) * 100
        emoji = INFLATION_EMOJIS.get(env, "🟡")
        bar   = "█" * int(pct / 3)
        print(f"  {emoji} {env:<22} {count:>5} days  {pct:>5.1f}%  {bar}")

    print(f"\n{'═'*W}\n")


# ─────────────────────────────────────────────────────────────
# CRISIS ANALYSIS
# ─────────────────────────────────────────────────────────────

def print_crisis_analysis(ma: MacroAnalyzer) -> None:
    """
    For each known crisis, shows macro conditions
    3 and 6 months before the crisis start.
    Useful for understanding what preceded each event.
    """
    history    = ma.get_history()
    indicators = ma.get_indicators()

    if history.empty:
        print("No data available.")
        return

    W = 65
    print(f"\n{'═'*W}")
    print(f"  CRISIS PRE-CONDITIONS ANALYSIS")
    print(f"  What signals showed BEFORE each crisis:")
    print(f"{'═'*W}")

    for event_name, (start_str, end_str) in HISTORICAL_EVENTS.items():
        start_ts = pd.Timestamp(start_str)
        end_ts   = pd.Timestamp(end_str)

        # Check if we have data for this period
        if start_ts < history.index[0] or start_ts > history.index[-1]:
            continue

        print(f"\n  {'─'*60}")
        print(f"  📌 {event_name.upper()}")
        print(f"     {start_str} -> {end_str}")
        print(f"  {'─'*60}")

        # Snapshot 6 months before
        for months_before, label in [(6, "6 months before"), (3, "3 months before"), (1, "1 month before")]:
            lookback_date = start_ts - pd.DateOffset(months=months_before)
            avail = history.index[history.index <= lookback_date]
            if avail.empty:
                continue
            actual = avail[-1]
            row    = history.loc[actual]

            phase_e = PHASE_EMOJIS.get(row["cycle_phase"], "⚪")
            risk_e  = RISK_EMOJIS.get(row["risk_mode"], "🟡")

            print(f"\n  [{label} - {actual.strftime('%Y-%m-%d')}]")
            print(f"    {phase_e} Phase:   {_phase_label(row['cycle_phase'])}")
            print(f"    {risk_e} Risk:    {row['risk_mode'].replace('_',' ').upper()}")
            print(f"    📊 Macro score: {row['macro_score']:.1f}")

            # Key ratios from indicators
            if actual in indicators.index:
                ind_row = indicators.loc[actual]
                if "yield_curve" in ind_row and pd.notna(ind_row["yield_curve"]):
                    inv = "⚠️ INVERTED" if ind_row["yield_curve"] < 0 else ""
                    print(f"    📉 Yield curve: {ind_row['yield_curve']:.2f}%  {inv}")
                if "copper_gold_zscore" in ind_row and pd.notna(ind_row["copper_gold_zscore"]):
                    print(f"    🔶 Cu/Gold z:   {ind_row['copper_gold_zscore']:.2f}sd")
                if "credit_spread_zscore" in ind_row and pd.notna(ind_row["credit_spread_zscore"]):
                    print(f"    📊 Credit z:    {ind_row['credit_spread_zscore']:.2f}sd")
                if "vix" in ind_row and pd.notna(ind_row["vix"]):
                    print(f"    😨 VIX:         {ind_row['vix']:.1f}")

        # Conditions during the crisis
        crisis_data = history[
            (history.index >= start_ts) & (history.index <= end_ts)
        ]
        if not crisis_data.empty:
            dominant_phase = crisis_data["cycle_phase"].mode().iloc[0]
            avg_score      = crisis_data["macro_score"].mean()
            print(f"\n  [During the crisis]")
            print(f"    Dominant phase: {_phase_label(dominant_phase)}")
            print(f"    Avg macro score: {avg_score:.1f}")

    print(f"\n{'═'*W}\n")


# ─────────────────────────────────────────────────────────────
# PHASE PERFORMANCE (BACKTESTING)
# ─────────────────────────────────────────────────────────────

def print_phase_performance(
    ma:             MacroAnalyzer,
    asset_prices:   dict[str, pd.Series] = None,
) -> None:
    """
    Computes and prints mean asset performance per macro phase.

    asset_prices: dict with {"SPY": series, "GLD": series, ...}
    If None -> uses the MacroAnalyzer internal prices.
    """
    history = ma.get_history()

    if history.empty:
        print("No data available.")
        return

    # Use internal prices if none provided
    if asset_prices is None:
        asset_prices = {
            "SPY (Equities)": ma._prices.get("spy"),
            "GLD (Gold)":     ma._prices.get("gold"),
            "TLT (Bonds)":    None,   # not available - skip
            "SLV (Silver)":   ma._prices.get("silver"),
            "XLE (Energy)":   ma._prices.get("xle"),
            "XLK (Tech)":     ma._prices.get("xlk"),
            "XLU (Utilities)": ma._prices.get("xlu"),
        }
        asset_prices = {k: v for k, v in asset_prices.items() if v is not None}

    if not asset_prices:
        print("No asset prices available for analysis.")
        return

    W = 75
    phases = ["early_expansion", "late_expansion", "early_contraction", "late_contraction"]

    print(f"\n{'═'*W}")
    print(f"  ASSET PERFORMANCE BY MACRO PHASE")
    print(f"  Mean annualized return per cycle phase")
    print(f"{'═'*W}\n")

    # Header
    asset_names = list(asset_prices.keys())
    header = f"  {'Phase':<22}"
    for name in asset_names:
        short = name.split("(")[0].strip()[:6]
        header += f" {short:>8}"
    print(header)
    print(f"  {'-'*(22 + 9*len(asset_names))}")

    # Build unified DataFrame: phase_label aligned with each asset
    # Uses merge_asof to handle different trading calendars
    phase_series = history["cycle_phase"].copy()

    for phase in phases:
        phase_mask  = history["cycle_phase"] == phase
        phase_dates = history.index[phase_mask]

        if len(phase_dates) < 20:
            continue

        row_str = f"  {PHASE_EMOJIS.get(phase,'')} {_phase_label(phase):<20}"

        for name, prices in asset_prices.items():
            if prices is None:
                row_str += f"  {'N/A':>6}"
                continue

            try:
                # Align prices with phase_series using reindex + ffill
                # Correctly handles different trading calendars
                aligned_phase = phase_series.reindex(
                    prices.index, method="ffill"
                ).fillna("unknown")

                # Find phase segments from aligned series
                is_in_phase = (aligned_phase == phase)
                transitions = is_in_phase.astype(int).diff().fillna(0)

                segment_starts = prices.index[transitions == 1].tolist()
                segment_ends   = prices.index[transitions == -1].tolist()

                # Handle if series starts or ends mid-phase
                if is_in_phase.iloc[0]:
                    segment_starts = [prices.index[0]] + segment_starts
                if is_in_phase.iloc[-1]:
                    segment_ends = segment_ends + [prices.index[-1]]

                ann_returns = []
                for start, end in zip(segment_starts, segment_ends):
                    segment = prices[start:end]
                    if len(segment) < 40:
                        continue
                    n_days = (segment.index[-1] - segment.index[0]).days
                    if n_days < 60:
                        continue
                    total_ret = (segment.iloc[-1] / segment.iloc[0]) - 1
                    # Annualize: (1 + ret)^(365/days) - 1
                    ann = (1 + total_ret) ** (365.0 / n_days) - 1
                    # Cap at ±200% to avoid extreme outliers
                    ann = max(-2.0, min(2.0, ann))
                    ann_returns.append(ann)

                if not ann_returns:
                    row_str += f"  {'N/A':>6}"
                    continue

                # Median -- robust to outliers
                annualized = float(np.median(ann_returns))
                sign = "+" if annualized > 0 else ""
                row_str += f"  {sign}{annualized*100:>5.1f}%"

            except Exception:
                row_str += f"  {'ERR':>6}"
                continue

        print(row_str)

    print(f"\n  Note: Annualized returns — rough estimate based on phase segments.")
    print(f"  Use get_history() for more detailed analysis.")
    print(f"\n{'═'*W}\n")


# ─────────────────────────────────────────────────────────────
# COMPACT SUMMARY (for daily use)
# ─────────────────────────────────────────────────────────────

def print_macro_summary(snap: MacroSnapshot) -> None:
    """
    Compact print — for use in trading notebook header.
    """
    phase_e = PHASE_EMOJIS.get(snap.cycle_phase, "⚪")
    risk_e  = RISK_EMOJIS.get(snap.risk_mode, "🟡")
    inf_e   = INFLATION_EMOJIS.get(snap.inflation_env, "🟡")

    curve_str = f"{snap.yield_curve:.2f}%" if snap.yield_curve else "N/A"
    inv_str   = " ⚠️" if snap.yield_curve and snap.yield_curve < 0 else ""

    print(f"\n{'─'*65}")
    print(f"  MACRO CONTEXT - {snap.date.strftime('%d/%m/%Y')}")
    print(f"  {phase_e} {_phase_label(snap.cycle_phase)}  |  "
          f"{risk_e} {snap.risk_mode.replace('_',' ').upper()}  |  "
          f"{inf_e} Inflation: {snap.inflation_env.upper()}")
    print(f"  Score: {snap.macro_score:.1f}/10  |  "
          f"Curve: {curve_str}{inv_str}  |  "
          f"VIX: {snap.vix:.0f}" if snap.vix else f"  Score: {snap.macro_score:.1f}/10  |  Curve: {curve_str}{inv_str}")
    print(f"  Favor: {', '.join(snap.favor_sectors[:3])}")
    print(f"{'─'*65}\n")


# ─────────────────────────────────────────────────────────────
# RECENT PHASE TIMELINE
# ─────────────────────────────────────────────────────────────

def print_phase_timeline(
    ma:     MacroAnalyzer,
    years:  int = 5,
) -> None:
    """
    Prints phase timeline for the last N years.
    Ideal for spotting transitions and patterns.
    """
    history = ma.get_history()
    if history.empty:
        return

    cutoff  = history.index[-1] - pd.DateOffset(years=years)
    recent  = history[history.index >= cutoff].copy()

    # Resample to monthly for cleaner timeline
    monthly = recent.resample("ME").agg({
        "cycle_phase":   lambda x: x.mode().iloc[0] if len(x) > 0 and len(x.mode()) > 0 else "unknown",
        "risk_mode":     lambda x: x.mode().iloc[0] if len(x) > 0 and len(x.mode()) > 0 else "neutral",
        "macro_score":   "mean",
        "yield_curve":   "last",
        "vix":           "mean",
    })

    W = 65
    print(f"\n{'═'*W}")
    print(f"  PHASE TIMELINE — last {years} years")
    print(f"{'═'*W}\n")
    print(f"  {'Date':<10}  {'Phase':<22}  {'Risk':<10}  {'Score':>5}  {'Curve':>7}  {'VIX':>5}")
    print(f"  {'-'*60}")

    prev_phase = None
    for date, row in monthly.iterrows():
        phase   = row["cycle_phase"]
        emoji   = PHASE_EMOJIS.get(phase, "⚪")
        risk_e  = RISK_EMOJIS.get(row["risk_mode"], "🟡")
        score   = row["macro_score"]
        curve   = row["yield_curve"]
        vix     = row["vix"]

        # Highlight phase changes
        marker = " ◄ CHANGE" if phase != prev_phase else ""

        curve_str = f"{curve:.2f}" if pd.notna(curve) else "N/A"
        vix_str   = f"{vix:.0f}"   if pd.notna(vix)   else "N/A"

        print(
            f"  {date.strftime('%Y-%m'):<10}  "
            f"{emoji} {_phase_label(phase):<20}  "
            f"{risk_e} {row['risk_mode'].replace('_',' '):<8}  "
            f"{score:>5.1f}  "
            f"{curve_str:>7}  "
            f"{vix_str:>5}"
            f"{marker}"
        )

        prev_phase = phase

    print(f"\n{'═'*W}\n")