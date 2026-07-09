# -*- coding: utf-8 -*-
"""
macro_charts_print.py
---------------------
Print-ready static charts for the Macro Analyzer.
- White background (Word / PDF / report friendly)
- Matplotlib only (static, no Plotly)
- Export to PNG at 300 dpi

33 charts total, organized in 9 groups matching macro_charts.py.

------------------------------------------
  NOTEBOOK USAGE
------------------------------------------

    from macro_charts_print import print_all

    ma   = MacroAnalyzer(); ma.load()
    snap = ma.get_snapshot()

    # Display only (no save)
    print_all(ma, snap, save=False)

    # Display and save to folder
    print_all(ma, snap, save=True, output_dir="charts_print")

------------------------------------------
  INDIVIDUAL CHARTS
------------------------------------------

  Group 1 — Current Snapshot
    fig = signal_radar_print(snap)               # 01  Signal radar
    fig = macro_gauge_print(snap)                # 02  Macro score gauge
    fig = allocation_pie_print(snap)             # 03  Asset allocation pie

  Group 2-3 — Phase Timeline & History
    fig = phase_gantt_print(ma)                  # 04a Phase Gantt — overview
    fig = phase_gantt_print(ma, highlight_phase="early_expansion")   # 04b
    fig = phase_gantt_print(ma, highlight_phase="late_expansion")    # 04c
    fig = phase_gantt_print(ma, highlight_phase="early_contraction") # 04d
    fig = phase_gantt_print(ma, highlight_phase="late_contraction")  # 04e
    fig = phase_timeline_print(ma)               # 05  Phase timeline
    fig = ratios_history_print(ma)               # 06  Commodity ratios
    fig = yield_curve_history_print(ma)          # 07  Yield curve
    fig = vix_history_print(ma)                  # 08  VIX history
    fig = macro_score_history_print(ma)          # 09  Macro score history

  Group 3 — Crisis Analysis
    fig = crisis_heatmap_print(ma)               # 10  Pre-crisis signal heatmap
    fig = zscore_bar_print(snap)                 # 11  Current z-scores bar

  Group 4 — Asset Performance
    fig = phase_performance_bar_print(ma)        # 12  Returns per cycle phase
    fig = sector_rotation_wheel_print(snap)      # 13  Sector rotation wheel

  Group 5-6 — Divergence, Recession & Volatility
    fig = credit_spread_print(ma)                # 14  HYG/LQD credit spread
    fig = cg_yield_divergence_print(ma)          # 15  Cu/Gold vs yield divergence
    fig = probit_recession_print(ma)             # 16  Probit recession probability
    fig = vix_mean_reversion_print(ma)           # 17  VIX mean reversion
    fig = fred_signals_print(ma)                 # 18  ISM + TED (requires FRED key)

  Group 7-8 — Metals, Yields & Valuation
    fig = gold_silver_chart_print(ma)            # 19  Gold & Silver
    fig = gold_silver_mean_reversion_print(ma)   # 20  Gold/Silver mean reversion
    fig = earnings_yield_gap_print(ma)           # 21  Earnings yield gap
    fig = dxy_history_print(ma)                  # 22  DXY dollar index
    fig = real_yields_print(ma)                  # 23  Real yields (DFII10 / TIP)
    fig = spy_sma200_print(ma)                   # 24  SPY vs SMA200
    fig = inflation_environment_print(ma)        # 25  Inflation environment

  Group 9 — Crisis Timelines  (run individually per crisis)
    fig = crisis_timeline_print(ma, "GFC 2008",   "2007-10-01", "2009-03-01")
    fig = crisis_timeline_print(ma, "Euro Crisis", "2011-07-01", "2012-07-01")
    fig = crisis_timeline_print(ma, "China Shock", "2015-08-01", "2016-02-01")
    fig = crisis_timeline_print(ma, "2018 Q4",    "2018-10-01", "2018-12-31")
    fig = crisis_timeline_print(ma, "COVID",      "2020-02-01", "2020-04-01")
    fig = crisis_timeline_print(ma, "Bear 2022",  "2022-01-01", "2022-10-01")

  Manual save (any individual chart)
    fig.savefig("my_chart.png", dpi=300, bbox_inches="tight")

------------------------------------------
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, Wedge
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as mticker
from typing import Optional

warnings.filterwarnings("ignore")

try:
    from IPython.display import display as _ipy_display
except ImportError:
    _ipy_display = None

from macro_analyzer import (
    MacroAnalyzer, MacroSnapshot,
    TED_NORMAL, TED_ELEVATED, TED_CRISIS,
    ISM_EXPANSION, ISM_STRONG, ISM_WEAK,
)


# -------------------------------------------------------------
#  PRINT PALETTE  (white background, print-friendly)
# -------------------------------------------------------------

PC = {
    # Backgrounds
    "bg":        "#FFFFFF",
    "panel":     "#F8F9FA",
    "grid":      "#E5E8EC",

    # Text
    "title":     "#1A2332",
    "label":     "#2C3E50",
    "tick":      "#555555",
    "muted":     "#7F8C8D",

    # Phases
    "early_exp":  "#27AE60",   # green
    "late_exp":   "#E67E22",   # orange
    "early_con":  "#D35400",   # dark orange
    "late_con":   "#C0392B",   # red

    # Signals
    "bullish":   "#1A7A4A",
    "bearish":   "#C0392B",
    "neutral":   "#7F8C8D",

    # Risk
    "risk_on":   "#1A7A4A",
    "risk_off":  "#C0392B",
    "neutral_r": "#E67E22",

    # Accents
    "blue":      "#2980B9",
    "purple":    "#8E44AD",
    "teal":      "#16A085",
    "yellow":    "#F39C12",
    "red":       "#C0392B",
    "green":     "#27AE60",

    # Lines
    "border":    "#BDC3C7",
    "zero":      "#2C3E50",
}

PHASE_COLORS_P = {
    "early_expansion":   PC["early_exp"],
    "late_expansion":    PC["late_exp"],
    "early_contraction": PC["early_con"],
    "late_contraction":  PC["late_con"],
    "unknown":           PC["muted"],
}

PHASE_LABELS = {
    "early_expansion":   "Early Expansion",
    "late_expansion":    "Late Expansion",
    "early_contraction": "Early Contraction",
    "late_contraction":  "Late Contraction",
}

CRISIS_EVENTS = {
    "GFC 2008":    ("2007-10-01", "2009-03-01"),
    "Euro Crisis": ("2011-07-01", "2012-07-01"),
    "China Shock": ("2015-08-01", "2016-02-01"),
    "2018 Q4":     ("2018-10-01", "2018-12-31"),
    "COVID":       ("2020-02-01", "2020-04-01"),
    "Bear 2022":   ("2022-01-01", "2022-10-01"),
}

PRINT_STYLE = {
    "figure.facecolor":  PC["bg"],
    "axes.facecolor":    PC["bg"],
    "axes.edgecolor":    PC["border"],
    "axes.labelcolor":   PC["label"],
    "axes.titlecolor":   PC["title"],
    "text.color":        PC["label"],
    "xtick.color":       PC["tick"],
    "ytick.color":       PC["tick"],
    "grid.color":        PC["grid"],
    "grid.alpha":        1.0,
    "legend.facecolor":  PC["bg"],
    "legend.edgecolor":  PC["border"],
    "font.family":       "Arial",
    "axes.spines.top":   False,
    "axes.spines.right": False,
}

# Figure width for A4/Letter (in inches): content width ~6.5"
FIG_W = 10.0
DPI   = 300


def _apply_style():
    plt.rcParams.update(PRINT_STYLE)


def _save(fig: plt.Figure, name: str, output_dir: str):
    """Saves the figure as PNG at 300dpi."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight",
                facecolor=PC["bg"], edgecolor="none")
    print(f"  OK Saved: {path}")


def _add_phase_bands(ax, history_weekly: pd.DataFrame):
    """Adds colored phase zones to the axis."""
    if "cycle_phase" not in history_weekly.columns:
        return
    ph = history_weekly["cycle_phase"].dropna()
    if ph.empty:
        return
    prev_phase = None
    start_date = None
    for date, phase in ph.items():
        if phase != prev_phase:
            if prev_phase is not None and start_date is not None:
                color = PHASE_COLORS_P.get(prev_phase, PC["muted"])
                ax.axvspan(start_date, date, alpha=0.08, color=color,
                           linewidth=0, zorder=0)
            start_date = date
            prev_phase = phase
    if prev_phase and start_date:
        color = PHASE_COLORS_P.get(prev_phase, PC["muted"])
        ax.axvspan(start_date, ph.index[-1], alpha=0.08, color=color,
                   linewidth=0, zorder=0)


def _add_crisis_markers(ax, history_start=None):
    """Adds vertical lines + labels for crisis events."""
    ymax = ax.get_ylim()[1]
    for name, (start_str, _) in CRISIS_EVENTS.items():
        s = pd.Timestamp(start_str)
        if history_start and s < pd.Timestamp(history_start):
            continue
        ax.axvline(s, color=PC["red"], linewidth=0.8,
                   linestyle="--", alpha=0.5, zorder=1)
        ax.text(s, ymax * 0.97, name, rotation=90,
                fontsize=6.5, color=PC["red"], alpha=0.7,
                va="top", ha="right")


def _fig_title(fig, text: str, subtitle: str = ""):
    """Unified title style for all charts."""
    if subtitle:
        fig.suptitle(f"{text}\n{subtitle}", fontsize=13, fontweight="bold",
                     color=PC["title"], y=1.01)
    else:
        fig.suptitle(text, fontsize=13, fontweight="bold",
                     color=PC["title"], y=1.01)


# -------------------------------------------------------------
#  CHART 1 — Signal Radar
# -------------------------------------------------------------

def signal_radar_print(snap: MacroSnapshot) -> plt.Figure:
    """Radar chart for current signals. White background."""
    _apply_style()
    signals = snap.signals
    if not signals:
        return None

    names, values, colors = [], [], []
    for key, sig in signals.items():
        z = sig.zscore or 0.0
        if np.isnan(z):
            z = 0.0
        short_name = (sig.name
                      .replace(" (TIP proxy)", "")
                      .replace(" (10Y-3M)", ""))
        names.append(short_name)
        if sig.direction == "bearish" and z > 0:
            z = -z
        elif sig.direction == "bullish" and z < 0:
            z = -z
        values.append(np.clip(z, -3, 3))
        colors.append(PC.get(sig.direction, PC["neutral"]))

    N = len(names)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    vals_norm = [(v + 3) / 6 for v in values]
    vals_norm += vals_norm[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True),
                           facecolor=PC["bg"])
    ax.set_facecolor(PC["panel"])

    # Risk mode -> fill color
    fill_color = (PC["risk_on"]  if snap.risk_mode == "risk_on"  else
                  PC["risk_off"] if snap.risk_mode == "risk_off" else
                  PC["neutral_r"])

    ax.fill(angles, vals_norm, alpha=0.18, color=fill_color)
    ax.plot(angles, vals_norm, linewidth=2.0, color=fill_color)

    # Dots per signal
    for i, (angle, val, c) in enumerate(zip(angles[:-1], vals_norm[:-1], colors)):
        ax.scatter(angle, val, color=c, s=80, zorder=5, edgecolors="white",
                   linewidths=0.5)

    # Neutral dashed ring
    ax.plot(angles, [0.5] * len(angles), color=PC["border"],
            linewidth=1.0, linestyle="--", alpha=0.8)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(names, size=8.5, color=PC["label"], fontweight="bold")
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["Extreme\nBear", "", "Neutral", "", "Extreme\nBull"],
                       size=7, color=PC["muted"])
    ax.set_ylim(0, 1)
    ax.grid(color=PC["grid"], alpha=0.8)
    ax.tick_params(pad=8)

    phase_label = PHASE_LABELS.get(snap.cycle_phase, snap.cycle_phase)
    _fig_title(fig,
               f"Signal Radar — {snap.date.strftime('%d/%m/%Y')}",
               f"{phase_label}  |  Macro Score: {snap.macro_score:.1f}/10  |  Risk: {snap.risk_mode.replace('_',' ').title()}")

    plt.tight_layout()
    return fig


# -------------------------------------------------------------
#  CHART 2 — Macro Gauge
# -------------------------------------------------------------

def macro_gauge_print(snap: MacroSnapshot) -> plt.Figure:
    """Gauge 0-10 for the macro score."""
    _apply_style()
    score = snap.macro_score

    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=PC["bg"])
    ax.set_facecolor(PC["bg"])
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.2, 1.3)
    ax.axis("off")

    # Background arc
    theta_bg = np.linspace(np.pi, 0, 300)
    ax.plot(np.cos(theta_bg), np.sin(theta_bg), color=PC["grid"],
            linewidth=32, solid_capstyle="round")

    # Color zones
    zones = [
        (0.00, 0.40, PC["red"]),
        (0.40, 0.75, PC["yellow"]),
        (0.75, 1.00, PC["green"]),
    ]
    for start, end, color in zones:
        t = np.linspace(np.pi - start * np.pi, np.pi - end * np.pi, 100)
        ax.plot(np.cos(t), np.sin(t), color=color, linewidth=30, alpha=0.75,
                solid_capstyle="butt")

    # Needle
    angle = np.pi - (score / 10) * np.pi
    nx, ny = 0.78 * np.cos(angle), 0.78 * np.sin(angle)
    ax.annotate("", xy=(nx, ny), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=PC["title"],
                                lw=2.5, mutation_scale=18))
    ax.add_patch(plt.Circle((0, 0), 0.06, color=PC["title"], zorder=5))

    # Score text
    ax.text(0, -0.10, f"{score:.1f}", ha="center", va="center",
            fontsize=28, fontweight="bold", color=PC["title"])
    ax.text(0, -0.25, "/ 10", ha="center", va="center",
            fontsize=12, color=PC["muted"])

    # Zone labels
    for x_pos, label, color in [
        (-0.95, "Bearish\n0–3", PC["red"]),
        (0.20,  "Mixed\n4–7.5", PC["yellow"]),
        (0.80,  "Bullish\n7.5–10", PC["green"]),
    ]:
        ax.text(x_pos, -0.12, label, ha="center", va="center",
                fontsize=7, color=color, fontweight="bold")

    # Tick marks
    for val in range(0, 11):
        a = np.pi - (val / 10) * np.pi
        r1, r2 = (0.85, 0.95) if val % 5 == 0 else (0.88, 0.93)
        ax.plot([r1 * np.cos(a), r2 * np.cos(a)],
                [r1 * np.sin(a), r2 * np.sin(a)],
                color=PC["tick"], linewidth=1.2 if val % 5 == 0 else 0.6)
        if val % 5 == 0:
            ax.text(0.72 * np.cos(a), 0.72 * np.sin(a), str(val),
                    ha="center", va="center", fontsize=8, color=PC["muted"])

    phase_label = PHASE_LABELS.get(snap.cycle_phase, snap.cycle_phase)
    _fig_title(fig,
               f"Macro Score Gauge — {snap.date.strftime('%d/%m/%Y')}",
               f"Phase: {phase_label}  |  Risk Mode: {snap.risk_mode.replace('_',' ').title()}")

    plt.tight_layout()
    return fig


# -------------------------------------------------------------
#  CHART 3 — Allocation Pie
# -------------------------------------------------------------

def allocation_pie_print(snap: MacroSnapshot) -> plt.Figure:
    """Pie chart for asset allocation."""
    _apply_style()
    alloc = snap.asset_allocation
    if not alloc:
        return None

    labels = [k.capitalize() for k in alloc.keys()]
    sizes  = [v * 100 for v in alloc.values()]
    colors_map = {
        "equities": "#2980B9",
        "bonds":    "#27AE60",
        "gold":     "#F39C12",
        "cash":     "#7F8C8D",
    }
    pie_colors = [colors_map.get(k, "#BDC3C7") for k in alloc.keys()]
    explode    = [0.04] * len(labels)

    fig, ax = plt.subplots(figsize=(6.5, 5.5), facecolor=PC["bg"])
    ax.set_facecolor(PC["bg"])

    wedges, texts, autotexts = ax.pie(
        sizes, labels=None, colors=pie_colors,
        explode=explode, autopct="%1.0f%%",
        pctdistance=0.72, startangle=90,
        wedgeprops=dict(edgecolor="white", linewidth=2),
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_fontweight("bold")
        at.set_color("white")

    # Legend
    legend_patches = [mpatches.Patch(color=c, label=f"{l}  {s:.0f}%")
                      for l, c, s in zip(labels, pie_colors, sizes)]
    ax.legend(handles=legend_patches, loc="lower center",
              bbox_to_anchor=(0.5, -0.10), ncol=2,
              fontsize=10, frameon=True,
              edgecolor=PC["border"], facecolor=PC["bg"])

    phase_label = PHASE_LABELS.get(snap.cycle_phase, snap.cycle_phase)
    _fig_title(fig,
               "Asset Allocation",
               f"Phase: {phase_label}  |  Risk: {snap.risk_mode.replace('_',' ').title()}")

    plt.tight_layout()
    return fig


# -------------------------------------------------------------
#  CHART 4 — Phase Timeline
# -------------------------------------------------------------

def phase_timeline_print(ma: MacroAnalyzer, years: int = None) -> plt.Figure:
    """
    3-panel timeline: Macro Score / Yield Curve / VIX
    with phase color bands and crisis markers.
    """
    _apply_style()
    history = ma.get_history()
    inds    = ma.get_indicators()
    if history.empty:
        return None

    if years:
        cutoff  = history.index[-1] - pd.DateOffset(years=years)
        history = history[history.index >= cutoff]
        inds    = inds[inds.index >= cutoff]

    weekly_h = history.resample("W").agg({
        "cycle_phase": lambda x: x.mode().iloc[0] if len(x) > 0 and len(x.mode()) > 0 else "unknown",
        "macro_score": "mean",
    })
    weekly_i = inds.resample("W").last()

    fig, axes = plt.subplots(3, 1, figsize=(FIG_W, 9),
                             sharex=True, facecolor=PC["bg"],
                             gridspec_kw={"height_ratios": [2.2, 1.2, 1.2],
                                          "hspace": 0.12})

    hist_start = str(history.index[0].date())

    # -- Panel 1: Macro Score ------------------------------
    ax1 = axes[0]
    ax1.set_facecolor(PC["bg"])
    ms = weekly_h["macro_score"].dropna()
    _add_phase_bands(ax1, weekly_h)

    ax1.fill_between(ms.index, ms.values, alpha=0.15, color=PC["blue"])
    ax1.plot(ms.index, ms.values, color=PC["blue"], linewidth=1.8, label="Macro Score")
    ax1.axhline(4.0, color=PC["yellow"], linewidth=1.0, linestyle="--",
                alpha=0.7, label="Watchlist (4.0)")
    ax1.axhline(7.5, color=PC["green"],  linewidth=1.0, linestyle="--",
                alpha=0.7, label="Bullish (7.5)")
    ax1.set_ylim(0, 10)
    ax1.set_ylabel("Macro Score (0–10)", fontsize=9, color=PC["label"])
    ax1.legend(fontsize=7.5, loc="upper right",
               frameon=True, edgecolor=PC["border"])
    ax1.grid(True, alpha=0.5)
    ax1.set_title("Macro Score", fontsize=9, color=PC["muted"],
                  loc="left", pad=4)
    _add_crisis_markers(ax1, hist_start)

    # Phase legend
    patches = [mpatches.Patch(color=c, label=PHASE_LABELS.get(p, p), alpha=0.7)
               for p, c in PHASE_COLORS_P.items() if p != "unknown"]
    ax1.legend(handles=patches, fontsize=7, loc="upper left",
               frameon=True, edgecolor=PC["border"], ncol=2)

    # -- Panel 2: Yield Curve ------------------------------
    ax2 = axes[1]
    ax2.set_facecolor(PC["bg"])
    if "yield_curve" in weekly_i.columns:
        yc = weekly_i["yield_curve"].dropna()
        yc_pos = yc.clip(lower=0)
        yc_neg = yc.clip(upper=0)
        ax2.fill_between(yc.index, yc_pos, alpha=0.20, color=PC["green"])
        ax2.fill_between(yc.index, yc_neg, alpha=0.30, color=PC["red"])
        ax2.plot(yc.index, yc.values, color=PC["label"], linewidth=1.4)
        ax2.axhline(0, color=PC["red"], linewidth=1.2, linestyle="--", alpha=0.8)
        ax2.axhline(1.5, color=PC["green"], linewidth=0.8, linestyle=":",
                    alpha=0.6, label="Steep (1.5%)")
    ax2.set_ylabel("Yield Curve %", fontsize=9, color=PC["label"])
    ax2.grid(True, alpha=0.5)
    ax2.set_title("Yield Curve (10Y - 3M)", fontsize=9, color=PC["muted"],
                  loc="left", pad=4)

    # -- Panel 3: VIX -------------------------------------
    ax3 = axes[2]
    ax3.set_facecolor(PC["bg"])
    if "vix" in weekly_i.columns:
        vix = weekly_i["vix"].dropna()
        ax3.fill_between(vix.index, vix.values, alpha=0.15, color=PC["purple"])
        ax3.plot(vix.index, vix.values, color=PC["purple"], linewidth=1.4)
        for level, color, label in [
            (15, PC["green"],  "Low (15)"),
            (25, PC["yellow"], "Elevated (25)"),
            (35, PC["red"],    "Stress (35)"),
        ]:
            ax3.axhline(level, color=color, linewidth=0.9,
                        linestyle="--", alpha=0.7)
            ax3.text(vix.index[-1], level + 0.5, label,
                     fontsize=6.5, color=color, ha="right", va="bottom")
    ax3.set_ylabel("VIX", fontsize=9, color=PC["label"])
    ax3.grid(True, alpha=0.5)
    ax3.set_title("VIX", fontsize=9, color=PC["muted"], loc="left", pad=4)

    for ax in axes:
        ax.tick_params(labelsize=8)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    _fig_title(fig, "Macro Phase Timeline",
               f"History {history.index[0].year}–{history.index[-1].year}"
               f"  |  Current Phase: {PHASE_LABELS.get(ma.get_snapshot().cycle_phase,'')}")
    plt.tight_layout()
    return fig


# -------------------------------------------------------------
#  CHART 5 — Ratios History
# -------------------------------------------------------------

def ratios_history_print(ma: MacroAnalyzer, years: int = None) -> plt.Figure:
    """Copper/Gold, Gold/Silver, Gold/Oil — 3-panel."""
    _apply_style()
    history = ma.get_history()
    inds    = ma.get_indicators()
    if history.empty:
        return None

    if years:
        cutoff  = history.index[-1] - pd.DateOffset(years=years)
        history = history[history.index >= cutoff]
        inds    = inds[inds.index >= cutoff]

    weekly_h = history.resample("W").last()
    weekly_i = inds.resample("W").last()

    fig, axes = plt.subplots(3, 1, figsize=(FIG_W, 9),
                             sharex=True, facecolor=PC["bg"],
                             gridspec_kw={"hspace": 0.14})

    configs = [
        ("copper_gold",   "Copper/Gold Ratio",         PC["late_exp"], "Growth Signal — high = bullish growth"),
        ("gold_silver",   "Gold/Silver Ratio",          PC["purple"],  "Risk Appetite — low = risk-on"),
        ("gold_oil",      "Gold/Oil Ratio",             PC["teal"],    "Macro Stress — high = demand collapse"),
    ]

    hist_start = str(history.index[0].date())

    for ax, (col, title, color, subtitle) in zip(axes, configs):
        ax.set_facecolor(PC["bg"])
        src = weekly_h if col in weekly_h.columns else weekly_i
        if col not in src.columns:
            ax.set_visible(False)
            continue

        series = src[col].dropna()
        _add_phase_bands(ax, weekly_h)
        ax.fill_between(series.index, series.values, alpha=0.12, color=color)
        ax.plot(series.index, series.values, color=color, linewidth=1.6)

        # Rolling mean for context
        rm = series.rolling(52, min_periods=10).mean()
        ax.plot(rm.index, rm.values, color=color, linewidth=0.8,
                linestyle="--", alpha=0.5, label="1Y avg")

        ax.set_ylabel(col.replace("_", "/").title(), fontsize=8.5,
                      color=PC["label"])
        ax.set_title(f"{title}  —  {subtitle}", fontsize=9,
                     color=PC["muted"], loc="left", pad=4)
        ax.legend(fontsize=7.5, loc="upper right",
                  frameon=True, edgecolor=PC["border"])
        ax.grid(True, alpha=0.5)
        ax.tick_params(labelsize=8)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        _add_crisis_markers(ax, hist_start)

    _fig_title(fig, "Key Macro Ratios History",
               f"History {history.index[0].year}–{history.index[-1].year}")
    plt.tight_layout()
    return fig


# -------------------------------------------------------------
#  CHART 6 — Yield Curve History
# -------------------------------------------------------------

def yield_curve_history_print(ma: MacroAnalyzer) -> plt.Figure:
    """Yield curve spread + 10Y vs 3M yields — 2-panel."""
    _apply_style()
    inds    = ma.get_indicators()
    history = ma.get_history()
    if inds.empty or "yield_curve" not in inds.columns:
        return None

    weekly_i = inds.resample("W").last()
    weekly_h = history.resample("W").last()

    fig, axes = plt.subplots(2, 1, figsize=(FIG_W, 7),
                             sharex=True, facecolor=PC["bg"],
                             gridspec_kw={"height_ratios": [1.6, 1],
                                          "hspace": 0.12})

    # -- Panel 1: Spread -----------------------------------
    ax1 = axes[0]
    ax1.set_facecolor(PC["bg"])
    yc = weekly_i["yield_curve"].dropna()
    _add_phase_bands(ax1, weekly_h)
    ax1.fill_between(yc.index, yc.clip(lower=0), alpha=0.20, color=PC["green"],
                     label="Normal (positive)")
    ax1.fill_between(yc.index, yc.clip(upper=0), alpha=0.35, color=PC["red"],
                     label="Inversion (negative)")
    ax1.plot(yc.index, yc.values, color=PC["label"], linewidth=1.5)
    ax1.axhline(0,   color=PC["red"],   linewidth=1.5, linestyle="--")
    ax1.axhline(1.5, color=PC["green"], linewidth=0.8, linestyle=":",
                alpha=0.7, label="Steep threshold (1.5%)")
    ax1.set_ylabel("Spread %", fontsize=9, color=PC["label"])
    ax1.set_title("Yield Curve Spread (10Y - 3M)  |  Red = inversion",
                  fontsize=9, color=PC["muted"], loc="left", pad=4)
    ax1.legend(fontsize=7.5, loc="upper right",
               frameon=True, edgecolor=PC["border"])
    ax1.grid(True, alpha=0.5)
    _add_crisis_markers(ax1, str(history.index[0].date()))

    # -- Panel 2: 10Y & 3M yields -------------------------
    ax2 = axes[1]
    ax2.set_facecolor(PC["bg"])
    if "yield_10y" in weekly_i.columns:
        y10 = weekly_i["yield_10y"].dropna()
        ax2.plot(y10.index, y10.values, color=PC["blue"],
                 linewidth=1.4, label="10Y Yield")
    if "yield_3m" in weekly_i.columns:
        y3m = weekly_i["yield_3m"].dropna()
        ax2.plot(y3m.index, y3m.values, color=PC["red"],
                 linewidth=1.4, label="3M Yield")
    ax2.set_ylabel("Yield %", fontsize=9, color=PC["label"])
    ax2.set_title("10Y vs 3M Treasury Yields", fontsize=9,
                  color=PC["muted"], loc="left", pad=4)
    ax2.legend(fontsize=7.5, loc="upper right",
               frameon=True, edgecolor=PC["border"])
    ax2.grid(True, alpha=0.5)

    for ax in axes:
        ax.tick_params(labelsize=8)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    _fig_title(fig, "Yield Curve History",
               "Inversion = historically reliable recession predictor (6–18m lag)")
    plt.tight_layout()
    return fig


# -------------------------------------------------------------
#  CHART 7 — VIX History
# -------------------------------------------------------------

def vix_history_print(ma: MacroAnalyzer, years: int = None) -> plt.Figure:
    """VIX with phase overlay and threshold lines."""
    _apply_style()
    history = ma.get_history()
    inds    = ma.get_indicators()
    if history.empty or "vix" not in inds.columns:
        return None

    if years:
        cutoff  = history.index[-1] - pd.DateOffset(years=years)
        history = history[history.index >= cutoff]
        inds    = inds[inds.index >= cutoff]

    weekly_h = history.resample("W").last()
    weekly_i = inds.resample("W").last()
    vix      = weekly_i["vix"].dropna()

    fig, ax = plt.subplots(figsize=(FIG_W, 5), facecolor=PC["bg"])
    ax.set_facecolor(PC["bg"])

    _add_phase_bands(ax, weekly_h)
    ax.fill_between(vix.index, vix.values, alpha=0.15, color=PC["purple"])
    ax.plot(vix.index, vix.values, color=PC["purple"], linewidth=1.5)

    for level, color, label in [
        (15, PC["green"],  "Low Volatility (15)"),
        (25, PC["yellow"], "Elevated (25)"),
        (35, PC["red"],    "Stress (35)"),
    ]:
        ax.axhline(level, color=color, linewidth=1.0,
                   linestyle="--", alpha=0.75)
        ax.text(vix.index[-1], level + 0.5, label,
                fontsize=7.5, color=color, ha="right", va="bottom")

    # "Now" annotation
    last_vix   = vix.iloc[-1]
    last_date  = vix.index[-1]
    vix_color  = (PC["green"] if last_vix < 15 else
                  PC["yellow"] if last_vix < 25 else
                  PC["red"])
    ax.annotate(f"Now: {last_vix:.1f}",
                xy=(last_date, last_vix),
                xytext=(-60, 20), textcoords="offset points",
                fontsize=9, color=vix_color, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=vix_color, lw=1.2))

    _add_crisis_markers(ax, str(history.index[0].date()))
    ax.set_ylabel("VIX Level", fontsize=9, color=PC["label"])
    ax.grid(True, alpha=0.5)
    ax.tick_params(labelsize=8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    _fig_title(fig, "VIX History with Phase Overlay",
               f"History {history.index[0].year}–{history.index[-1].year}")
    plt.tight_layout()
    return fig


# -------------------------------------------------------------
#  CHART 8 — Macro Score History
# -------------------------------------------------------------

def macro_score_history_print(ma: MacroAnalyzer) -> plt.Figure:
    """Macro score over time with crisis markers and risk mode dots."""
    _apply_style()
    history = ma.get_history()
    if history.empty:
        return None

    weekly = history.resample("W").agg({
        "macro_score": "mean",
        "cycle_phase": lambda x: x.mode().iloc[0] if len(x) > 0 and len(x.mode()) > 0 else "unknown",
        "risk_mode":   lambda x: x.mode().iloc[0] if len(x) > 0 and len(x.mode()) > 0 else "neutral",
    })

    fig, ax = plt.subplots(figsize=(FIG_W, 5.5), facecolor=PC["bg"])
    ax.set_facecolor(PC["bg"])
    _add_phase_bands(ax, weekly)

    ms = weekly["macro_score"].dropna()
    ax.fill_between(ms.index, ms.values, alpha=0.15, color=PC["blue"])
    ax.plot(ms.index, ms.values, color=PC["blue"], linewidth=1.8,
            label="Macro Score")

    # Threshold lines
    ax.axhline(4.0, color=PC["yellow"], linewidth=1.0, linestyle="--",
               alpha=0.75, label="Watchlist (4.0)")
    ax.axhline(7.5, color=PC["green"],  linewidth=1.0, linestyle="--",
               alpha=0.75, label="Bullish (7.5)")

    # Risk mode dots
    risk_color_map = {"risk_on":  PC["risk_on"],
                      "risk_off": PC["risk_off"],
                      "neutral":  PC["neutral_r"]}
    for risk, rcolor in risk_color_map.items():
        mask = weekly["risk_mode"] == risk
        if mask.any():
            ax.scatter(weekly.index[mask], weekly["macro_score"][mask],
                       s=8, color=rcolor, alpha=0.6, zorder=4,
                       label=risk.replace("_", " ").title())

    # Crisis event labels
    for name, (start_str, _) in CRISIS_EVENTS.items():
        s = pd.Timestamp(start_str)
        if s < history.index[0]:
            continue
        avail = weekly.index[weekly.index >= s]
        if avail.empty:
            continue
        idx   = avail[0]
        score = weekly.loc[idx, "macro_score"]
        ax.annotate(name,
                    xy=(idx, score),
                    xytext=(0, -28), textcoords="offset points",
                    fontsize=7, color=PC["red"],
                    arrowprops=dict(arrowstyle="-", color=PC["red"],
                                   lw=0.8, alpha=0.6),
                    ha="center")

    ax.set_ylim(0, 10)
    ax.set_ylabel("Macro Score (0–10)", fontsize=9, color=PC["label"])
    ax.legend(fontsize=7.5, loc="upper right",
              frameon=True, edgecolor=PC["border"], ncol=2)
    ax.grid(True, alpha=0.5)
    ax.tick_params(labelsize=8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    _fig_title(fig, "Macro Score History with Crisis Markers",
               f"History {history.index[0].year}–{history.index[-1].year}")
    plt.tight_layout()
    return fig


# -------------------------------------------------------------
#  CHART 9 — Crisis Pre-conditions Heatmap
# -------------------------------------------------------------

def crisis_heatmap_print(ma: MacroAnalyzer) -> plt.Figure:
    """
    Heatmap: what signals showed 6M / 3M / 1M before each crisis.
    """
    _apply_style()
    history    = ma.get_history()
    indicators = ma.get_indicators()
    if history.empty:
        return None

    signal_keys = ["macro_score", "yield_curve", "copper_gold", "vix", "credit_ratio"]
    signal_labels_map = {
        "macro_score":  "Macro\nScore",
        "yield_curve":  "Yield\nCurve",
        "copper_gold":  "Cu/\nGold",
        "vix":          "VIX",
        "credit_ratio": "Credit\nSpread",
    }
    lookbacks   = [6, 3, 1]
    col_labels  = []
    for lb in lookbacks:
        for sk in signal_keys:
            col_labels.append(f"{lb}M\n{signal_labels_map.get(sk, sk)}")

    rows_labels, matrix = [], []
    for event_name, (start_str, _) in CRISIS_EVENTS.items():
        start_ts = pd.Timestamp(start_str)
        if start_ts < history.index[0]:
            continue
        row_vals = []
        for months_before in lookbacks:
            lb_date = start_ts - pd.DateOffset(months=months_before)
            avail   = history.index[history.index <= lb_date]
            if avail.empty:
                row_vals.extend([np.nan] * len(signal_keys))
                continue
            actual = avail[-1]
            h_row  = history.loc[actual]
            i_row  = indicators.loc[actual] if actual in indicators.index else pd.Series()
            for sk in signal_keys:
                val = h_row.get(sk, np.nan)
                if pd.isna(val) and sk in i_row:
                    val = i_row[sk]
                row_vals.append(float(val) if pd.notna(val) else np.nan)
        rows_labels.append(event_name)
        matrix.append(row_vals)

    if not matrix:
        return None

    z_matrix = np.array(matrix, dtype=float)
    z_norm   = np.zeros_like(z_matrix)
    for j in range(z_matrix.shape[1]):
        col   = z_matrix[:, j]
        valid = col[~np.isnan(col)]
        if len(valid) > 1:
            mu  = np.nanmean(col)
            std = np.nanstd(col)
            z_norm[:, j] = (col - mu) / (std if std > 0 else 1)

    h = max(5, len(rows_labels) * 1.0 + 2.5)
    w = max(FIG_W, len(col_labels) * 0.75 + 2)
    fig, ax = plt.subplots(figsize=(w, h), facecolor=PC["bg"])
    ax.set_facecolor(PC["bg"])

    # Custom colormap: red (bearish) -> white -> green (bullish)
    cmap = LinearSegmentedColormap.from_list(
        "print_heatmap",
        [PC["red"], "#F5B7B1", "#FDFEFE", "#A9DFBF", PC["green"]]
    )

    im = ax.imshow(z_norm, cmap=cmap, aspect="auto",
                   vmin=-2, vmax=2, interpolation="nearest")

    # Cell annotations
    for i in range(len(rows_labels)):
        for j in range(len(col_labels)):
            val = z_matrix[i, j]
            txt = "N/A" if np.isnan(val) else (
                f"{val:.1f}" if abs(val) < 10 else f"{val:.0f}")
            bg  = z_norm[i, j]
            fg  = "white" if abs(bg) > 1.0 else PC["label"]
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=7.5, color=fg, fontweight="bold")

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=8, color=PC["label"])
    ax.set_yticks(range(len(rows_labels)))
    ax.set_yticklabels(rows_labels, fontsize=9, color=PC["label"])

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Normalized vs avg across crises", fontsize=8,
                   color=PC["label"])
    cbar.ax.tick_params(labelsize=7, colors=PC["tick"])

    # Vertical separators between lookback groups
    for sep in [4.5, 9.5]:
        ax.axvline(sep, color=PC["border"], linewidth=1.5)

    # Group labels above heatmap
    for i, lb in enumerate(lookbacks):
        center = i * len(signal_keys) + (len(signal_keys) - 1) / 2
        ax.text(center, -0.9, f"{lb} Months before",
                ha="center", va="top", fontsize=8.5,
                color=PC["muted"], fontweight="bold")

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    _fig_title(fig, "Pre-Crisis Signal Heatmap",
               "Signal readings 6M / 3M / 1M before each crisis")
    plt.tight_layout()
    return fig


# -------------------------------------------------------------
#  CHART 10 — Z-score Bar
# -------------------------------------------------------------

def zscore_bar_print(snap: MacroSnapshot) -> plt.Figure:
    """Horizontal bar chart of current z-scores with percentile labels."""
    _apply_style()
    signals = snap.signals
    if not signals:
        return None

    names, zscores, colors, pcts = [], [], [], []
    for key, sig in signals.items():
        z = sig.zscore
        if z is None or (isinstance(z, float) and np.isnan(z)):
            continue
        short_name = (sig.name
                      .replace(" (TIP proxy)", "")
                      .replace(" (10Y-3M)", ""))
        names.append(short_name)
        zscores.append(round(z, 2))
        colors.append(PC.get(sig.direction, PC["neutral"]))
        p = sig.percentile
        pcts.append(f"{p:.0f}th" if p and not np.isnan(p) else "")

    # ── SPY vs SMA200 -- uses 60d ROC as proxy z-score
    spy_roc = getattr(snap, "spy_roc", None)
    # Fallback: derive from signals if available
    spy_z = None
    if hasattr(snap, "signals") and "equity_trend" in snap.signals:
        spy_z = snap.signals["equity_trend"].zscore
    # Use scaled ROC as approximate z-score (ROC / 0.05 ~ z-score)
    if spy_z is None:
        # Check if we can get from snapshot attributes
        for attr in ["spy_roc_60", "spy_momentum"]:
            val = getattr(snap, attr, None)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                spy_z = round(val / 0.05, 2)  # scale: 5% ROC ~ 1 std dev
                break

    if spy_z is not None:
        direction = "bullish" if spy_z > 0 else "bearish"
        names.append("SPY vs SMA200")
        zscores.append(round(spy_z, 2))
        colors.append(PC.get(direction, PC["neutral"]))
        pcts.append("")

    # ── Real Yields -- DFII10 z-score (100d) if available, else TIP pressure
    ry_z = getattr(snap, "real_yield_zscore_100d", None)
    if ry_z is None or (isinstance(ry_z, float) and np.isnan(ry_z)):
        # Fallback: scale real_yield_pressure
        ryp = getattr(snap, "real_yield_pressure", None)
        if ryp is not None and not (isinstance(ryp, float) and np.isnan(ryp)):
            ry_z = round(ryp / 0.01, 2)  # scale: 0.01 pressure ~ 1 std dev

    if ry_z is not None and not (isinstance(ry_z, float) and np.isnan(ry_z)):
        # Rising real yields = bearish, falling = bullish
        direction = "bearish" if ry_z > 0 else "bullish"
        label = "Real Yields (DFII10)" if getattr(snap, "real_yield_10y", None) else "Real Yields (TIP)"
        names.append(label)
        zscores.append(round(ry_z, 2))
        colors.append(PC.get(direction, PC["neutral"]))
        pcts.append("")

    if not names:
        return None

    fig, ax = plt.subplots(figsize=(8, 0.7 * len(names) + 2.5),
                           facecolor=PC["bg"])
    ax.set_facecolor(PC["panel"])

    y_pos = range(len(names))
    bars  = ax.barh(list(y_pos), zscores, color=colors,
                    alpha=0.80, edgecolor="white", linewidth=0.8, height=0.62)

    # Value + percentile labels
    for i, (bar, z, p) in enumerate(zip(bars, zscores, pcts)):
        xoffset = 0.08 if z >= 0 else -0.08
        ha      = "left" if z >= 0 else "right"
        label   = f"  {z:+.2f}sd  {p}"
        ax.text(z + xoffset, i, label, ha=ha, va="center",
                fontsize=8.5, color=PC["label"], fontweight="bold")

    # Reference lines
    for xv, color, label in [(-2, PC["red"],    "-2sd"),
                               (-1, PC["late_exp"], "-1sd"),
                               (1,  PC["late_exp"], "+1sd"),
                               (2,  PC["green"],    "+2sd")]:
        ax.axvline(xv, color=color, linewidth=0.9,
                   linestyle="--", alpha=0.65)
        ax.text(xv, len(names) - 0.2, label, ha="center",
                fontsize=7, color=color, alpha=0.85)

    ax.axvline(0, color=PC["zero"], linewidth=1.5, alpha=0.9)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(names, fontsize=9, color=PC["label"])
    ax.set_xlabel("Z-score (5Y rolling window)", fontsize=9, color=PC["label"])
    ax.grid(axis="x", alpha=0.4)
    ax.tick_params(labelsize=8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    phase_label = PHASE_LABELS.get(snap.cycle_phase, snap.cycle_phase)
    _fig_title(fig,
               f"Current Signal Z-Scores — {snap.date.strftime('%d/%m/%Y')}",
               f"Red = bearish | Green = bullish | >2sd = extreme  |  Phase: {phase_label}")
    plt.tight_layout()
    return fig


# -------------------------------------------------------------
#  CHART 11 — Phase Performance Bar
# -------------------------------------------------------------

def phase_performance_bar_print(ma: MacroAnalyzer) -> plt.Figure:
    """Grouped bar chart — annualized asset returns per cycle phase."""
    _apply_style()
    history = ma.get_history()
    if history.empty:
        return None

    assets = {
        "SPY":  ma._prices.get("spy"),
        "GLD":  ma._prices.get("gold_spot", ma._prices.get("gold")),
        "TLT":  ma._prices.get("tlt"),
        "XLE":  ma._prices.get("xle"),
        "XLV":  ma._prices.get("xlv"),
        "XLU":  ma._prices.get("xlu"),
    }
    asset_colors = {
        "SPY": PC["blue"],
        "GLD": PC["yellow"],
        "TLT": PC["green"],
        "XLE": PC["late_exp"],
        "XLV": PC["teal"],
        "XLU": PC["purple"],
    }
    phases = ["early_expansion", "late_expansion",
              "early_contraction", "late_contraction"]
    results = {name: {} for name in assets}

    for name, prices in assets.items():
        if prices is None:
            continue
        for phase in phases:
            phase_dates    = history.index[history["cycle_phase"] == phase]
            monthly_returns = []
            in_phase       = False
            period_start   = None
            for date in prices.index:
                if date not in history.index:
                    continue
                is_phase = history.loc[date, "cycle_phase"] == phase
                if is_phase and not in_phase:
                    period_start = date
                    in_phase     = True
                elif not is_phase and in_phase:
                    if period_start:
                        seg = prices[period_start:date]
                        if len(seg) > 1:
                            monthly_returns.append((seg.iloc[-1] / seg.iloc[0]) - 1)
                    in_phase = False
            if in_phase and period_start:
                seg = prices[period_start:]
                if len(seg) > 1:
                    monthly_returns.append((seg.iloc[-1] / seg.iloc[0]) - 1)
            if monthly_returns:
                avg_ret  = np.mean(monthly_returns)
                avg_days = len(phase_dates) / max(len(monthly_returns), 1)
                ann      = avg_ret * (252 / max(avg_days, 1))
                ann      = np.clip(ann, -1.5, 1.5)
                results[name][phase] = round(ann * 100, 1)

    phase_x = [PHASE_LABELS.get(p, p) for p in phases]
    valid_assets = [n for n in assets if results[n]]
    if not valid_assets:
        return None

    n_assets  = len(valid_assets)
    bar_width = 0.8 / n_assets
    fig, ax   = plt.subplots(figsize=(FIG_W, 5.5), facecolor=PC["bg"])
    ax.set_facecolor(PC["panel"])

    x = np.arange(len(phases))
    for i, name in enumerate(valid_assets):
        y_vals  = [results[name].get(p) for p in phases]
        offsets = x + (i - n_assets / 2 + 0.5) * bar_width
        bars    = ax.bar(offsets, y_vals, bar_width * 0.92,
                         color=asset_colors.get(name, PC["blue"]),
                         label=name, alpha=0.82, edgecolor="white",
                         linewidth=0.5)
        for bar, val in zip(bars, y_vals):
            if val is not None and abs(val) > 3:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + (1 if val > 0 else -3),
                        f"{val:+.0f}%", ha="center", va="bottom",
                        fontsize=6.5, color=PC["label"])

    ax.axhline(0, color=PC["zero"], linewidth=1.3)
    ax.set_xticks(x)
    ax.set_xticklabels(phase_x, fontsize=9, color=PC["label"])
    ax.set_ylabel("Annualized Return %", fontsize=9, color=PC["label"])
    ax.legend(fontsize=8, loc="upper right",
              frameon=True, edgecolor=PC["border"])
    ax.grid(axis="y", alpha=0.5)
    ax.tick_params(labelsize=8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    _fig_title(fig, "Asset Performance by Macro Phase",
               "Annualized returns per phase  |  Outliers capped +/-150%")
    plt.tight_layout()
    return fig


# -------------------------------------------------------------
#  CHART 12 — Sector Rotation Wheel
# -------------------------------------------------------------

def sector_rotation_wheel_print(snap: MacroSnapshot) -> plt.Figure:
    """Sector rotation wheel — which sectors are favored per cycle phase."""
    _apply_style()

    phases_ordered = ["early_expansion", "late_expansion",
                      "early_contraction", "late_contraction"]
    phase_sectors  = {
        "early_expansion":   ["Financials", "Industrials", "Tech", "Materials"],
        "late_expansion":    ["Energy", "Materials", "Industrials", "Value"],
        "early_contraction": ["Healthcare", "Staples", "Utilities", "Gold"],
        "late_contraction":  ["Staples", "Utilities", "Gold", "Bonds"],
    }

    fig, ax = plt.subplots(figsize=(8, 8), facecolor=PC["bg"])
    ax.set_facecolor(PC["bg"])
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.axis("off")

    current_phase = snap.cycle_phase

    for i, phase in enumerate(phases_ordered):
        angle_start = i * 90
        angle_end   = (i + 1) * 90
        color       = PHASE_COLORS_P.get(phase, PC["muted"])
        is_current  = (phase == current_phase)

        # Wedge fill
        theta = np.linspace(np.radians(angle_start), np.radians(angle_end), 80)
        x_arc = np.concatenate([[0], np.cos(theta), [0]])
        y_arc = np.concatenate([[0], np.sin(theta), [0]])
        alpha = 0.30 if is_current else 0.12
        ax.fill(x_arc, y_arc, color=color, alpha=alpha, zorder=1)
        ax.plot(np.cos(theta), np.sin(theta), color=color,
                linewidth=2.5 if is_current else 1.2, zorder=2)

        # Divider
        ax.plot([0, np.cos(np.radians(angle_start))],
                [0, np.sin(np.radians(angle_start))],
                color=PC["border"], linewidth=1.0, zorder=3)

        # Phase label
        mid_angle = np.radians((angle_start + angle_end) / 2)
        lx = 0.60 * np.cos(mid_angle)
        ly = 0.60 * np.sin(mid_angle)
        label     = PHASE_LABELS.get(phase, phase).replace(" ", "\n")
        fontsize  = 9.5 if is_current else 8.5
        fontweight = "bold" if is_current else "normal"
        ax.text(lx, ly, label, ha="center", va="center",
                fontsize=fontsize, color=color,
                fontweight=fontweight, zorder=4)

        # Sector tags
        sectors = phase_sectors.get(phase, [])
        for j, sector in enumerate(sectors):
            frac      = (j + 0.5) / len(sectors)
            sec_angle = np.radians(angle_start + frac * 90)
            r         = 1.15
            sx = r * np.cos(sec_angle)
            sy = r * np.sin(sec_angle)
            bg_alpha  = 0.25 if is_current else 0.12
            ax.text(sx, sy, sector, ha="center", va="center",
                    fontsize=7.5 if is_current else 7,
                    color=color if is_current else PC["muted"],
                    fontweight="bold" if is_current else "normal",
                    bbox=dict(boxstyle="round,pad=0.25",
                              facecolor=color, alpha=bg_alpha,
                              edgecolor=color, linewidth=0.8))

    # Arrow for current phase
    if current_phase in phases_ordered:
        idx   = phases_ordered.index(current_phase)
        mid_a = np.radians(idx * 90 + 45)
        ax.annotate("",
                    xy     = (0.82 * np.cos(mid_a), 0.82 * np.sin(mid_a)),
                    xytext = (0, 0),
                    arrowprops=dict(arrowstyle="-|>",
                                    color=PC["title"],
                                    lw=2.8, mutation_scale=22),
                    zorder=5)

    # Center circle
    circle = plt.Circle((0, 0), 0.32, color=PC["panel"],
                         ec=PC["border"], linewidth=1.5, zorder=4)
    ax.add_patch(circle)
    ax.text(0, 0.07, "Sector", ha="center", va="center",
            fontsize=10, color=PC["title"], fontweight="bold")
    ax.text(0, -0.07, "Rotation", ha="center", va="center",
            fontsize=10, color=PC["title"], fontweight="bold")

    # Outer border circle
    theta_full = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(theta_full), np.sin(theta_full),
            color=PC["border"], linewidth=1.0, zorder=2)

    # Cycle direction arrow
    ax.annotate("", xy=(0, 1.42), xytext=(-0.18, 1.42),
                arrowprops=dict(arrowstyle="->", color=PC["muted"], lw=1.3))
    ax.text(0.12, 1.47, "Cycle direction", fontsize=8, color=PC["muted"])

    # Current phase label bottom
    c_color = PHASE_COLORS_P.get(current_phase, PC["muted"])
    ax.text(0, -1.50,
            f">  Current: {PHASE_LABELS.get(current_phase, '')}",
            ha="center", va="center",
            fontsize=11, color=c_color, fontweight="bold")

    _fig_title(fig, "Sector Rotation Wheel",
               f"Arrow = current phase  |  {snap.date.strftime('%d/%m/%Y')}") 
    plt.tight_layout()
    return fig


# -------------------------------------------------------------
#  MAIN ENTRY POINT
# -------------------------------------------------------------

def print_all(
    ma:         MacroAnalyzer,
    snap:       MacroSnapshot,
    save:       bool = False,
    output_dir: str  = "charts_print",
) -> dict:
    """
    Generates all print-ready charts.

    Parameters
    ----------
    ma          : MacroAnalyzer instance (with load() already called)
    snap        : MacroSnapshot from ma.get_snapshot()
    save        : True  -> saves PNG 300dpi to output_dir
                  False -> display only in notebook
    output_dir  : output folder (created if it does not exist)

    Returns
    -------
    dict with {chart_name: matplotlib.Figure}
    """

    # -- CONFIG -----------------------------------------------
    SAVE_CHARTS = save          # ← True/False from user
    OUTPUT_DIR  = output_dir
    # ---------------------------------------------------------

    print("=" * 55)
    print("  MACRO CHARTS — Print Edition")
    print(f"  Save: {'YES -> ' + OUTPUT_DIR if SAVE_CHARTS else 'NO'}")
    print("=" * 55)

    chart_funcs = {
        "01_signal_radar":           (signal_radar_print,           (snap,)),
        "02_macro_gauge":            (macro_gauge_print,            (snap,)),
        "03_allocation_pie":         (allocation_pie_print,         (snap,)),
        "04a_phase_gantt_overview":           (phase_gantt_print,            (ma,)),
        "04b_phase_gantt_early_expansion":    (phase_gantt_print,            (ma, None, "early_expansion")),
        "04c_phase_gantt_late_expansion":     (phase_gantt_print,            (ma, None, "late_expansion")),
        "04d_phase_gantt_early_contraction":  (phase_gantt_print,            (ma, None, "early_contraction")),
        "04e_phase_gantt_late_contraction":   (phase_gantt_print,            (ma, None, "late_contraction")),
        "05_phase_timeline":         (phase_timeline_print,         (ma,)),
        "06_ratios_history":         (ratios_history_print,         (ma,)),
        "07_yield_curve_history":    (yield_curve_history_print,    (ma,)),
        "08_vix_history":            (vix_history_print,            (ma,)),
        "09_macro_score_history":    (macro_score_history_print,    (ma,)),
        "10_crisis_heatmap":         (crisis_heatmap_print,         (ma,)),
        "11_zscore_bar":             (zscore_bar_print,             (snap,)),
        "12_phase_performance_bar":  (phase_performance_bar_print,  (ma,)),
        "13_sector_rotation_wheel":  (sector_rotation_wheel_print,  (snap,)),
        "14_credit_spread":           (credit_spread_print,          (ma,)),
        "15_cg_yield_divergence":     (cg_yield_divergence_print,    (ma,)),
        "16_probit_recession":         (probit_recession_print,       (ma,)),
        "17_vix_mean_reversion":       (vix_mean_reversion_print,     (ma,)),
        "18_fred_signals":             (fred_signals_print,           (ma,)),
        "19_gold_silver":              (gold_silver_chart_print,       (ma,)),
        "20_gold_silver_mean_rev":     (gold_silver_mean_reversion_print, (ma,)),
        "21_earnings_yield_gap":       (earnings_yield_gap_print,     (ma,)),
        "22_dxy_history":              (dxy_history_print,            (ma,)),
        "23_real_yields":              (real_yields_print,            (ma,)),
        "24_spy_sma200":               (spy_sma200_print,             (ma,)),
        "25_inflation_environment":    (inflation_environment_print,  (ma,)),
        "26a_crisis_gfc_2008": (crisis_timeline_print, (ma, "GFC 2008", "2007-10-01", "2009-03-01")),
        "26b_crisis_euro": (crisis_timeline_print, (ma, "Euro Crisis", "2011-07-01", "2012-07-01")),
        "26c_crisis_china": (crisis_timeline_print, (ma, "China Shock", "2015-08-01", "2016-02-01")),
        "26d_crisis_2018q4": (crisis_timeline_print, (ma, "2018 Q4", "2018-10-01", "2018-12-31")),
        "26e_crisis_covid": (crisis_timeline_print, (ma, "COVID", "2020-02-01", "2020-04-01")),
        "26f_crisis_bear2022": (crisis_timeline_print, (ma, "Bear 2022", "2022-01-01", "2022-10-01")),
    }

    figures = {}
    for name, (func, args) in chart_funcs.items():
        print(f"\n> {name} ...", end=" ")
        try:
            fig = func(*args)
            if fig is not None:
                figures[name] = fig
                if SAVE_CHARTS:
                    _save(fig, name, OUTPUT_DIR)
                if _ipy_display is not None:
                    _ipy_display(fig)
                else:
                    plt.show()
                plt.close(fig)
                print("OK")
            else:
                print("skipped (no data)")
        except Exception as e:
            print(f"ERROR — {e}")

    print("\n" + "=" * 55)
    if SAVE_CHARTS:
        print(f"  ✅ Saved {len(figures)} charts -> '{OUTPUT_DIR}/'")
    else:
        print(f"  ✅ Displayed {len(figures)} charts (no save)")
    print("=" * 55)

    return figures


# -------------------------------------------------------------
#  CHART 13 — Phase Gantt / Distribution
# -------------------------------------------------------------

import matplotlib.dates as _mdates

def _to_num(dt):
    """Timestamp -> matplotlib date number."""
    return _mdates.date2num(dt.to_pydatetime())


def phase_gantt_print(
    ma:              MacroAnalyzer,
    years:           int  = None,
    highlight_phase: str  = None,   # e.g. "early_expansion" — None = all phases normal
) -> plt.Figure:
    """
    Gantt-style chart — alternation of the 4 cycle phases over time.

    Parameters
    ----------
    ma               : MacroAnalyzer (with load() executed)
    years            : years of history to show (None = all)
    highlight_phase  : if set, the specified phase is highlighted
                       and all others are dimmed.
                       Options: "early_expansion" | "late_expansion"
                              "early_contraction" | "late_contraction"

    Usage for Word document export (4 variants):
        fig = phase_gantt_print(ma, highlight_phase="early_expansion")
        fig = phase_gantt_print(ma, highlight_phase="late_expansion")
        fig = phase_gantt_print(ma, highlight_phase="early_contraction")
        fig = phase_gantt_print(ma, highlight_phase="late_contraction")
    """
    _apply_style()
    history = ma.get_history()
    if history.empty:
        return None

    if years:
        cutoff  = history.index[-1] - pd.DateOffset(years=years)
        history = history[history.index >= cutoff]

    weekly = history.resample("W").agg({
        "cycle_phase": lambda x: x.mode().iloc[0] if len(x) > 0 and len(x.mode()) > 0 else "unknown",
    })
    weekly = weekly[weekly["cycle_phase"] != "unknown"]

    # -- Find consecutive phase periods --------------------------
    periods      = []
    prev_phase   = None
    period_start = None

    for date, row in weekly.iterrows():
        phase = row["cycle_phase"]
        if phase != prev_phase:
            if prev_phase is not None and period_start is not None:
                periods.append((period_start, date, prev_phase))
            period_start = date
            prev_phase   = phase
    if prev_phase and period_start:
        periods.append((period_start, weekly.index[-1], prev_phase))

    if not periods:
        return None

    # -- Phase distribution -----------------------------------
    phase_counts   = weekly["cycle_phase"].value_counts()
    total          = phase_counts.sum()
    phases_ordered = ["early_expansion", "late_expansion",
                      "early_contraction", "late_contraction"]
    phase_pct      = {p: (phase_counts.get(p, 0) / total * 100)
                      for p in phases_ordered}

    # -- Highlight logic --------------------------------------
    # If highlight_phase is set:
    #   - target phase: normal color, alpha=0.90, bold edgecolor
    #   - others: grey (#BBBBBB), alpha=0.25
    #   - duration label: only for highlighted phase
    #   - Y-axis label: bold + larger for highlighted phase
    HL = highlight_phase  # shorthand

    def _bar_color(phase):
        if HL is None:
            return PHASE_COLORS_P.get(phase, PC["muted"])
        return PHASE_COLORS_P.get(phase, PC["muted"]) if phase == HL else "#CCCCCC"

    def _bar_alpha(phase):
        if HL is None:
            return 0.82
        return 0.88 if phase == HL else 0.28

    def _bar_edge(phase):
        if HL is None:
            return ("white", 0.4)
        if phase == HL:
            return (PHASE_COLORS_P.get(phase, PC["muted"]), 1.8)
        return ("#AAAAAA", 0.3)

    # -- Layout -----------------------------------------------
    fig = plt.figure(figsize=(FIG_W, 6.5), facecolor=PC["bg"])
    gs  = gridspec.GridSpec(2, 1, height_ratios=[4.2, 1.4],
                            hspace=0.38, figure=fig)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax1.set_facecolor(PC["bg"])
    ax2.set_facecolor(PC["bg"])

    y_pos      = {p: i for i, p in enumerate(phases_ordered)}
    bar_height = 0.62

    # ==== PANEL 1 — Gantt ====================================

    for start, end, phase in periods:
        if phase not in y_pos:
            continue
        y          = y_pos[phase]
        color      = _bar_color(phase)
        alpha      = _bar_alpha(phase)
        edge_c, edge_w = _bar_edge(phase)

        ax1.barh(
            y,
            _to_num(end) - _to_num(start),
            left      = _to_num(start),
            height    = bar_height,
            color     = color,
            alpha     = alpha,
            edgecolor = edge_c,
            linewidth = edge_w,
            zorder    = 3 if phase == HL else 2,
        )

        # Duration label — always for highlighted, >6m for others
        dur_m    = (end - start).days / 30.4
        show_lbl = (phase == HL and dur_m >= 3) or (HL is None and dur_m >= 6)
        if show_lbl:
            mid       = start + (end - start) / 2
            lbl_color = "white" if phase == HL else PC["muted"]
            lbl_alpha = 0.95    if phase == HL else 0.0   # hide labels for non-highlighted
            if HL is None:
                lbl_alpha = 0.9
            ax1.text(
                _to_num(mid), y,
                f"{dur_m:.0f}m",
                ha="center", va="center",
                fontsize=6.5 if HL is None else 7.5,
                color=lbl_color,
                fontweight="bold",
                alpha=lbl_alpha,
                zorder=4,
            )

    # Highlighted phase -> shade the entire row (background band)
    if HL and HL in y_pos:
        hy = y_pos[HL]
        ax1.axhspan(hy - bar_height / 2 - 0.04,
                    hy + bar_height / 2 + 0.04,
                    color=PHASE_COLORS_P.get(HL, PC["muted"]),
                    alpha=0.06, zorder=0)

    # Crisis zones
    for name, (s_str, e_str) in CRISIS_EVENTS.items():
        s = pd.Timestamp(s_str)
        e = pd.Timestamp(e_str)
        if s < weekly.index[0]:
            continue
        ax1.axvspan(_to_num(s), _to_num(e),
                    alpha=0.08, color=PC["red"], zorder=1)
        ax1.text(
            (_to_num(s) + _to_num(e)) / 2,
            len(phases_ordered) - 0.36,
            name,
            ha="center", va="bottom",
            fontsize=6.5, color=PC["red"], alpha=0.70,
        )

    # NOW line
    now_num = _to_num(weekly.index[-1])
    ax1.axvline(now_num, color=PC["title"],
                linewidth=1.8, linestyle="-", alpha=0.50, zorder=5)
    ax1.text(now_num, -0.48, "^ Now",
             ha="center", va="bottom",
             fontsize=7.5, color=PC["title"], fontweight="bold")

    # Y axis labels — highlighted = bold + color, rest = muted
    ax1.set_yticks(list(y_pos.values()))
    ylabels = ax1.set_yticklabels(
        [PHASE_LABELS.get(p, p) for p in phases_ordered],
        fontsize=9.5,
    )
    for lbl, phase in zip(ylabels, phases_ordered):
        if HL is None or phase == HL:
            lbl.set_color(PHASE_COLORS_P.get(phase, PC["label"]))
            lbl.set_fontweight("bold")
            lbl.set_fontsize(10.5 if phase == HL else 9.5)
        else:
            lbl.set_color("#AAAAAA")
            lbl.set_fontweight("normal")
            lbl.set_fontsize(9.0)

    # X axis
    ax1.xaxis.set_major_locator(_mdates.YearLocator(2))
    ax1.xaxis.set_major_formatter(_mdates.DateFormatter("%Y"))
    ax1.tick_params(axis="x", labelsize=8.5, colors=PC["tick"])
    ax1.set_xlim(_to_num(weekly.index[0]),
                 _to_num(weekly.index[-1]))
    ax1.set_ylim(-0.52, len(phases_ordered) - 0.48)

    ax1.grid(axis="x", alpha=0.35, color=PC["grid"])
    ax1.set_axisbelow(True)
    for sp in ["top", "right", "left"]:
        ax1.spines[sp].set_visible(False)
    ax1.spines["bottom"].set_color(PC["border"])

    # Title — if highlight set, show which phase is highlighted
    if HL:
        hl_label = PHASE_LABELS.get(HL, HL)
        hl_color = PHASE_COLORS_P.get(HL, PC["muted"])
        ax1.set_title(f"Phase Rotation — Business Cycle Timeline  "
                      f"[ {hl_label} ]",
                      fontsize=10, color=hl_color,
                      loc="left", pad=5, fontweight="bold")
    else:
        ax1.set_title("Phase Rotation — Business Cycle Timeline",
                      fontsize=10, color=PC["muted"], loc="left", pad=5)

    # ==== PANEL 2 — Distribution stacked bar =================

    left = 0.0
    for phase in phases_ordered:
        pct   = phase_pct.get(phase, 0)
        color = _bar_color(phase)
        alpha = _bar_alpha(phase)
        ax2.barh(0, pct, left=left, height=0.52,
                 color=color, alpha=alpha,
                 edgecolor="white", linewidth=0.8)
        if pct > 5:
            label     = PHASE_LABELS.get(phase, "").replace(" ", "\n")
            txt_alpha = 1.0 if (HL is None or phase == HL) else 0.35
            ax2.text(
                left + pct / 2, 0,
                f"{label}\n{pct:.0f}%",
                ha="center", va="center",
                fontsize=7.0, color="white",
                fontweight="bold" if (HL is None or phase == HL) else "normal",
                linespacing=1.3, alpha=txt_alpha,
            )
        left += pct

    ax2.set_xlim(0, 100)
    ax2.set_ylim(-0.5, 0.5)
    ax2.set_xlabel("% of total time", fontsize=8.5,
                   color=PC["label"])
    ax2.set_yticks([])
    ax2.set_title("Phase Distribution", fontsize=9,
                  color=PC["muted"], loc="left", pad=4)
    ax2.tick_params(axis="x", labelsize=8)
    for sp in ["top", "right", "left"]:
        ax2.spines[sp].set_visible(False)
    ax2.spines["bottom"].set_color(PC["border"])
    ax2.grid(axis="x", alpha=0.3)

    # -- Suptitle ---------------------------------------------
    snap       = ma.get_snapshot()
    hl_suffix  = f"  |  Highlight: {PHASE_LABELS.get(HL,'')}" if HL else ""
    _fig_title(
        fig,
        "Business Cycle Phase Timeline",
        f"History {weekly.index[0].year}–{weekly.index[-1].year}"
        f"  |  Current: {PHASE_LABELS.get(snap.cycle_phase, '')}"
        f"{hl_suffix}",
    )
    plt.tight_layout()
    return fig

# -----------------------------------------------------------------
#  CHART 14 -- Credit Spread (HYG/LQD) Deep Dive
# -----------------------------------------------------------------

def credit_spread_print(ma: MacroAnalyzer) -> plt.Figure:
    """
    3-panel chart for the HYG/LQD credit spread ratio.

    Panel 1: HYG/LQD ratio history with phase bands and crisis markers
    Panel 2: HYG/LQD vs SPY normalized -- leading signal analysis
    Panel 3: Credit spread z-score with threshold lines
    """
    _apply_style()
    history = ma.get_history()
    inds    = ma.get_indicators()
    if history.empty:
        return None

    weekly_h = history.resample("W").last()
    weekly_i = inds.resample("W").last()

    # Columns
    cr_col  = "credit_spread_ratio"
    cz_col  = "credit_spread_zscore"
    spy_col = "spy_roc_60"

    if cr_col not in weekly_i.columns:
        print("  credit_spread_ratio not found in indicators")
        return None

    cr  = weekly_i[cr_col].dropna()
    cz  = weekly_i[cz_col].dropna() if cz_col in weekly_i.columns else None
    spy = weekly_i[spy_col].dropna() if spy_col in weekly_i.columns else None

    hist_start = str(history.index[0].date())

    fig, axes = plt.subplots(3, 1, figsize=(FIG_W, 10),
                             sharex=True, facecolor=PC["bg"],
                             gridspec_kw={"height_ratios": [2.2, 1.6, 1.4],
                                          "hspace": 0.14})

    # ================================================================
    # PANEL 1 -- HYG/LQD Ratio History
    # ================================================================
    ax1 = axes[0]
    ax1.set_facecolor(PC["bg"])
    _add_phase_bands(ax1, weekly_h)

    ax1.fill_between(cr.index, cr.values, alpha=0.12, color=PC["teal"])
    ax1.plot(cr.index, cr.values, color=PC["teal"], linewidth=1.8,
             label="HYG/LQD Ratio")

    # Rolling mean
    rm = cr.rolling(52, min_periods=10).mean()
    ax1.plot(rm.index, rm.values, color=PC["teal"], linewidth=0.9,
             linestyle="--", alpha=0.55, label="1Y avg")

    # Annotate key crisis lows
    for name, (s_str, e_str) in CRISIS_EVENTS.items():
        s = pd.Timestamp(s_str)
        e = pd.Timestamp(e_str)
        if s < history.index[0]:
            continue
        seg = cr[(cr.index >= s) & (cr.index <= e)]
        if seg.empty:
            continue
        low_date = seg.idxmin()
        low_val  = seg.min()
        ax1.annotate(
            f"{name}\nlow: {low_val:.3f}",
            xy=(low_date, low_val),
            xytext=(0, -36), textcoords="offset points",
            fontsize=6.5, color=PC["red"], ha="center",
            arrowprops=dict(arrowstyle="->", color=PC["red"], lw=0.8),
        )

    ax1.set_ylabel("HYG / LQD", fontsize=9, color=PC["label"])
    ax1.legend(fontsize=7.5, loc="lower left",
               frameon=True, edgecolor=PC["border"])
    ax1.grid(True, alpha=0.4)
    ax1.set_title(
        "HYG/LQD Ratio  --  falling = spreads widen = credit stress",
        fontsize=9, color=PC["muted"], loc="left", pad=4)
    ax1.tick_params(labelsize=8)
    for sp in ["top", "right"]:
        ax1.spines[sp].set_visible(False)

    # ================================================================
    # PANEL 2 -- HYG/LQD vs SPY (normalized, leading signal)
    # ================================================================
    ax2 = axes[1]
    ax2.set_facecolor(PC["bg"])

    # Normalize both to 100 at start
    def norm100(s):
        s = s.dropna()
        return (s / s.iloc[0]) * 100 if len(s) > 0 else s

    cr_norm  = norm100(cr)
    ax2.plot(cr_norm.index, cr_norm.values, color=PC["teal"],
             linewidth=1.6, label="HYG/LQD (norm. 100)")

    if spy is not None:
        # Use spy_above_sma200 as proxy for SPY level direction
        spy_above = weekly_i.get("spy_above_sma200", None)
        if spy_above is not None:
            spy_above = spy_above.dropna()

        # Better: reconstruct SPY level from ma._prices if available
        spy_prices = getattr(ma, "_prices", {})
        spy_p = spy_prices.get("spy") if spy_prices else None
        if spy_p is not None:
            spy_w   = spy_p.resample("W").last()
            # align to cr index range
            spy_w   = spy_w[spy_w.index >= cr.index[0]]
            spy_n   = norm100(spy_w)
            ax2.plot(spy_n.index, spy_n.values, color=PC["blue"],
                     linewidth=1.6, label="SPY (norm. 100)",
                     linestyle="--", alpha=0.85)

    # Shade where credit leads equity lower (credit drops before SPY)
    ax2.axhline(100, color=PC["border"], linewidth=0.8,
                linestyle=":", alpha=0.7)

    _add_crisis_markers(ax2, hist_start)
    ax2.set_ylabel("Norm. Level (base=100)", fontsize=9, color=PC["label"])
    ax2.legend(fontsize=7.5, loc="lower left",
               frameon=True, edgecolor=PC["border"])
    ax2.grid(True, alpha=0.4)
    ax2.set_title(
        "HYG/LQD vs SPY (norm.)  --  credit leads equity in credit-driven crises",
        fontsize=9, color=PC["muted"], loc="left", pad=4)
    ax2.tick_params(labelsize=8)
    for sp in ["top", "right"]:
        ax2.spines[sp].set_visible(False)

    # ================================================================
    # PANEL 3 -- Z-score
    # ================================================================
    ax3 = axes[2]
    ax3.set_facecolor(PC["bg"])

    if cz is not None:
        cz_w = cz.reindex(weekly_i.index, method="ffill").dropna()
        cz_pos = cz_w.clip(lower=0)
        cz_neg = cz_w.clip(upper=0)
        ax3.fill_between(cz_w.index, cz_pos, alpha=0.20, color=PC["green"])
        ax3.fill_between(cz_w.index, cz_neg, alpha=0.30, color=PC["red"])
        ax3.plot(cz_w.index, cz_w.values, color=PC["label"],
                 linewidth=1.4)

        # Threshold lines
        for lv, color, lbl in [
            (-0.5, PC["red"],    "Bearish (-0.5)"),
            (-0.3, PC["late_exp"], "Watch (-0.3)"),
            ( 0.5, PC["green"],  "Bullish (+0.5)"),
        ]:
            ax3.axhline(lv, color=color, linewidth=0.9,
                        linestyle="--", alpha=0.7)
            ax3.text(cz_w.index[-1], lv + 0.05, lbl,
                     fontsize=6.5, color=color, ha="right", va="bottom")

        ax3.axhline(0, color=PC["zero"], linewidth=1.2, alpha=0.8)

        # "Now" annotation
        last_z    = cz_w.iloc[-1]
        last_date = cz_w.index[-1]
        z_color   = (PC["green"] if last_z > 0.5 else
                     PC["red"]   if last_z < -0.5 else PC["late_exp"])
        ax3.annotate(f"Now: {last_z:+.2f}s",
                     xy=(last_date, last_z),
                     xytext=(-70, 20), textcoords="offset points",
                     fontsize=9, color=z_color, fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color=z_color, lw=1.1))

    ax3.set_ylabel("Z-score (5Y rolling)", fontsize=9, color=PC["label"])
    ax3.grid(True, alpha=0.4)
    ax3.set_title(
        "Credit Spread Z-score  --  < -0.5 = stress | > +0.5 = tight (healthy)",
        fontsize=9, color=PC["muted"], loc="left", pad=4)
    ax3.tick_params(labelsize=8)
    for sp in ["top", "right"]:
        ax3.spines[sp].set_visible(False)

    # X axis dates
    axes[2].xaxis.set_major_locator(_mdates.YearLocator(2))
    axes[2].xaxis.set_major_formatter(_mdates.DateFormatter("%Y"))

    _fig_title(
        fig,
        "Credit Spread Analysis -- HYG/LQD",
        f"History {history.index[0].year}-{history.index[-1].year}"
        f"  |  Liquidity Black Hole effect + Leading Signal vs Equities"
    )
    plt.tight_layout()
    return fig

# -----------------------------------------------------------------
#  CHART 15 -- Cu/Gold vs 10Y Yield Divergence
# -----------------------------------------------------------------

def cg_yield_divergence_print(ma: MacroAnalyzer) -> plt.Figure:
    """
    2-panel chart for Cu/Gold ratio vs 10Y yield correlation.

    Panel 1: Cu/Gold ratio and 10Y yield normalized (dual axis)
             -- shows historical correlation and divergence periods
    Panel 2: Divergence signal (Cu/Gold z - 10Y z) with shading
             -- bullish/bearish divergence zones
    """
    _apply_style()
    history = ma.get_history()
    inds    = ma.get_indicators()
    if history.empty:
        return None

    weekly_h = history.resample("W").last()
    weekly_i = inds.resample("W").last()

    # Required columns
    required = ["copper_gold", "yield_10y", "cg_yield_divergence",
                "cg_yield_corr_60"]
    missing = [c for c in required if c not in weekly_i.columns]
    if missing:
        print(f"  Missing columns: {missing} -- run ma.load() with updated macro_analyzer.py")
        return None

    cg      = weekly_i["copper_gold"].dropna()
    y10     = weekly_i["yield_10y"].dropna()
    div     = weekly_i["cg_yield_divergence"].dropna()
    corr60  = weekly_i["cg_yield_corr_60"].dropna()
    hist_start = str(history.index[0].date())

    fig, axes = plt.subplots(2, 1, figsize=(FIG_W, 8.5),
                             sharex=True, facecolor=PC["bg"],
                             gridspec_kw={"height_ratios": [2.2, 1.4],
                                          "hspace": 0.12})

    # ================================================================
    # PANEL 1 -- Overlay: Cu/Gold vs 10Y Yield (normalized)
    # ================================================================
    ax1  = axes[0]
    ax1b = ax1.twinx()   # second Y axis for 10Y yield
    ax1.set_facecolor(PC["bg"])
    ax1b.set_facecolor(PC["bg"])

    _add_phase_bands(ax1, weekly_h)

    # Cu/Gold ratio -- left axis
    ax1.plot(cg.index, cg.values, color=PC["late_exp"],
             linewidth=1.8, label="Cu/Gold Ratio", zorder=3)
    ax1.set_ylabel("Copper/Gold Ratio", fontsize=9,
                   color=PC["late_exp"])
    ax1.tick_params(axis="y", labelcolor=PC["late_exp"], labelsize=8)

    # 10Y Yield -- right axis
    ax1b.plot(y10.index, y10.values, color=PC["blue"],
              linewidth=1.8, linestyle="--", label="10Y Yield %",
              alpha=0.85, zorder=3)
    ax1b.set_ylabel("10Y Treasury Yield %", fontsize=9,
                    color=PC["blue"])
    ax1b.tick_params(axis="y", labelcolor=PC["blue"], labelsize=8)

    # Divergence shading -- zones where they visibly diverge
    # Using normalized versions to identify divergences
    if "cg_yield_divergence" in weekly_i.columns:
        div_w = weekly_i["cg_yield_divergence"].reindex(
            weekly_i.index, method="ffill"
        ).dropna()
        # Shade at extreme divergence
        bull_mask = div_w > 1.5
        bear_mask = div_w < -1.5
        for mask, color, label in [
            (bull_mask, PC["green"], "Bullish Div"),
            (bear_mask, PC["red"],   "Bearish Div"),
        ]:
            if mask.any():
                ax1.fill_between(
                    div_w.index, ax1.get_ylim()[0], ax1.get_ylim()[1],
                    where=mask.reindex(div_w.index, fill_value=False),
                    alpha=0.08, color=color, zorder=0, label=label
                )

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               fontsize=7.5, loc="upper left",
               frameon=True, edgecolor=PC["border"])

    ax1.grid(True, alpha=0.35)
    ax1.set_title(
        "Copper/Gold Ratio vs 10Y Treasury Yield  --  "
        "rho(Cu/Gold, 10Y) >> 0 in normal regimes",
        fontsize=9, color=PC["muted"], loc="left", pad=4
    )
    for sp in ["top"]:
        ax1.spines[sp].set_visible(False)
    ax1b.spines["top"].set_visible(False)
    ax1.tick_params(labelsize=8)
    _add_crisis_markers(ax1, hist_start)

    # ================================================================
    # PANEL 2 -- Divergence Signal + Rolling Correlation
    # ================================================================
    ax2  = axes[1]
    ax2b = ax2.twinx()
    ax2.set_facecolor(PC["bg"])
    ax2b.set_facecolor(PC["bg"])

    # Divergence bars
    div_pos = div.clip(lower=0)
    div_neg = div.clip(upper=0)
    ax2.fill_between(div.index, div_pos, alpha=0.30, color=PC["green"],
                     label="Bullish Div (Cu/Gold leads)")
    ax2.fill_between(div.index, div_neg, alpha=0.35, color=PC["red"],
                     label="Bearish Div (Long Duration signal)")
    ax2.plot(div.index, div.values, color=PC["label"],
             linewidth=1.2, alpha=0.7)

    # Threshold lines
    for lv, color, lbl in [
        ( 1.5, PC["green"], "+1.5 Bullish"),
        (-1.5, PC["red"],   "-1.5 Bearish"),
    ]:
        ax2.axhline(lv, color=color, linewidth=1.0,
                    linestyle="--", alpha=0.75)
        ax2.text(div.index[-1], lv + 0.05, lbl,
                 fontsize=6.5, color=color, ha="right", va="bottom")

    ax2.axhline(0, color=PC["zero"], linewidth=1.2, alpha=0.8)
    ax2.set_ylabel("Divergence\n(CG_z - Y10_z)", fontsize=8,
                   color=PC["label"])
    ax2.tick_params(axis="y", labelsize=8)

    # Rolling correlation -- right axis
    ax2b.plot(corr60.index, corr60.values, color=PC["purple"],
              linewidth=1.3, linestyle=":", alpha=0.8,
              label="60d Corr")
    ax2b.axhline(0.3, color=PC["purple"], linewidth=0.8,
                 linestyle="--", alpha=0.5)
    ax2b.text(corr60.index[-1], 0.32, "Corr breakdown (0.3)",
              fontsize=6.5, color=PC["purple"], ha="right")
    ax2b.set_ylabel("Rolling Correlation", fontsize=8,
                    color=PC["purple"])
    ax2b.set_ylim(-1.1, 1.1)
    ax2b.tick_params(axis="y", labelcolor=PC["purple"], labelsize=8)

    # "Now" annotation
    last_div  = div.iloc[-1]
    last_date = div.index[-1]
    if abs(last_div) > 1.5:
        sig_label = "BULLISH DIV" if last_div > 0 else "BEARISH DIV"
        sig_color = PC["green"] if last_div > 0 else PC["red"]
    else:
        sig_label = "NEUTRAL"
        sig_color = PC["muted"]
    ax2.annotate(
        f"Now: {last_div:+.2f} ({sig_label})",
        xy=(last_date, last_div),
        xytext=(-90, 20 if last_div > 0 else -30),
        textcoords="offset points",
        fontsize=9, color=sig_color, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=sig_color, lw=1.1)
    )

    # Combined legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2,
               fontsize=7, loc="lower left",
               frameon=True, edgecolor=PC["border"])

    ax2.grid(True, alpha=0.35)
    ax2.set_title(
        "Divergence Signal  --  > +1.5: yields likely to rise  |"
        "  < -1.5: Long Duration signal",
        fontsize=9, color=PC["muted"], loc="left", pad=4
    )
    for sp in ["top"]:
        ax2.spines[sp].set_visible(False)
    ax2b.spines["top"].set_visible(False)
    ax2.tick_params(labelsize=8)

    # X axis dates
    axes[1].xaxis.set_major_locator(_mdates.YearLocator(2))
    axes[1].xaxis.set_major_formatter(_mdates.DateFormatter("%Y"))

    # Get current signal from snapshot
    try:
        snap = ma.get_snapshot()
        sig  = getattr(snap, "cg_yield_signal", "neutral")
        sig_str = {"bullish_div": "BULLISH DIVERGENCE -- yields likely to rise",
                   "bearish_div": "BEARISH DIVERGENCE -- Long Duration signal",
                   "neutral":     "NEUTRAL -- normal correlation regime"
                   }.get(sig, sig)
    except Exception:
        sig_str = ""

    _fig_title(
        fig,
        "Cu/Gold vs 10Y Yield -- Divergence Analysis",
        f"History {history.index[0].year}-{history.index[-1].year}"
        f"  |  Current signal: {sig_str}"
    )
    plt.tight_layout()
    return fig

# -----------------------------------------------------------------
#  CHART 16 -- Probit Recession Probability (Estrella & Mishkin)
# -----------------------------------------------------------------

# NBER recession periods -- hardcoded
_NBER_RECESSIONS = [
    ("2001-03-01", "2001-11-01"),
    ("2007-12-01", "2009-06-01"),
    ("2020-02-01", "2020-04-01"),
]

def probit_recession_print(ma: MacroAnalyzer) -> plt.Figure:
    """
    3-panel chart for the Probit Recession Probability model.

    Panel 1: Recession probability (0-100%) with NBER shading
             -- historical model performance
    Panel 2: Yield curve spread + Probit overlay (dual axis)
             -- how the spread maps to probability
    Panel 3: Yield curve spread with inversion shading
             -- context for the reader
    """
    _apply_style()
    history = ma.get_history()
    inds    = ma.get_indicators()
    if history.empty:
        return None

    if "probit_recession_prob" not in inds.columns:
        print("  probit_recession_prob not found -- re-run ma.load() with updated macro_analyzer.py")
        return None

    weekly_i = inds.resample("W").last()
    weekly_h = history.resample("W").last()

    prob  = weekly_i["probit_recession_prob"].dropna()
    yc    = weekly_i["yield_curve"].dropna() if "yield_curve" in weekly_i.columns else None
    start = history.index[0]
    end   = history.index[-1]

    fig, axes = plt.subplots(3, 1, figsize=(FIG_W, 10.5),
                             sharex=True, facecolor=PC["bg"],
                             gridspec_kw={"height_ratios": [2.4, 1.4, 1.2],
                                          "hspace": 0.12})

    def _add_nber(ax):
        """NBER recession shading."""
        for s_str, e_str in _NBER_RECESSIONS:
            s = pd.Timestamp(s_str)
            e = pd.Timestamp(e_str)
            if e < start:
                continue
            ax.axvspan(max(s, start), min(e, end),
                       alpha=0.18, color=PC["red"],
                       zorder=0, label="_nber")

    # ================================================================
    # PANEL 1 -- Probit Probability
    # ================================================================
    ax1 = axes[0]
    ax1.set_facecolor(PC["bg"])
    _add_nber(ax1)

    # Color fill by risk zone
    prob_arr = prob.values
    ax1.fill_between(prob.index, prob_arr, 0,
                     where=(prob_arr < 15),
                     alpha=0.25, color=PC["green"], label="Low (<15%)")
    ax1.fill_between(prob.index, prob_arr, 0,
                     where=((prob_arr >= 15) & (prob_arr < 30)),
                     alpha=0.30, color=PC["yellow"], label="Elevated (15-30%)")
    ax1.fill_between(prob.index, prob_arr, 0,
                     where=((prob_arr >= 30) & (prob_arr < 50)),
                     alpha=0.35, color=PC["late_exp"], label="High (30-50%)")
    ax1.fill_between(prob.index, prob_arr, 0,
                     where=(prob_arr >= 50),
                     alpha=0.40, color=PC["red"], label="Critical (>50%)")

    ax1.plot(prob.index, prob_arr, color=PC["label"],
             linewidth=1.5, zorder=3)

    # Threshold lines
    for lv, color, lbl in [
        (15, PC["green"],    "Low (15%)"),
        (30, PC["late_exp"], "NY Fed threshold (30%)"),
        (50, PC["red"],      "High risk (50%)"),
    ]:
        ax1.axhline(lv, color=color, linewidth=1.0,
                    linestyle="--", alpha=0.75)
        ax1.text(prob.index[-1], lv + 0.8, lbl,
                 fontsize=6.5, color=color, ha="right", va="bottom")

    # NBER legend patch
    nber_patch = mpatches.Patch(
        color=PC["red"], alpha=0.18, label="NBER Recession"
    )
    ax1.legend(handles=[nber_patch] + ax1.get_legend_handles_labels()[0][:4],
               fontsize=7.5, loc="upper right",
               frameon=True, edgecolor=PC["border"])

    # Now annotation
    last_prob  = prob.iloc[-1]
    last_date  = prob.index[-1]
    p_color    = (PC["green"]   if last_prob < 15 else
                  PC["yellow"]  if last_prob < 30 else
                  PC["late_exp"] if last_prob < 50 else PC["red"])
    ax1.annotate(
        f"Now: {last_prob:.1f}%",
        xy=(last_date, last_prob),
        xytext=(-80, 20), textcoords="offset points",
        fontsize=9.5, color=p_color, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=p_color, lw=1.2)
    )

    ax1.set_ylim(0, 100)
    ax1.set_ylabel("Recession Probability %", fontsize=9, color=PC["label"])
    ax1.grid(True, alpha=0.4)
    ax1.set_title(
        "Probit Recession Probability (12-month horizon)  --  "
        "Estrella & Mishkin (1998)  |  Grey = NBER Recessions",
        fontsize=9, color=PC["muted"], loc="left", pad=4
    )
    ax1.tick_params(labelsize=8)
    for sp in ["top", "right"]:
        ax1.spines[sp].set_visible(False)

    # ================================================================
    # PANEL 2 -- Yield Curve + Probit overlay (dual axis)
    # ================================================================
    ax2  = axes[1]
    ax2b = ax2.twinx()
    ax2.set_facecolor(PC["bg"])
    ax2b.set_facecolor(PC["bg"])
    _add_nber(ax2)

    if yc is not None:
        ax2.plot(yc.index, yc.values, color=PC["label"],
                 linewidth=1.4, label="Yield Curve (10Y-3M)")
        ax2.axhline(0, color=PC["red"], linewidth=1.2,
                    linestyle="--", alpha=0.8)
        ax2.axhline(1.5, color=PC["green"], linewidth=0.8,
                    linestyle=":", alpha=0.6)
        # Inversion shading
        yc_arr = yc.values
        ax2.fill_between(yc.index, yc_arr, 0,
                         where=(yc_arr < 0),
                         alpha=0.20, color=PC["red"])
        ax2.set_ylabel("Spread %", fontsize=9, color=PC["label"])
        ax2.tick_params(axis="y", labelsize=8)

    # Probit on right axis
    ax2b.plot(prob.index, prob_arr, color=PC["purple"],
              linewidth=1.3, linestyle="--", alpha=0.8,
              label="Probit Prob %")
    ax2b.axhline(30, color=PC["purple"], linewidth=0.8,
                 linestyle=":", alpha=0.5)
    ax2b.set_ylabel("Recession Prob %", fontsize=9, color=PC["purple"])
    ax2b.set_ylim(0, 100)
    ax2b.tick_params(axis="y", labelcolor=PC["purple"], labelsize=8)

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2,
               fontsize=7.5, loc="upper right",
               frameon=True, edgecolor=PC["border"])

    ax2.grid(True, alpha=0.4)
    ax2.set_title(
        "Yield Curve Spread vs Probit Probability  --  "
        "inversion -> probability rises with 6-18m lag",
        fontsize=9, color=PC["muted"], loc="left", pad=4
    )
    ax2.tick_params(labelsize=8)
    for sp in ["top"]:
        ax2.spines[sp].set_visible(False)
    ax2b.spines["top"].set_visible(False)

    # ================================================================
    # PANEL 3 -- Phase overlay
    # ================================================================
    ax3 = axes[2]
    ax3.set_facecolor(PC["bg"])
    _add_phase_bands(ax3, weekly_h)
    _add_nber(ax3)

    if yc is not None:
        ax3.plot(yc.index, yc.values, color=PC["label"],
                 linewidth=1.2)
        ax3.axhline(0, color=PC["red"], linewidth=1.2,
                    linestyle="--", alpha=0.8)
        ax3.fill_between(yc.index, yc.values, 0,
                         where=(yc.values < 0),
                         alpha=0.25, color=PC["red"])

    ax3.set_ylabel("Spread %", fontsize=9, color=PC["label"])
    ax3.grid(True, alpha=0.4)
    ax3.set_title(
        "Yield Curve with Macro Phase Overlay",
        fontsize=9, color=PC["muted"], loc="left", pad=4
    )
    ax3.tick_params(labelsize=8)
    for sp in ["top", "right"]:
        ax3.spines[sp].set_visible(False)

    # X axis
    axes[2].xaxis.set_major_locator(_mdates.YearLocator(2))
    axes[2].xaxis.set_major_formatter(_mdates.DateFormatter("%Y"))

    # Get current signal
    try:
        snap = ma.get_snapshot()
        sig  = getattr(snap, "probit_signal", "unknown")
        prob_now = getattr(snap, "probit_recession_prob", 0) or 0
        subtitle = f"Recession probability (12m): {prob_now:.1f}%  |  Signal: {sig.upper()}"
    except Exception:
        subtitle = f"P(recession 12m) = Phi({-0.6045:.4f} + {-0.7374:.4f} x spread)"

    _fig_title(
        fig,
        "Probit Recession Probability -- Estrella & Mishkin (1998)",
        subtitle
    )
    plt.tight_layout()
    return fig

# -----------------------------------------------------------------
#  CHART 17 -- VIX Mean Reversion
# -----------------------------------------------------------------

def vix_mean_reversion_print(ma: MacroAnalyzer) -> plt.Figure:
    """
    3-panel VIX mean reversion chart.

    Panel 1: VIX vs long-term mean with 1/2 std dev bands
             -- shows when VIX is extremely elevated
    Panel 2: VIX mean reversion z-score
             -- distance from LT mean in std devs
    Panel 3: Half-life of mean reversion (60d rolling)
             -- how fast VIX reverts to the mean
    """
    _apply_style()
    history = ma.get_history()
    inds    = ma.get_indicators()
    if history.empty:
        return None

    required = ["vix", "vix_lt_mean", "vix_mr_zscore",
                "vix_half_life", "vix_band_1up", "vix_band_2up"]
    missing = [c for c in required if c not in inds.columns]
    if missing:
        print(f"  Missing: {missing} -- re-run ma.load()")
        return None

    weekly_h = history.resample("W").last()
    weekly_i = inds.resample("W").last()

    vix     = weekly_i["vix"].dropna()
    lt_mean = weekly_i["vix_lt_mean"].dropna()
    mr_z    = weekly_i["vix_mr_zscore"].dropna()
    hl      = weekly_i["vix_half_life"].dropna()
    b1up    = weekly_i["vix_band_1up"].dropna()
    b2up    = weekly_i["vix_band_2up"].dropna()
    b1dn    = weekly_i.get("vix_band_1dn", lt_mean - (b1up - lt_mean)).dropna()
    hist_start = str(history.index[0].date())

    fig, axes = plt.subplots(3, 1, figsize=(FIG_W, 10),
                             sharex=True, facecolor=PC["bg"],
                             gridspec_kw={"height_ratios": [2.4, 1.4, 1.2],
                                          "hspace": 0.12})

    # ================================================================
    # PANEL 1 -- VIX vs Bands
    # ================================================================
    ax1 = axes[0]
    ax1.set_facecolor(PC["bg"])
    _add_phase_bands(ax1, weekly_h)

    # 2 std dev band fill
    ax1.fill_between(b2up.index, b2up.values, b1up.values,
                     alpha=0.12, color=PC["red"], label="1-2 std dev zone")
    ax1.fill_between(b1up.index, b1up.values, lt_mean.reindex(b1up.index, method="ffill"),
                     alpha=0.10, color=PC["yellow"], label="0-1 std dev zone")
    ax1.fill_between(lt_mean.index, lt_mean.values, b1dn.reindex(lt_mean.index, method="ffill"),
                     alpha=0.08, color=PC["green"], label="Below mean")

    # VIX line
    ax1.plot(vix.index, vix.values, color=PC["purple"],
             linewidth=1.8, label="VIX", zorder=4)

    # Long-term mean
    ax1.plot(lt_mean.index, lt_mean.values, color=PC["label"],
             linewidth=1.2, linestyle="--", alpha=0.8,
             label="5Y Rolling Mean")

    # Upper bands
    ax1.plot(b1up.index, b1up.values, color=PC["yellow"],
             linewidth=0.8, linestyle=":", alpha=0.7, label="Mean+1sd")
    ax1.plot(b2up.index, b2up.values, color=PC["red"],
             linewidth=0.8, linestyle=":", alpha=0.7, label="Mean+2sd")

    # Threshold lines
    for lv, color, lbl in [
        (15, PC["green"],   "Low (15)"),
        (25, PC["yellow"],  "Elevated (25)"),
        (35, PC["red"],     "Stress (35)"),
    ]:
        ax1.axhline(lv, color=color, linewidth=0.7,
                    linestyle="--", alpha=0.5)

    # Now annotation
    last_vix  = vix.iloc[-1]
    last_mean = lt_mean.reindex([vix.index[-1]], method="ffill").iloc[0]
    ax1.annotate(
        f"Now: {last_vix:.1f}  (Mean: {last_mean:.1f})",
        xy=(vix.index[-1], last_vix),
        xytext=(-90, 20), textcoords="offset points",
        fontsize=9, color=PC["purple"], fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=PC["purple"], lw=1.1)
    )

    _add_crisis_markers(ax1, hist_start)
    ax1.set_ylabel("VIX Level", fontsize=9, color=PC["label"])
    ax1.legend(fontsize=7, loc="upper right",
               frameon=True, edgecolor=PC["border"], ncol=3)
    ax1.grid(True, alpha=0.4)
    ax1.set_title(
        "VIX vs Long-term Mean (5Y) with Bollinger Bands  --  "
        ">+2sd = mean reversion opportunity",
        fontsize=9, color=PC["muted"], loc="left", pad=4
    )
    ax1.tick_params(labelsize=8)
    for sp in ["top", "right"]:
        ax1.spines[sp].set_visible(False)

    # ================================================================
    # PANEL 2 -- Mean Reversion Z-score
    # ================================================================
    ax2 = axes[1]
    ax2.set_facecolor(PC["bg"])

    mr_pos = mr_z.clip(lower=0)
    mr_neg = mr_z.clip(upper=0)
    ax2.fill_between(mr_z.index, mr_pos, alpha=0.25, color=PC["red"],
                     label="Above mean (sell vol)")
    ax2.fill_between(mr_z.index, mr_neg, alpha=0.25, color=PC["green"],
                     label="Below mean (complacency)")
    ax2.plot(mr_z.index, mr_z.values, color=PC["label"],
             linewidth=1.3, alpha=0.8)

    # Thresholds
    for lv, color, lbl in [
        ( 2.0, PC["red"],    "+2sd Short Vol"),
        (-1.0, PC["green"],  "-1sd Complacency"),
    ]:
        ax2.axhline(lv, color=color, linewidth=1.0,
                    linestyle="--", alpha=0.75)
        ax2.text(mr_z.index[-1], lv + 0.05, lbl,
                 fontsize=6.5, color=color, ha="right", va="bottom")
    ax2.axhline(0, color=PC["zero"], linewidth=1.2, alpha=0.8)

    # Now
    last_mrz   = mr_z.iloc[-1]
    mr_signal  = ("EXTENDED HIGH" if last_mrz > 2.0 else
                  "COMPLACENCY"   if last_mrz < -1.0 else "NORMAL")
    mr_color   = (PC["red"] if last_mrz > 2.0 else
                  PC["yellow"] if last_mrz < -1.0 else PC["muted"])
    ax2.annotate(
        f"Now: {last_mrz:+.2f}sd  ({mr_signal})",
        xy=(mr_z.index[-1], last_mrz),
        xytext=(-100, 20 if last_mrz > 0 else -30),
        textcoords="offset points",
        fontsize=9, color=mr_color, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=mr_color, lw=1.1)
    )

    ax2.set_ylabel("MR Z-score", fontsize=9, color=PC["label"])
    ax2.legend(fontsize=7.5, loc="upper right",
               frameon=True, edgecolor=PC["border"])
    ax2.grid(True, alpha=0.4)
    ax2.set_title(
        "VIX Distance from Long-term Mean  --  "
        "> +2sd: short volatility signal  |  < -1sd: complacency warning",
        fontsize=9, color=PC["muted"], loc="left", pad=4
    )
    ax2.tick_params(labelsize=8)
    for sp in ["top", "right"]:
        ax2.spines[sp].set_visible(False)

    # ================================================================
    # PANEL 3 -- Half-life
    # ================================================================
    ax3 = axes[2]
    ax3.set_facecolor(PC["bg"])

    # Cap half-life for readability
    hl_capped = hl.clip(upper=120)
    ax3.fill_between(hl_capped.index, hl_capped.values,
                     alpha=0.20, color=PC["teal"])
    ax3.plot(hl_capped.index, hl_capped.values, color=PC["teal"],
             linewidth=1.4)

    # Reference lines
    for lv, lbl in [(14, "14d"), (30, "30d"), (60, "60d")]:
        ax3.axhline(lv, color=PC["border"], linewidth=0.8,
                    linestyle="--", alpha=0.7)
        ax3.text(hl_capped.index[-1], lv + 1, lbl,
                 fontsize=6.5, color=PC["muted"], ha="right")

    last_hl = hl.iloc[-1]
    ax3.annotate(
        f"Now: {last_hl:.0f}d",
        xy=(hl_capped.index[-1], min(last_hl, 120)),
        xytext=(-70, 10), textcoords="offset points",
        fontsize=9, color=PC["teal"], fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=PC["teal"], lw=1.0)
    )

    ax3.set_ylabel("Half-life (days)", fontsize=9, color=PC["label"])
    ax3.grid(True, alpha=0.4)
    ax3.set_title(
        "Mean Reversion Half-life (60d rolling)  --  "
        "days to revert VIX back to its long-term mean",
        fontsize=9, color=PC["muted"], loc="left", pad=4
    )
    ax3.tick_params(labelsize=8)
    for sp in ["top", "right"]:
        ax3.spines[sp].set_visible(False)

    axes[2].xaxis.set_major_locator(_mdates.YearLocator(2))
    axes[2].xaxis.set_major_formatter(_mdates.DateFormatter("%Y"))

    try:
        snap = ma.get_snapshot()
        sig  = getattr(snap, "vix_mr_signal", "normal")
        mrz  = getattr(snap, "vix_mr_zscore", 0) or 0
        ltm  = getattr(snap, "vix_lt_mean", 0) or 0
        hlv  = getattr(snap, "vix_half_life", 0) or 0
        subtitle = (f"VIX={snap.vix:.1f}  |  LT Mean={ltm:.1f}  |  "
                    f"MR z={mrz:+.2f}  |  Half-life={hlv:.0f}d  |  "
                    f"Signal: {sig.upper().replace('_',' ')}")
    except Exception:
        subtitle = "Ornstein-Uhlenbeck mean reversion proxy"

    _fig_title(fig,
               "VIX Mean Reversion Analysis",
               subtitle)
    plt.tight_layout()
    return fig

# -----------------------------------------------------------------
#  CHART 18 -- ISM Manufacturing + TED Spread (FRED data)
# -----------------------------------------------------------------

def fred_signals_print(ma: MacroAnalyzer) -> plt.Figure:
    """
    2-panel chart for FRED signals.

    Panel 1: ISM Manufacturing PMI history with phase bands
             -- 50 line, expansion/contraction zones
    Panel 2: TED Spread or FRA-OIS spread history
             -- bank systemic risk over time
    """
    _apply_style()
    history = ma.get_history()
    inds    = ma.get_indicators()
    if history.empty:
        return None

    has_ism = "ism" in inds.columns
    has_ted = "ted_spread" in inds.columns
    has_fra = "fra_ois_spread" in inds.columns

    if not has_ism and not has_ted and not has_fra:
        print("  No FRED data found -- run with fred_api_key")
        return None

    weekly_h = history.resample("W").last()
    weekly_i = inds.resample("W").last()

    n_panels = sum([has_ism, has_ted or has_fra])
    if n_panels == 0:
        return None

    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(FIG_W, 5.0 * n_panels),
        sharex=False, facecolor=PC["bg"],
        gridspec_kw={"hspace": 0.35}
    )
    if n_panels == 1:
        axes = [axes]

    panel = 0

    # ================================================================
    # PANEL 1 -- ISM Manufacturing PMI
    # ================================================================
    if has_ism:
        ax = axes[panel]
        ax.set_facecolor(PC["bg"])

        ism = weekly_i["ism"].dropna()
        _add_phase_bands(ax, weekly_h)

        # Expansion / contraction zones
        ax.fill_between(ism.index, ism.values, 50,
                        where=(ism.values >= 50),
                        alpha=0.20, color=PC["green"],
                        label="Expansion (>50)")
        ax.fill_between(ism.index, ism.values, 50,
                        where=(ism.values < 50),
                        alpha=0.25, color=PC["red"],
                        label="Contraction (<50)")

        ax.plot(ism.index, ism.values, color=PC["label"],
                linewidth=1.8, zorder=3)

        # Key threshold lines
        for lv, color, lbl in [
            (55, PC["green"],  "Strong (55)"),
            (50, PC["label"],  "Neutral (50)"),
            (45, PC["red"],    "Weak (45)"),
        ]:
            ax.axhline(lv, color=color, linewidth=1.0,
                       linestyle="--", alpha=0.75)
            ax.text(ism.index[-1], lv + 0.3, lbl,
                    fontsize=6.5, color=color, ha="right", va="bottom")

        # Crisis markers
        _add_crisis_markers(ax, str(history.index[0].date()))

        # Now annotation
        last_ism  = ism.iloc[-1]
        last_date = ism.index[-1]
        ism_color = (PC["green"] if last_ism > 55 else
                     PC["late_exp"] if last_ism > 50 else
                     PC["red"])
        ism_sig   = ("EXPANDING" if last_ism > 50 else "CONTRACTING")
        ax.annotate(
            f"Now: {last_ism:.1f}  ({ism_sig})",
            xy=(last_date, last_ism),
            xytext=(-90, 20 if last_ism > 50 else -30),
            textcoords="offset points",
            fontsize=9, color=ism_color, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=ism_color, lw=1.1)
        )

        ax.set_ylabel("ISM PMI", fontsize=9, color=PC["label"])
        ax.legend(fontsize=7.5, loc="lower left",
                  frameon=True, edgecolor=PC["border"])
        ax.grid(True, alpha=0.4)
        ax.set_title(
            "ISM Manufacturing PMI (FRED)  --  "
            ">50 = expanding  |  <50 = contracting  |  >55 = strongly expanding",
            fontsize=9, color=PC["muted"], loc="left", pad=4
        )
        ax.tick_params(labelsize=8)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)

        ax.xaxis.set_major_locator(_mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(_mdates.DateFormatter("%Y"))

        panel += 1

    # ================================================================
    # PANEL 2 -- TED Spread / FRA-OIS
    # ================================================================
    if has_ted or has_fra:
        ax = axes[panel]
        ax.set_facecolor(PC["bg"])

        if has_ted:
            spread      = weekly_i["ted_spread"].dropna()
            spread_name = "TED Spread"
            note        = "3M LIBOR - 3M T-Bill  |  historical until 2023"
        else:
            spread      = weekly_i["fra_ois_spread"].dropna()
            spread_name = "FRA-OIS Spread (SOFR - 3M T-Bill)"
            note        = "Post-LIBOR proxy  |  2018-present"

        _add_phase_bands(ax, weekly_h)

        # Zone fills
        ax.fill_between(spread.index, spread.values, 0,
                        where=(spread.values < TED_NORMAL),
                        alpha=0.20, color=PC["green"],
                        label=f"Normal (<{TED_NORMAL:.0%})")
        ax.fill_between(spread.index, spread.values, 0,
                        where=((spread.values >= TED_NORMAL) &
                               (spread.values < TED_ELEVATED)),
                        alpha=0.20, color=PC["yellow"],
                        label="Watch")
        ax.fill_between(spread.index, spread.values, 0,
                        where=(spread.values >= TED_ELEVATED),
                        alpha=0.30, color=PC["red"],
                        label="Elevated/Crisis")

        ax.plot(spread.index, spread.values, color=PC["label"],
                linewidth=1.6, zorder=3)

        # Threshold lines
        for lv, color, lbl in [
            (TED_NORMAL,   PC["yellow"], f"Watch ({TED_NORMAL:.2f}%)"),
            (TED_ELEVATED, PC["late_exp"], f"Elevated ({TED_ELEVATED:.2f}%)"),
            (TED_CRISIS,   PC["red"],    f"Crisis ({TED_CRISIS:.2f}%)"),
        ]:
            ax.axhline(lv, color=color, linewidth=1.0,
                       linestyle="--", alpha=0.75)
            ax.text(spread.index[-1], lv + 0.02, lbl,
                    fontsize=6.5, color=color, ha="right", va="bottom")

        _add_crisis_markers(ax, str(history.index[0].date()))

        # Now annotation
        last_sp   = spread.iloc[-1]
        last_date = spread.index[-1]
        sp_color  = (PC["green"]   if last_sp < TED_NORMAL else
                     PC["yellow"]  if last_sp < TED_ELEVATED else
                     PC["red"])
        sp_sig    = ("NORMAL"   if last_sp < TED_NORMAL else
                     "WATCH"    if last_sp < TED_ELEVATED else
                     "ELEVATED" if last_sp < TED_CRISIS else "CRISIS")
        ax.annotate(
            f"Now: {last_sp:.2f}%  ({sp_sig})",
            xy=(last_date, last_sp),
            xytext=(-90, 20), textcoords="offset points",
            fontsize=9, color=sp_color, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=sp_color, lw=1.1)
        )

        ax.set_ylabel("Spread %", fontsize=9, color=PC["label"])
        ax.legend(fontsize=7.5, loc="upper right",
                  frameon=True, edgecolor=PC["border"])
        ax.grid(True, alpha=0.4)
        ax.set_title(
            f"{spread_name} (FRED)  --  {note}",
            fontsize=9, color=PC["muted"], loc="left", pad=4
        )
        ax.tick_params(labelsize=8)
        for sp2 in ["top", "right"]:
            ax.spines[sp2].set_visible(False)

        ax.xaxis.set_major_locator(_mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(_mdates.DateFormatter("%Y"))

    # ── Suptitle ─────────────────────────────────────────────
    ism_str = ""
    ted_str = ""
    try:
        snap = ma.get_snapshot()
        if snap.ism:
            ism_str = f"ISM: {snap.ism:.1f} ({(snap.ism_signal or '').replace('_',' ')})"
        ted_val = snap.ted_spread or snap.fra_ois_spread
        if ted_val:
            ted_str = f"TED/FRA-OIS: {ted_val:.2f}% ({snap.ted_signal or ''})"
    except Exception:
        pass

    subtitle = "  |  ".join(filter(None, [ism_str, ted_str]))
    _fig_title(fig,
               "FRED Signals -- ISM Manufacturing + Bank Systemic Risk",
               subtitle or "Requires fred_api_key")
    plt.tight_layout()
    return fig

# -----------------------------------------------------------------
#  CHART 19 -- Gold vs Silver (dual axis)
# -----------------------------------------------------------------

def gold_silver_chart_print(ma: MacroAnalyzer) -> plt.Figure:
    """
    2-panel chart for Gold and Silver.

    Panel 1: Gold price + Silver price (dual axis)
             -- co-movement and divergences
    Panel 2: Gold/Silver ratio history
             -- risk appetite indicator
    """
    _apply_style()
    history = ma.get_history()
    inds    = ma.get_indicators()
    prices  = getattr(ma, "_prices", {})

    if history.empty:
        return None

    # Prices
    # GLD/SLV for chart lines (smooth historical series)
    gold_p = prices.get("gold")
    silv_p = prices.get("silver")
    # GC=F/SI=F for displayed prices (spot $/oz)
    gold_spot_p = prices.get("gold_spot", gold_p)
    silv_spot_p = prices.get("silver_spot", silv_p)

    if gold_p is None or silv_p is None:
        print("  Gold or Silver prices not found")
        return None

    weekly_h = history.resample("W").last()
    weekly_i = inds.resample("W").last()

    gold_w      = gold_p.resample("W").last()
    silv_w      = silv_p.resample("W").last()
    gold_spot_w = gold_spot_p.resample("W").last()
    silv_spot_w = silv_spot_p.resample("W").last()

    # Align
    gold_w, silv_w = gold_w.align(silv_w, join="inner")
    hist_start = str(history.index[0].date())

    fig, axes = plt.subplots(2, 1, figsize=(FIG_W, 8.5),
                             sharex=True, facecolor=PC["bg"],
                             gridspec_kw={"height_ratios": [2.2, 1.4],
                                          "hspace": 0.12})

    # ================================================================
    # PANEL 1 -- Gold + Silver dual axis
    # ================================================================
    ax1  = axes[0]
    ax1b = ax1.twinx()
    ax1.set_facecolor(PC["bg"])
    ax1b.set_facecolor(PC["bg"])

    _add_phase_bands(ax1, weekly_h)

    # Gold -- left axis
    ax1.plot(gold_w.index, gold_w.values,
             color=PC["yellow"], linewidth=1.8,
             label="Gold (USD/oz)", zorder=3)
    ax1.fill_between(gold_w.index, gold_w.values,
                     alpha=0.08, color=PC["yellow"])
    ax1.set_ylabel("Gold (USD/oz)", fontsize=9,
                   color=PC["yellow"])
    ax1.tick_params(axis="y", labelcolor=PC["yellow"], labelsize=8)

    # Silver -- right axis
    ax1b.plot(silv_w.index, silv_w.values,
              color=PC["muted"], linewidth=1.5,
              linestyle="--", alpha=0.85,
              label="Silver (USD/oz)", zorder=3)
    ax1b.set_ylabel("Silver (USD/oz)", fontsize=9,
                    color=PC["muted"])
    ax1b.tick_params(axis="y", labelcolor=PC["muted"], labelsize=8)

    # Crisis markers
    _add_crisis_markers(ax1, hist_start)

    # Now annotations
    last_gold = gold_w.iloc[-1]
    last_silv = silv_w.iloc[-1]
    ax1.annotate(
        f"Gold: ${gold_spot_w.iloc[-1]:.0f}",
        xy=(gold_w.index[-1], last_gold),
        xytext=(-90, 15), textcoords="offset points",
        fontsize=9, color=PC["yellow"], fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=PC["yellow"], lw=1.0)
    )
    ax1b.annotate(
        f"Silver: ${silv_spot_w.iloc[-1]:.2f}",
        xy=(silv_w.index[-1], last_silv),
        xytext=(-90, -25), textcoords="offset points",
        fontsize=9, color=PC["muted"], fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=PC["muted"], lw=1.0)
    )

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               fontsize=7.5, loc="upper left",
               frameon=True, edgecolor=PC["border"])

    ax1.grid(True, alpha=0.4)
    ax1.set_title(
        "Gold vs Silver  --  co-movement and divergences",
        fontsize=9, color=PC["muted"], loc="left", pad=4
    )
    ax1.tick_params(labelsize=8)
    for sp in ["top"]:
        ax1.spines[sp].set_visible(False)
    ax1b.spines["top"].set_visible(False)

    # ================================================================
    # PANEL 2 -- Gold/Silver ratio
    # ================================================================
    ax2 = axes[1]
    ax2.set_facecolor(PC["bg"])

    if "gold_silver" in weekly_i.columns:
        gs_ratio = weekly_i["gold_silver"].dropna()
        _add_phase_bands(ax2, weekly_h)

        ax2.plot(gs_ratio.index, gs_ratio.values,
                 color=PC["purple"], linewidth=1.6,
                 label="Gold/Silver Ratio")
        ax2.fill_between(gs_ratio.index, gs_ratio.values,
                         alpha=0.10, color=PC["purple"])

        # Rolling mean for context
        rm = gs_ratio.rolling(52, min_periods=10).mean()
        ax2.plot(rm.index, rm.values, color=PC["purple"],
                 linewidth=0.9, linestyle="--", alpha=0.6,
                 label="1Y avg")

        # High ratio = fear/defensive, Low ratio = risk-on
        last_gs = gs_ratio.iloc[-1]
        gs_color = PC["red"] if last_gs > rm.iloc[-1] * 1.1 else PC["green"]
        ax2.annotate(
            f"Now: {last_gs:.1f}",
            xy=(gs_ratio.index[-1], last_gs),
            xytext=(-80, 15), textcoords="offset points",
            fontsize=9, color=gs_color, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=gs_color, lw=1.0)
        )

        _add_crisis_markers(ax2, hist_start)
        ax2.set_ylabel("Gold/Silver Ratio", fontsize=9, color=PC["label"])
        ax2.legend(fontsize=7.5, loc="upper left",
                   frameon=True, edgecolor=PC["border"])
        ax2.grid(True, alpha=0.4)
        ax2.set_title(
            "Gold/Silver Ratio  --  high = defensive/fear  |  low = risk appetite",
            fontsize=9, color=PC["muted"], loc="left", pad=4
        )

    ax2.tick_params(labelsize=8)
    for sp in ["top", "right"]:
        ax2.spines[sp].set_visible(False)

    axes[1].xaxis.set_major_locator(_mdates.YearLocator(2))
    axes[1].xaxis.set_major_formatter(_mdates.DateFormatter("%Y"))

    _fig_title(fig,
               "Gold & Silver Analysis",
               f"History {history.index[0].year}-{history.index[-1].year}"
               f"  |  Gold: ${gold_spot_w.iloc[-1]:.0f}/oz  |  Silver: ${silv_spot_w.iloc[-1]:.2f}/oz  (spot GC=F/SI=F)")
    plt.tight_layout()
    return fig


# -----------------------------------------------------------------
#  CHART 20 -- Gold/Silver Ratio Mean Reversion
# -----------------------------------------------------------------

def gold_silver_mean_reversion_print(ma: MacroAnalyzer) -> plt.Figure:
    """
    3-panel chart -- Gold/Silver ratio mean reversion.

    Panel 1: Ratio vs LT mean with 1/2 std dev bands
    Panel 2: Mean reversion z-score
    Panel 3: Rolling half-life
    """
    _apply_style()
    history = ma.get_history()
    inds    = ma.get_indicators()
    if history.empty or "gold_silver" not in inds.columns:
        return None

    weekly_h = history.resample("W").last()
    weekly_i = inds.resample("W").last()

    gs = weekly_i["gold_silver"].dropna()

    # Compute LT mean + bands (5Y rolling)
    LT_WINDOW = 252 * 5
    lt_mean = gs.rolling(LT_WINDOW, min_periods=LT_WINDOW // 4).mean()
    lt_std  = gs.rolling(LT_WINDOW, min_periods=LT_WINDOW // 4).std()
    b1up    = lt_mean + lt_std
    b2up    = lt_mean + 2 * lt_std
    b1dn    = lt_mean - lt_std
    b2dn    = lt_mean - 2 * lt_std

    # MR z-score
    mr_z = ((gs - lt_mean) / lt_std.replace(0, np.nan)).round(3)

    # Half-life via AR(1)
    gs_lag   = gs.shift(1)
    gs_a, gs_l = gs.align(gs_lag, join="inner")
    roll_corr = gs_a.rolling(60, min_periods=30).corr(gs_l)
    rho_safe  = roll_corr.abs().clip(0.01, 0.999)
    half_life = (-np.log(2) / np.log(rho_safe)).clip(upper=120)

    hist_start = str(history.index[0].date())

    fig, axes = plt.subplots(3, 1, figsize=(FIG_W, 10),
                             sharex=True, facecolor=PC["bg"],
                             gridspec_kw={"height_ratios": [2.4, 1.4, 1.2],
                                          "hspace": 0.12})

    # ================================================================
    # PANEL 1 -- Ratio vs Bands
    # ================================================================
    ax1 = axes[0]
    ax1.set_facecolor(PC["bg"])
    _add_phase_bands(ax1, weekly_h)

    # Bands fill
    ax1.fill_between(b2up.index, b2up.values, b1up.values,
                     alpha=0.12, color=PC["red"])
    ax1.fill_between(b1up.index, b1up.values,
                     lt_mean.reindex(b1up.index, method="ffill"),
                     alpha=0.08, color=PC["yellow"])
    ax1.fill_between(lt_mean.index,
                     lt_mean.values,
                     b1dn.reindex(lt_mean.index, method="ffill"),
                     alpha=0.08, color=PC["green"])
    ax1.fill_between(b1dn.index,
                     b1dn.values,
                     b2dn.reindex(b1dn.index, method="ffill"),
                     alpha=0.12, color=PC["green"],
                     label="2sd below (extreme low)")

    # Ratio line
    ax1.plot(gs.index, gs.values, color=PC["purple"],
             linewidth=1.8, label="Gold/Silver Ratio", zorder=4)

    # LT mean + bands
    ax1.plot(lt_mean.index, lt_mean.values,
             color=PC["label"], linewidth=1.2,
             linestyle="--", alpha=0.8, label="5Y Mean")
    ax1.plot(b1up.index, b1up.values,
             color=PC["yellow"], linewidth=0.8,
             linestyle=":", alpha=0.7, label="+1sd")
    ax1.plot(b2up.index, b2up.values,
             color=PC["red"], linewidth=0.8,
             linestyle=":", alpha=0.7, label="+2sd (extreme fear)")

    # Crisis markers
    _add_crisis_markers(ax1, hist_start)

    # Now annotation
    last_gs   = gs.iloc[-1]
    last_mean = lt_mean.reindex([gs.index[-1]], method="ffill").iloc[0]
    gs_color  = (PC["red"] if last_gs > last_mean * 1.15 else
                 PC["green"] if last_gs < last_mean * 0.85 else
                 PC["muted"])
    ax1.annotate(
        f"Now: {last_gs:.1f}  (Mean: {last_mean:.1f})",
        xy=(gs.index[-1], last_gs),
        xytext=(-110, 20), textcoords="offset points",
        fontsize=9, color=gs_color, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=gs_color, lw=1.1)
    )

    ax1.legend(fontsize=7, loc="upper left",
               frameon=True, edgecolor=PC["border"], ncol=3)
    ax1.set_ylabel("Gold/Silver Ratio", fontsize=9, color=PC["label"])
    ax1.grid(True, alpha=0.4)
    ax1.set_title(
        "Gold/Silver Ratio vs 5Y Mean with Bollinger Bands  --  "
        "high = fear/defensive  |  low = risk appetite",
        fontsize=9, color=PC["muted"], loc="left", pad=4
    )
    ax1.tick_params(labelsize=8)
    for sp in ["top", "right"]:
        ax1.spines[sp].set_visible(False)

    # ================================================================
    # PANEL 2 -- MR Z-score
    # ================================================================
    ax2 = axes[1]
    ax2.set_facecolor(PC["bg"])

    mr_pos = mr_z.clip(lower=0)
    mr_neg = mr_z.clip(upper=0)
    ax2.fill_between(mr_z.index, mr_pos, alpha=0.25,
                     color=PC["red"], label="Above mean (fear)")
    ax2.fill_between(mr_z.index, mr_neg, alpha=0.25,
                     color=PC["green"], label="Below mean (risk-on)")
    ax2.plot(mr_z.index, mr_z.values,
             color=PC["label"], linewidth=1.3, alpha=0.8)

    for lv, color, lbl in [
        ( 2.0, PC["red"],   "+2sd Extreme Fear"),
        (-2.0, PC["green"], "-2sd Extreme Risk-On"),
    ]:
        ax2.axhline(lv, color=color, linewidth=1.0,
                    linestyle="--", alpha=0.75)
        ax2.text(mr_z.index[-1], lv + 0.05, lbl,
                 fontsize=6.5, color=color, ha="right", va="bottom")
    ax2.axhline(0, color=PC["zero"], linewidth=1.2, alpha=0.8)

    last_mrz = mr_z.iloc[-1]
    mr_color = (PC["red"]   if last_mrz > 2.0 else
                PC["green"] if last_mrz < -2.0 else PC["muted"])
    mr_sig   = ("EXTREME FEAR"    if last_mrz > 2.0 else
                "EXTREME RISK-ON" if last_mrz < -2.0 else "NORMAL")
    ax2.annotate(
        f"Now: {last_mrz:+.2f}sd  ({mr_sig})",
        xy=(mr_z.index[-1], last_mrz),
        xytext=(-110, 20 if last_mrz > 0 else -30),
        textcoords="offset points",
        fontsize=9, color=mr_color, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=mr_color, lw=1.1)
    )

    ax2.set_ylabel("MR Z-score", fontsize=9, color=PC["label"])
    ax2.legend(fontsize=7.5, loc="upper right",
               frameon=True, edgecolor=PC["border"])
    ax2.grid(True, alpha=0.4)
    ax2.set_title(
        "Distance from 5Y Mean  --  > +2sd: ratio likely to fall  |  < -2sd: ratio likely to rise",
        fontsize=9, color=PC["muted"], loc="left", pad=4
    )
    ax2.tick_params(labelsize=8)
    for sp in ["top", "right"]:
        ax2.spines[sp].set_visible(False)

    # ================================================================
    # PANEL 3 -- Half-life
    # ================================================================
    ax3 = axes[2]
    ax3.set_facecolor(PC["bg"])

    ax3.fill_between(half_life.index, half_life.values,
                     alpha=0.18, color=PC["teal"])
    ax3.plot(half_life.index, half_life.values,
             color=PC["teal"], linewidth=1.4)

    for lv, lbl in [(14, "14d"), (30, "30d"), (60, "60d")]:
        ax3.axhline(lv, color=PC["border"], linewidth=0.8,
                    linestyle="--", alpha=0.6)
        ax3.text(half_life.index[-1], lv + 1, lbl,
                 fontsize=6.5, color=PC["muted"], ha="right")

    last_hl = half_life.iloc[-1]
    ax3.annotate(
        f"Now: {last_hl:.0f}d",
        xy=(half_life.index[-1], last_hl),
        xytext=(-70, 10), textcoords="offset points",
        fontsize=9, color=PC["teal"], fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=PC["teal"], lw=1.0)
    )

    ax3.set_ylabel("Half-life (days)", fontsize=9, color=PC["label"])
    ax3.grid(True, alpha=0.4)
    ax3.set_title(
        "Mean Reversion Half-life (60d rolling)  --  "
        "days for the ratio to revert to its long-term mean",
        fontsize=9, color=PC["muted"], loc="left", pad=4
    )
    ax3.tick_params(labelsize=8)
    for sp in ["top", "right"]:
        ax3.spines[sp].set_visible(False)

    axes[2].xaxis.set_major_locator(_mdates.YearLocator(2))
    axes[2].xaxis.set_major_formatter(_mdates.DateFormatter("%Y"))

    last_mrz_val = mr_z.iloc[-1] if not mr_z.empty else 0
    last_hl_val  = half_life.iloc[-1] if not half_life.empty else 0
    last_gs_val  = gs.iloc[-1] if not gs.empty else 0

    _fig_title(fig,
               "Gold/Silver Ratio -- Mean Reversion Analysis",
               f"Ratio: {last_gs_val:.1f}  |  MR z: {last_mrz_val:+.2f}  |  "
               f"Half-life: {last_hl_val:.0f}d  |  "
               f"{'FEAR' if last_mrz_val > 1 else 'RISK-ON' if last_mrz_val < -1 else 'NEUTRAL'}")
    plt.tight_layout()
    return fig

# -----------------------------------------------------------------
#  CHART 21 -- Earnings Yield Gap
# -----------------------------------------------------------------

def earnings_yield_gap_print(ma: MacroAnalyzer) -> plt.Figure:
    """
    2-panel chart for the Earnings Yield Gap.

    Panel 1: EYG history (earnings yield - 10Y yield)
             with threshold zones and phase bands
    Panel 2: Earnings Yield vs 10Y Yield (dual axis)
             -- shows when bonds "beat" equities
    """
    _apply_style()
    history = ma.get_history()
    inds    = ma.get_indicators()
    if history.empty or "eyg" not in inds.columns:
        print("  EYG not found -- re-run ma.load()")
        return None

    weekly_h = history.resample("W").last()
    weekly_i = inds.resample("W").last()

    eyg      = weekly_i["eyg"].dropna() * 100        # as %
    ey       = weekly_i["earnings_yield"].dropna() * 100
    y10      = weekly_i["yield_10y"].dropna()
    hist_start = str(history.index[0].date())

    fig, axes = plt.subplots(2, 1, figsize=(FIG_W, 8.5),
                             sharex=True, facecolor=PC["bg"],
                             gridspec_kw={"height_ratios": [2.0, 1.4],
                                          "hspace": 0.12})

    # ================================================================
    # PANEL 1 -- EYG History
    # ================================================================
    ax1 = axes[0]
    ax1.set_facecolor(PC["bg"])
    _add_phase_bands(ax1, weekly_h)

    # Zone fills -- based on historical analysis
    # >3%: Bullish Equity Regime
    # 1.5-2.5%: Neutral/Stable
    # <0.5%: Compressed/High Risk
    # <0%: Crisis
    ax1.fill_between(eyg.index, eyg.values, 0,
                     where=(eyg.values > 3),
                     alpha=0.22, color=PC["green"],
                     label="Bullish Equity Regime (>3%)")
    ax1.fill_between(eyg.index, eyg.values, 0,
                     where=((eyg.values >= 1.5) & (eyg.values <= 3)),
                     alpha=0.12, color=PC["yellow"],
                     label="Neutral / Stable (1.5-3%)")
    ax1.fill_between(eyg.index, eyg.values, 0,
                     where=((eyg.values >= 0) & (eyg.values < 1.5)),
                     alpha=0.20, color=PC["late_exp"],
                     label="Compressed (<1.5%)")
    ax1.fill_between(eyg.index, eyg.values, 0,
                     where=(eyg.values < 0),
                     alpha=0.30, color=PC["red"],
                     label="Crisis Signal (<0%)")

    ax1.plot(eyg.index, eyg.values, color=PC["label"],
             linewidth=1.6, zorder=3)
    ax1.axhline(0, color=PC["zero"], linewidth=1.3, alpha=0.8)

    # Threshold lines
    for lv, color, lbl in [
        ( 3.0, PC["green"],    "+3% Bullish"),
        ( 1.5, PC["yellow"],   "+1.5% Neutral floor"),
        ( 0.5, PC["late_exp"], "+0.5% Compressed"),
        ( 0.0, PC["red"],      "0% Crisis"),
    ]:
        ax1.axhline(lv, color=color, linewidth=1.0,
                    linestyle="--", alpha=0.75)
        ax1.text(eyg.index[-1], lv + 0.1, lbl,
                 fontsize=6.5, color=color, ha="right", va="bottom")

    # Crisis markers
    _add_crisis_markers(ax1, hist_start)

    # Now annotation
    last_eyg  = eyg.iloc[-1]
    last_date = eyg.index[-1]
    eyg_color = (PC["green"]   if last_eyg > 2   else
                 PC["late_exp"] if last_eyg > -2  else
                 PC["red"])
    ax1.annotate(
        f"Now: {last_eyg:+.2f}%",
        xy=(last_date, last_eyg),
        xytext=(-90, 20 if last_eyg > 0 else -30),
        textcoords="offset points",
        fontsize=9.5, color=eyg_color, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=eyg_color, lw=1.2)
    )

    ax1.set_ylabel("EYG %", fontsize=9, color=PC["label"])
    ax1.legend(fontsize=7.5, loc="upper right",
               frameon=True, edgecolor=PC["border"], ncol=2)
    ax1.grid(True, alpha=0.4)
    ax1.set_title(
        "Earnings Yield Gap  =  SPY Earnings Yield  -  10Y Treasury Yield  |  "
        ">3% Bullish  |  1.5-2.5% Neutral  |  <0.5% Compressed  |  <0% Crisis",
        fontsize=9, color=PC["muted"], loc="left", pad=4
    )
    ax1.tick_params(labelsize=8)
    for sp in ["top", "right"]:
        ax1.spines[sp].set_visible(False)

    # ================================================================
    # PANEL 2 -- Earnings Yield vs 10Y Yield
    # ================================================================
    ax2 = axes[1]
    ax2.set_facecolor(PC["bg"])

    # Earnings yield line
    ey_aligned = ey.reindex(y10.index, method="ffill").dropna()
    y10_aligned = y10.reindex(ey_aligned.index, method="ffill").dropna()
    ey_aligned, y10_aligned = ey_aligned.align(y10_aligned, join="inner")

    ax2.plot(ey_aligned.index, ey_aligned.values,
             color=PC["blue"], linewidth=1.6,
             label="Earnings Yield (1/PE %)")
    ax2.plot(y10_aligned.index, y10_aligned.values,
             color=PC["red"], linewidth=1.6,
             linestyle="--", label="10Y Treasury Yield %")

    # Shade where bonds > earnings yield
    ax2.fill_between(ey_aligned.index,
                     ey_aligned.values, y10_aligned.values,
                     where=(y10_aligned.values > ey_aligned.values),
                     alpha=0.18, color=PC["red"],
                     label="Bonds > Equities")
    ax2.fill_between(ey_aligned.index,
                     ey_aligned.values, y10_aligned.values,
                     where=(ey_aligned.values >= y10_aligned.values),
                     alpha=0.12, color=PC["green"],
                     label="Equities > Bonds")

    ax2.set_ylabel("Yield %", fontsize=9, color=PC["label"])
    ax2.legend(fontsize=7.5, loc="upper right",
               frameon=True, edgecolor=PC["border"], ncol=2)
    ax2.grid(True, alpha=0.4)
    ax2.set_title(
        "SPY Earnings Yield vs 10Y Treasury Yield  --  "
        "red shading = bonds more attractive than equities",
        fontsize=9, color=PC["muted"], loc="left", pad=4
    )
    ax2.tick_params(labelsize=8)
    for sp in ["top", "right"]:
        ax2.spines[sp].set_visible(False)

    axes[1].xaxis.set_major_locator(_mdates.YearLocator(2))
    axes[1].xaxis.set_major_formatter(_mdates.DateFormatter("%Y"))

    # Get snapshot info
    try:
        snap = ma.get_snapshot()
        pe_str  = f"P/E: {snap.spy_pe:.1f}x" if snap.spy_pe else ""
        ey_str  = f"Earnings Yield: {(snap.earnings_yield or 0)*100:.2f}%"
        y10_str = f"10Y: {(getattr(snap, 'yield_10y_abs', None) or snap.yield_curve or 0):.2f}%" 
        eyg_str = f"EYG: {(snap.eyg or 0)*100:+.2f}%"
        sig_str = (snap.eyg_signal or "").upper()
        subtitle = f"{pe_str}  |  {ey_str}  |  {y10_str}  |  {eyg_str}  |  {sig_str}"
    except Exception:
        subtitle = "SPY Earnings Yield - 10Y Treasury Yield"

    _fig_title(fig, "Earnings Yield Gap (EYG)", subtitle)
    plt.tight_layout()
    return fig

# -----------------------------------------------------------------
#  CHART 22 -- DXY Dollar Index
# -----------------------------------------------------------------

def dxy_history_print(ma: MacroAnalyzer) -> plt.Figure:
    """
    3-panel chart for the DXY Dollar Index — enhanced for report.

    Panel 1: DXY level + 1Y/5Y rolling avg + cycle annotations + phase bands
    Panel 2: DXY 20d ROC (classifier signal, ±3% thresholds)
    Panel 3: DXY 60d ROC + Z-score (medium-term trend + historical context)
    """
    _apply_style()
    history = ma.get_history()
    inds    = ma.get_indicators()
    if history.empty or "dxy" not in inds.columns:
        return None

    weekly_h = history.resample("W").last()
    weekly_i = inds.resample("W").last()

    dxy      = weekly_i["dxy"].dropna()
    dxy_roc  = weekly_i["dxy_roc_20"].dropna()
    dxy_z    = weekly_i["dxy_zscore"].dropna() if "dxy_zscore" in weekly_i.columns else None
    hist_start = str(history.index[0].date())

    fig, axes = plt.subplots(2, 1, figsize=(FIG_W, 7.5),
                             sharex=True, facecolor=PC["bg"],
                             gridspec_kw={"height_ratios": [2.0, 1.2],
                                          "hspace": 0.12})

    # ================================================================
    # PANEL 1 -- DXY Level
    # ================================================================
    ax1 = axes[0]
    ax1.set_facecolor(PC["bg"])
    _add_phase_bands(ax1, weekly_h)

    ax1.fill_between(dxy.index, dxy.values,
                     alpha=0.10, color=PC["blue"])
    ax1.plot(dxy.index, dxy.values,
             color=PC["blue"], linewidth=1.8,
             label="DXY Index")

    # Rolling mean
    rm = dxy.rolling(52, min_periods=10).mean()
    ax1.plot(rm.index, rm.values, color=PC["blue"],
             linewidth=0.9, linestyle="--", alpha=0.55,
             label="1Y avg")

    _add_crisis_markers(ax1, hist_start)

    # Now annotation
    last_dxy  = dxy.iloc[-1]
    last_mean = rm.iloc[-1]
    dxy_color = (PC["red"]   if last_dxy > last_mean * 1.05 else
                 PC["green"] if last_dxy < last_mean * 0.95 else
                 PC["muted"])
    dxy_lbl   = ("Strong" if last_dxy > last_mean * 1.05 else
                 "Weak"   if last_dxy < last_mean * 0.95 else "Neutral")
    ax1.annotate(
        f"Now: {last_dxy:.1f}  ({dxy_lbl})",
        xy=(dxy.index[-1], last_dxy),
        xytext=(-100, 15), textcoords="offset points",
        fontsize=9, color=dxy_color, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=dxy_color, lw=1.1)
    )

    ax1.set_ylabel("DXY Level", fontsize=9, color=PC["label"])
    ax1.legend(fontsize=7.5, loc="upper left",
               frameon=True, edgecolor=PC["border"])
    ax1.grid(True, alpha=0.4)
    ax1.set_title(
        "DXY Dollar Index  --  strong dollar = pressure on commodities, EM, gold",
        fontsize=9, color=PC["muted"], loc="left", pad=4
    )
    ax1.tick_params(labelsize=8)
    for sp in ["top", "right"]:
        ax1.spines[sp].set_visible(False)

    # ================================================================
    # PANEL 2 -- DXY Momentum (20d ROC)
    # ================================================================
    ax2 = axes[1]
    ax2.set_facecolor(PC["bg"])

    roc_pct = dxy_roc * 100
    roc_pos = roc_pct.clip(lower=0)
    roc_neg = roc_pct.clip(upper=0)
    ax2.fill_between(roc_pct.index, roc_pos,
                     alpha=0.25, color=PC["red"],
                     label="Strong dollar (>+3%)")
    ax2.fill_between(roc_pct.index, roc_neg,
                     alpha=0.25, color=PC["green"],
                     label="Weak dollar (<-3%)")
    ax2.plot(roc_pct.index, roc_pct.values,
             color=PC["label"], linewidth=1.3, alpha=0.8)

    # Threshold lines
    for lv, color, lbl in [
        ( 3.0, PC["red"],   "+3% Strong"),
        (-3.0, PC["green"], "-3% Weak"),
    ]:
        ax2.axhline(lv, color=color, linewidth=1.0,
                    linestyle="--", alpha=0.75)
        ax2.text(roc_pct.index[-1], lv + 0.1, lbl,
                 fontsize=6.5, color=color, ha="right", va="bottom")
    ax2.axhline(0, color=PC["zero"], linewidth=1.2, alpha=0.8)

    last_roc  = roc_pct.iloc[-1]
    roc_color = (PC["red"]   if last_roc > 3  else
                 PC["green"] if last_roc < -3 else PC["muted"])
    roc_lbl   = ("STRONG" if last_roc > 3 else
                 "WEAK"   if last_roc < -3 else "STABLE")
    ax2.annotate(
        f"Now: {last_roc:+.1f}%  ({roc_lbl})",
        xy=(roc_pct.index[-1], last_roc),
        xytext=(-100, 20 if last_roc > 0 else -30),
        textcoords="offset points",
        fontsize=9, color=roc_color, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=roc_color, lw=1.1)
    )

    ax2.set_ylabel("20d ROC %", fontsize=9, color=PC["label"])
    ax2.legend(fontsize=7.5, loc="upper right",
               frameon=True, edgecolor=PC["border"])
    ax2.grid(True, alpha=0.4)
    ax2.set_title(
        "DXY Momentum (20d ROC)  --  >+3%: Strong  |  <-3%: Weak  |  "
        "overlay on sector rotation",
        fontsize=9, color=PC["muted"], loc="left", pad=4
    )
    ax2.tick_params(labelsize=8)
    for sp in ["top", "right"]:
        ax2.spines[sp].set_visible(False)

    axes[1].xaxis.set_major_locator(_mdates.YearLocator(2))
    axes[1].xaxis.set_major_formatter(_mdates.DateFormatter("%Y"))

    _fig_title(fig, "DXY Dollar Index History",
               f"History {history.index[0].year}-{history.index[-1].year}"
               f"  |  DXY: {last_dxy:.1f}  |  Momentum: {last_roc:+.1f}%  ({roc_lbl})")
    plt.tight_layout()
    return fig


# -----------------------------------------------------------------
#  CHART 23 -- Real Yields (TIP proxy)
# -----------------------------------------------------------------

def real_yields_print(ma: MacroAnalyzer) -> plt.Figure:
    """
    2-panel chart for Real Yields (TIP proxy).

    Panel 1: TIP ETF price history with phase bands
             -- falling TIP = rising real yields
    Panel 2: Real Yield Pressure (60d ROC TIP, inverted)
             -- the signal used by the scorer
    """
    _apply_style()
    history = ma.get_history()
    inds    = ma.get_indicators()
    prices  = getattr(ma, "_prices", {})
    if history.empty:
        return None

    weekly_h  = history.resample("W").last()
    weekly_i  = inds.resample("W").last()
    hist_start = str(history.index[0].date())

    tip_prices = prices.get("tips")
    ryp        = weekly_i["real_yield_pressure"].dropna() if "real_yield_pressure" in weekly_i.columns else None

    if tip_prices is None and ryp is None:
        print("  TIP data not found")
        return None

    fig, axes = plt.subplots(2, 1, figsize=(FIG_W, 7.5),
                             sharex=True, facecolor=PC["bg"],
                             gridspec_kw={"height_ratios": [2.0, 1.2],
                                          "hspace": 0.12})

    # ================================================================
    # PANEL 1 -- TIP ETF Price
    # ================================================================
    ax1 = axes[0]
    ax1.set_facecolor(PC["bg"])
    _add_phase_bands(ax1, weekly_h)

    if tip_prices is not None:
        tip_w = tip_prices.resample("W").last()
        ax1.fill_between(tip_w.index, tip_w.values,
                         alpha=0.10, color=PC["teal"])
        ax1.plot(tip_w.index, tip_w.values,
                 color=PC["teal"], linewidth=1.8,
                 label="TIP ETF (USD)")

        # 2022 annotation -- the rate crisis
        crisis_start = pd.Timestamp("2022-01-01")
        crisis_end   = pd.Timestamp("2022-10-01")
        if crisis_start >= tip_w.index[0]:
            ax1.axvspan(_to_num(crisis_start), _to_num(crisis_end),
                        alpha=0.12, color=PC["red"], zorder=0)
            ax1.text(
                _to_num(crisis_start + (crisis_end - crisis_start) / 2),
                tip_w.max() * 0.97,
                "2022\nRate Crisis",
                ha="center", va="top",
                fontsize=7, color=PC["red"], fontweight="bold"
            )

        last_tip = tip_w.iloc[-1]
        ax1.annotate(
            f"Now: ${last_tip:.1f}",
            xy=(tip_w.index[-1], last_tip),
            xytext=(-80, 15), textcoords="offset points",
            fontsize=9, color=PC["teal"], fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=PC["teal"], lw=1.1)
        )

    _add_crisis_markers(ax1, hist_start)
    ax1.set_ylabel("TIP ETF Price (USD)", fontsize=9, color=PC["label"])
    ax1.legend(fontsize=7.5, loc="upper left",
               frameon=True, edgecolor=PC["border"])
    ax1.grid(True, alpha=0.4)
    ax1.set_title(
        "TIP ETF (TIPS ETF)  --  falling = rising real yields = headwind for equities/gold",
        fontsize=9, color=PC["muted"], loc="left", pad=4
    )
    ax1.tick_params(labelsize=8)
    for sp in ["top", "right"]:
        ax1.spines[sp].set_visible(False)

    # ================================================================
    # PANEL 2 -- DFII10 Z-score (primary) or Real Yield Pressure (fallback)
    # ================================================================
    ax2 = axes[1]
    ax2.set_facecolor(PC["bg"])

    has_dfii10 = "real_yield_zscore_100d" in weekly_i.columns

    if has_dfii10:
        # PRIMARY: DFII10 z-score vs 100d mean
        ry_z = weekly_i["real_yield_zscore_100d"].dropna()
        rz_pos = ry_z.clip(lower=0)
        rz_neg = ry_z.clip(upper=0)
        ax2.fill_between(ry_z.index, rz_pos,
                         alpha=0.28, color=PC["red"],
                         label="Tightening (z > 0)")
        ax2.fill_between(ry_z.index, rz_neg,
                         alpha=0.22, color=PC["green"],
                         label="Easing (z < 0)")
        ax2.plot(ry_z.index, ry_z.values,
                 color=PC["label"], linewidth=1.3, alpha=0.8)

        # Threshold lines -- ChatGPT approach: z > 1.5 = tightening
        for lv, color, lbl in [
            ( 1.5, PC["red"],   "+1.5sd Tightening"),
            (-1.5, PC["green"], "-1.5sd Easing"),
        ]:
            ax2.axhline(lv, color=color, linewidth=1.0,
                        linestyle="--", alpha=0.75)
            ax2.text(ry_z.index[-1], lv + 0.05, lbl,
                     fontsize=6.5, color=color, ha="right", va="bottom")
        ax2.axhline(0, color=PC["zero"], linewidth=1.2, alpha=0.8)

        last_ryz  = ry_z.iloc[-1]
        ryz_color = (PC["red"]   if last_ryz > 1.5  else
                     PC["green"] if last_ryz < -1.5 else PC["muted"])
        ryz_lbl   = ("TIGHTENING" if last_ryz > 1.5  else
                     "EASING"     if last_ryz < -1.5 else "NEUTRAL")
        ax2.annotate(
            f"Now: {last_ryz:+.2f}sd  ({ryz_lbl})",
            xy=(ry_z.index[-1], last_ryz),
            xytext=(-110, 20 if last_ryz > 0 else -30),
            textcoords="offset points",
            fontsize=9, color=ryz_color, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=ryz_color, lw=1.1)
        )
        ax2.set_ylabel("DFII10 Z-score (100d)",
                       fontsize=9, color=PC["label"])
        ax2.legend(fontsize=7.5, loc="upper right",
                   frameon=True, edgecolor=PC["border"])
        p2_title = ("DFII10 Z-score (100d mean)  --  "
                    ">+1.5sd: liquidity tightening  |  <-1.5sd: easing")

    elif ryp is not None:
        # FALLBACK: TIP proxy ROC
        ryp_pos = ryp.clip(lower=0)
        ryp_neg = ryp.clip(upper=0)
        ax2.fill_between(ryp.index, ryp_pos,
                         alpha=0.28, color=PC["red"],
                         label="Rising real yields (bearish)")
        ax2.fill_between(ryp.index, ryp_neg,
                         alpha=0.22, color=PC["green"],
                         label="Falling real yields (bullish)")
        ax2.plot(ryp.index, ryp.values,
                 color=PC["label"], linewidth=1.3, alpha=0.8)

        for lv, color, lbl in [
            ( 0.02, PC["red"],   "+0.02 Bearish"),
            (-0.02, PC["green"], "-0.02 Bullish"),
        ]:
            ax2.axhline(lv, color=color, linewidth=1.0,
                        linestyle="--", alpha=0.75)
            ax2.text(ryp.index[-1], lv + 0.001, lbl,
                     fontsize=6.5, color=color, ha="right", va="bottom")
        ax2.axhline(0, color=PC["zero"], linewidth=1.2, alpha=0.8)

        last_ryp  = ryp.iloc[-1]
        ryp_color = (PC["red"]   if last_ryp > 0.02  else
                     PC["green"] if last_ryp < -0.02 else PC["muted"])
        ryp_lbl   = ("RISING (bearish)"  if last_ryp > 0.02  else
                     "FALLING (bullish)" if last_ryp < -0.02 else "STABLE")
        ax2.annotate(
            f"Now: {last_ryp:+.3f}  ({ryp_lbl})",
            xy=(ryp.index[-1], last_ryp),
            xytext=(-110, 20 if last_ryp > 0 else -30),
            textcoords="offset points",
            fontsize=9, color=ryp_color, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=ryp_color, lw=1.1)
        )
        ax2.set_ylabel("Real Yield Pressure\n(-ROC TIP 60d)",
                       fontsize=9, color=PC["label"])
        ax2.legend(fontsize=7.5, loc="upper right",
                   frameon=True, edgecolor=PC["border"])
        p2_title = ("Real Yield Pressure (TIP proxy)  --  "
                    ">+0.02: bearish  |  <-0.02: bullish")
    else:
        p2_title = "Real Yield Pressure  --  no data"

    ax2.grid(True, alpha=0.4)
    ax2.set_title(p2_title, fontsize=9, color=PC["muted"],
                  loc="left", pad=4)
    ax2.tick_params(labelsize=8)
    for sp in ["top", "right"]:
        ax2.spines[sp].set_visible(False)

    axes[1].xaxis.set_major_locator(_mdates.YearLocator(2))
    axes[1].xaxis.set_major_formatter(_mdates.DateFormatter("%Y"))

    last_ryp_val = ryp.iloc[-1] if ryp is not None else 0
    # Add DFII10 panel if available
    ry_10y = weekly_i.get("real_yield_10y") if hasattr(weekly_i, "get") else None
    if "real_yield_10y" in weekly_i.columns:
        ry10  = weekly_i["real_yield_10y"].dropna()
        ry_z  = weekly_i["real_yield_zscore_100d"].dropna() if "real_yield_zscore_100d" in weekly_i.columns else None
        last_ry  = ry10.iloc[-1] if len(ry10) > 0 else 0
        last_ryz = ry_z.iloc[-1] if ry_z is not None and len(ry_z) > 0 else 0
        ry_sig   = ("TIGHTENING" if last_ryz > 1.5 else
                    "EASING"     if last_ryz < -1.5 else "NEUTRAL")
        subtitle = (f"DFII10: {last_ry:.2f}%  |  Z(100d): {last_ryz:+.2f}  |  {ry_sig}"
                    f"  |  Pressure: {last_ryp_val:+.3f}")
    else:
        subtitle = f"TIP proxy  |  Pressure: {last_ryp_val:+.3f}"

    _fig_title(fig, "Real Yields Analysis (DFII10 primary / TIP fallback)",
               f"History {history.index[0].year}-{history.index[-1].year}"
               f"  |  {subtitle}")
    plt.tight_layout()
    return fig


# -----------------------------------------------------------------
#  CHART 24 -- SPY vs SMA200
# -----------------------------------------------------------------

def spy_sma200_print(ma: MacroAnalyzer) -> plt.Figure:
    """
    2-panel chart for SPY vs SMA200.

    Panel 1: SPY price vs SMA200 -- bull/bear regime
             -- shading when SPY is below SMA200
    Panel 2: SPY 60d momentum (ROC)
             -- accelerating or decelerating
    """
    _apply_style()
    history = ma.get_history()
    inds    = ma.get_indicators()
    prices  = getattr(ma, "_prices", {})
    if history.empty:
        return None

    weekly_h  = history.resample("W").last()
    weekly_i  = inds.resample("W").last()
    hist_start = str(history.index[0].date())

    spy_prices = prices.get("spy")
    spy_roc    = weekly_i["spy_roc_60"].dropna() if "spy_roc_60" in weekly_i.columns else None
    spy_above  = weekly_i["spy_above_sma200"].dropna() if "spy_above_sma200" in weekly_i.columns else None

    if spy_prices is None:
        print("  SPY prices not found")
        return None

    spy_w   = spy_prices.resample("W").last()
    sma200  = spy_prices.rolling(200).mean().resample("W").last()
    sma200  = sma200.reindex(spy_w.index, method="ffill")

    fig, axes = plt.subplots(2, 1, figsize=(FIG_W, 7.5),
                             sharex=True, facecolor=PC["bg"],
                             gridspec_kw={"height_ratios": [2.2, 1.0],
                                          "hspace": 0.12})

    # ================================================================
    # PANEL 1 -- SPY vs SMA200
    # ================================================================
    ax1 = axes[0]
    ax1.set_facecolor(PC["bg"])

    # Bear market shading -- when SPY is below SMA200
    ax1.fill_between(spy_w.index, spy_w.values, sma200.values,
                     where=(spy_w.values < sma200.values),
                     alpha=0.20, color=PC["red"],
                     label="Below SMA200 (Bear)")
    ax1.fill_between(spy_w.index, spy_w.values, sma200.values,
                     where=(spy_w.values >= sma200.values),
                     alpha=0.08, color=PC["green"],
                     label="Above SMA200 (Bull)")

    # SPY line
    ax1.plot(spy_w.index, spy_w.values,
             color=PC["blue"], linewidth=1.8,
             label="SPY", zorder=3)

    # SMA200 line
    ax1.plot(sma200.index, sma200.values,
             color=PC["label"], linewidth=1.2,
             linestyle="--", alpha=0.8,
             label="SMA200", zorder=3)

    _add_crisis_markers(ax1, hist_start)

    # Now annotation
    last_spy   = spy_w.iloc[-1]
    last_sma   = sma200.iloc[-1]
    is_bull    = last_spy >= last_sma
    spy_color  = PC["green"] if is_bull else PC["red"]
    spy_regime = "BULL" if is_bull else "BEAR"
    pct_from_sma = ((last_spy / last_sma) - 1) * 100
    ax1.annotate(
        f"SPY: ${last_spy:.0f}  ({spy_regime} | {pct_from_sma:+.1f}% vs SMA200)",
        xy=(spy_w.index[-1], last_spy),
        xytext=(-140, 15), textcoords="offset points",
        fontsize=9, color=spy_color, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=spy_color, lw=1.1)
    )

    ax1.set_ylabel("SPY Price (USD)", fontsize=9, color=PC["label"])
    ax1.legend(fontsize=7.5, loc="upper left",
               frameon=True, edgecolor=PC["border"], ncol=2)
    ax1.grid(True, alpha=0.4)
    ax1.set_title(
        "SPY vs SMA200  --  below SMA200 = confirmed bear market  |  "
        "red shading = bear regime",
        fontsize=9, color=PC["muted"], loc="left", pad=4
    )
    ax1.tick_params(labelsize=8)
    for sp in ["top", "right"]:
        ax1.spines[sp].set_visible(False)

    # ================================================================
    # PANEL 2 -- SPY 60d Momentum
    # ================================================================
    ax2 = axes[1]
    ax2.set_facecolor(PC["bg"])

    if spy_roc is not None:
        roc_pct = spy_roc * 100
        roc_pos = roc_pct.clip(lower=0)
        roc_neg = roc_pct.clip(upper=0)
        ax2.fill_between(roc_pct.index, roc_pos,
                         alpha=0.22, color=PC["green"])
        ax2.fill_between(roc_pct.index, roc_neg,
                         alpha=0.25, color=PC["red"])
        ax2.plot(roc_pct.index, roc_pct.values,
                 color=PC["label"], linewidth=1.3, alpha=0.8)
        ax2.axhline(0, color=PC["zero"], linewidth=1.2, alpha=0.8)

        last_roc  = roc_pct.iloc[-1]
        roc_color = PC["green"] if last_roc > 0 else PC["red"]
        ax2.annotate(
            f"Now: {last_roc:+.1f}%",
            xy=(roc_pct.index[-1], last_roc),
            xytext=(-80, 15 if last_roc > 0 else -25),
            textcoords="offset points",
            fontsize=9, color=roc_color, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=roc_color, lw=1.0)
        )

    ax2.set_ylabel("60d ROC %", fontsize=9, color=PC["label"])
    ax2.grid(True, alpha=0.4)
    ax2.set_title(
        "SPY 60-day Momentum (ROC)  --  positive + above SMA200 = Bullish",
        fontsize=9, color=PC["muted"], loc="left", pad=4
    )
    ax2.tick_params(labelsize=8)
    for sp in ["top", "right"]:
        ax2.spines[sp].set_visible(False)

    axes[1].xaxis.set_major_locator(_mdates.YearLocator(2))
    axes[1].xaxis.set_major_formatter(_mdates.DateFormatter("%Y"))

    regime_str = "BULL" if last_spy >= last_sma else "BEAR"
    _fig_title(fig, "SPY vs SMA200 -- Equity Regime",
               f"History {history.index[0].year}-{history.index[-1].year}"
               f"  |  SPY: ${spy_w.iloc[-1]:.0f}  |  Regime: {regime_str}"
               f"  |  {pct_from_sma:+.1f}% vs SMA200")
    plt.tight_layout()
    return fig

# -----------------------------------------------------------------
#  CHART 25 — Inflation Environment (4 panels, print-ready)
#  Chapter 6: Inflation Analysis
# -----------------------------------------------------------------

def inflation_environment_print(ma: MacroAnalyzer) -> Optional[plt.Figure]:
    """
    4-panel Inflation Environment chart — print-ready matplotlib.

    Panel 1: Gold Price + 60d ROC (level vs momentum)
    Panel 2: Gold/Oil Ratio + Z-score
    Panel 3: TIPS 60d ROC  vs  10Y Yield 60d ROC
    Panel 4: Inflation State Timeline (categorical color band)
    """
    _apply_style()

    inds    = ma.get_indicators()
    history = ma.get_history()
    p       = ma._prices

    if inds is None or inds.empty:
        return None

    # ── Extract series ──────────────────────────────────────────
    gold_price      = p.get("gold")
    gold_roc        = inds["gold_roc_60"]       if "gold_roc_60"       in inds.columns else None
    gold_oil_ratio  = inds["gold_oil"]           if "gold_oil"          in inds.columns else None
    gold_oil_zscore = inds["gold_oil_zscore"]    if "gold_oil_zscore"   in inds.columns else None
    tips_roc        = inds["tips_roc_60"]        if "tips_roc_60"       in inds.columns else None
    y10y_roc        = inds["yield_10y_roc_60"]   if "yield_10y_roc_60"  in inds.columns else None
    inf_env         = history["inflation_env"]   if (not history.empty and "inflation_env" in history.columns) else None

    if gold_price is None and gold_roc is None:
        return None

    def _w(s):
        if s is None:
            return None
        return s.resample("W").last().dropna()

    gold_w     = _w(gold_price)
    roc_w      = (_w(gold_roc) * 100)      if gold_roc is not None else None
    go_ratio_w = _w(gold_oil_ratio)
    go_z_w     = _w(gold_oil_zscore)
    tips_w     = (_w(tips_roc) * 100)      if tips_roc is not None else None
    y10y_w     = (_w(y10y_roc) * 100)      if y10y_roc is not None else None
    inf_m      = inf_env.resample("ME").last().dropna() if inf_env is not None else None

    hist_start = inds.index[0] if len(inds) > 0 else None

    INF_COLORS = {
        "low":         PC["blue"],
        "rising":      PC["yellow"],
        "high":        PC["red"],
        "falling":     PC["green"],
        "stagflation": PC["purple"],
    }
    INF_LABELS = {
        "low":         "Low",
        "rising":      "Rising",
        "high":        "High",
        "falling":     "Falling / Deflationary",
        "stagflation": "Stagflation",
    }

    fig, axes = plt.subplots(
        4, 1, figsize=(FIG_W, 13),
        gridspec_kw={"height_ratios": [3, 2.5, 2.5, 1.5]},
        sharex=True,
    )
    fig.patch.set_facecolor(PC["bg"])

    # ── Panel 1: Gold price + ROC ────────────────────────────────
    ax1 = axes[0]
    ax1.set_facecolor(PC["bg"])

    if gold_w is not None and len(gold_w) > 0:
        ax1.plot(gold_w.index, gold_w.values,
                 color=PC["yellow"], linewidth=2.0, label="Gold (GLD)")
        ax1.fill_between(gold_w.index, gold_w.values, alpha=0.07, color=PC["yellow"])
        last_g = gold_w.iloc[-1]
        ax1.annotate(f"${last_g:.0f}",
            xy=(gold_w.index[-1], last_g), xytext=(-60, 12),
            textcoords="offset points", fontsize=9,
            color=PC["yellow"], fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=PC["yellow"], lw=0.9))

    ax1.set_ylabel("Gold (USD)", fontsize=9, color=PC["label"])
    ax1.legend(fontsize=8, loc="upper left", frameon=True, edgecolor=PC["border"])
    ax1.grid(True, alpha=0.4)

    if roc_w is not None and len(roc_w) > 0:
        ax1b = ax1.twinx()
        ax1b.set_facecolor(PC["bg"])
        bar_colors = [PC["red"] if v < 0 else PC["green"] for v in roc_w.values]
        ax1b.bar(roc_w.index, roc_w.values, width=6,
                 color=bar_colors, alpha=0.45, label="ROC 60d (%)")
        ax1b.axhline(0, color=PC["zero"],  linewidth=0.9, alpha=0.7)
        ax1b.axhline(5, color=PC["green"], linewidth=0.7, linestyle=":", alpha=0.5)
        ax1b.axhline(8, color=PC["red"],   linewidth=0.7, linestyle=":", alpha=0.5)
        ax1b.set_ylabel("ROC 60d (%)", fontsize=8, color=PC["muted"])
        ax1b.tick_params(labelsize=7, colors=PC["muted"])
        for sp in ["top"]:
            ax1b.spines[sp].set_visible(False)
        last_roc = roc_w.iloc[-1]
        ax1b.annotate(f"ROC: {last_roc:+.1f}%",
            xy=(roc_w.index[-1], last_roc),
            xytext=(-70, -20 if last_roc > 0 else 15),
            textcoords="offset points", fontsize=8,
            color=PC["green"] if last_roc > 0 else PC["red"],
            arrowprops=dict(arrowstyle="->",
                color=PC["green"] if last_roc > 0 else PC["red"], lw=0.9))

    ax1.set_title(
        "Gold Price (USD)  &  60d ROC  |  ROC > 5% = inflation signal  |  ROC > 8% = high",
        fontsize=9, color=PC["muted"], loc="left", pad=4)
    _add_crisis_markers(ax1, hist_start)
    ax1.tick_params(labelsize=8)
    for sp in ["top", "right"]:
        ax1.spines[sp].set_visible(False)

    # ── Panel 2: Gold/Oil ratio + Z-score ───────────────────────
    ax2 = axes[1]
    ax2.set_facecolor(PC["bg"])

    if go_ratio_w is not None and len(go_ratio_w) > 0:
        ax2.plot(go_ratio_w.index, go_ratio_w.values,
                 color=PC["teal"], linewidth=1.8, label="Gold/Oil Ratio")
        ax2.fill_between(go_ratio_w.index, go_ratio_w.values,
                         alpha=0.07, color=PC["teal"])
        last_go = go_ratio_w.iloc[-1]
        ax2.annotate(f"{last_go:.2f}",
            xy=(go_ratio_w.index[-1], last_go), xytext=(-60, 12),
            textcoords="offset points", fontsize=9,
            color=PC["teal"], fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=PC["teal"], lw=0.9))

    ax2.set_ylabel("Gold/Oil", fontsize=9, color=PC["label"])
    ax2.grid(True, alpha=0.4)

    if go_z_w is not None and len(go_z_w) > 0:
        ax2b = ax2.twinx()
        ax2b.set_facecolor(PC["bg"])
        ax2b.plot(go_z_w.index, go_z_w.values,
                  color=PC["purple"], linewidth=1.5, linestyle="--",
                  alpha=0.85, label="Z-score")
        ax2b.axhline(0,    color=PC["zero"],  linewidth=0.8, alpha=0.6)
        ax2b.axhline(1.0,  color=PC["red"],   linewidth=0.7, linestyle=":", alpha=0.5)
        ax2b.axhline(-1.0, color=PC["green"], linewidth=0.7, linestyle=":", alpha=0.5)
        ax2b.set_ylabel("Z-score", fontsize=8, color=PC["muted"])
        ax2b.tick_params(labelsize=7, colors=PC["muted"])
        for sp in ["top"]:
            ax2b.spines[sp].set_visible(False)
        last_z = go_z_w.iloc[-1]
        ax2b.text(go_z_w.index[-1], last_z,
                  f"  z={last_z:.2f}", fontsize=8,
                  color=PC["red"] if last_z > 1.0 else
                        PC["green"] if last_z < -1.0 else PC["purple"])

    ax2.set_title(
        "Gold/Oil Ratio  |  z > +1.0 = deflationary fear  |  z < −1.0 = oil surge / inflation",
        fontsize=9, color=PC["muted"], loc="left", pad=4)
    ax2.legend(fontsize=8, loc="upper left", frameon=True, edgecolor=PC["border"])
    _add_crisis_markers(ax2, hist_start)
    ax2.tick_params(labelsize=8)
    for sp in ["top", "right"]:
        ax2.spines[sp].set_visible(False)

    # ── Panel 3: TIPS ROC vs 10Y Yield ROC ──────────────────────
    ax3 = axes[2]
    ax3.set_facecolor(PC["bg"])

    if tips_w is not None and len(tips_w) > 0:
        ax3.plot(tips_w.index, tips_w.values,
                 color=PC["teal"], linewidth=1.8, label="TIPS 60d ROC (%)")
    if y10y_w is not None and len(y10y_w) > 0:
        ax3.plot(y10y_w.index, y10y_w.values,
                 color=PC["blue"], linewidth=1.8, linestyle="--",
                 label="10Y Yield 60d ROC (%)")

    ax3.axhline(0, color=PC["zero"], linewidth=1.0, alpha=0.8)
    ax3.set_ylabel("ROC (%)", fontsize=9, color=PC["label"])
    ax3.grid(True, alpha=0.4)
    ax3.legend(fontsize=8, loc="upper left", frameon=True, edgecolor=PC["border"])
    ax3.set_title(
        "TIPS ROC & 10Y Yield ROC (60d)  |  both rising = high inflation",
        fontsize=9, color=PC["muted"], loc="left", pad=4)
    _add_crisis_markers(ax3, hist_start)
    ax3.tick_params(labelsize=8)
    for sp in ["top", "right"]:
        ax3.spines[sp].set_visible(False)

    # ── Panel 4: Inflation State timeline ───────────────────────
    ax4 = axes[3]
    ax4.set_facecolor(PC["bg"])

    if inf_m is not None and len(inf_m) > 0:
        prev_state = None
        seg_start  = None
        seen       = set()
        patches_legend = []

        for dt, state in inf_m.items():
            if state != prev_state:
                if prev_state is not None and seg_start is not None:
                    clr = INF_COLORS.get(prev_state, PC["muted"])
                    ax4.axvspan(seg_start, dt, alpha=0.70, color=clr, linewidth=0)
                    if prev_state not in seen:
                        patches_legend.append(
                            mpatches.Patch(color=clr, alpha=0.70,
                                label=INF_LABELS.get(prev_state, prev_state.title())))
                        seen.add(prev_state)
                seg_start  = dt
                prev_state = state

        if prev_state and seg_start:
            clr = INF_COLORS.get(prev_state, PC["muted"])
            ax4.axvspan(seg_start, inf_m.index[-1], alpha=0.70, color=clr, linewidth=0)
            if prev_state not in seen:
                patches_legend.append(
                    mpatches.Patch(color=clr, alpha=0.70,
                        label=INF_LABELS.get(prev_state, prev_state.title())))

        ax4.set_ylim(0, 1)
        ax4.set_yticks([])
        ax4.legend(handles=patches_legend, fontsize=7.5, loc="upper left",
                   frameon=True, edgecolor=PC["border"], ncol=5)
        ax4.set_title("Inflation State  (monthly classification)",
                      fontsize=9, color=PC["muted"], loc="left", pad=4)

    ax4.tick_params(labelsize=8)
    for sp in ["top", "right", "left"]:
        ax4.spines[sp].set_visible(False)

    # ── X axis formatting ────────────────────────────────────────
    axes[-1].xaxis.set_major_locator(_mdates.YearLocator(2))
    axes[-1].xaxis.set_major_formatter(_mdates.DateFormatter("%Y"))

    # ── Title ────────────────────────────────────────────────────
    try:
        snap      = ma.get_snapshot()
        inf_state = snap.inflation_env.upper()
        gold_last = p.get("gold_spot", p.get("gold", pd.Series())).iloc[-1] if (p.get("gold_spot") is not None or p.get("gold") is not None) else None
        roc_last  = inds["gold_roc_60"].iloc[-1] * 100 if "gold_roc_60" in inds.columns else None
        go_z_last = inds["gold_oil_zscore"].iloc[-1] if "gold_oil_zscore" in inds.columns else None
        parts = [f"State: {inf_state}"]
        if gold_last:          parts.append(f"Gold: ${gold_last:.0f}")
        if roc_last is not None: parts.append(f"ROC-60d: {roc_last:+.1f}%")
        if go_z_last is not None: parts.append(f"Gold/Oil z: {go_z_last:.2f}")
        subtitle = "  |  ".join(parts)
    except Exception:
        subtitle = ""

    _fig_title(fig,
               "Inflation Environment — Signals & Classification",
               subtitle)
    plt.tight_layout()
    return fig

# -----------------------------------------------------------------
#  CHART 26 — Crisis Signal Timeline (4-panel, print-ready)
#  Chapter 11: System Crisis Analysis
# -----------------------------------------------------------------

def crisis_timeline_print(
    ma:          "MacroAnalyzer",
    crisis_name: str,
    start_str:   str,
    end_str:     str,
    pre_months:  int = 12,
) -> Optional[plt.Figure]:
    """
    4-panel crisis signal timeline chart.

    Shows the evolution of macro signals BEFORE and DURING a crisis,
    with shaded zone for the crisis period and pre-crisis window.

    Parameters
    ----------
    ma           : MacroAnalyzer (with load() executed)
    crisis_name  : e.g. "GFC 2008"
    start_str    : crisis start date e.g. "2007-10-01"
    end_str      : crisis end date e.g. "2009-03-01"
    pre_months   : months of pre-crisis history to show (default 12)

    Panels
    ------
    1: Macro Score + Risk Mode background
    2: Yield Curve 10Y-3M (inversion highlighted)
    3: Cu/Gold z-score + VIX
    4: SPY price (context)
    """
    _apply_style()

    history = ma.get_history()
    inds    = ma.get_indicators()
    prices  = ma._prices

    if history.empty or inds.empty:
        return None

    # ── Date range ───────────────────────────────────────────────
    crisis_start = pd.Timestamp(start_str)
    crisis_end   = pd.Timestamp(end_str)
    window_start = crisis_start - pd.DateOffset(months=pre_months)
    # Clamp to available data
    window_start = max(window_start, history.index[0])
    window_end   = min(crisis_end + pd.DateOffset(months=3), history.index[-1])

    # Slice history and indicators to window
    h = history[(history.index >= window_start) & (history.index <= window_end)]
    i = inds[(inds.index >= window_start) & (inds.index <= window_end)]

    if h.empty:
        print(f"  crisis_timeline_print: no data for {crisis_name}")
        return None

    # Weekly resample
    hw = h.resample("W").last()
    iw = i.resample("W").last()

    # ── Series ───────────────────────────────────────────────────
    macro_score = hw["macro_score"].dropna()   if "macro_score"           in hw.columns else None
    risk_mode   = hw["risk_mode"].dropna()     if "risk_mode"             in hw.columns else None
    yield_curve = iw["yield_curve"].dropna()   if "yield_curve"           in iw.columns else None
    cg_z        = iw["copper_gold_zscore"].dropna() if "copper_gold_zscore" in iw.columns else None
    vix_s       = iw["vix"].dropna()           if "vix"                   in iw.columns else None

    # SPY from prices
    spy = None
    if "spy" in prices:
        sp = prices["spy"]
        spy = sp[(sp.index >= window_start) & (sp.index <= window_end)].resample("W").last()

    # ── Risk mode colors per date ─────────────────────────────────
    RISK_COLORS = {
        "risk_on":  PC["green"],
        "neutral":  PC["yellow"],
        "risk_off": PC["red"],
    }

    # ── Figure ───────────────────────────────────────────────────
    fig, axes = plt.subplots(
        4, 1, figsize=(FIG_W, 12),
        sharex=True, facecolor=PC["bg"],
        gridspec_kw={"height_ratios": [2.5, 2.0, 2.0, 1.5], "hspace": 0.08}
    )

    def _shade_crisis(ax):
        """Crisis shading (red) and pre-crisis window (yellow)."""
        ax.axvspan(crisis_start, min(crisis_end, window_end),
                   alpha=0.10, color=PC["red"], linewidth=0, zorder=0)
        ax.axvline(crisis_start, color=PC["red"], linewidth=1.2,
                   linestyle="--", alpha=0.7)
        ax.axvline(crisis_end if crisis_end <= window_end else window_end,
                   color=PC["red"], linewidth=0.8, linestyle=":", alpha=0.5)

    # ── Panel 1: Macro Score + Risk Mode background ───────────────
    ax1 = axes[0]
    ax1.set_facecolor(PC["bg"])

    # Risk mode colored background strips
    if risk_mode is not None and len(risk_mode) > 0:
        prev_rm  = None
        seg_s    = None
        for dt, rm in risk_mode.items():
            if rm != prev_rm:
                if prev_rm is not None and seg_s is not None:
                    ax1.axvspan(seg_s, dt,
                                alpha=0.12,
                                color=RISK_COLORS.get(prev_rm, PC["muted"]),
                                linewidth=0)
                seg_s   = dt
                prev_rm = rm
        if prev_rm and seg_s:
            ax1.axvspan(seg_s, risk_mode.index[-1],
                        alpha=0.12,
                        color=RISK_COLORS.get(prev_rm, PC["muted"]),
                        linewidth=0)

    if macro_score is not None and len(macro_score) > 0:
        ax1.fill_between(macro_score.index, macro_score.values,
                         alpha=0.15, color=PC["blue"])
        ax1.plot(macro_score.index, macro_score.values,
                 color=PC["blue"], linewidth=2.0, label="Macro Score")
        ax1.axhline(4.0, color=PC["yellow"], linewidth=0.9,
                    linestyle="--", alpha=0.7)
        ax1.axhline(7.5, color=PC["green"],  linewidth=0.9,
                    linestyle="--", alpha=0.7)
        ax1.set_ylim(0, 10)

        # Annotate last value
        last_ms = macro_score.iloc[-1]
        ax1.text(macro_score.index[-1], last_ms + 0.3,
                 f"{last_ms:.1f}", fontsize=8, color=PC["blue"])

    _shade_crisis(ax1)

    # Risk mode legend patches
    import matplotlib.patches as mpatches
    rm_patches = [mpatches.Patch(color=c, alpha=0.5, label=lbl)
                  for lbl, c in [("Risk On", PC["green"]),
                                 ("Neutral", PC["yellow"]),
                                 ("Risk Off", PC["red"])]]
    ms_line    = plt.Line2D([0], [0], color=PC["blue"], linewidth=2, label="Macro Score")
    ax1.legend(handles=[ms_line] + rm_patches,
               fontsize=7.5, loc="upper left",
               frameon=True, edgecolor=PC["border"], ncol=4)
    ax1.set_ylabel("Macro Score (0–10)", fontsize=9, color=PC["label"])
    ax1.grid(True, alpha=0.35)
    ax1.set_title(f"Macro Score & Risk Mode  |  background = Risk Mode",
                  fontsize=9, color=PC["muted"], loc="left", pad=4)
    ax1.tick_params(labelsize=8)
    for sp in ["top", "right"]:
        ax1.spines[sp].set_visible(False)

    # ── Panel 2: Yield Curve ─────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor(PC["bg"])

    if yield_curve is not None and len(yield_curve) > 0:
        ax2.fill_between(yield_curve.index,
                         yield_curve.clip(upper=0), 0,
                         alpha=0.30, color=PC["red"],   label="Inverted")
        ax2.fill_between(yield_curve.index,
                         yield_curve.clip(lower=0), 0,
                         alpha=0.15, color=PC["green"], label="Normal")
        ax2.plot(yield_curve.index, yield_curve.values,
                 color=PC["label"], linewidth=1.8)
        ax2.axhline(0, color=PC["red"], linewidth=1.2, alpha=0.8)
        ax2.axhline(1.5, color=PC["green"], linewidth=0.8,
                    linestyle=":", alpha=0.6)

        # Inversion label
        inv_mask = yield_curve < 0
        if inv_mask.any():
            first_inv = yield_curve[inv_mask].index[0]
            ax2.annotate("Inversion",
                         xy=(first_inv, -0.05),
                         xytext=(first_inv, -0.6),
                         fontsize=8, color=PC["red"],
                         arrowprops=dict(arrowstyle="->",
                                         color=PC["red"], lw=0.9))

    _shade_crisis(ax2)
    ax2.legend(fontsize=7.5, loc="upper left",
               frameon=True, edgecolor=PC["border"])
    ax2.set_ylabel("Yield Curve 10Y-3M (%)", fontsize=9, color=PC["label"])
    ax2.grid(True, alpha=0.35)
    ax2.set_title("Yield Curve 10Y−3M  |  red = inversion  |  threshold +1.5% (steep)",
                  fontsize=9, color=PC["muted"], loc="left", pad=4)
    ax2.tick_params(labelsize=8)
    for sp in ["top", "right"]:
        ax2.spines[sp].set_visible(False)

    # ── Panel 3: Cu/Gold z-score + VIX ───────────────────────────
    ax3  = axes[2]
    ax3b = ax3.twinx()
    ax3.set_facecolor(PC["bg"])
    ax3b.set_facecolor(PC["bg"])

    if cg_z is not None and len(cg_z) > 0:
        ax3.fill_between(cg_z.index,
                         cg_z.clip(lower=0), 0,
                         alpha=0.20, color=PC["green"])
        ax3.fill_between(cg_z.index,
                         cg_z.clip(upper=0), 0,
                         alpha=0.20, color=PC["red"])
        ax3.plot(cg_z.index, cg_z.values,
                 color=PC["teal"], linewidth=1.8, label="Cu/Gold z-score")
        ax3.axhline(0,    color=PC["zero"],  linewidth=1.0, alpha=0.8)
        ax3.axhline(0.3,  color=PC["green"], linewidth=0.7, linestyle=":", alpha=0.6)
        ax3.axhline(-0.5, color=PC["red"],   linewidth=0.7, linestyle=":", alpha=0.6)

    if vix_s is not None and len(vix_s) > 0:
        ax3b.plot(vix_s.index, vix_s.values,
                  color=PC["purple"], linewidth=1.5,
                  linestyle="--", alpha=0.80, label="VIX")
        ax3b.axhline(20, color=PC["purple"], linewidth=0.7,
                     linestyle=":", alpha=0.5)
        ax3b.axhline(35, color=PC["red"],    linewidth=0.7,
                     linestyle=":", alpha=0.5)
        ax3b.set_ylabel("VIX", fontsize=8, color=PC["purple"])
        ax3b.tick_params(labelsize=7, colors=PC["purple"])
        for s in ["top"]:
            ax3b.spines[s].set_visible(False)

    _shade_crisis(ax3)
    # Filter internal matplotlib artists (e.g. _child3, _child4)
    lines3  = [(l, l.get_label()) for l in ax3.get_lines() + ax3b.get_lines()
                if not l.get_label().startswith("_")]
    if lines3:
        _lines, _labels = zip(*lines3)
        ax3.legend(_lines, _labels, fontsize=7.5, loc="upper left",
                   frameon=True, edgecolor=PC["border"])
    ax3.set_ylabel("Cu/Gold z-score", fontsize=9, color=PC["label"])
    ax3.grid(True, alpha=0.35)
    ax3.set_title("Copper/Gold z-score (growth)  +  VIX (fear)",
                  fontsize=9, color=PC["muted"], loc="left", pad=4)
    ax3.tick_params(labelsize=8)
    for sp in ["top", "right"]:
        ax3.spines[sp].set_visible(False)

    # ── Panel 4: SPY ─────────────────────────────────────────────
    ax4 = axes[3]
    ax4.set_facecolor(PC["bg"])

    if spy is not None and len(spy) > 0:
        ax4.fill_between(spy.index, spy.values, alpha=0.10, color=PC["blue"])
        ax4.plot(spy.index, spy.values,
                 color=PC["blue"], linewidth=1.8, label="SPY")

        peak_idx  = spy.idxmax()
        peak_val  = spy.max()
        trough_val = spy.min()
        drawdown  = (trough_val - peak_val) / peak_val * 100

        ax4.annotate(f"Peak: ${peak_val:.0f}",
                     xy=(peak_idx, peak_val),
                     xytext=(peak_idx, peak_val * 1.05),
                     fontsize=8, color=PC["green"],
                     ha="center",
                     arrowprops=dict(arrowstyle="->",
                                     color=PC["green"], lw=0.9))
        ax4.text(0.99, 0.12, f"Max drawdown: {drawdown:.1f}%",
                 transform=ax4.transAxes,
                 fontsize=8, color=PC["red"],
                 ha="right", va="bottom")

    _shade_crisis(ax4)
    ax4.set_ylabel("SPY ($)", fontsize=9, color=PC["label"])
    ax4.legend(fontsize=7.5, loc="upper left",
               frameon=True, edgecolor=PC["border"])
    ax4.grid(True, alpha=0.35)
    ax4.set_title("SPY — price context  |  shading = crisis period",
                  fontsize=9, color=PC["muted"], loc="left", pad=4)
    ax4.tick_params(labelsize=8)
    for sp in ["top", "right"]:
        ax4.spines[sp].set_visible(False)

    # ── X axis ───────────────────────────────────────────────────
    axes[-1].xaxis.set_major_locator(_mdates.YearLocator(1))
    axes[-1].xaxis.set_major_formatter(_mdates.DateFormatter("%Y"))

    # ── Pre-crisis label ─────────────────────────────────────────
    axes[0].axvspan(window_start, crisis_start,
                    alpha=0.05, color=PC["yellow"], linewidth=0)
    mid_pre = window_start + (crisis_start - window_start) / 2
    axes[0].text(mid_pre, 9.2, "Pre-crisis window",
                 fontsize=7.5, color=PC["yellow"],
                 ha="center", va="top", alpha=0.85)

    # ── Main title ───────────────────────────────────────────────
    _fig_title(
        fig,
        f"Crisis Signal Timeline — {crisis_name}",
        f"{window_start.strftime('%b %Y')} → {window_end.strftime('%b %Y')}"
        f"  |  Crisis: {crisis_start.strftime('%b %Y')} – {crisis_end.strftime('%b %Y')}"
        f"  |  Pre-crisis window: {pre_months} months",
    )
    plt.tight_layout()
    return fig