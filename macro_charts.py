# -*- coding: utf-8 -*-
# macro_charts.py
"""
Macro Charts — Interactive & Static Visualizations for the Macro Analyzer
--------------------------------------------------------------------------
24 charts across 9 groups.

  Group 1 — Current Snapshot  (matplotlib, static)
    1.  signal_radar()               - Signal radar chart (current readings)
    2.  macro_gauge()                - Macro score gauge (0-10)
    3.  allocation_pie()             - Suggested asset allocation pie

  Group 2 — Historical Timeline  (Plotly, interactive)
    4.  phase_timeline()             - Cycle phase zones over time
    5.  ratios_history()             - Copper/Gold, Gold/Silver, Gold/Oil
    6.  yield_curve_history()        - Yield curve with inversion shading
    7.  vix_history()                - VIX with phase overlay
    8.  macro_score_history()        - Macro score with crisis markers

  Group 3 — Crisis Analysis  (Plotly)
    9.  crisis_heatmap()             - Pre-crisis signal heatmap
    10. zscore_bar()                 - Current z-scores vs history

  Group 4 — Asset Performance  (matplotlib + Plotly)
    11. phase_performance_bar()      - Annualized returns per cycle phase
    12. sector_rotation_wheel()      - Sector rotation by phase

  Group 5 — Divergence & Macro Signals  (Plotly)
    13. cg_yield_divergence()        - Cu/Gold vs 10Y yield divergence

  Group 6 — Recession & Volatility  (Plotly)
    14. probit_recession()           - Probit recession probability (Estrella & Mishkin)
    15. vix_mean_reversion()         - VIX mean reversion with half-life

  Group 7 — Credit, Dollar & Equity Regime  (Plotly)
    16. credit_spread()              - HYG/LQD credit spread (3 panels)
    17. dxy_history()                - DXY dollar index + momentum
    18. spy_sma200()                 - SPY vs SMA200 (bull/bear regime)

  Group 8 — Yields & Valuation  (Plotly)
    19. earnings_yield_gap()         - Earnings yield gap (EYG)
    20. real_yields()                - Real yields (DFII10 primary / TIP proxy fallback)

  Group 9 — Metals & Inflation  (Plotly)
    21. gold_silver_chart()          - Gold & Silver (dual axis + ratio)
    22. gold_silver_mean_reversion() - Gold/Silver ratio mean reversion
    23. inflation_environment()      - Inflation classification (4 panels)
    24. phase_gantt()                - Business cycle phase timeline (Gantt)

  Individual — Crisis Timeline  (Plotly, run per crisis)
      crisis_timeline()              - Signal evolution before/during a specific crisis

  FRED Signals  (Plotly, requires fred_api_key)
      fred_signals()                 - ISM Manufacturing PMI + TED Spread / FRA-OIS

Usage:
    from macro_charts import plot_all

    ma   = MacroAnalyzer(); ma.load()
    snap = ma.get_snapshot()

    plot_all(ma, snap)               # all 24 charts

    # by group
    plot_snapshot(snap)              # Group 1  (matplotlib)
    plot_history(ma)                 # Groups 2-3
    plot_performance(ma, snap)       # Group 4
    cg_yield_divergence(ma)          # Group 5
    probit_recession(ma)             # Group 6
    vix_mean_reversion(ma)           # Group 6
    credit_spread(ma)                # Group 7
    dxy_history(ma)                  # Group 7
    spy_sma200(ma)                   # Group 7
    earnings_yield_gap(ma)           # Group 8
    real_yields(ma)                  # Group 8
    gold_silver_chart(ma)            # Group 9
    gold_silver_mean_reversion(ma)   # Group 9
    inflation_environment(ma)        # Group 9
    phase_gantt(ma)                  # Group 9

    # crisis timeline (run individually)
    crisis_timeline(ma, "GFC 2008",   "2007-10-01", "2009-03-01")
    crisis_timeline(ma, "COVID",      "2020-02-01", "2020-04-01")
    crisis_timeline(ma, "Bear 2022",  "2022-01-01", "2022-10-01")
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import plotly.graph_objects as go
import plotly.subplots as ps
from plotly.subplots import make_subplots
from typing import Optional

warnings.filterwarnings("ignore")

try:
    from IPython.display import display as _ipy_display
    _IN_JUPYTER = True
except ImportError:
    _ipy_display = None
    _IN_JUPYTER = False

from macro_analyzer import (
    MacroAnalyzer, MacroSnapshot,
    TED_NORMAL, TED_ELEVATED, TED_CRISIS,
    ISM_EXPANSION, ISM_STRONG, ISM_WEAK,
)


#-----------------------------------------------------------------
# COLORS & STYLE
#-----------------------------------------------------------------

PHASE_COLORS = {
    "early_expansion":   "#2ecc71",   # green
    "late_expansion":    "#f39c12",   # orange-yellow
    "early_contraction": "#e67e22",   # orange
    "late_contraction":  "#e74c3c",   # red
    "unknown":           "#95a5a6",   # grey
}

PHASE_LABELS = {
    "early_expansion":   "Early Expansion",
    "late_expansion":    "Late Expansion",
    "early_contraction": "Early Contraction",
    "late_contraction":  "Late Contraction",
}

RISK_COLORS = {
    "risk_on":  "#27ae60",
    "neutral":  "#f39c12",
    "risk_off": "#c0392b",
}

DIRECTION_COLORS = {
    "bullish": "#27ae60",
    "bearish": "#c0392b",
    "neutral": "#7f8c8d",
}

CRISIS_EVENTS = {
    "Dot-com":    ("2000-03-01", "2002-10-01"),
    "9/11":       ("2001-09-01", "2001-12-01"),
    "GFC 2008":   ("2007-10-01", "2009-03-01"),
    "Euro":       ("2011-07-01", "2012-07-01"),
    "China":      ("2015-08-01", "2016-02-01"),
    "Q4 2018":    ("2018-10-01", "2018-12-31"),
    "COVID":      ("2020-02-01", "2020-04-01"),
    "Bear 2022":  ("2022-01-01", "2022-10-01"),
}

MPL_STYLE = {
    "figure.facecolor":  "#0d1117",
    "axes.facecolor":    "#161b22",
    "axes.edgecolor":    "#30363d",
    "axes.labelcolor":   "#c9d1d9",
    "axes.titlecolor":   "#f0f6fc",
    "text.color":        "#c9d1d9",
    "xtick.color":       "#8b949e",
    "ytick.color":       "#8b949e",
    "grid.color":        "#21262d",
    "grid.alpha":        0.6,
    "legend.facecolor":  "#161b22",
    "legend.edgecolor":  "#30363d",
}

PLOTLY_TEMPLATE = dict(
    paper_bgcolor = "#0d1117",
    plot_bgcolor  = "#161b22",
    font          = dict(color="#c9d1d9", family="monospace"),
)

# Applied via fig.update_xaxes() / fig.update_yaxes() to avoid conflicts
PLOTLY_AXIS_STYLE = dict(gridcolor="#21262d", linecolor="#30363d", color="#c9d1d9")


def _apply_mpl_style():
    plt.rcParams.update(MPL_STYLE)


#-----------------------------------------------------------------
# GROUP 1 - CURRENT SNAPSHOT
#-----------------------------------------------------------------

def signal_radar(snap: MacroSnapshot, ax=None) -> plt.Figure:
    """
    Chart 1: Radar chart for current signals.
    Each signal plotted o? z-score (clamped -3 to +3).
    """
    _apply_mpl_style()

    signals = snap.signals
    if not signals:
        print("No signals to plot.")
        return None

    names  = []
    values = []
    colors = []

    for key, sig in signals.items():
        z = sig.zscore
        if z is None or (isinstance(z, float) and np.isnan(z)):
            z = 0.0
        names.append(sig.name.replace(" (TIP proxy)", "").replace(" (10Y-3M)", ""))
        # Normalize: bullish = positive, bearish = negative
        # # If bearish signal ____ ______ z (_._. Gold/Silver high = bearish) _ __________
        if sig.direction == "bearish" and z > 0:
            z = -z
        elif sig.direction == "bullish" and z < 0:
            z = -z
        values.append(np.clip(z, -3, 3))
        colors.append(DIRECTION_COLORS.get(sig.direction, "#7f8c8d"))

    N = len(names)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    # Normalize values to 0-1 scale for radar
    vals_norm = [(v + 3) / 6 for v in values]
    vals_norm += vals_norm[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True),
                           facecolor="#0d1117")
    ax.set_facecolor("#161b22")

    # Fill
    ax.fill(angles, vals_norm, alpha=0.25,
            color="#3498db" if snap.risk_mode == "risk_on" else
                  "#e74c3c" if snap.risk_mode == "risk_off" else "#f39c12")
    ax.plot(angles, vals_norm, linewidth=2,
            color="#3498db" if snap.risk_mode == "risk_on" else
                  "#e74c3c" if snap.risk_mode == "risk_off" else "#f39c12")

    # Dots per signal with _____ direction
    for i, (angle, val, color) in enumerate(zip(angles[:-1], vals_norm[:-1], colors)):
        ax.scatter(angle, val, color=color, s=100, zorder=5)

    # Neutral line (center = 0.5)
    ax.plot(angles, [0.5] * len(angles), color="#30363d",
            linewidth=1, linestyle="--", alpha=0.5)

    # Labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(names, size=9, color="#c9d1d9")
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["Extreme\nBear", "", "Neutral", "", "Extreme\nBull"],
                       size=7, color="#8b949e")
    ax.set_ylim(0, 1)
    ax.grid(color="#21262d", alpha=0.6)

    phase_label = PHASE_LABELS.get(snap.cycle_phase, snap.cycle_phase)
    ax.set_title(
        f"Signal Radar - {snap.date.strftime('%d/%m/%Y')}\n"
        f"{phase_label}  |  Macro Score: {snap.macro_score:.1f}/10",
        color="#f0f6fc", size=13, pad=20, fontweight="bold"
    )

    plt.tight_layout()
    return fig


def macro_gauge(snap: MacroSnapshot) -> plt.Figure:
    """
    Chart 2: Gauge chart for macro score (0-10).
    """
    _apply_mpl_style()

    score = snap.macro_score
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.1, 1.2)
    ax.axis("off")

    # Background arc (full semicircle)
    theta = np.linspace(np.pi, 0, 200)
    ax.plot(np.cos(theta), np.sin(theta), color="#21262d", linewidth=30,
            solid_capstyle="round")

    # Color zones
    zones = [
        (0.0, 0.40, "#c0392b"),  # 0-4: bearish
        (0.40, 0.75, "#f39c12"),  # 4-7.5: mixed
        (0.75, 1.0,  "#27ae60"),  # 7.5-10: bull
    ]
    for start, end, color in zones:
        t = np.linspace(np.pi - start * np.pi, np.pi - end * np.pi, 100)
        ax.plot(np.cos(t), np.sin(t), color=color, linewidth=28, alpha=0.7,
                solid_capstyle="butt")

    # Needle
    needle_angle = np.pi - (score / 10) * np.pi
    needle_len   = 0.75
    ax.annotate("",
        xy     = (needle_len * np.cos(needle_angle), needle_len * np.sin(needle_angle)),
        xytext = (0, 0),
        arrowprops = dict(arrowstyle="-|>", color="white",
                          lw=3, mutation_scale=20)
    )
    ax.add_patch(plt.Circle((0, 0), 0.06, color="white", zorder=5))

    # Score text
    ax.text(0, -0.08, f"{score:.1f}", ha="center", va="center",
            fontsize=42, fontweight="bold", color="white")
    ax.text(0, -0.25, "/ 10", ha="center", va="center",
            fontsize=16, color="#8b949e")

    # Zone labels
    ax.text(-1.05, 0.05, "BEAR", ha="center", va="center",
            fontsize=9, color="#c0392b", fontweight="bold")

    ax.text(0.0, 0.88, "MIXED", ha="center", va="center",
            fontsize=9, color="#f39c12", fontweight="bold")
    ax.text(1.05, 0.05, "BULL", ha="center", va="center",
            fontsize=9, color="#27ae60", fontweight="bold")

    # Tick marks
    for i in range(11):
        angle = np.pi - (i / 10) * np.pi
        r_in, r_out = 0.82, 0.92
        ax.plot([r_in * np.cos(angle), r_out * np.cos(angle)],
                [r_in * np.sin(angle), r_out * np.sin(angle)],
                color="#8b949e", linewidth=1.5)
        ax.text(1.05 * np.cos(angle), 1.05 * np.sin(angle), str(i),
                ha="center", va="center", fontsize=8, color="#8b949e")

    phase_label = PHASE_LABELS.get(snap.cycle_phase, snap.cycle_phase)
    risk_color  = RISK_COLORS.get(snap.risk_mode, "#f39c12")
    ax.set_title(
        f"Macro Score Gauge\n"
        f"{phase_label}  |  {snap.risk_mode.replace('_',' ').upper()}",
        color="#f0f6fc", size=13, fontweight="bold", pad=10
    )

    plt.tight_layout()
    return fig


def allocation_pie(snap: MacroSnapshot) -> plt.Figure:
    """
    Chart 3: Asset allocation pie chart.
    """
    _apply_mpl_style()

    alloc = snap.asset_allocation
    if not alloc:
        return None

    labels = [k.capitalize() for k in alloc.keys()]
    sizes  = list(alloc.values())
    colors = ["#3498db", "#2ecc71", "#f39c12", "#95a5a6"][:len(labels)]
    explode = [0.05] * len(labels)

    fig, ax = plt.subplots(figsize=(7, 7), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")

    wedges, texts, autotexts = ax.pie(
        sizes,
        labels      = labels,
        colors      = colors,
        explode     = explode,
        autopct     = "%1.1f%%",
        startangle  = 140,
        textprops   = dict(color="#c9d1d9", fontsize=12),
        wedgeprops  = dict(edgecolor="#0d1117", linewidth=2),
        pctdistance = 0.75,
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_color("white")
        at.set_fontweight("bold")

    phase_label = PHASE_LABELS.get(snap.cycle_phase, snap.cycle_phase)
    ax.set_title(
        f"Suggested Asset Allocation\n{phase_label}",
        color="#f0f6fc", size=13, fontweight="bold", pad=15
    )

    plt.tight_layout()
    return fig


def plot_snapshot(snap: MacroSnapshot) -> None:
    """_________ ___ __ 3 snapshot charts."""
    print("Generating snapshot charts...")
    fig1 = signal_radar(snap)
    fig2 = macro_gauge(snap)
    fig3 = allocation_pie(snap)
    plt.show()
    print("Done.")


#-----------------------------------------------------------------
# GROUP 2 - HISTORICAL TIMELINE (PLOTLY)
#-----------------------------------------------------------------

def _add_crisis_shading(fig, row=1, col=1, history_start=None, history_end=None):
    """Helper: adds crisis shading - no annotations for render speed."""
    for name, (start, end) in CRISIS_EVENTS.items():
        s = pd.Timestamp(start)
        if history_start and s < pd.Timestamp(history_start):
            continue
        fig.add_vrect(
            x0=start, x1=end,
            fillcolor="rgba(231,76,60,0.06)",
            layer="below", line_width=0,
            row=row, col=col,
        )


def _add_phase_shading(fig, history: pd.DataFrame, row=1, col=1):
    """
    Adds phase color shading to plotly figure.
    Uses monthly resampling to keep vrect count low (<= ~60).
    More vrects = slower plotly render.
    """
    if history.empty:
        return

    # Monthly resample keeps transition count very low
    phases = history["cycle_phase"].resample("ME").last().ffill().dropna()
    mask   = phases != phases.shift(1)
    transitions = phases[mask]

    dates      = transitions.index.tolist()
    phase_list = transitions.values.tolist()

    # Cap at 80 vrects max to keep render fast
    if len(dates) > 80:
        # Keep only every Nth transition
        step  = len(dates) // 80 + 1
        dates      = dates[::step]
        phase_list = phase_list[::step]

    for i, (start_date, phase) in enumerate(zip(dates, phase_list)):
        end_date = dates[i+1] if i+1 < len(dates) else history.index[-1]
        color = PHASE_COLORS.get(phase, "#95a5a6")
        fig.add_vrect(
            x0=str(start_date.date()),
            x1=str(end_date.date()),
            fillcolor=color,
            opacity=0.08,
            layer="below",
            line_width=0,
            row=row, col=col,
        )


def phase_timeline(ma: MacroAnalyzer, years: int = None) -> go.Figure:
    """
    Chart 4: Interactive phase timeline.
    Chromatiste? zone? for kathe phase, macro score line.
    """
    history = ma.get_history()
    if history.empty:
        return None

    if years:
        cutoff  = history.index[-1] - pd.DateOffset(years=years)
        history = history[history.index >= cutoff]

    # Resample __ ___________ for ________
    weekly = history.resample("W").agg({
        "cycle_phase":   lambda x: x.mode().iloc[0] if len(x) > 0 and len(x.mode()) > 0 else "unknown",
        "risk_mode":     lambda x: x.mode().iloc[0] if len(x) > 0 and len(x.mode()) > 0 else "neutral",
        "macro_score":   "mean",
        "yield_curve":   "last",
        "vix":           "mean",
        "copper_gold":   "last",
    })

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=("Macro Score", "Yield Curve (10Y-3M) %", "VIX"),
        row_heights=[0.5, 0.25, 0.25],
    )

    # Phase color map for scatter
    phase_color_series = weekly["cycle_phase"].map(PHASE_COLORS).fillna("#95a5a6")

    # Row 1: Macro Score
    fig.add_trace(go.Scatter(
        x    = weekly.index,
        y    = weekly["macro_score"],
        mode = "lines",
        name = "Macro Score",
        line = dict(color="#3498db", width=2),
        fill = "tozeroy",
        fillcolor = "rgba(52,152,219,0.1)",
    ), row=1, col=1)

    # Phase dots on macro score
    for phase, color in PHASE_COLORS.items():
        mask = weekly["cycle_phase"] == phase
        if mask.any():
            fig.add_trace(go.Scatter(
                x    = weekly.index[mask],
                y    = weekly["macro_score"][mask],
                mode = "markers",
                name = PHASE_LABELS.get(phase, phase),
                marker = dict(color=color, size=4, opacity=0.6),
                showlegend = True,
            ), row=1, col=1)

    # Row 2: Yield Curve
    yc = weekly["yield_curve"].dropna()
    fig.add_trace(go.Scatter(
        x    = yc.index,
        y    = yc.values,
        mode = "lines",
        name = "Yield Curve",
        line = dict(color="#f39c12", width=1.5),
        showlegend = False,
    ), row=2, col=1)

    # Inversion shading
    fig.add_hline(y=0, line_dash="dash", line_color="#e74c3c",
                  line_width=1, row=2, col=1)

    # Fill inversion areas
    yc_df = yc.to_frame("yc")
    yc_df["neg"] = yc_df["yc"].clip(upper=0)
    yc_df["pos"] = yc_df["yc"].clip(lower=0)

    fig.add_trace(go.Scatter(
        x    = yc_df.index,
        y    = yc_df["neg"],
        fill = "tozeroy",
        fillcolor = "rgba(231,76,60,0.3)",
        line = dict(width=0),
        name = "Inverted",
        showlegend = False,
    ), row=2, col=1)

    # Row 3: VIX
    vix = weekly["vix"].dropna()
    fig.add_trace(go.Scatter(
        x    = vix.index,
        y    = vix.values,
        mode = "lines",
        name = "VIX",
        line = dict(color="#9b59b6", width=1.5),
        fill = "tozeroy",
        fillcolor = "rgba(155,89,182,0.1)",
        showlegend = False,
    ), row=3, col=1)

    fig.add_hline(y=20, line_dash="dot", line_color="#f39c12",
                  line_width=1, row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="#e74c3c",
                  line_width=1, row=3, col=1)

    # Crisis shading on row 1 only for render speed
    _add_crisis_shading(fig, row=1, col=1,
                        history_start=str(history.index[0].date()))

    fig.update_layout(
        title=dict(text="Macro Phase Timeline", font=dict(size=18, color="#f0f6fc")),
        height=700,
        hovermode="x unified",
        **PLOTLY_TEMPLATE,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#21262d")
    fig.update_yaxes(showgrid=True, gridcolor="#21262d")

    return fig


def ratios_history(ma: MacroAnalyzer, years: int = None) -> go.Figure:
    """
    Chart 5: Copper/Gold, Gold/Silver, Gold/Oil over time.
    """
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

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(
            "Copper/Gold Ratio (Growth Signal)",
            "Gold/Silver Ratio (Risk Appetite)",
            "Gold/Oil Ratio (Stress Indicator)",
        ),
    )

    # Copper/Gold
    if "copper_gold" in weekly_h.columns:
        cg = weekly_h["copper_gold"].dropna()
        fig.add_trace(go.Scatter(
            x=cg.index, y=cg.values, mode="lines",
            name="Cu/Gold", line=dict(color="#e67e22", width=2),
        ), row=1, col=1)
        if "copper_gold_zscore" in weekly_i.columns:
            cg_z = weekly_i["copper_gold_zscore"].dropna()
            fig.add_trace(go.Scatter(
                x=cg_z.index, y=cg_z.values, mode="lines",
                name="Cu/Gold Z-score",
                line=dict(color="#f39c12", width=1, dash="dot"),
                yaxis="y4",
            ), row=1, col=1)

    # Gold/Silver
    if "gold_silver" in weekly_h.columns:
        gs = weekly_h["gold_silver"].dropna()
        fig.add_trace(go.Scatter(
            x=gs.index, y=gs.values, mode="lines",
            name="Gold/Silver", line=dict(color="#f1c40f", width=2),
        ), row=2, col=1)

    # Gold/Oil
    if "gold_oil" in weekly_h.columns:
        go_r = weekly_h["gold_oil"].dropna()
        fig.add_trace(go.Scatter(
            x=go_r.index, y=go_r.values, mode="lines",
            name="Gold/Oil", line=dict(color="#1abc9c", width=2),
            fill="tozeroy", fillcolor="rgba(26,188,156,0.1)",
        ), row=3, col=1)

    # Phase shading on row 1 only for render speed
    _add_phase_shading(fig, history, row=1, col=1)
    _add_crisis_shading(fig, row=1, col=1,
                        history_start=str(history.index[0].date()))

    fig.update_layout(
        title=dict(text="Key Macro Ratios History", font=dict(size=18, color="#f0f6fc")),
        height=750,
        hovermode="x unified",
        **PLOTLY_TEMPLATE,
    )

    return fig


def yield_curve_history(ma: MacroAnalyzer) -> go.Figure:
    """
    Chart 6: Yield curve history with inversion shading kai recession markers.
    """
    inds    = ma.get_indicators()
    history = ma.get_history()

    if inds.empty or "yield_curve" not in inds.columns:
        return None

    weekly_i = inds.resample("W").last()
    weekly_h = history.resample("W").last()

    yc  = weekly_i["yield_curve"].dropna()
    y10 = weekly_i["yield_10y"].dropna() if "yield_10y" in weekly_i.columns else None
    y3m = weekly_i["yield_3m"].dropna()  if "yield_3m"  in weekly_i.columns else None

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=(
            "Yield Curve Spread (10Y - 3M)  - red = inversion",
            "10Y vs 3M Treasury Yields",
        ),
        row_heights=[0.55, 0.45],
    )

    # Yield curve spread
    yc_pos = yc.clip(lower=0)
    yc_neg = yc.clip(upper=0)

    fig.add_trace(go.Scatter(
        x=yc.index, y=yc_pos.values, mode="lines",
        fill="tozeroy", fillcolor="rgba(39,174,96,0.2)",
        line=dict(color="#27ae60", width=0), name="Positive spread",
        showlegend=False,
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=yc.index, y=yc_neg.values, mode="lines",
        fill="tozeroy", fillcolor="rgba(231,76,60,0.35)",
        line=dict(color="#e74c3c", width=0), name="Inversion",
        showlegend=False,
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=yc.index, y=yc.values, mode="lines",
        line=dict(color="#f0f6fc", width=1.5), name="10Y-3M Spread",
    ), row=1, col=1)

    fig.add_hline(y=0, line_dash="dash", line_color="#e74c3c",
                  line_width=1.5, row=1, col=1)

    # Individual yields
    if y10 is not None:
        fig.add_trace(go.Scatter(
            x=y10.index, y=y10.values, mode="lines",
            line=dict(color="#3498db", width=1.5), name="10Y Yield",
        ), row=2, col=1)

    if y3m is not None:
        fig.add_trace(go.Scatter(
            x=y3m.index, y=y3m.values, mode="lines",
            line=dict(color="#e74c3c", width=1.5), name="3M Yield",
        ), row=2, col=1)

    # Crisis shading on row 1 only
    _add_crisis_shading(fig, row=1, col=1)

    fig.update_layout(
        title=dict(text="Yield Curve History", font=dict(size=18, color="#f0f6fc")),
        height=650,
        hovermode="x unified",
        **PLOTLY_TEMPLATE,
    )

    return fig


def vix_history(ma: MacroAnalyzer, years: int = None) -> go.Figure:
    """
    Chart 7: VIX history with phase overlay kai threshold lines.
    """
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

    fig = go.Figure()

    vix = weekly_i["vix"].dropna() if "vix" in weekly_i.columns else None
    if vix is None:
        return None

    # VIX line
    fig.add_trace(go.Scatter(
        x=vix.index, y=vix.values, mode="lines",
        name="VIX", line=dict(color="#9b59b6", width=2),
        fill="tozeroy", fillcolor="rgba(155,89,182,0.12)",
    ))

    # Threshold lines
    fig.add_hline(y=15, line_dash="dot", line_color="#27ae60",
                  line_width=1, annotation_text="Low Volatility (15)",
                  annotation_font_color="#27ae60")
    fig.add_hline(y=25, line_dash="dot", line_color="#f39c12",
                  line_width=1.5, annotation_text="Elevated (25)",
                  annotation_font_color="#f39c12")
    fig.add_hline(y=35, line_dash="dot", line_color="#e74c3c",
                  line_width=2, annotation_text="Stress (35)",
                  annotation_font_color="#e74c3c")

    # Phase shading
    _add_phase_shading(fig, weekly_h)

    # Crisis shading
    _add_crisis_shading(fig, history_start=str(history.index[0].date()))

    # Current VIX annotation
    last_vix = vix.iloc[-1]
    fig.add_annotation(
        x=vix.index[-1], y=last_vix,
        text=f"Now: {last_vix:.1f}",
        showarrow=True, arrowhead=2,
        font=dict(color="white", size=12),
        bgcolor="#9b59b6", bordercolor="#9b59b6",
    )

    fig.update_layout(
        title=dict(text="VIX History with Phase Overlay", font=dict(size=18, color="#f0f6fc")),
        yaxis_title="VIX Level",
        height=500,
        hovermode="x unified",
        **PLOTLY_TEMPLATE,
    )

    return fig


def macro_score_history(ma: MacroAnalyzer) -> go.Figure:
    """
    Chart 8: Macro score over time with crisis markers.
    """
    history = ma.get_history()
    if history.empty:
        return None

    weekly = history.resample("W").agg({
        "macro_score": "mean",
        "cycle_phase": lambda x: x.mode().iloc[0] if len(x) > 0 and len(x.mode()) > 0 else "unknown",
        "risk_mode":   lambda x: x.mode().iloc[0] if len(x) > 0 and len(x.mode()) > 0 else "neutral",
    })

    fig = go.Figure()

    # Macro score area
    fig.add_trace(go.Scatter(
        x=weekly.index, y=weekly["macro_score"].values,
        mode="lines", name="Macro Score",
        line=dict(color="#3498db", width=2),
        fill="tozeroy", fillcolor="rgba(52,152,219,0.12)",
    ))

    # Risk mode colored dots
    for risk, color in RISK_COLORS.items():
        mask = weekly["risk_mode"] == risk
        if mask.any():
            fig.add_trace(go.Scatter(
                x=weekly.index[mask],
                y=weekly["macro_score"][mask],
                mode="markers",
                name=risk.replace("_", " ").title(),
                marker=dict(color=color, size=5, opacity=0.7),
            ))

    # Threshold lines
    fig.add_hline(y=4.0, line_dash="dash", line_color="#f39c12",
                  line_width=1, annotation_text="Watchlist threshold (4.0)",
                  annotation_font_color="#f39c12")
    fig.add_hline(y=7.5, line_dash="dash", line_color="#27ae60",
                  line_width=1, annotation_text="Strong threshold (7.5)",
                  annotation_font_color="#27ae60")

    # Crisis markers
    for name, (start, end) in CRISIS_EVENTS.items():
        s = pd.Timestamp(start)
        if s < history.index[0]:
            continue
        avail = weekly.index[weekly.index >= s]
        if avail.empty:
            continue
        idx   = avail[0]
        score = weekly.loc[idx, "macro_score"]
        fig.add_annotation(
            x=idx, y=score,
            text=name,
            showarrow=True,
            arrowhead=2,
            arrowcolor="#e74c3c",
            font=dict(size=9, color="#e74c3c"),
            bgcolor="rgba(13,17,23,0.8)",
            bordercolor="#e74c3c",
        )

    fig.update_layout(
        title=dict(text="Macro Score History with Crisis Markers",
                   font=dict(size=18, color="#f0f6fc")),
        yaxis_title="Macro Score (0-10)",
        yaxis=dict(range=[0, 10], gridcolor="#21262d", linecolor="#30363d"),
        height=500,
        hovermode="x unified",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        font=dict(color="#c9d1d9", family="monospace"),
        xaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
        legend=dict(bgcolor="#161b22"),
    )

    return fig


def plot_history(ma: MacroAnalyzer, years: int = None) -> None:
    """_________ ___ __ historical charts."""
    print("Generating historical charts...")
    fig4 = phase_timeline(ma, years=years)
    fig5 = ratios_history(ma, years=years)
    fig6 = yield_curve_history(ma)
    fig7 = vix_history(ma, years=years)
    fig8 = macro_score_history(ma)
    for fig in [fig4, fig5, fig6, fig7, fig8]:
        if fig:
            fig.show()
    print("Done.")


#-----------------------------------------------------------------
# GROUP 3 - CRISIS ANALYSIS
#-----------------------------------------------------------------

def crisis_heatmap(ma: MacroAnalyzer) -> go.Figure:
    """
    Chart 9: Heatmap - ti edeichnan ta signals prin apo kathe krish.
    Rows: krisei?, Columns: signals se 6M/3M/1M prin.
    """
    history    = ma.get_history()
    indicators = ma.get_indicators()

    if history.empty:
        return None

    signal_keys = ["macro_score", "yield_curve", "copper_gold", "vix", "credit_ratio"]
    signal_labels = {
        "macro_score":   "Macro Score",
        "yield_curve":   "Yield Curve",
        "copper_gold":   "Cu/Gold",
        "vix":           "VIX",
        "credit_ratio":  "Credit (HYG/LQD)",
    }

    lookbacks = [6, 3, 1]
    columns   = []
    for lb in lookbacks:
        for sk in signal_keys:
            columns.append(f"{lb}M - {signal_labels.get(sk, sk)}")

    rows_labels = []
    matrix      = []

    for event_name, (start_str, end_str) in CRISIS_EVENTS.items():
        start_ts = pd.Timestamp(start_str)
        if start_ts < history.index[0]:
            continue

        row_vals = []
        for months_before in lookbacks:
            lookback_date = start_ts - pd.DateOffset(months=months_before)
            avail = history.index[history.index <= lookback_date]
            if avail.empty:
                for _ in signal_keys:
                    row_vals.append(np.nan)
                continue

            actual  = avail[-1]
            h_row   = history.loc[actual]
            i_row   = indicators.loc[actual] if actual in indicators.index else pd.Series()

            for sk in signal_keys:
                val = h_row.get(sk, np.nan)
                if pd.isna(val) and sk in i_row:
                    val = i_row[sk]
                row_vals.append(val if pd.notna(val) else np.nan)

        rows_labels.append(event_name)
        matrix.append(row_vals)

    if not matrix:
        return None

    z_matrix = np.array(matrix, dtype=float)

    # Normalize ____ column for colorscale (z-score across events)
    z_norm = np.zeros_like(z_matrix)
    for j in range(z_matrix.shape[1]):
        col = z_matrix[:, j]
        valid = col[~np.isnan(col)]
        if len(valid) > 1:
            mu  = np.nanmean(col)
            std = np.nanstd(col)
            z_norm[:, j] = (col - mu) / (std if std > 0 else 1)
        else:
            z_norm[:, j] = 0

    # Text annotations
    text_matrix = []
    for row in matrix:
        text_row = []
        for val in row:
            if np.isnan(val):
                text_row.append("N/A")
            elif abs(val) < 10:
                text_row.append(f"{val:.2f}")
            else:
                text_row.append(f"{val:.0f}")
        text_matrix.append(text_row)

    fig = go.Figure(data=go.Heatmap(
        z           = z_norm,
        x           = columns,
        y           = rows_labels,
        text        = text_matrix,
        texttemplate = "%{text}",
        textfont    = dict(size=9),
        colorscale  = [
            [0.0,  "#c0392b"],
            [0.35, "#e67e22"],
            [0.5,  "#f0f6fc"],
            [0.65, "#27ae60"],
            [1.0,  "#1a5276"],
        ],
        zmid        = 0,
        showscale   = True,
        colorbar    = dict(
            title=dict(text="Normalized", font=dict(color="#c9d1d9")),
            tickfont=dict(color="#c9d1d9"),
        ),
    ))

    fig.update_layout(
        title=dict(
            text="Pre-Crisis Signal Heatmap<br><sup>Ti edeichnan ta signals 6M / 3M / 1M prin apo kathe krish</sup>",
            font=dict(size=16, color="#f0f6fc"),
        ),
        height=max(400, len(rows_labels) * 60 + 150),
        **PLOTLY_TEMPLATE,
    )
    fig.update_xaxes(tickangle=-35, tickfont=dict(size=9), **PLOTLY_AXIS_STYLE)
    fig.update_yaxes(**PLOTLY_AXIS_STYLE)

    return fig


def zscore_bar(snap: MacroSnapshot) -> go.Figure:
    """
    Chart 10: Bar chart trechonton z-scores.
    Deichnei poso akraia einai kathe signal vs istoria.
    """
    signals = snap.signals
    if not signals:
        return None

    names  = []
    zscores = []
    colors  = []
    pcts    = []

    for key, sig in signals.items():
        z = sig.zscore
        if z is None or (isinstance(z, float) and np.isnan(z)):
            continue
        names.append(sig.name.replace(" (TIP proxy)", "").replace(" (10Y-3M)", ""))
        zscores.append(round(z, 2))
        colors.append(DIRECTION_COLORS.get(sig.direction, "#7f8c8d"))
        p = sig.percentile
        pcts.append(f"{p:.0f}th pct" if p and not np.isnan(p) else "")

    if not names:
        return None

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x           = zscores,
        y           = names,
        orientation = "h",
        marker      = dict(color=colors, opacity=0.85),
        text        = [f"{z:+.2f}s  {p}" for z, p in zip(zscores, pcts)],
        textposition = "auto",
        textfont    = dict(color="white", size=10),
    ))

    # Reference lines
    for x_val, label, color in [(-2, "-2s", "#e74c3c"), (2, "+2s", "#27ae60"),
                                  (-1, "-1s", "#e67e22"), (1, "+1s", "#f39c12")]:
        fig.add_vline(x=x_val, line_dash="dot", line_color=color,
                      line_width=1, opacity=0.6)

    fig.add_vline(x=0, line_color="#c9d1d9", line_width=1.5)

    fig.update_layout(
        title=dict(
            text=f"Current Signal Z-Scores - {snap.date.strftime('%d/%m/%Y')}<br>"
                 f"<sup> red = bearish / green = bullish  |  >2sd = extreme </sup>",
            font=dict(size=15, color="#f0f6fc"),
        ),
        bargap=0.25,
        **PLOTLY_TEMPLATE,
    )
    fig.update_xaxes(**PLOTLY_AXIS_STYLE)
    fig.update_yaxes(**PLOTLY_AXIS_STYLE)

    return fig


#-----------------------------------------------------------------
# GROUP 4 - ASSET PERFORMANCE
#-----------------------------------------------------------------

def phase_performance_bar(ma: MacroAnalyzer) -> go.Figure:
    """
    Chart 11: Grouped bar chart - return assets per phase.
    """
    history = ma.get_history()
    if history.empty:
        return None

    assets = {
        "SPY":  ma._prices.get("spy"),
        "GLD":  ma._prices.get("gold_spot", ma._prices.get("gold")),
        "SLV":  ma._prices.get("silver_spot", ma._prices.get("silver")),
        "XLE":  ma._prices.get("xle"),
        "XLK":  ma._prices.get("xlk"),
        "XLU":  ma._prices.get("xlu"),
        "XLF":  ma._prices.get("xlf"),
        "XLV":  ma._prices.get("xlv"),
    }
    assets = {k: v for k, v in assets.items() if v is not None}

    phases = ["early_expansion", "late_expansion", "early_contraction", "late_contraction"]
    asset_colors = {
        "SPY": "#3498db", "GLD": "#f1c40f", "SLV": "#95a5a6",
        "XLE": "#e67e22", "XLK": "#9b59b6", "XLU": "#1abc9c",
        "XLF": "#e74c3c", "XLV": "#2ecc71",
    }

    # ___________ ________ per phase
    results = {asset: {} for asset in assets}

    for phase in phases:
        phase_mask  = history["cycle_phase"] == phase
        phase_dates = history.index[phase_mask]

        for name, prices in assets.items():
            monthly_returns = []
            in_phase     = False
            period_start = None

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

            if monthly_returns:
                avg_ret  = np.mean(monthly_returns)
                avg_days = len(phase_dates) / max(len(monthly_returns), 1)
                ann      = avg_ret * (252 / max(avg_days, 1))
                # Cap outliers (COVID recovery distortion)
                ann = np.clip(ann, -1.5, 1.5)
                results[name][phase] = round(ann * 100, 1)
            else:
                results[name][phase] = None

    fig = go.Figure()

    for name, color in asset_colors.items():
        if name not in results:
            continue
        y_vals = [results[name].get(p) for p in phases]
        fig.add_trace(go.Bar(
            name  = name,
            x     = [PHASE_LABELS.get(p, p) for p in phases],
            y     = y_vals,
            marker_color = color,
            text  = [f"{v:+.1f}%" if v is not None else "N/A" for v in y_vals],
            textposition = "outside",
            textfont = dict(size=9),
        ))

    fig.add_hline(y=0, line_color="#c9d1d9", line_width=1)

    fig.update_layout(
        title=dict(
            text="Asset Performance by Macro Phase<br><sup>Annualized returns - outliers capped at ?150%</sup>",
            font=dict(size=16, color="#f0f6fc"),
        ),
        barmode    = "group",
        yaxis_title = "Annualized Return %",
        height     = 550,
        bargap     = 0.15,
        bargroupgap = 0.05,
        **PLOTLY_TEMPLATE,
    )

    return fig


def sector_rotation_wheel(snap: MacroSnapshot) -> plt.Figure:
    """
    Chart 12: Sector rotation wheel - which sectors favored at each phase.
    Static matplotlib.
    """
    _apply_mpl_style()

    phases_ordered = [
        "early_expansion",
        "late_expansion",
        "early_contraction",
        "late_contraction",
    ]

    phase_sectors = {
        "early_expansion":   ["Financials", "Industrials", "Tech", "Materials"],
        "late_expansion":    ["Energy", "Materials", "Industrials", "Value"],
        "early_contraction": ["Healthcare", "Staples", "Utilities", "Gold"],
        "late_contraction":  ["Staples", "Utilities", "Gold", "Bonds"],
    }

    fig, ax = plt.subplots(figsize=(10, 10), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.axis("off")

    # Draw wheel
    n_phases = len(phases_ordered)
    for i, phase in enumerate(phases_ordered):
        angle_start = i * 90
        angle_end   = (i + 1) * 90
        color       = PHASE_COLORS.get(phase, "#95a5a6")

        # Pie wedge
        theta = np.linspace(np.radians(angle_start), np.radians(angle_end), 50)
        x_arc = np.concatenate([[0], np.cos(theta), [0]])
        y_arc = np.concatenate([[0], np.sin(theta), [0]])
        ax.fill(x_arc, y_arc, color=color, alpha=0.25, zorder=1)
        ax.plot(np.cos(theta), np.sin(theta), color=color, linewidth=2, zorder=2)

        # Divider lines
        ax.plot([0, np.cos(np.radians(angle_start))],
                [0, np.sin(np.radians(angle_start))],
                color="#0d1117", linewidth=2, zorder=3)

        # Phase label
        mid_angle = np.radians((angle_start + angle_end) / 2)
        lx = 0.65 * np.cos(mid_angle)
        ly = 0.65 * np.sin(mid_angle)
        label = PHASE_LABELS.get(phase, phase).replace(" ", "\n")
        ax.text(lx, ly, label, ha="center", va="center",
                fontsize=9, color=color, fontweight="bold", zorder=4)

        # Sectors in outer ring
        sectors = phase_sectors.get(phase, [])
        for j, sector in enumerate(sectors):
            frac      = (j + 0.5) / len(sectors)
            sec_angle = np.radians(angle_start + frac * 90)
            r         = 1.1
            sx = r * np.cos(sec_angle)
            sy = r * np.sin(sec_angle)
            ax.text(sx, sy, sector, ha="center", va="center",
                    fontsize=8, color="#c9d1d9",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=color,
                              alpha=0.3, edgecolor=color))

    # Current phase indicator
    current_phase = snap.cycle_phase
    if current_phase in phases_ordered:
        idx   = phases_ordered.index(current_phase)
        mid_a = np.radians(idx * 90 + 45)
        ax.annotate("",
            xy     = (0.85 * np.cos(mid_a), 0.85 * np.sin(mid_a)),
            xytext = (0, 0),
            arrowprops = dict(
                arrowstyle="-|>",
                color="white",
                lw=3,
                mutation_scale=25,
            ),
            zorder=5,
        )

    # Center circle
    circle = plt.Circle((0, 0), 0.35, color="#161b22", zorder=4)
    ax.add_patch(circle)
    ax.text(0, 0.06, "Sector", ha="center", va="center",
            fontsize=10, color="#f0f6fc", fontweight="bold")
    ax.text(0, -0.06, "Rotation", ha="center", va="center",
            fontsize=10, color="#f0f6fc", fontweight="bold")

    # Outer circle border
    theta_full = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta_full), np.sin(theta_full),
            color="#30363d", linewidth=1.5, zorder=2)

    # Clock direction arrow hint
    ax.annotate("", xy=(0, 1.38), xytext=(-0.15, 1.38),
                arrowprops=dict(arrowstyle="->", color="#8b949e", lw=1.5))
    ax.text(0.1, 1.42, "Cycle direction", fontsize=8, color="#8b949e")

    # Current phase legend
    c_color = PHASE_COLORS.get(snap.cycle_phase, "#95a5a6")
    ax.text(0, -1.42,
            f"Current: {PHASE_LABELS.get(snap.cycle_phase, '')}",
            ha="center", va="center",
            fontsize=11, color=c_color, fontweight="bold")

    ax.set_title("Sector Rotation Wheel\nArrow = current phase",
                 color="#f0f6fc", size=13, fontweight="bold", pad=10)

    plt.tight_layout()
    return fig


#-----------------------------------------------------------------
# CONVENIENCE WRAPPERS
#-----------------------------------------------------------------

def plot_performance(ma: MacroAnalyzer, snap: MacroSnapshot) -> None:
    """Group 4: performance + rotation wheel."""
    fig11 = phase_performance_bar(ma)
    if fig11:
        fig11.show()
    fig12 = sector_rotation_wheel(snap)
    if fig12:
        plt.show()


def plot_crisis(ma: MacroAnalyzer, snap: MacroSnapshot) -> None:
    """Group 3: crisis heatmap + zscore bar."""
    fig9  = crisis_heatmap(ma)
    fig10 = zscore_bar(snap)
    if fig9:
        fig9.show()
    if fig10:
        fig10.show()


def plot_all(ma: MacroAnalyzer, snap: MacroSnapshot) -> None:
    """
    Renders all macro charts (25 total across 9 groups).

    Usage in notebook:
        plot_all(ma, snap)
    """
    print("=" * 55)
    print("  MACRO CHARTS - Full Report")
    print("=" * 55)

    print("\n[Group 1] Current Snapshot...")
    fig1 = signal_radar(snap)
    fig2 = macro_gauge(snap)
    fig3 = allocation_pie(snap)
    for f in [fig1, fig2, fig3]:
        if f:
            plt.figure(f.number)
            if _ipy_display:
                _ipy_display(f)
                plt.close(f)
            else:
                plt.show()

    print("\n[Group 2] Historical Timeline...")
    fig4 = phase_timeline(ma)
    fig5 = ratios_history(ma)
    fig6 = yield_curve_history(ma)
    fig7 = vix_history(ma)
    fig8 = macro_score_history(ma)
    for f in [fig4, fig5, fig6, fig7, fig8]:
        if f:
            _ipy_display(f) if _ipy_display else f.show()

    print("\n[Group 3] Crisis Analysis...")
    fig9  = crisis_heatmap(ma)
    fig10 = zscore_bar(snap)
    for f in [fig9, fig10]:
        if f:
            _ipy_display(f) if _ipy_display else f.show()

    print("\n[Group 4] Asset Performance...")
    fig11 = phase_performance_bar(ma)
    if fig11:
        _ipy_display(fig11) if _ipy_display else fig11.show()
    fig12 = sector_rotation_wheel(snap)
    if fig12:
        plt.figure(fig12.number)
        if _ipy_display:
            _ipy_display(fig12)
            plt.close(fig12)
        else:
            plt.show()
    plt.close("all")  # clear pending matplotlib figures

    print("\n[Group 5] Divergence & Macro Signals...")
    fig13 = cg_yield_divergence(ma)
    if fig13:
        _ipy_display(fig13) if _ipy_display else fig13.show()

    print("\n[Group 6] Recession & VIX Analysis...")
    fig14 = probit_recession(ma)
    if fig14:
        _ipy_display(fig14) if _ipy_display else fig14.show()
    fig15 = vix_mean_reversion(ma)
    if fig15:
        _ipy_display(fig15) if _ipy_display else fig15.show()
    plt.close("all")  # clear any pending matplotlib figures

    print("\n[Group 7] Credit, Dollar & Equity Regime...")
    fig16 = credit_spread(ma)
    if fig16:
        _ipy_display(fig16) if _ipy_display else fig16.show()
    fig17 = dxy_history(ma)
    if fig17:
        _ipy_display(fig17) if _ipy_display else fig17.show()
    fig18 = spy_sma200(ma)
    if fig18:
        _ipy_display(fig18) if _ipy_display else fig18.show()

    print("\n[Group 8] Yields & Valuation...")
    fig19 = earnings_yield_gap(ma)
    if fig19:
        _ipy_display(fig19) if _ipy_display else fig19.show()
    fig20 = real_yields(ma)
    if fig20:
        _ipy_display(fig20) if _ipy_display else fig20.show()

    print("\n[Group 9] Metals & Inflation...")
    fig21 = gold_silver_chart(ma)
    if fig21:
        _ipy_display(fig21) if _ipy_display else fig21.show()
    fig22 = gold_silver_mean_reversion(ma)
    if fig22:
        _ipy_display(fig22) if _ipy_display else fig22.show()
    fig23 = inflation_environment(ma)
    if fig23:
        _ipy_display(fig23) if _ipy_display else fig23.show()
    fig24 = phase_gantt(ma)
    if fig24:
        _ipy_display(fig24) if _ipy_display else fig24.show()

    print("\nDone - all 24 charts generated.")
    print("Note: crisis_timeline(ma, 'GFC 2008', '2007-10-01', '2009-03-01') runs separately per crisis.")

#-----------------------------------------------------------------
# GROUP 5 - Divergence Analysis  (plotly, interactive)
#-----------------------------------------------------------------

def cg_yield_divergence(ma: MacroAnalyzer) -> Optional[go.Figure]:
    """
    Interactive plotly chart -- Cu/Gold vs 10Y Yield Divergence.

    Panel 1: Cu/Gold ratio and 10Y yield (dual axis)
    Panel 2: Divergence signal + rolling correlation 60d
    """
    inds    = ma.get_indicators()
    history = ma.get_history()
    if inds.empty:
        return None

    required = ["copper_gold", "yield_10y", "cg_yield_divergence",
                "cg_yield_corr_60"]
    missing = [c for c in required if c not in inds.columns]
    if missing:
        print(f"  Missing columns: {missing}")
        print("  Re-run ma.load() with updated macro_analyzer.py")
        return None

    weekly_i = inds.resample("W").last()
    weekly_h = history.resample("W").last()

    cg     = weekly_i["copper_gold"].dropna()
    y10    = weekly_i["yield_10y"].dropna()
    div    = weekly_i["cg_yield_divergence"].dropna()
    corr60 = weekly_i["cg_yield_corr_60"].dropna()

    # Phase colors for background bands
    phase_band_colors = {
        "early_expansion":   "rgba(46,204,113,0.07)",
        "late_expansion":    "rgba(243,156,18,0.07)",
        "early_contraction": "rgba(230,126,34,0.07)",
        "late_contraction":  "rgba(231,76,60,0.07)",
    }

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.58, 0.42],
        vertical_spacing=0.06,
        specs=[[{"secondary_y": True}],
               [{"secondary_y": True}]],
        subplot_titles=[
            "Cu/Gold Ratio vs 10Y Treasury Yield",
            "Divergence Signal  (Cu/Gold z - 10Y z)  +  Rolling Correlation 60d",
        ]
    )

    # ---- Phase bands (Panel 1) ----
    if not weekly_h.empty and "cycle_phase" in weekly_h.columns:
        ph = weekly_h["cycle_phase"].dropna()
        prev_phase, start_date = None, None
        for date, phase in ph.items():
            if phase != prev_phase:
                if prev_phase and start_date:
                    color = phase_band_colors.get(prev_phase, "rgba(0,0,0,0.03)")
                    fig.add_vrect(
                        x0=str(start_date.date()),
                        x1=str(date.date()),
                        fillcolor=color, opacity=1,
                        layer="below", line_width=0,
                        row=1, col=1
                    )
                start_date = date
                prev_phase = phase
        if prev_phase and start_date:
            color = phase_band_colors.get(prev_phase, "rgba(0,0,0,0.03)")
            fig.add_vrect(
                x0=str(start_date.date()),
                x1=str(ph.index[-1].date()),
                fillcolor=color, opacity=1,
                layer="below", line_width=0,
                row=1, col=1
            )

    # ---- Cu/Gold ratio (primary Y, Panel 1) ----
    fig.add_trace(
        go.Scatter(
            x=cg.index, y=cg.values,
            name="Cu/Gold Ratio",
            line=dict(color="#E67E22", width=2),
            hovertemplate="%{x|%d/%m/%Y}<br>Cu/Gold: %{y:.4f}<extra></extra>",
        ),
        row=1, col=1, secondary_y=False
    )

    # ---- 10Y Yield (secondary Y, Panel 1) ----
    fig.add_trace(
        go.Scatter(
            x=y10.index, y=y10.values,
            name="10Y Yield %",
            line=dict(color="#2980B9", width=2, dash="dash"),
            opacity=0.85,
            hovertemplate="%{x|%d/%m/%Y}<br>10Y: %{y:.2f}%<extra></extra>",
        ),
        row=1, col=1, secondary_y=True
    )

    # ---- Crisis markers (Panel 1) ----
    CRISIS_EVENTS_LOCAL = {
        "Dot-com":   ("2000-03-01", "2002-10-01"),
        "GFC 2008":  ("2007-10-01", "2009-03-01"),
        "COVID":     ("2020-02-01", "2020-04-01"),
        "Bear 2022": ("2022-01-01", "2022-10-01"),
    }
    for name, (s_str, _) in CRISIS_EVENTS_LOCAL.items():
        s = pd.Timestamp(s_str)
        if s < inds.index[0]:
            continue
        fig.add_vline(
            x=s.timestamp() * 1000,
            line=dict(color="rgba(192,57,43,0.4)", width=1, dash="dot"),
            annotation_text=name,
            annotation_position="top left",
            annotation_font=dict(size=9, color="rgba(192,57,43,0.7)"),
            row=1, col=1
        )

    # ---- Divergence bars (Panel 2) ----
    div_colors = [
        "#27AE60" if v > 0 else "#C0392B"
        for v in div.values
    ]
    fig.add_trace(
        go.Bar(
            x=div.index, y=div.values,
            name="Divergence (CG_z - Y10_z)",
            marker_color=div_colors,
            opacity=0.55,
            hovertemplate="%{x|%d/%m/%Y}<br>Div: %{y:+.2f}<extra></extra>",
        ),
        row=2, col=1, secondary_y=False
    )

    # Divergence line
    fig.add_trace(
        go.Scatter(
            x=div.index, y=div.values,
            name="Divergence",
            line=dict(color="#2C3E50", width=1.2),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=2, col=1, secondary_y=False
    )

    # Threshold lines
    for lv, color, label in [
        ( 1.5, "#27AE60", "+1.5 Bullish Div"),
        (-1.5, "#C0392B", "-1.5 Bearish Div"),
    ]:
        fig.add_hline(
            y=lv, line=dict(color=color, width=1.2, dash="dash"),
            annotation_text=label,
            annotation_position="right",
            annotation_font=dict(size=9, color=color),
            row=2, col=1
        )

    # Zero line
    fig.add_hline(
        y=0, line=dict(color="#2C3E50", width=1.0),
        row=2, col=1
    )

    # ---- Rolling correlation 60d (secondary Y, Panel 2) ----
    fig.add_trace(
        go.Scatter(
            x=corr60.index, y=corr60.values,
            name="Corr 60d",
            line=dict(color="#8E44AD", width=1.5, dash="dot"),
            opacity=0.85,
            hovertemplate="%{x|%d/%m/%Y}<br>Corr: %{y:.2f}<extra></extra>",
        ),
        row=2, col=1, secondary_y=True
    )

    # Correlation breakdown threshold
    fig.add_hline(
        y=0.3,
        line=dict(color="rgba(142,68,173,0.4)", width=1.0, dash="dot"),
        annotation_text="Corr breakdown (0.3)",
        annotation_position="right",
        annotation_font=dict(size=8, color="rgba(142,68,173,0.7)"),
        row=2, col=1
    )

    # ---- Current signal annotation ----
    try:
        snap = ma.get_snapshot()
        sig  = getattr(snap, "cg_yield_signal", "neutral")
        div_now = getattr(snap, "cg_yield_divergence", 0) or 0
        sig_labels = {
            "bullish_div": "BULLISH DIVERGENCE -- yields likely to rise",
            "bearish_div": "BEARISH DIVERGENCE -- Long Duration signal",
            "neutral":     "NEUTRAL -- normal correlation regime",
        }
        sig_colors = {
            "bullish_div": "#27AE60",
            "bearish_div": "#C0392B",
            "neutral":     "#7F8C8D",
        }
        sig_str   = sig_labels.get(sig, sig)
        sig_color = sig_colors.get(sig, "#7F8C8D")
        subtitle  = f"Current signal: {sig_str}  |  Divergence: {div_now:+.2f}"
    except Exception:
        subtitle = ""

    # ---- Layout ----
    fig.update_layout(
        title=dict(
            text=f"Cu/Gold vs 10Y Yield -- Divergence Analysis<br>"
                 f"<sup>{subtitle}</sup>",
            font=dict(size=14),
            x=0.02,
        ),
        template="plotly_dark",
        height=700,
        hovermode="x unified",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01,
            xanchor="left", x=0, font=dict(size=10)
        ),
        margin=dict(t=100, b=40, l=60, r=80),
    )

    # Y axis labels
    fig.update_yaxes(title_text="Cu/Gold Ratio", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="10Y Yield %",   row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Divergence",    row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Correlation",   row=2, col=1, secondary_y=True,
                     range=[-1.1, 1.1])

    return fig


#-----------------------------------------------------------------
# CHART 14 -- Probit Recession Probability  (plotly, interactive)
#-----------------------------------------------------------------

_NBER_RECESSIONS_PLOTLY = [
    ("2001-03-01", "2001-11-01"),
    ("2007-12-01", "2009-06-01"),
    ("2020-02-01", "2020-04-01"),
]

def probit_recession(ma: MacroAnalyzer) -> Optional[go.Figure]:
    """
    Interactive Probit recession probability chart.
    Panel 1: Probability (0-100%) with NBER shading
    Panel 2: Yield curve spread + Probit overlay
    """
    inds    = ma.get_indicators()
    history = ma.get_history()
    if inds.empty:
        return None

    if "probit_recession_prob" not in inds.columns:
        print("  probit_recession_prob not found -- re-run ma.load()")
        return None

    weekly_i = inds.resample("W").last()
    weekly_h = history.resample("W").last()

    prob = weekly_i["probit_recession_prob"].dropna()
    yc   = weekly_i["yield_curve"].dropna() if "yield_curve" in weekly_i.columns else None

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.62, 0.38],
        vertical_spacing=0.06,
        specs=[[{"secondary_y": False}],
               [{"secondary_y": True}]],
        subplot_titles=[
            "Probit Recession Probability (12-month horizon)",
            "Yield Curve Spread vs Probit Probability",
        ]
    )

    # ---- NBER recession bands ----
    for s_str, e_str in _NBER_RECESSIONS_PLOTLY:
        fig.add_vrect(
            x0=s_str, x1=e_str,
            fillcolor="rgba(192,57,43,0.15)",
            opacity=1, layer="below", line_width=0,
            annotation_text="NBER",
            annotation_position="top left",
            annotation_font=dict(size=8, color="rgba(192,57,43,0.6)"),
            row=1, col=1
        )
        fig.add_vrect(
            x0=s_str, x1=e_str,
            fillcolor="rgba(192,57,43,0.15)",
            opacity=1, layer="below", line_width=0,
            row=2, col=1
        )

    # ---- Phase bands (Panel 1) ----
    if not weekly_h.empty and "cycle_phase" in weekly_h.columns:
        phase_colors_p = {
            "early_expansion":   "rgba(46,204,113,0.06)",
            "late_expansion":    "rgba(243,156,18,0.06)",
            "early_contraction": "rgba(230,126,34,0.06)",
            "late_contraction":  "rgba(231,76,60,0.06)",
        }
        ph = weekly_h["cycle_phase"].dropna()
        prev_phase, start_date = None, None
        for date, phase in ph.items():
            if phase != prev_phase:
                if prev_phase and start_date:
                    color = phase_colors_p.get(prev_phase, "rgba(0,0,0,0.03)")
                    fig.add_vrect(
                        x0=str(start_date.date()), x1=str(date.date()),
                        fillcolor=color, opacity=1,
                        layer="below", line_width=0,
                        row=1, col=1
                    )
                start_date = date
                prev_phase = phase

    # ---- Probability trace ----
    # Color by zone
    zone_colors = []
    for p in prob.values:
        if p < 15:
            zone_colors.append("#27AE60")
        elif p < 30:
            zone_colors.append("#F39C12")
        elif p < 50:
            zone_colors.append("#E67E22")
        else:
            zone_colors.append("#C0392B")

    fig.add_trace(
        go.Scatter(
            x=prob.index, y=prob.values,
            name="Recession Probability %",
            fill="tozeroy",
            fillcolor="rgba(142,68,173,0.12)",
            line=dict(color="#8E44AD", width=2),
            hovertemplate="%{x|%d/%m/%Y}<br>P(recession): %{y:.1f}%<extra></extra>",
        ),
        row=1, col=1
    )

    # Threshold lines
    for lv, color, lbl in [
        (15, "#27AE60", "Low (15%)"),
        (30, "#E67E22", "NY Fed threshold (30%)"),
        (50, "#C0392B", "High risk (50%)"),
    ]:
        fig.add_hline(
            y=lv, line=dict(color=color, width=1.2, dash="dash"),
            annotation_text=lbl,
            annotation_position="right",
            annotation_font=dict(size=9, color=color),
            row=1, col=1
        )

    # ---- Yield curve ----
    if yc is not None:
        fig.add_trace(
            go.Scatter(
                x=yc.index, y=yc.values,
                name="Yield Curve (10Y-3M) %",
                line=dict(color="#2C3E50", width=1.8),
                hovertemplate="%{x|%d/%m/%Y}<br>Spread: %{y:.2f}%<extra></extra>",
            ),
            row=2, col=1, secondary_y=False
        )
        # Inversion fill
        yc_neg = yc.clip(upper=0)
        fig.add_trace(
            go.Scatter(
                x=yc.index, y=yc_neg.values,
                fill="tozeroy",
                fillcolor="rgba(192,57,43,0.20)",
                line=dict(width=0),
                showlegend=False, hoverinfo="skip",
            ),
            row=2, col=1, secondary_y=False
        )
        fig.add_hline(
            y=0, line=dict(color="#C0392B", width=1.2, dash="dash"),
            row=2, col=1
        )

    # Probit on secondary Y (Panel 2)
    fig.add_trace(
        go.Scatter(
            x=prob.index, y=prob.values,
            name="Probit Prob %",
            line=dict(color="#8E44AD", width=1.5, dash="dot"),
            opacity=0.85,
            hovertemplate="%{x|%d/%m/%Y}<br>Prob: %{y:.1f}%<extra></extra>",
        ),
        row=2, col=1, secondary_y=True
    )

    # Current signal
    try:
        snap     = ma.get_snapshot()
        prob_now = getattr(snap, "probit_recession_prob", 0) or 0
        sig      = getattr(snap, "probit_signal", "unknown")
        subtitle = (f"Recession probability (12m): {prob_now:.1f}%  |  "
                    f"Signal: {sig.upper()}  |  "
                    f"P = Phi({-0.6045:.4f} + {-0.7374:.4f} x spread)")
    except Exception:
        subtitle = "Estrella & Mishkin (1998)"

    fig.update_layout(
        title=dict(
            text=f"Probit Recession Probability<br><sup>{subtitle}</sup>",
            font=dict(size=14), x=0.02,
        ),
        template="plotly_dark",
        height=680,
        hovermode="x unified",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01,
            xanchor="left", x=0, font=dict(size=10)
        ),
        margin=dict(t=100, b=40, l=60, r=80),
    )

    fig.update_yaxes(title_text="Recession Probability %",
                     range=[0, 100], row=1, col=1)
    fig.update_yaxes(title_text="Spread %",     row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Probability %",
                     range=[0, 100], row=2, col=1, secondary_y=True)
    return fig


#-----------------------------------------------------------------
# CHART 15 -- VIX Mean Reversion  (plotly, interactive)
#-----------------------------------------------------------------

def vix_mean_reversion(ma: MacroAnalyzer) -> Optional[go.Figure]:
    """
    Interactive VIX mean reversion chart.
    Panel 1: VIX vs LT mean with bands
    Panel 2: Mean reversion z-score
    Panel 3: Half-life
    """
    inds    = ma.get_indicators()
    history = ma.get_history()
    if inds.empty:
        return None

    required = ["vix", "vix_lt_mean", "vix_mr_zscore",
                "vix_half_life", "vix_band_1up", "vix_band_2up"]
    missing = [c for c in required if c not in inds.columns]
    if missing:
        print(f"  Missing: {missing} -- re-run ma.load()")
        return None

    weekly_i = inds.resample("W").last()
    weekly_h = history.resample("W").last()

    vix     = weekly_i["vix"].dropna()
    lt_mean = weekly_i["vix_lt_mean"].dropna()
    mr_z    = weekly_i["vix_mr_zscore"].dropna()
    hl      = weekly_i["vix_half_life"].clip(upper=120).dropna()
    b1up    = weekly_i["vix_band_1up"].dropna()
    b2up    = weekly_i["vix_band_2up"].dropna()
    b1dn    = weekly_i.get("vix_band_1dn",
                           lt_mean - (b1up - lt_mean)).dropna()

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.50, 0.28, 0.22],
        vertical_spacing=0.05,
        subplot_titles=[
            "VIX vs Long-term Mean (5Y) + Bollinger Bands",
            "Mean Reversion Z-score",
            "Mean Reversion Half-life (days, 60d rolling)",
        ]
    )

    # ---- Phase bands ----
    if not weekly_h.empty and "cycle_phase" in weekly_h.columns:
        phase_colors_p = {
            "early_expansion":   "rgba(46,204,113,0.06)",
            "late_expansion":    "rgba(243,156,18,0.06)",
            "early_contraction": "rgba(230,126,34,0.06)",
            "late_contraction":  "rgba(231,76,60,0.06)",
        }
        ph = weekly_h["cycle_phase"].dropna()
        prev_phase, start_date = None, None
        for date, phase in ph.items():
            if phase != prev_phase:
                if prev_phase and start_date:
                    color = phase_colors_p.get(prev_phase, "rgba(0,0,0,0.03)")
                    fig.add_vrect(
                        x0=str(start_date.date()), x1=str(date.date()),
                        fillcolor=color, opacity=1,
                        layer="below", line_width=0, row=1, col=1
                    )
                start_date = date
                prev_phase = phase

    # ---- Panel 1: VIX + Bands ----
    # 2 std dev band fill
    fig.add_trace(go.Scatter(
        x=pd.concat([b2up.index.to_series(), b1up.index.to_series()[::-1]]),
        y=pd.concat([b2up, b1up[::-1]]),
        fill="toself",
        fillcolor="rgba(192,57,43,0.10)",
        line=dict(width=0), showlegend=True,
        name="1-2 std dev zone",
        hoverinfo="skip",
    ), row=1, col=1)

    # LT mean line
    fig.add_trace(go.Scatter(
        x=lt_mean.index, y=lt_mean.values,
        name="5Y Rolling Mean",
        line=dict(color="#95A5A6", width=1.5, dash="dash"),
        hovertemplate="%{x|%d/%m/%Y}<br>Mean: %{y:.1f}<extra></extra>",
    ), row=1, col=1)

    # 1 std dev bands
    fig.add_trace(go.Scatter(
        x=b1up.index, y=b1up.values,
        name="Mean+1 std dev",
        line=dict(color="#F39C12", width=1.0, dash="dot"),
        hovertemplate="%{x|%d/%m/%Y}<br>+1sd: %{y:.1f}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=b2up.index, y=b2up.values,
        name="Mean+2 std devs",
        line=dict(color="#E74C3C", width=1.0, dash="dot"),
        hovertemplate="%{x|%d/%m/%Y}<br>+2sd: %{y:.1f}<extra></extra>",
    ), row=1, col=1)

    # VIX
    fig.add_trace(go.Scatter(
        x=vix.index, y=vix.values,
        name="VIX",
        line=dict(color="#8E44AD", width=2.0),
        hovertemplate="%{x|%d/%m/%Y}<br>VIX: %{y:.1f}<extra></extra>",
    ), row=1, col=1)

    # Threshold lines
    for lv, color in [(15, "#27AE60"), (25, "#F39C12"), (35, "#E74C3C")]:
        fig.add_hline(y=lv, line=dict(color=color, width=0.8, dash="dash"),
                      row=1, col=1)

    # ---- Panel 2: MR Z-score ----
    fig.add_trace(go.Bar(
        x=mr_z.index, y=mr_z.values,
        name="MR Z-score",
        marker_color=["#E74C3C" if v > 0 else "#27AE60" for v in mr_z.values],
        opacity=0.55,
        hovertemplate="%{x|%d/%m/%Y}<br>Z: %{y:+.2f}<extra></extra>",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=mr_z.index, y=mr_z.values,
        line=dict(color="#2C3E50", width=1.0),
        showlegend=False, hoverinfo="skip",
    ), row=2, col=1)

    for lv, color, lbl in [
        ( 2.0, "#E74C3C", "+2sd Short Vol"),
        (-1.0, "#27AE60", "-1sd Complacency"),
    ]:
        fig.add_hline(y=lv, line=dict(color=color, width=1.2, dash="dash"),
                      annotation_text=lbl,
                      annotation_position="right",
                      annotation_font=dict(size=9, color=color),
                      row=2, col=1)
    fig.add_hline(y=0, line=dict(color="#2C3E50", width=1.0), row=2, col=1)

    # ---- Panel 3: Half-life ----
    fig.add_trace(go.Scatter(
        x=hl.index, y=hl.values,
        name="Half-life (days)",
        fill="tozeroy",
        fillcolor="rgba(22,160,133,0.15)",
        line=dict(color="#16A085", width=1.5),
        hovertemplate="%{x|%d/%m/%Y}<br>HL: %{y:.0f}d<extra></extra>",
    ), row=3, col=1)

    for lv in [14, 30, 60]:
        fig.add_hline(y=lv,
                      line=dict(color="rgba(149,165,166,0.5)",
                                width=0.8, dash="dot"),
                      row=3, col=1)

    # Current signal
    try:
        snap     = ma.get_snapshot()
        sig      = getattr(snap, "vix_mr_signal", "normal")
        mrz      = getattr(snap, "vix_mr_zscore", 0) or 0
        ltm      = getattr(snap, "vix_lt_mean", 0) or 0
        hlv      = getattr(snap, "vix_half_life", 0) or 0
        subtitle = (f"VIX={snap.vix:.1f}  |  LT Mean={ltm:.1f}  |  "
                    f"MR z={mrz:+.2f}  |  Half-life={hlv:.0f}d  |  "
                    f"Signal: {sig.upper().replace('_',' ')}")
    except Exception:
        subtitle = "Ornstein-Uhlenbeck mean reversion proxy"

    fig.update_layout(
        title=dict(
            text=f"VIX Mean Reversion Analysis<br><sup>{subtitle}</sup>",
            font=dict(size=14), x=0.02,
        ),
        template="plotly_dark",
        height=750,
        hovermode="x unified",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01,
            xanchor="left", x=0, font=dict(size=10)
        ),
        margin=dict(t=100, b=40, l=60, r=80),
    )

    fig.update_yaxes(title_text="VIX Level",       row=1, col=1)
    fig.update_yaxes(title_text="Z-score",          row=2, col=1)
    fig.update_yaxes(title_text="Half-life (days)", row=3, col=1)
    return fig


#-----------------------------------------------------------------
# CHART 16 -- FRED Signals: ISM + TED  (plotly, interactive)
#-----------------------------------------------------------------

def fred_signals(ma: MacroAnalyzer) -> Optional[go.Figure]:
    """
    Interactive FRED signals chart.
    Panel 1: ISM Manufacturing PMI
    Panel 2: TED Spread / FRA-OIS
    """
    inds    = ma.get_indicators()
    history = ma.get_history()
    if inds.empty:
        return None

    has_ism = "ism" in inds.columns
    has_ted = "ted_spread" in inds.columns
    has_fra = "fra_ois_spread" in inds.columns

    if not has_ism and not has_ted and not has_fra:
        print("  No FRED data -- run with fred_api_key")
        return None

    weekly_i = inds.resample("W").last()
    weekly_h = history.resample("W").last()

    n_rows   = sum([has_ism, has_ted or has_fra])
    subtitles = []
    if has_ism:
        subtitles.append("ISM Manufacturing PMI (FRED)")
    if has_ted or has_fra:
        subtitles.append("TED Spread / FRA-OIS (FRED)")

    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        row_heights=[1/n_rows] * n_rows,
        vertical_spacing=0.08,
        subplot_titles=subtitles,
    )

    # Phase bands helper
    phase_colors_p = {
        "early_expansion":   "rgba(46,204,113,0.06)",
        "late_expansion":    "rgba(243,156,18,0.06)",
        "early_contraction": "rgba(230,126,34,0.06)",
        "late_contraction":  "rgba(231,76,60,0.06)",
    }

    def _add_phase_bands_plotly(row_num):
        if weekly_h.empty or "cycle_phase" not in weekly_h.columns:
            return
        ph = weekly_h["cycle_phase"].dropna()
        prev_phase, start_date = None, None
        for date, phase in ph.items():
            if phase != prev_phase:
                if prev_phase and start_date:
                    color = phase_colors_p.get(prev_phase, "rgba(0,0,0,0.03)")
                    fig.add_vrect(
                        x0=str(start_date.date()), x1=str(date.date()),
                        fillcolor=color, opacity=1,
                        layer="below", line_width=0,
                        row=row_num, col=1
                    )
                start_date = date
                prev_phase = phase

    row = 1

    # ---- ISM Panel ----
    if has_ism:
        ism = weekly_i["ism"].dropna()
        _add_phase_bands_plotly(row)

        # Expansion fill
        fig.add_trace(go.Scatter(
            x=ism.index, y=ism.values,
            fill="tonexty" if False else "tozeroy",
            fillcolor="rgba(39,174,96,0.0)",
            line=dict(color="#2C3E50", width=1.8),
            name="ISM PMI",
            hovertemplate="%{x|%d/%m/%Y}<br>ISM: %{y:.1f}<extra></extra>",
        ), row=row, col=1)

        # 50 line
        fig.add_hline(y=50, line=dict(color="#2C3E50", width=1.5, dash="dash"),
                      annotation_text="Neutral (50)",
                      annotation_position="right",
                      annotation_font=dict(size=9),
                      row=row, col=1)
        # 55 line
        fig.add_hline(y=55, line=dict(color="#27AE60", width=1.0, dash="dot"),
                      annotation_text="Strong (55)",
                      annotation_position="right",
                      annotation_font=dict(size=9, color="#27AE60"),
                      row=row, col=1)
        # 45 line
        fig.add_hline(y=45, line=dict(color="#C0392B", width=1.0, dash="dot"),
                      annotation_text="Weak (45)",
                      annotation_position="right",
                      annotation_font=dict(size=9, color="#C0392B"),
                      row=row, col=1)

        fig.update_yaxes(title_text="ISM PMI", row=row, col=1)
        row += 1

    # ---- TED/FRA-OIS Panel ----
    if has_ted or has_fra:
        if has_ted:
            spread      = weekly_i["ted_spread"].dropna()
            spread_name = "TED Spread"
        else:
            spread      = weekly_i["fra_ois_spread"].dropna()
            spread_name = "FRA-OIS"

        _add_phase_bands_plotly(row)

        spread_colors = []
        for v in spread.values:
            if v >= TED_CRISIS:
                spread_colors.append("#C0392B")
            elif v >= TED_ELEVATED:
                spread_colors.append("#E67E22")
            elif v >= TED_NORMAL:
                spread_colors.append("#F39C12")
            else:
                spread_colors.append("#27AE60")

        fig.add_trace(go.Scatter(
            x=spread.index, y=spread.values,
            fill="tozeroy",
            fillcolor="rgba(142,68,173,0.12)",
            line=dict(color="#8E44AD", width=1.8),
            name=spread_name,
            hovertemplate=f"%{{x|%d/%m/%Y}}<br>{spread_name}: %{{y:.3f}}%<extra></extra>",
        ), row=row, col=1)

        for lv, color, lbl in [
            (TED_NORMAL,   "#F39C12", f"Watch ({TED_NORMAL:.2f}%)"),
            (TED_ELEVATED, "#E67E22", f"Elevated ({TED_ELEVATED:.2f}%)"),
            (TED_CRISIS,   "#C0392B", f"Crisis ({TED_CRISIS:.2f}%)"),
        ]:
            fig.add_hline(
                y=lv, line=dict(color=color, width=1.0, dash="dash"),
                annotation_text=lbl,
                annotation_position="right",
                annotation_font=dict(size=9, color=color),
                row=row, col=1
            )

        fig.update_yaxes(title_text="Spread %", row=row, col=1)

    # Current signals
    try:
        snap    = ma.get_snapshot()
        ism_str = f"ISM: {snap.ism:.1f}" if snap.ism else ""
        ted_val = snap.ted_spread or snap.fra_ois_spread
        ted_str = f"TED: {ted_val:.2f}%" if ted_val else ""
        subtitle = "  |  ".join(filter(None, [ism_str, ted_str]))
    except Exception:
        subtitle = "FRED data"

    fig.update_layout(
        title=dict(
            text=f"FRED Signals -- ISM Manufacturing + Bank Systemic Risk<br>"
                 f"<sup>{subtitle}</sup>",
            font=dict(size=14), x=0.02,
        ),
        template="plotly_dark",
        height=350 * n_rows + 100,
        hovermode="x unified",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01,
            xanchor="left", x=0, font=dict(size=10)
        ),
        margin=dict(t=100, b=40, l=60, r=80),
    )
    return fig

# =================================================================
# GROUP 5 — ADDITIONAL INTERACTIVE CHARTS (Plotly equivalents)
# =================================================================

# -----------------------------------------------------------------
#  CHART 14 — Credit Spread (HYG/LQD)
# -----------------------------------------------------------------

def credit_spread(ma: MacroAnalyzer) -> Optional[go.Figure]:
    """
    Interactive 3-panel credit spread chart.

    Panel 1: HYG/LQD ratio history with phase bands and crisis markers
    Panel 2: HYG/LQD vs SPY normalized (leading signal analysis)
    Panel 3: Credit spread z-score with threshold lines
    """
    inds    = ma.get_indicators()
    history = ma.get_history()
    if inds.empty or "credit_spread_ratio" not in inds.columns:
        print("  credit_spread_ratio not found -- re-run ma.load()")
        return None

    weekly_i = inds.resample("W").last()
    weekly_h = history.resample("W").last()

    cr  = weekly_i["credit_spread_ratio"].dropna()
    cz  = weekly_i["credit_spread_zscore"].dropna() if "credit_spread_zscore" in weekly_i.columns else None

    # SPY prices for panel 2
    spy_prices = getattr(ma, "_prices", {}).get("spy")
    spy_w = spy_prices.resample("W").last() if spy_prices is not None else None

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.42, 0.32, 0.26],
        subplot_titles=[
            "HYG/LQD Ratio — falling = credit stress",
            "HYG/LQD vs SPY Normalized (base=100) — credit leads equity in credit-driven crises",
            "Credit Spread Z-score (5Y rolling) — < -0.5 stress | > +0.5 healthy",
        ],
        vertical_spacing=0.06,
    )

    # ── Panel 1: HYG/LQD ratio ──────────────────────────────────
    rm = cr.rolling(52, min_periods=10).mean()

    fig.add_trace(go.Scatter(
        x=cr.index, y=cr.values,
        fill="tozeroy", fillcolor="rgba(0,188,212,0.10)",
        line=dict(color="#00BCD4", width=1.8),
        name="HYG/LQD Ratio",
        hovertemplate="%{x|%d/%m/%Y}<br>Ratio: %{y:.4f}<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=rm.index, y=rm.values,
        line=dict(color="#00BCD4", width=1.0, dash="dash"),
        opacity=0.55, name="1Y avg",
        hovertemplate="%{x|%d/%m/%Y}<br>1Y avg: %{y:.4f}<extra></extra>",
    ), row=1, col=1)

    # Crisis low annotations
    for name, (s_str, e_str) in CRISIS_EVENTS.items():
        s = pd.Timestamp(s_str)
        e = pd.Timestamp(e_str)
        if s < cr.index[0]:
            continue
        seg = cr[(cr.index >= s) & (cr.index <= e)]
        if seg.empty:
            continue
        low_date = seg.idxmin()
        low_val  = seg.min()
        fig.add_annotation(
            x=low_date, y=low_val,
            text=f"{name}<br>low:{low_val:.3f}",
            showarrow=True, arrowhead=2,
            arrowcolor="#E74C3C", font=dict(size=8, color="#E74C3C"),
            ax=0, ay=-35, row=1, col=1,
        )

    fig.update_yaxes(title_text="HYG/LQD", row=1, col=1)

    # ── Panel 2: Normalized comparison ──────────────────────────
    def norm100(s):
        s = s.dropna()
        return (s / s.iloc[0]) * 100 if len(s) > 0 else s

    cr_n = norm100(cr)
    fig.add_trace(go.Scatter(
        x=cr_n.index, y=cr_n.values,
        line=dict(color="#00BCD4", width=1.6),
        name="HYG/LQD (norm.100)",
        hovertemplate="%{x|%d/%m/%Y}<br>Credit: %{y:.1f}<extra></extra>",
    ), row=2, col=1)

    if spy_w is not None:
        spy_w_aligned = spy_w[spy_w.index >= cr.index[0]]
        spy_n = norm100(spy_w_aligned)
        fig.add_trace(go.Scatter(
            x=spy_n.index, y=spy_n.values,
            line=dict(color="#3498DB", width=1.6, dash="dash"),
            name="SPY (norm.100)",
            hovertemplate="%{x|%d/%m/%Y}<br>SPY: %{y:.1f}<extra></extra>",
        ), row=2, col=1)

    fig.add_hline(y=100, line=dict(color="#30363d", width=0.8, dash="dot"), row=2, col=1)
    fig.update_yaxes(title_text="Norm. Level", row=2, col=1)

    # ── Panel 3: Z-score ─────────────────────────────────────────
    if cz is not None:
        cz_w = cz.reindex(weekly_i.index, method="ffill").dropna()
        colors = ["#E74C3C" if v < -0.5 else "#2ECC71" if v > 0.5 else "#7F8C8D"
                  for v in cz_w.values]

        cz_w_pos = cz_w.clip(lower=0)
        cz_w_neg = cz_w.clip(upper=0)
        fig.add_trace(go.Scatter(
            x=cz_w.index, y=cz_w_pos.values,
            fill="tozeroy", fillcolor="rgba(46,204,113,0.22)",
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=cz_w.index, y=cz_w_neg.values,
            fill="tozeroy", fillcolor="rgba(231,76,60,0.28)",
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=cz_w.index, y=cz_w.values,
            line=dict(color="#c9d1d9", width=1.3),
            name="Z-score",
            hovertemplate="%{x|%d/%m/%Y}<br>Z: %{y:.2f}<extra></extra>",
        ), row=3, col=1)

        for lv, color, lbl in [
            (-0.5, "#E74C3C", "Stress (-0.5)"),
            ( 0.5, "#2ECC71", "Healthy (+0.5)"),
        ]:
            fig.add_hline(
                y=lv, line=dict(color=color, width=1.0, dash="dash"),
                annotation_text=lbl,
                annotation_position="right",
                annotation_font=dict(size=9, color=color),
                row=3, col=1,
            )

        last_z = cz_w.iloc[-1]
        last_lbl = "STRESS" if last_z < -0.5 else "HEALTHY" if last_z > 0.5 else "NEUTRAL"
        fig.update_yaxes(title_text="Z-score (5Y)", row=3, col=1)

    # Crisis shading all panels
    for _, (s_str, e_str) in CRISIS_EVENTS.items():
        s = pd.Timestamp(s_str)
        e = pd.Timestamp(e_str)
        if s < cr.index[0]:
            continue
        for row_n in [1, 2, 3]:
            fig.add_vrect(
                x0=s, x1=e,
                fillcolor="rgba(231,76,60,0.07)",
                layer="below", line_width=0,
                row=row_n, col=1,
            )

    # Subtitle
    try:
        snap = ma.get_snapshot()
        last_z_val = cz_w.iloc[-1] if cz is not None else 0
        subtitle = (f"HYG/LQD: {cr.iloc[-1]:.4f}  |  1Y avg: {rm.iloc[-1]:.4f}  |  "
                    f"Z-score: {last_z_val:+.2f}  |  {last_lbl}")
    except Exception:
        subtitle = "HYG/LQD Credit Spread Analysis"

    fig.update_layout(
        title=dict(
            text=f"Credit Spread Analysis — HYG/LQD<br><sup>{subtitle}</sup>",
            font=dict(size=14), x=0.02,
        ),
        template="plotly_dark",
        height=700,
        hovermode="x unified",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01,
            xanchor="left", x=0, font=dict(size=10),
        ),
        margin=dict(t=100, b=40, l=60, r=80),
    )
    fig.update_xaxes(**PLOTLY_AXIS_STYLE)
    fig.update_yaxes(**PLOTLY_AXIS_STYLE)
    return fig


# -----------------------------------------------------------------
#  CHART 15 — DXY Dollar Index
# -----------------------------------------------------------------

def dxy_history(ma: MacroAnalyzer) -> Optional[go.Figure]:
    """
    Interactive 2-panel DXY Dollar Index chart.

    Panel 1: DXY level + 1Y rolling average with phase context
    Panel 2: DXY 20d ROC (momentum signal, ±3% thresholds)
    """
    inds    = ma.get_indicators()
    history = ma.get_history()
    if inds.empty or "dxy" not in inds.columns:
        print("  dxy not found in indicators")
        return None

    weekly_i = inds.resample("W").last()
    weekly_h = history.resample("W").last()

    dxy     = weekly_i["dxy"].dropna()
    dxy_roc = (weekly_i["dxy_roc_20"].dropna() * 100) if "dxy_roc_20" in weekly_i.columns else None

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.62, 0.38],
        subplot_titles=[
            "DXY Dollar Index — strong dollar = pressure on commodities, EM, gold",
            "DXY 20d Momentum (ROC) — > +3%: Strong  |  < -3%: Weak",
        ],
        vertical_spacing=0.07,
    )

    # ── Panel 1: DXY level ───────────────────────────────────────
    rm = dxy.rolling(52, min_periods=10).mean()

    fig.add_trace(go.Scatter(
        x=dxy.index, y=dxy.values,
        fill="tozeroy", fillcolor="rgba(52,152,219,0.08)",
        line=dict(color="#3498DB", width=1.8),
        name="DXY Index",
        hovertemplate="%{x|%d/%m/%Y}<br>DXY: %{y:.2f}<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=rm.index, y=rm.values,
        line=dict(color="#3498DB", width=1.0, dash="dash"),
        opacity=0.55, name="1Y avg",
        hovertemplate="%{x|%d/%m/%Y}<br>1Y avg: %{y:.2f}<extra></extra>",
    ), row=1, col=1)

    # Phase shading
    _add_phase_shading(fig, weekly_h, row=1)

    # Crisis markers
    for name, (s_str, e_str) in CRISIS_EVENTS.items():
        s = pd.Timestamp(s_str)
        e = pd.Timestamp(e_str)
        if s < dxy.index[0]:
            continue
        fig.add_vrect(
            x0=s, x1=e,
            fillcolor="rgba(231,76,60,0.07)",
            layer="below", line_width=0, row=1, col=1,
        )

    last_dxy   = dxy.iloc[-1]
    last_mean  = rm.iloc[-1]
    dxy_lbl    = "Strong" if last_dxy > last_mean * 1.05 else "Weak" if last_dxy < last_mean * 0.95 else "Neutral"
    dxy_color  = "#E74C3C" if dxy_lbl == "Strong" else "#2ECC71" if dxy_lbl == "Weak" else "#7F8C8D"

    fig.add_annotation(
        x=dxy.index[-1], y=last_dxy,
        text=f"<b>{last_dxy:.1f} ({dxy_lbl})</b>",
        showarrow=True, arrowhead=2,
        arrowcolor=dxy_color, font=dict(size=10, color=dxy_color),
        ax=-80, ay=-30, row=1, col=1,
    )
    fig.update_yaxes(title_text="DXY Level", row=1, col=1)

    # ── Panel 2: DXY Momentum ────────────────────────────────────
    if dxy_roc is not None:
        # Use Scatter with fill instead of Bar -- Bar + shared_xaxes + Timestamps
        # can fail to render in some Plotly versions
        dxy_roc_pos = dxy_roc.clip(lower=0)
        dxy_roc_neg = dxy_roc.clip(upper=0)
        fig.add_trace(go.Scatter(
            x=dxy_roc.index, y=dxy_roc_pos.values,
            fill="tozeroy", fillcolor="rgba(231,76,60,0.25)",
            line=dict(width=0), showlegend=False,
            hoverinfo="skip",
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=dxy_roc.index, y=dxy_roc_neg.values,
            fill="tozeroy", fillcolor="rgba(46,204,113,0.25)",
            line=dict(width=0), showlegend=False,
            hoverinfo="skip",
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=dxy_roc.index, y=dxy_roc.values,
            line=dict(color="#c9d1d9", width=1.3),
            name="20d ROC %",
            hovertemplate="%{x|%d/%m/%Y}<br>ROC: %{y:+.2f}%<extra></extra>",
        ), row=2, col=1)

        for lv, color, lbl in [
            ( 3.0, "#E74C3C", "+3% Strong"),
            (-3.0, "#2ECC71", "-3% Weak"),
        ]:
            fig.add_hline(
                y=lv, line=dict(color=color, width=1.0, dash="dash"),
                annotation_text=lbl,
                annotation_position="right",
                annotation_font=dict(size=9, color=color),
                row=2, col=1,
            )
        fig.add_hline(y=0, line=dict(color="#4a5568", width=1.0), row=2, col=1)

        last_roc   = dxy_roc.iloc[-1]
        roc_lbl    = "STRONG" if last_roc > 3 else "WEAK" if last_roc < -3 else "STABLE"
        roc_color  = "#E74C3C" if last_roc > 3 else "#2ECC71" if last_roc < -3 else "#7F8C8D"
        fig.add_annotation(
            x=dxy_roc.index[-1], y=last_roc,
            text=f"<b>{last_roc:+.1f}% ({roc_lbl})</b>",
            showarrow=True, arrowhead=2,
            arrowcolor=roc_color, font=dict(size=10, color=roc_color),
            ax=-80, ay=-30, row=2, col=1,
        )
        fig.update_yaxes(title_text="20d ROC %", row=2, col=1)

    subtitle = (f"DXY: {last_dxy:.1f}  |  1Y avg: {last_mean:.1f}  |  "
                f"Trend: {dxy_lbl}  |  Momentum: {last_roc:+.1f}% ({roc_lbl})")

    fig.update_layout(
        title=dict(
            text=f"DXY Dollar Index — History & Momentum<br><sup>{subtitle}</sup>",
            font=dict(size=14), x=0.02,
        ),
        template="plotly_dark",
        height=600,
        hovermode="x unified",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01,
            xanchor="left", x=0, font=dict(size=10),
        ),
        margin=dict(t=100, b=40, l=60, r=80),
    )
    fig.update_xaxes(**PLOTLY_AXIS_STYLE)
    fig.update_yaxes(**PLOTLY_AXIS_STYLE)
    return fig


# -----------------------------------------------------------------
#  CHART 16 — SPY vs SMA200
# -----------------------------------------------------------------

def spy_sma200(ma: MacroAnalyzer) -> Optional[go.Figure]:
    """
    Interactive 2-panel SPY vs SMA200 chart.

    Panel 1: SPY price vs SMA200 with bull/bear shading
    Panel 2: SPY 60d momentum (ROC)
    """
    inds    = ma.get_indicators()
    history = ma.get_history()
    prices  = getattr(ma, "_prices", {})
    if history.empty:
        return None

    weekly_i   = inds.resample("W").last()
    spy_prices = prices.get("spy")

    if spy_prices is None:
        print("  SPY prices not found")
        return None

    spy_w  = spy_prices.resample("W").last()
    sma200 = spy_prices.rolling(200).mean().resample("W").last()
    sma200 = sma200.reindex(spy_w.index, method="ffill")
    spy_roc = (weekly_i["spy_roc_60"] * 100).dropna() if "spy_roc_60" in weekly_i.columns else None

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.35],
        subplot_titles=[
            "SPY vs SMA200 — red shading = below SMA200 (bear regime)",
            "SPY 60-day Momentum (ROC) — positive + above SMA200 = Bullish",
        ],
        vertical_spacing=0.07,
    )

    # ── Panel 1: SPY vs SMA200 ───────────────────────────────────
    # Bear shading (SPY below SMA200)
    bear_mask = spy_w < sma200
    bear_y_upper = spy_w.where(bear_mask, sma200)
    bear_y_lower = sma200.where(bear_mask, spy_w)

    fig.add_trace(go.Scatter(
        x=spy_w.index, y=spy_w.values,
        line=dict(color="#3498DB", width=1.8),
        name="SPY",
        hovertemplate="%{x|%d/%m/%Y}<br>SPY: $%{y:.0f}<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=sma200.index, y=sma200.values,
        line=dict(color="#c9d1d9", width=1.2, dash="dash"),
        opacity=0.80, name="SMA200",
        hovertemplate="%{x|%d/%m/%Y}<br>SMA200: $%{y:.0f}<extra></extra>",
    ), row=1, col=1)

    # Fill between for bear periods
    fig.add_trace(go.Scatter(
        x=list(spy_w.index) + list(spy_w.index[::-1]),
        y=list(spy_w.values) + list(sma200.values[::-1]),
        fill="toself",
        fillcolor="rgba(231,76,60,0.12)",
        line=dict(width=0),
        showlegend=True,
        name="Below SMA200 (Bear)",
        hoverinfo="skip",
    ), row=1, col=1)

    # Crisis vrects
    for name, (s_str, e_str) in CRISIS_EVENTS.items():
        s = pd.Timestamp(s_str)
        if s < spy_w.index[0]:
            continue
        fig.add_vrect(
            x0=s, x1=pd.Timestamp(e_str),
            fillcolor="rgba(231,76,60,0.07)",
            layer="below", line_width=0, row=1, col=1,
        )

    last_spy  = spy_w.iloc[-1]
    last_sma  = sma200.iloc[-1]
    is_bull   = last_spy >= last_sma
    regime    = "BULL" if is_bull else "BEAR"
    pct_vs    = (last_spy / last_sma - 1) * 100
    r_color   = "#2ECC71" if is_bull else "#E74C3C"

    fig.add_annotation(
        x=spy_w.index[-1], y=last_spy,
        text=f"<b>SPY ${last_spy:.0f} — {regime} ({pct_vs:+.1f}% vs SMA200)</b>",
        showarrow=True, arrowhead=2,
        arrowcolor=r_color, font=dict(size=10, color=r_color),
        ax=-130, ay=-30, row=1, col=1,
    )
    fig.update_yaxes(title_text="SPY Price (USD)", row=1, col=1)

    # ── Panel 2: 60d ROC ─────────────────────────────────────────
    if spy_roc is not None:
        # Use explicit zero baseline + tonexty to avoid tozeroy issues
        # with shared_xaxes subplots
        import numpy as np
        zeros = np.zeros(len(spy_roc))
        fig.add_trace(go.Scatter(
            x=spy_roc.index, y=zeros,
            line=dict(width=0, color="rgba(0,0,0,0)"),
            showlegend=False, hoverinfo="skip",
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=spy_roc.index, y=spy_roc.clip(lower=0).values,
            fill="tonexty", fillcolor="rgba(46,204,113,0.30)",
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=spy_roc.index, y=zeros,
            line=dict(width=0, color="rgba(0,0,0,0)"),
            showlegend=False, hoverinfo="skip",
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=spy_roc.index, y=spy_roc.clip(upper=0).values,
            fill="tonexty", fillcolor="rgba(231,76,60,0.30)",
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=spy_roc.index, y=spy_roc.values,
            line=dict(color="#c9d1d9", width=1.3),
            name="60d ROC %",
            hovertemplate="%{x|%d/%m/%Y}<br>ROC: %{y:+.1f}%<extra></extra>",
        ), row=2, col=1)
        fig.add_hline(y=0, line=dict(color="#4a5568", width=1.0), row=2, col=1)

        last_roc   = spy_roc.iloc[-1]
        roc_color  = "#2ECC71" if last_roc >= 0 else "#E74C3C"
        fig.add_annotation(
            x=spy_roc.index[-1], y=last_roc,
            text=f"<b>{last_roc:+.1f}%</b>",
            showarrow=True, arrowhead=2,
            arrowcolor=roc_color, font=dict(size=10, color=roc_color),
            ax=-60, ay=-25, row=2, col=1,
        )
        fig.update_yaxes(title_text="60d ROC %", row=2, col=1)

    subtitle = f"SPY: ${last_spy:.0f}  |  SMA200: ${last_sma:.0f}  |  Regime: {regime}  |  {pct_vs:+.1f}% vs SMA200"

    fig.update_layout(
        title=dict(
            text=f"SPY vs SMA200 — Equity Regime<br><sup>{subtitle}</sup>",
            font=dict(size=14), x=0.02,
        ),
        template="plotly_dark",
        height=600,
        hovermode="x unified",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01,
            xanchor="left", x=0, font=dict(size=10),
        ),
        margin=dict(t=100, b=40, l=60, r=80),
    )
    fig.update_xaxes(**PLOTLY_AXIS_STYLE)
    fig.update_yaxes(**PLOTLY_AXIS_STYLE)
    return fig


# -----------------------------------------------------------------
#  CHART 17 — Earnings Yield Gap
# -----------------------------------------------------------------

def earnings_yield_gap(ma: MacroAnalyzer) -> Optional[go.Figure]:
    """
    Interactive 2-panel Earnings Yield Gap chart.

    Panel 1: EYG history with zone fills
    Panel 2: Earnings Yield vs 10Y Treasury Yield
    """
    inds    = ma.get_indicators()
    history = ma.get_history()
    if inds.empty or "eyg" not in inds.columns:
        print("  eyg not found -- re-run ma.load()")
        return None

    weekly_i = inds.resample("W").last()
    weekly_h = history.resample("W").last()

    eyg = (weekly_i["eyg"].dropna() * 100)
    ey  = (weekly_i["earnings_yield"].dropna() * 100) if "earnings_yield" in weekly_i.columns else None
    y10 = weekly_i["yield_10y"].dropna() if "yield_10y" in weekly_i.columns else None

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.58, 0.42],
        subplot_titles=[
            "Earnings Yield Gap (EYG) = SPY Earnings Yield − 10Y Treasury Yield",
            "SPY Earnings Yield vs 10Y Treasury — red shading = bonds more attractive",
        ],
        vertical_spacing=0.07,
    )

    # ── Panel 1: EYG History ──────────────────────────────────────
    # Zone coloring via bar chart
    colors_eyg = []
    for v in eyg.values:
        if v > 3:
            colors_eyg.append("#2ECC71")
        elif v >= 1.5:
            colors_eyg.append("#F39C12")
        elif v >= 0:
            colors_eyg.append("#E67E22")
        else:
            colors_eyg.append("#E74C3C")

    eyg_pos = eyg.clip(lower=0)
    eyg_neg = eyg.clip(upper=0)
    fig.add_trace(go.Scatter(
        x=eyg.index, y=eyg_pos.values,
        fill="tozeroy", fillcolor="rgba(46,204,113,0.18)",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=eyg.index, y=eyg_neg.values,
        fill="tozeroy", fillcolor="rgba(231,76,60,0.25)",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=eyg.index, y=eyg.values,
        line=dict(color="#c9d1d9", width=1.5),
        name="EYG %",
        hovertemplate="%{x|%d/%m/%Y}<br>EYG: %{y:+.2f}%<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=eyg.index, y=eyg.values,
        line=dict(color="#c9d1d9", width=1.2),
        showlegend=False,
        hoverinfo="skip",
    ), row=1, col=1)

    for lv, color, lbl in [
        ( 3.0, "#2ECC71",  ">3% Bullish"),
        ( 1.5, "#F39C12",  "1.5% Neutral floor"),
        ( 0.0, "#E74C3C",  "0% Crisis signal"),
    ]:
        fig.add_hline(
            y=lv, line=dict(color=color, width=1.0, dash="dash"),
            annotation_text=lbl,
            annotation_position="right",
            annotation_font=dict(size=9, color=color),
            row=1, col=1,
        )

    last_eyg   = eyg.iloc[-1]
    eyg_color  = "#2ECC71" if last_eyg > 2 else "#E74C3C" if last_eyg < 0 else "#E67E22"
    eyg_signal = "BULLISH" if last_eyg > 3 else "NEUTRAL" if last_eyg >= 1.5 else "COMPRESSED" if last_eyg >= 0 else "CRISIS"
    fig.add_annotation(
        x=eyg.index[-1], y=last_eyg,
        text=f"<b>{last_eyg:+.2f}% ({eyg_signal})</b>",
        showarrow=True, arrowhead=2,
        arrowcolor=eyg_color, font=dict(size=10, color=eyg_color),
        ax=-90, ay=-30, row=1, col=1,
    )
    fig.update_yaxes(title_text="EYG %", row=1, col=1)

    # ── Panel 2: Earnings Yield vs 10Y ───────────────────────────
    if ey is not None and y10 is not None:
        ey_a, y10_a = ey.align(y10, join="inner")

        fig.add_trace(go.Scatter(
            x=ey_a.index, y=ey_a.values,
            line=dict(color="#3498DB", width=1.6),
            name="Earnings Yield (1/PE %)",
            hovertemplate="%{x|%d/%m/%Y}<br>Earnings Yield: %{y:.2f}%<extra></extra>",
        ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=y10_a.index, y=y10_a.values,
            line=dict(color="#E74C3C", width=1.6, dash="dash"),
            name="10Y Treasury Yield %",
            hovertemplate="%{x|%d/%m/%Y}<br>10Y: %{y:.2f}%<extra></extra>",
        ), row=2, col=1)

        # Shading where bonds > equities
        fig.add_trace(go.Scatter(
            x=list(ey_a.index) + list(ey_a.index[::-1]),
            y=list(y10_a.values) + list(ey_a.values[::-1]),
            fill="toself",
            fillcolor="rgba(231,76,60,0.12)",
            line=dict(width=0),
            name="Bonds > Equities",
            hoverinfo="skip",
        ), row=2, col=1)

        fig.update_yaxes(title_text="Yield %", row=2, col=1)

    # Crisis vrects
    for _, (s_str, e_str) in CRISIS_EVENTS.items():
        s = pd.Timestamp(s_str)
        if s < eyg.index[0]:
            continue
        for r in [1, 2]:
            fig.add_vrect(
                x0=s, x1=pd.Timestamp(e_str),
                fillcolor="rgba(231,76,60,0.07)",
                layer="below", line_width=0, row=r, col=1,
            )

    try:
        snap = ma.get_snapshot()
        pe_str  = f"P/E: {snap.spy_pe:.1f}x" if hasattr(snap, "spy_pe") and snap.spy_pe else ""
        ey_str  = f"EY: {(getattr(snap,'earnings_yield',None) or 0)*100:.2f}%"
        eyg_str = f"EYG: {(getattr(snap,'eyg',None) or 0)*100:+.2f}%"
        subtitle = "  |  ".join(filter(None, [pe_str, ey_str, eyg_str, eyg_signal]))
    except Exception:
        subtitle = f"EYG: {last_eyg:+.2f}%  |  {eyg_signal}"

    fig.update_layout(
        title=dict(
            text=f"Earnings Yield Gap (EYG) — Equity vs Bond Attractiveness<br><sup>{subtitle}</sup>",
            font=dict(size=14), x=0.02,
        ),
        template="plotly_dark",
        height=650,
        hovermode="x unified",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01,
            xanchor="left", x=0, font=dict(size=10),
        ),
        margin=dict(t=100, b=40, l=60, r=80),
    )
    fig.update_xaxes(**PLOTLY_AXIS_STYLE)
    fig.update_yaxes(**PLOTLY_AXIS_STYLE)
    return fig


# -----------------------------------------------------------------
#  CHART 18 — Real Yields (TIP proxy / DFII10)
# -----------------------------------------------------------------

def real_yields(ma: MacroAnalyzer) -> Optional[go.Figure]:
    """
    Interactive 2-panel real yields chart.

    Panel 1: TIP ETF price history with phase context
    Panel 2: DFII10 z-score (primary) or Real Yield Pressure TIP proxy (fallback)
    """
    inds   = ma.get_indicators()
    history = ma.get_history()
    prices  = getattr(ma, "_prices", {})
    if history.empty:
        return None

    weekly_i   = inds.resample("W").last()
    weekly_h   = history.resample("W").last()
    tip_prices = prices.get("tips")
    ryp        = weekly_i["real_yield_pressure"].dropna() if "real_yield_pressure" in weekly_i.columns else None
    has_dfii10 = "real_yield_zscore_100d" in weekly_i.columns

    if tip_prices is None and ryp is None:
        print("  TIP data not found")
        return None

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.58, 0.42],
        subplot_titles=[
            "TIP ETF — falling = rising real yields = headwind for equities & gold",
            ("DFII10 Z-score (100d) — > +1.5sd tightening  |  < -1.5sd easing"
             if has_dfii10 else
             "Real Yield Pressure (TIP proxy ROC) — > +0.02 bearish  |  < -0.02 bullish"),
        ],
        vertical_spacing=0.07,
    )

    # ── Panel 1: TIP price ───────────────────────────────────────
    if tip_prices is not None:
        tip_w = tip_prices.resample("W").last()

        fig.add_trace(go.Scatter(
            x=tip_w.index, y=tip_w.values,
            fill="tozeroy", fillcolor="rgba(0,188,212,0.08)",
            line=dict(color="#00BCD4", width=1.8),
            name="TIP ETF (USD)",
            hovertemplate="%{x|%d/%m/%Y}<br>TIP: $%{y:.2f}<extra></extra>",
        ), row=1, col=1)

        # 2022 rate crisis annotation
        crisis_start = pd.Timestamp("2022-01-01")
        crisis_end   = pd.Timestamp("2022-10-01")
        if crisis_start >= tip_w.index[0]:
            fig.add_vrect(
                x0=crisis_start, x1=crisis_end,
                fillcolor="rgba(231,76,60,0.12)",
                layer="below", line_width=0, row=1, col=1,
            )
            fig.add_annotation(
                x=crisis_start + (crisis_end - crisis_start) / 2,
                y=tip_w.max() * 0.97,
                text="2022 Rate Crisis",
                showarrow=False,
                font=dict(size=9, color="#E74C3C"),
                row=1, col=1,
            )

        last_tip = tip_w.iloc[-1]
        fig.add_annotation(
            x=tip_w.index[-1], y=last_tip,
            text=f"<b>${last_tip:.1f}</b>",
            showarrow=True, arrowhead=2,
            arrowcolor="#00BCD4", font=dict(size=10, color="#00BCD4"),
            ax=-60, ay=-25, row=1, col=1,
        )
        fig.update_yaxes(title_text="TIP ETF Price (USD)", row=1, col=1)

    _add_phase_shading(fig, weekly_h, row=1)

    # ── Panel 2: Z-score or pressure ─────────────────────────────
    if has_dfii10:
        ry_z = weekly_i["real_yield_zscore_100d"].dropna()
        colors_z = ["#E74C3C" if v > 1.5 else "#2ECC71" if v < -1.5 else "#7F8C8D"
                    for v in ry_z.values]
        ry_z_pos = ry_z.clip(lower=0)
        ry_z_neg = ry_z.clip(upper=0)
        fig.add_trace(go.Scatter(
            x=ry_z.index, y=ry_z_pos.values,
            fill="tozeroy", fillcolor="rgba(231,76,60,0.28)",
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=ry_z.index, y=ry_z_neg.values,
            fill="tozeroy", fillcolor="rgba(46,204,113,0.22)",
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=ry_z.index, y=ry_z.values,
            line=dict(color="#c9d1d9", width=1.3),
            name="DFII10 Z-score",
            hovertemplate="%{x|%d/%m/%Y}<br>Z: %{y:+.2f}sd<extra></extra>",
        ), row=2, col=1)

        for lv, color, lbl in [
            ( 1.5, "#E74C3C", "+1.5sd Tightening"),
            (-1.5, "#2ECC71", "-1.5sd Easing"),
        ]:
            fig.add_hline(
                y=lv, line=dict(color=color, width=1.0, dash="dash"),
                annotation_text=lbl,
                annotation_position="right",
                annotation_font=dict(size=9, color=color),
                row=2, col=1,
            )
        fig.add_hline(y=0, line=dict(color="#4a5568", width=1.0), row=2, col=1)

        last_ryz  = ry_z.iloc[-1]
        ryz_lbl   = "TIGHTENING" if last_ryz > 1.5 else "EASING" if last_ryz < -1.5 else "NEUTRAL"
        ryz_color = "#E74C3C" if last_ryz > 1.5 else "#2ECC71" if last_ryz < -1.5 else "#7F8C8D"
        fig.add_annotation(
            x=ry_z.index[-1], y=last_ryz,
            text=f"<b>{last_ryz:+.2f}sd ({ryz_lbl})</b>",
            showarrow=True, arrowhead=2,
            arrowcolor=ryz_color, font=dict(size=10, color=ryz_color),
            ax=-90, ay=-25, row=2, col=1,
        )
        fig.update_yaxes(title_text="DFII10 Z-score (100d)", row=2, col=1)
        p2_subtitle = f"DFII10 Z: {last_ryz:+.2f}sd  |  {ryz_lbl}"

    elif ryp is not None:
        colors_ryp = ["#E74C3C" if v > 0.02 else "#2ECC71" if v < -0.02 else "#7F8C8D"
                      for v in ryp.values]
        ryp_pos = ryp.clip(lower=0)
        ryp_neg = ryp.clip(upper=0)
        fig.add_trace(go.Scatter(
            x=ryp.index, y=ryp_pos.values,
            fill="tozeroy", fillcolor="rgba(231,76,60,0.28)",
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=ryp.index, y=ryp_neg.values,
            fill="tozeroy", fillcolor="rgba(46,204,113,0.22)",
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=ryp.index, y=ryp.values,
            line=dict(color="#c9d1d9", width=1.3),
            name="Real Yield Pressure",
            hovertemplate="%{x|%d/%m/%Y}<br>Pressure: %{y:+.4f}<extra></extra>",
        ), row=2, col=1)

        for lv, color, lbl in [
            ( 0.02, "#E74C3C", "+0.02 Bearish"),
            (-0.02, "#2ECC71", "-0.02 Bullish"),
        ]:
            fig.add_hline(
                y=lv, line=dict(color=color, width=1.0, dash="dash"),
                annotation_text=lbl,
                annotation_position="right",
                annotation_font=dict(size=9, color=color),
                row=2, col=1,
            )
        fig.add_hline(y=0, line=dict(color="#4a5568", width=1.0), row=2, col=1)
        fig.update_yaxes(title_text="Real Yield Pressure", row=2, col=1)

        last_ryp  = ryp.iloc[-1]
        ryp_lbl   = "RISING (bearish)" if last_ryp > 0.02 else "FALLING (bullish)" if last_ryp < -0.02 else "STABLE"
        p2_subtitle = f"Pressure: {last_ryp:+.4f}  |  {ryp_lbl}"
    else:
        p2_subtitle = ""

    subtitle = f"TIP proxy for real yield direction  |  {p2_subtitle}"

    fig.update_layout(
        title=dict(
            text=f"Real Yields Analysis (DFII10 primary / TIP fallback)<br><sup>{subtitle}</sup>",
            font=dict(size=14), x=0.02,
        ),
        template="plotly_dark",
        height=620,
        hovermode="x unified",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01,
            xanchor="left", x=0, font=dict(size=10),
        ),
        margin=dict(t=100, b=40, l=60, r=80),
    )
    fig.update_xaxes(**PLOTLY_AXIS_STYLE)
    fig.update_yaxes(**PLOTLY_AXIS_STYLE)
    return fig


# -----------------------------------------------------------------
#  CHART 19 — Gold & Silver
# -----------------------------------------------------------------

def gold_silver_chart(ma: MacroAnalyzer) -> Optional[go.Figure]:
    """
    Interactive 2-panel Gold and Silver chart.

    Panel 1: Gold + Silver (dual axis) with phase bands
    Panel 2: Gold/Silver ratio history
    """
    inds   = ma.get_indicators()
    history = ma.get_history()
    prices  = getattr(ma, "_prices", {})
    if history.empty:
        return None

    weekly_i = inds.resample("W").last()
    weekly_h = history.resample("W").last()

    gold_p      = prices.get("gold")
    silv_p      = prices.get("silver")
    gold_spot_p = prices.get("gold_spot", gold_p)
    silv_spot_p = prices.get("silver_spot", silv_p)

    if gold_p is None or silv_p is None:
        print("  Gold or Silver prices not found")
        return None

    gold_w      = gold_p.resample("W").last()
    silv_w      = silv_p.resample("W").last()
    gold_spot_w = gold_spot_p.resample("W").last()
    silv_spot_w = silv_spot_p.resample("W").last()
    gold_w, silv_w = gold_w.align(silv_w, join="inner")

    gs_ratio = weekly_i["gold_silver"].dropna() if "gold_silver" in weekly_i.columns else None

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.58, 0.42],
        subplot_titles=[
            "Gold vs Silver (dual axis) — divergences signal risk appetite shifts",
            "Gold/Silver Ratio — high = fear/defensive  |  low = risk appetite",
        ],
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
        vertical_spacing=0.07,
    )

    # ── Panel 1: Gold + Silver dual axis ─────────────────────────
    fig.add_trace(go.Scatter(
        x=gold_w.index, y=gold_w.values,
        line=dict(color="#F39C12", width=1.8),
        name="Gold (GLD)",
        hovertemplate="%{x|%d/%m/%Y}<br>Gold: $%{y:.0f}<extra></extra>",
    ), row=1, col=1, secondary_y=False)

    fig.add_trace(go.Scatter(
        x=silv_w.index, y=silv_w.values,
        line=dict(color="#8B949E", width=1.5, dash="dash"),
        name="Silver (SLV)",
        hovertemplate="%{x|%d/%m/%Y}<br>Silver: $%{y:.2f}<extra></extra>",
    ), row=1, col=1, secondary_y=True)

    _add_phase_shading(fig, weekly_h, row=1)

    # Crisis markers
    for name, (s_str, e_str) in CRISIS_EVENTS.items():
        s = pd.Timestamp(s_str)
        if s < gold_w.index[0]:
            continue
        fig.add_vrect(
            x0=s, x1=pd.Timestamp(e_str),
            fillcolor="rgba(231,76,60,0.07)",
            layer="below", line_width=0, row=1, col=1,
        )

    last_gold = gold_spot_w.iloc[-1]
    last_silv = silv_spot_w.iloc[-1]
    fig.add_annotation(
        x=gold_w.index[-1], y=gold_w.iloc[-1],
        text=f"<b>Gold ${last_gold:.0f}</b>",
        showarrow=True, arrowhead=2,
        arrowcolor="#F39C12", font=dict(size=10, color="#F39C12"),
        ax=-70, ay=-25, row=1, col=1,
    )

    fig.update_yaxes(title_text="Gold (USD)", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Silver (USD)", row=1, col=1, secondary_y=True)

    # ── Panel 2: Gold/Silver ratio ────────────────────────────────
    if gs_ratio is not None:
        rm_gs = gs_ratio.rolling(52, min_periods=10).mean()

        fig.add_trace(go.Scatter(
            x=gs_ratio.index, y=gs_ratio.values,
            line=dict(color="#9B59B6", width=1.6),
            name="Gold/Silver Ratio",
            hovertemplate="%{x|%d/%m/%Y}<br>Ratio: %{y:.1f}<extra></extra>",
        ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=rm_gs.index, y=rm_gs.values,
            line=dict(color="#9B59B6", width=1.0, dash="dash"),
            opacity=0.55, name="1Y avg",
            hovertemplate="%{x|%d/%m/%Y}<br>1Y avg: %{y:.1f}<extra></extra>",
        ), row=2, col=1)

        last_gs   = gs_ratio.iloc[-1]
        last_gs_m = rm_gs.iloc[-1]
        gs_color  = "#E74C3C" if last_gs > last_gs_m * 1.1 else "#2ECC71"
        fig.add_annotation(
            x=gs_ratio.index[-1], y=last_gs,
            text=f"<b>{last_gs:.1f}</b>",
            showarrow=True, arrowhead=2,
            arrowcolor=gs_color, font=dict(size=10, color=gs_color),
            ax=-60, ay=-25, row=2, col=1,
        )
        fig.update_yaxes(title_text="Gold/Silver Ratio", row=2, col=1)

    subtitle = f"Gold: ${last_gold:.0f}/oz  |  Silver: ${last_silv:.2f}/oz  (spot GC=F/SI=F)"

    fig.update_layout(
        title=dict(
            text=f"Gold & Silver Analysis<br><sup>{subtitle}</sup>",
            font=dict(size=14), x=0.02,
        ),
        template="plotly_dark",
        height=650,
        hovermode="x unified",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01,
            xanchor="left", x=0, font=dict(size=10),
        ),
        margin=dict(t=100, b=40, l=60, r=80),
        annotations=[
            dict(text="Gold vs Silver (dual axis) — divergences signal risk appetite shifts",
                 x=0.5, xref="paper", y=1.0, yref="paper",
                 xanchor="center", yanchor="bottom", showarrow=False, font=dict(size=13)),
            dict(text="Gold/Silver Ratio — high = fear/defensive  |  low = risk appetite",
                 x=0.5, xref="paper", y=0.34, yref="paper",
                 xanchor="center", yanchor="bottom", showarrow=False, font=dict(size=13)),
        ],
    )
    fig.update_xaxes(**PLOTLY_AXIS_STYLE)
    fig.update_yaxes(**PLOTLY_AXIS_STYLE)
    return fig


# -----------------------------------------------------------------
#  CHART 20 — Gold/Silver Ratio Mean Reversion
# -----------------------------------------------------------------

def gold_silver_mean_reversion(ma: MacroAnalyzer) -> Optional[go.Figure]:
    """
    Interactive 3-panel Gold/Silver ratio mean reversion chart.

    Panel 1: Ratio vs 5Y mean with Bollinger bands
    Panel 2: Mean reversion z-score
    Panel 3: Rolling half-life
    """
    inds    = ma.get_indicators()
    history = ma.get_history()
    if inds.empty or "gold_silver" not in inds.columns:
        return None

    weekly_i = inds.resample("W").last()
    weekly_h = history.resample("W").last()

    gs = weekly_i["gold_silver"].dropna()

    LT_WINDOW = 252 * 5
    lt_mean = gs.rolling(LT_WINDOW, min_periods=LT_WINDOW // 4).mean()
    lt_std  = gs.rolling(LT_WINDOW, min_periods=LT_WINDOW // 4).std()
    b1up    = lt_mean + lt_std
    b2up    = lt_mean + 2 * lt_std
    b1dn    = lt_mean - lt_std
    b2dn    = lt_mean - 2 * lt_std

    mr_z = ((gs - lt_mean) / lt_std.replace(0, np.nan)).round(3)

    gs_lag    = gs.shift(1)
    gs_a, gs_l = gs.align(gs_lag, join="inner")
    roll_corr  = gs_a.rolling(60, min_periods=30).corr(gs_l)
    rho_safe   = roll_corr.abs().clip(0.01, 0.999)
    half_life  = (-np.log(2) / np.log(rho_safe)).clip(upper=120)

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.45, 0.30, 0.25],
        subplot_titles=[
            "Gold/Silver Ratio vs 5Y Mean & Bollinger Bands — high = fear | low = risk-on",
            "Mean Reversion Z-score — > +2sd extreme fear | < -2sd extreme risk-on",
            "Mean Reversion Half-life (60d rolling) — days until ratio reverts to mean",
        ],
        vertical_spacing=0.06,
    )

    # ── Panel 1: Ratio + Bands ───────────────────────────────────
    fig.add_trace(go.Scatter(
        x=b2up.index, y=b2up.values,
        line=dict(color="#E74C3C", width=0.8, dash="dot"),
        opacity=0.6, name="+2sd (extreme fear)",
        hovertemplate="%{x|%d/%m/%Y}<br>+2sd: %{y:.1f}<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=b1up.index, y=b1up.values,
        fill="tonexty",
        fillcolor="rgba(231,76,60,0.08)",
        line=dict(color="#F39C12", width=0.8, dash="dot"),
        opacity=0.6, name="+1sd",
        hovertemplate="%{x|%d/%m/%Y}<br>+1sd: %{y:.1f}<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=lt_mean.index, y=lt_mean.values,
        line=dict(color="#c9d1d9", width=1.2, dash="dash"),
        opacity=0.8, name="5Y Mean",
        hovertemplate="%{x|%d/%m/%Y}<br>Mean: %{y:.1f}<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=b2dn.index, y=b2dn.values,
        line=dict(color="#2ECC71", width=0.8, dash="dot"),
        opacity=0.6, name="-2sd (extreme risk-on)",
        hovertemplate="%{x|%d/%m/%Y}<br>-2sd: %{y:.1f}<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=gs.index, y=gs.values,
        line=dict(color="#9B59B6", width=1.8),
        name="Gold/Silver Ratio",
        hovertemplate="%{x|%d/%m/%Y}<br>Ratio: %{y:.2f}<extra></extra>",
    ), row=1, col=1)

    last_gs   = gs.iloc[-1]
    last_mean = lt_mean.iloc[-1]
    gs_color  = "#E74C3C" if last_gs > last_mean * 1.15 else "#2ECC71" if last_gs < last_mean * 0.85 else "#7F8C8D"
    fig.add_annotation(
        x=gs.index[-1], y=last_gs,
        text=f"<b>{last_gs:.1f} (Mean: {last_mean:.1f})</b>",
        showarrow=True, arrowhead=2,
        arrowcolor=gs_color, font=dict(size=10, color=gs_color),
        ax=-100, ay=-25, row=1, col=1,
    )
    fig.update_yaxes(title_text="Gold/Silver", row=1, col=1)

    # ── Panel 2: MR Z-score ──────────────────────────────────────
    colors_z = ["#E74C3C" if v > 2 else "#2ECC71" if v < -2 else "#7F8C8D"
                for v in mr_z.values]
    mr_z_pos = mr_z.clip(lower=0)
    mr_z_neg = mr_z.clip(upper=0)
    fig.add_trace(go.Scatter(
        x=mr_z.index, y=mr_z_pos.values,
        fill="tozeroy", fillcolor="rgba(231,76,60,0.25)",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=mr_z.index, y=mr_z_neg.values,
        fill="tozeroy", fillcolor="rgba(46,204,113,0.22)",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=mr_z.index, y=mr_z.values,
        line=dict(color="#c9d1d9", width=1.3),
        name="MR Z-score",
        hovertemplate="%{x|%d/%m/%Y}<br>Z: %{y:+.2f}sd<extra></extra>",
    ), row=2, col=1)

    for lv, color, lbl in [
        ( 2.0, "#E74C3C", "+2sd Extreme Fear"),
        (-2.0, "#2ECC71", "-2sd Extreme Risk-On"),
    ]:
        fig.add_hline(
            y=lv, line=dict(color=color, width=1.0, dash="dash"),
            annotation_text=lbl,
            annotation_position="right",
            annotation_font=dict(size=9, color=color),
            row=2, col=1,
        )
    fig.add_hline(y=0, line=dict(color="#4a5568", width=1.0), row=2, col=1)

    last_mrz = mr_z.iloc[-1]
    mr_color = "#E74C3C" if last_mrz > 2 else "#2ECC71" if last_mrz < -2 else "#7F8C8D"
    mr_sig   = "EXTREME FEAR" if last_mrz > 2 else "EXTREME RISK-ON" if last_mrz < -2 else "NORMAL"
    fig.add_annotation(
        x=mr_z.index[-1], y=last_mrz,
        text=f"<b>{last_mrz:+.2f}sd ({mr_sig})</b>",
        showarrow=True, arrowhead=2,
        arrowcolor=mr_color, font=dict(size=10, color=mr_color),
        ax=-100, ay=-25, row=2, col=1,
    )
    fig.update_yaxes(title_text="MR Z-score", row=2, col=1)

    # ── Panel 3: Half-life ───────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=half_life.index, y=half_life.values,
        fill="tozeroy", fillcolor="rgba(0,188,212,0.10)",
        line=dict(color="#00BCD4", width=1.4),
        name="Half-life (days)",
        hovertemplate="%{x|%d/%m/%Y}<br>Half-life: %{y:.0f}d<extra></extra>",
    ), row=3, col=1)

    for lv, lbl in [(14, "14d"), (30, "30d"), (60, "60d")]:
        fig.add_hline(
            y=lv, line=dict(color="#30363d", width=0.8, dash="dash"),
            annotation_text=lbl,
            annotation_position="right",
            annotation_font=dict(size=8, color="#8b949e"),
            row=3, col=1,
        )

    last_hl = half_life.iloc[-1]
    fig.add_annotation(
        x=half_life.index[-1], y=last_hl,
        text=f"<b>{last_hl:.0f}d</b>",
        showarrow=True, arrowhead=2,
        arrowcolor="#00BCD4", font=dict(size=10, color="#00BCD4"),
        ax=-60, ay=-20, row=3, col=1,
    )
    fig.update_yaxes(title_text="Half-life (days)", row=3, col=1)

    subtitle = (f"Ratio: {last_gs:.1f}  |  5Y Mean: {last_mean:.1f}  |  "
                f"MR Z: {last_mrz:+.2f}sd  |  Half-life: {last_hl:.0f}d  |  {mr_sig}")

    fig.update_layout(
        title=dict(
            text=f"Gold/Silver Ratio — Mean Reversion Analysis<br><sup>{subtitle}</sup>",
            font=dict(size=14), x=0.02,
        ),
        template="plotly_dark",
        height=700,
        hovermode="x unified",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01,
            xanchor="left", x=0, font=dict(size=10),
        ),
        margin=dict(t=100, b=40, l=60, r=80),
    )
    fig.update_xaxes(**PLOTLY_AXIS_STYLE)
    fig.update_yaxes(**PLOTLY_AXIS_STYLE)
    return fig


# -----------------------------------------------------------------
#  CHART 21 — Inflation Environment
# -----------------------------------------------------------------

def inflation_environment(ma: MacroAnalyzer) -> Optional[go.Figure]:
    """
    Interactive 4-panel Inflation Environment chart.

    Panel 1: Gold price + 60d ROC (dual axis)
    Panel 2: Gold/Oil ratio + z-score
    Panel 3: TIPS ROC vs 10Y Yield ROC
    Panel 4: Inflation state classification timeline
    """
    inds    = ma.get_indicators()
    history = ma.get_history()
    p       = getattr(ma, "_prices", {})
    if inds is None or inds.empty:
        return None

    weekly_i = inds.resample("W").last()
    weekly_h = history.resample("W").last()

    gold_price     = p.get("gold")
    gold_roc       = (weekly_i["gold_roc_60"] * 100).dropna() if "gold_roc_60" in weekly_i.columns else None
    gold_oil_ratio = weekly_i["gold_oil"].dropna() if "gold_oil" in weekly_i.columns else None
    gold_oil_z     = weekly_i["gold_oil_zscore"].dropna() if "gold_oil_zscore" in weekly_i.columns else None
    tips_roc       = (weekly_i["tips_roc_60"] * 100).dropna() if "tips_roc_60" in weekly_i.columns else None
    y10y_roc       = (weekly_i["yield_10y_roc_60"] * 100).dropna() if "yield_10y_roc_60" in weekly_i.columns else None
    inf_env        = history["inflation_env"].resample("ME").last().dropna() if "inflation_env" in history.columns else None

    if gold_price is None and gold_roc is None:
        return None

    gold_w = gold_price.resample("W").last() if gold_price is not None else None

    INF_COLORS = {
        "low":         "#3498DB",
        "rising":      "#F39C12",
        "high":        "#E74C3C",
        "falling":     "#2ECC71",
        "stagflation": "#9B59B6",
    }

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        row_heights=[0.30, 0.25, 0.25, 0.20],
        subplot_titles=[
            "Gold Price (GLD) & 60d ROC — ROC > 5% inflation signal | > 8% high",
            "Gold/Oil Ratio + Z-score — z > +1.0 deflationary fear | z < -1.0 oil surge",
            "TIPS 60d ROC & 10Y Yield ROC — both rising = high inflation",
            "Inflation State Classification (monthly)",
        ],
        vertical_spacing=0.05,
        specs=[[{"secondary_y": True}],
               [{"secondary_y": True}],
               [{"secondary_y": False}],
               [{"secondary_y": False}]],
    )

    # ── Panel 1: Gold price + ROC ────────────────────────────────
    if gold_w is not None:
        fig.add_trace(go.Scatter(
            x=gold_w.index, y=gold_w.values,
            line=dict(color="#F39C12", width=2.0),
            name="Gold (GLD)",
            hovertemplate="%{x|%d/%m/%Y}<br>Gold: $%{y:.0f}<extra></extra>",
        ), row=1, col=1, secondary_y=False)

    if gold_roc is not None:
        colors_roc = ["#2ECC71" if v > 0 else "#E74C3C" for v in gold_roc.values]
        gold_roc_pos = gold_roc.clip(lower=0)
        gold_roc_neg = gold_roc.clip(upper=0)
        fig.add_trace(go.Scatter(
            x=gold_roc.index, y=gold_roc_pos.values,
            fill="tozeroy", fillcolor="rgba(46,204,113,0.20)",
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ), row=1, col=1, secondary_y=True)
        fig.add_trace(go.Scatter(
            x=gold_roc.index, y=gold_roc_neg.values,
            fill="tozeroy", fillcolor="rgba(231,76,60,0.22)",
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ), row=1, col=1, secondary_y=True)
        fig.add_trace(go.Scatter(
            x=gold_roc.index, y=gold_roc.values,
            line=dict(color="#8b949e", width=1.2),
            name="ROC 60d (%)",
            hovertemplate="%{x|%d/%m/%Y}<br>ROC: %{y:+.1f}%<extra></extra>",
        ), row=1, col=1, secondary_y=True)

        for lv, color, lbl in [(5, "#F39C12", "5% signal"), (8, "#E74C3C", "8% high")]:
            fig.add_hline(
                y=lv, line=dict(color=color, width=0.8, dash="dot"),
                annotation_text=lbl,
                annotation_position="right",
                annotation_font=dict(size=8, color=color),
                row=1, col=1,
            )

    fig.update_yaxes(title_text="Gold (USD)", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="ROC 60d %", row=1, col=1, secondary_y=True)

    # ── Panel 2: Gold/Oil + Z-score ──────────────────────────────
    if gold_oil_ratio is not None:
        fig.add_trace(go.Scatter(
            x=gold_oil_ratio.index, y=gold_oil_ratio.values,
            line=dict(color="#00BCD4", width=1.8),
            name="Gold/Oil Ratio",
            hovertemplate="%{x|%d/%m/%Y}<br>Gold/Oil: %{y:.2f}<extra></extra>",
        ), row=2, col=1, secondary_y=False)

    if gold_oil_z is not None:
        fig.add_trace(go.Scatter(
            x=gold_oil_z.index, y=gold_oil_z.values,
            line=dict(color="#9B59B6", width=1.5, dash="dash"),
            opacity=0.85, name="Gold/Oil Z-score",
            hovertemplate="%{x|%d/%m/%Y}<br>Z: %{y:.2f}<extra></extra>",
        ), row=2, col=1, secondary_y=True)

        for lv, color, lbl in [(1.0, "#E74C3C", "+1sd"), (-1.0, "#2ECC71", "-1sd")]:
            fig.add_hline(
                y=lv, line=dict(color=color, width=0.8, dash="dot"),
                annotation_text=lbl,
                annotation_position="right",
                annotation_font=dict(size=8, color=color),
                row=2, col=1,
            )

    fig.update_yaxes(title_text="Gold/Oil", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Z-score", row=2, col=1, secondary_y=True)

    # ── Panel 3: TIPS ROC + 10Y ROC ─────────────────────────────
    if tips_roc is not None:
        fig.add_trace(go.Scatter(
            x=tips_roc.index, y=tips_roc.values,
            line=dict(color="#00BCD4", width=1.8),
            name="TIPS 60d ROC (%)",
            hovertemplate="%{x|%d/%m/%Y}<br>TIPS ROC: %{y:+.1f}%<extra></extra>",
        ), row=3, col=1)

    if y10y_roc is not None:
        fig.add_trace(go.Scatter(
            x=y10y_roc.index, y=y10y_roc.values,
            line=dict(color="#3498DB", width=1.8, dash="dash"),
            name="10Y Yield 60d ROC (%)",
            hovertemplate="%{x|%d/%m/%Y}<br>10Y ROC: %{y:+.1f}%<extra></extra>",
        ), row=3, col=1)

    fig.add_hline(y=0, line=dict(color="#4a5568", width=1.0), row=3, col=1)
    fig.update_yaxes(title_text="ROC %", row=3, col=1)

    # ── Panel 4: Inflation state timeline ────────────────────────
    if inf_env is not None and len(inf_env) > 0:
        prev_state = None
        seg_start  = None
        seen_states = set()

        for dt, state in inf_env.items():
            if state != prev_state:
                if prev_state is not None and seg_start is not None:
                    color = INF_COLORS.get(prev_state, "#7F8C8D")
                    fig.add_vrect(
                        x0=seg_start, x1=dt,
                        fillcolor=f"rgba{tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.65,)}",
                        layer="below", line_width=0, row=4, col=1,
                    )
                seg_start  = dt
                prev_state = state

        if prev_state and seg_start:
            color = INF_COLORS.get(prev_state, "#7F8C8D")
            fig.add_vrect(
                x0=seg_start, x1=inf_env.index[-1],
                fillcolor=f"rgba{tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.65,)}",
                layer="below", line_width=0, row=4, col=1,
            )

    # Crisis markers all panels
    for _, (s_str, e_str) in CRISIS_EVENTS.items():
        s = pd.Timestamp(s_str)
        if gold_w is not None and s < gold_w.index[0]:
            continue
        for r in [1, 2, 3]:
            fig.add_vrect(
                x0=s, x1=pd.Timestamp(e_str),
                fillcolor="rgba(231,76,60,0.07)",
                layer="below", line_width=0, row=r, col=1,
            )

    try:
        snap       = ma.get_snapshot()
        inf_state  = snap.inflation_env.upper()
        gold_last  = (p.get("gold_spot") or p.get("gold", pd.Series())).iloc[-1]
        roc_last   = inds["gold_roc_60"].iloc[-1] * 100 if "gold_roc_60" in inds.columns else None
        go_z_last  = inds["gold_oil_zscore"].iloc[-1] if "gold_oil_zscore" in inds.columns else None
        parts = [f"State: {inf_state}", f"Gold: ${gold_last:.0f}"]
        if roc_last is not None: parts.append(f"ROC-60d: {roc_last:+.1f}%")
        if go_z_last is not None: parts.append(f"Gold/Oil z: {go_z_last:.2f}")
        subtitle = "  |  ".join(parts)
    except Exception:
        subtitle = "Inflation environment signals"

    fig.update_layout(
        title=dict(
            text=f"Inflation Environment — Signals & Classification<br><sup>{subtitle}</sup>",
            font=dict(size=14), x=0.02,
        ),
        template="plotly_dark",
        height=800,
        hovermode="x unified",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01,
            xanchor="left", x=0, font=dict(size=10),
        ),
        margin=dict(t=100, b=40, l=60, r=80),
    )
    fig.update_xaxes(**PLOTLY_AXIS_STYLE)
    fig.update_yaxes(**PLOTLY_AXIS_STYLE)
    return fig


# -----------------------------------------------------------------
#  CHART 22 — Crisis Signal Timeline
# -----------------------------------------------------------------

def crisis_timeline(
    ma:          "MacroAnalyzer",
    crisis_name: str,
    start_str:   str,
    end_str:     str,
    pre_months:  int = 12,
) -> Optional[go.Figure]:
    """
    Interactive 4-panel crisis signal timeline.

    Shows macro signal evolution before and during a specific crisis
    with shaded crisis zone and pre-crisis window.

    Parameters
    ----------
    ma           : MacroAnalyzer (with load() executed)
    crisis_name  : e.g. "GFC 2008"
    start_str    : crisis start e.g. "2007-10-01"
    end_str      : crisis end   e.g. "2009-03-01"
    pre_months   : months of pre-crisis history to show (default 12)

    Panels
    ------
    1: Macro Score + Risk Mode background
    2: Yield Curve 10Y-3M (inversion highlighted)
    3: Cu/Gold z-score + VIX
    4: SPY price (context)
    """
    inds    = ma.get_indicators()
    history = ma.get_history()
    prices  = getattr(ma, "_prices", {})

    if history.empty or inds.empty:
        return None

    crisis_start = pd.Timestamp(start_str)
    crisis_end   = pd.Timestamp(end_str)
    window_start = max(crisis_start - pd.DateOffset(months=pre_months), history.index[0])
    window_end   = min(crisis_end + pd.DateOffset(months=3), history.index[-1])

    h  = history[(history.index >= window_start) & (history.index <= window_end)]
    i  = inds[(inds.index >= window_start) & (inds.index <= window_end)]

    if h.empty:
        print(f"  crisis_timeline: no data for {crisis_name}")
        return None

    hw = h.resample("W").last()
    iw = i.resample("W").last()

    macro_score = hw["macro_score"].dropna() if "macro_score" in hw.columns else None
    risk_mode   = hw["risk_mode"].dropna()   if "risk_mode"   in hw.columns else None
    yield_curve = iw["yield_curve"].dropna() if "yield_curve" in iw.columns else None
    cg_z        = iw["copper_gold_zscore"].dropna() if "copper_gold_zscore" in iw.columns else None
    vix_s       = iw["vix"].dropna()         if "vix"         in iw.columns else None

    spy = None
    if "spy" in prices:
        sp  = prices["spy"]
        spy = sp[(sp.index >= window_start) & (sp.index <= window_end)].resample("W").last()

    RISK_PALETTE = {"risk_on": "#2ECC71", "neutral": "#F39C12", "risk_off": "#E74C3C"}

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        row_heights=[0.30, 0.25, 0.25, 0.20],
        subplot_titles=[
            "Macro Score (0–10) — background = Risk Mode",
            "Yield Curve 10Y−3M — red = inversion",
            "Copper/Gold Z-score (growth) + VIX (fear)",
            "SPY Price — context",
        ],
        vertical_spacing=0.06,
    )

    # Crisis + pre-crisis shading (all panels)
    for row_n in [1, 2, 3, 4]:
        # Pre-crisis (yellow)
        fig.add_vrect(
            x0=window_start, x1=crisis_start,
            fillcolor="rgba(243,156,18,0.06)",
            layer="below", line_width=0, row=row_n, col=1,
        )
        # Crisis (red)
        fig.add_vrect(
            x0=crisis_start, x1=min(crisis_end, window_end),
            fillcolor="rgba(231,76,60,0.10)",
            layer="below", line_width=0, row=row_n, col=1,
        )
        fig.add_shape(
            type="line",
            x0=str(crisis_start.date()), x1=str(crisis_start.date()),
            y0=0, y1=1, xref="x", yref="paper",
            line=dict(color="#E74C3C", width=1.5, dash="dash"),
            row=row_n, col=1,
        )

    # ── Panel 1: Macro Score + Risk Mode ─────────────────────────
    if risk_mode is not None:
        prev_rm  = None
        seg_s    = None
        for dt, rm in risk_mode.items():
            if rm != prev_rm:
                if prev_rm is not None and seg_s is not None:
                    c = RISK_PALETTE.get(prev_rm, "#7F8C8D")
                    fig.add_vrect(
                        x0=seg_s, x1=dt,
                        fillcolor=f"rgba{tuple(int(c.lstrip('#')[i:i+2], 16) for i in (0,2,4)) + (0.12,)}",
                        layer="below", line_width=0, row=1, col=1,
                    )
                seg_s  = dt
                prev_rm = rm
        if prev_rm and seg_s:
            c = RISK_PALETTE.get(prev_rm, "#7F8C8D")
            fig.add_vrect(
                x0=seg_s, x1=risk_mode.index[-1],
                fillcolor=f"rgba{tuple(int(c.lstrip('#')[i:i+2], 16) for i in (0,2,4)) + (0.12,)}",
                layer="below", line_width=0, row=1, col=1,
            )

    if macro_score is not None:
        fig.add_trace(go.Scatter(
            x=macro_score.index, y=macro_score.values,
            fill="tozeroy", fillcolor="rgba(52,152,219,0.12)",
            line=dict(color="#3498DB", width=2.0),
            name="Macro Score",
            hovertemplate="%{x|%d/%m/%Y}<br>Score: %{y:.1f}<extra></extra>",
        ), row=1, col=1)

        for lv, color, lbl in [(4.0, "#F39C12", "4.0"), (7.5, "#2ECC71", "7.5")]:
            fig.add_hline(
                y=lv, line=dict(color=color, width=0.8, dash="dash"),
                annotation_text=lbl,
                annotation_position="right",
                annotation_font=dict(size=8, color=color),
                row=1, col=1,
            )
        fig.update_yaxes(title_text="Score (0-10)", range=[0, 10], row=1, col=1)

    # ── Panel 2: Yield Curve ─────────────────────────────────────
    if yield_curve is not None:
        colors_yc = ["#E74C3C" if v < 0 else "#2ECC71" for v in yield_curve.values]
        yc_pos = yield_curve.clip(lower=0)
        yc_neg = yield_curve.clip(upper=0)
        fig.add_trace(go.Scatter(
            x=yield_curve.index, y=yc_pos.values,
            fill="tozeroy", fillcolor="rgba(46,204,113,0.18)",
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=yield_curve.index, y=yc_neg.values,
            fill="tozeroy", fillcolor="rgba(231,76,60,0.28)",
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=yield_curve.index, y=yield_curve.values,
            line=dict(color="#c9d1d9", width=1.5),
            name="Yield Curve 10Y-3M",
            hovertemplate="%{x|%d/%m/%Y}<br>Spread: %{y:.2f}%<extra></extra>",
        ), row=2, col=1)

        fig.add_hline(y=0, line=dict(color="#E74C3C", width=1.2), row=2, col=1)
        fig.add_hline(
            y=1.5, line=dict(color="#2ECC71", width=0.8, dash="dot"),
            annotation_text="Steep (1.5%)",
            annotation_position="right",
            annotation_font=dict(size=8, color="#2ECC71"),
            row=2, col=1,
        )
        fig.update_yaxes(title_text="10Y-3M %", row=2, col=1)

    # ── Panel 3: Cu/Gold z-score + VIX ───────────────────────────
    if cg_z is not None:
        fig.add_trace(go.Scatter(
            x=cg_z.index, y=cg_z.values,
            fill="tozeroy", fillcolor="rgba(0,188,212,0.10)",
            line=dict(color="#00BCD4", width=1.8),
            name="Cu/Gold z-score",
            hovertemplate="%{x|%d/%m/%Y}<br>Cu/Gold z: %{y:.2f}<extra></extra>",
        ), row=3, col=1)
        fig.add_hline(y=0, line=dict(color="#4a5568", width=1.0), row=3, col=1)

    if vix_s is not None:
        fig.add_trace(go.Scatter(
            x=vix_s.index, y=vix_s.values,
            line=dict(color="#9B59B6", width=1.5, dash="dash"),
            opacity=0.85, name="VIX",
            yaxis="y7",
            hovertemplate="%{x|%d/%m/%Y}<br>VIX: %{y:.1f}<extra></extra>",
        ), row=3, col=1)

        for lv, color in [(20, "#F39C12"), (35, "#E74C3C")]:
            fig.add_hline(
                y=lv, line=dict(color=color, width=0.7, dash="dot"),
                row=3, col=1,
            )

    fig.update_yaxes(title_text="Cu/Gold z-score", row=3, col=1)

    # ── Panel 4: SPY ─────────────────────────────────────────────
    if spy is not None and len(spy) > 0:
        fig.add_trace(go.Scatter(
            x=spy.index, y=spy.values,
            fill="tozeroy", fillcolor="rgba(52,152,219,0.08)",
            line=dict(color="#3498DB", width=1.8),
            name="SPY",
            hovertemplate="%{x|%d/%m/%Y}<br>SPY: $%{y:.0f}<extra></extra>",
        ), row=4, col=1)

        peak_val    = spy.max()
        trough_val  = spy.min()
        drawdown    = (trough_val - peak_val) / peak_val * 100
        peak_idx    = spy.idxmax()

        fig.add_annotation(
            x=peak_idx, y=peak_val,
            text=f"Peak ${peak_val:.0f}",
            showarrow=True, arrowhead=2,
            arrowcolor="#2ECC71", font=dict(size=9, color="#2ECC71"),
            ax=0, ay=-25, row=4, col=1,
        )
        fig.add_annotation(
            x=window_end, y=trough_val,
            text=f"Max DD: {drawdown:.1f}%",
            showarrow=False,
            font=dict(size=9, color="#E74C3C"),
            xanchor="right",
            row=4, col=1,
        )
        fig.update_yaxes(title_text="SPY ($)", row=4, col=1)

    # Pre-crisis label
    fig.add_annotation(
        x=window_start + (crisis_start - window_start) / 2,
        y=9.5,
        text=f"Pre-crisis<br>({pre_months}m window)",
        showarrow=False,
        font=dict(size=9, color="#F39C12"),
        row=1, col=1,
    )

    fig.update_layout(
        title=dict(
            text=(f"Crisis Signal Timeline — {crisis_name}<br>"
                  f"<sup>{window_start.strftime('%b %Y')} → {window_end.strftime('%b %Y')}  |  "
                  f"Crisis: {crisis_start.strftime('%b %Y')} – {crisis_end.strftime('%b %Y')}  |  "
                  f"Pre-crisis window: {pre_months} months</sup>"),
            font=dict(size=14), x=0.02,
        ),
        template="plotly_dark",
        height=750,
        hovermode="x unified",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01,
            xanchor="left", x=0, font=dict(size=10),
        ),
        margin=dict(t=110, b=40, l=60, r=80),
    )
    fig.update_xaxes(**PLOTLY_AXIS_STYLE)
    fig.update_yaxes(**PLOTLY_AXIS_STYLE)
    return fig


# -----------------------------------------------------------------
#  CHART 23 — Business Cycle Phase Gantt
# -----------------------------------------------------------------

def phase_gantt(
    ma:              "MacroAnalyzer",
    years:           int  = None,
    highlight_phase: str  = None,
) -> Optional[go.Figure]:
    """
    Interactive Gantt-style business cycle phase timeline.

    Parameters
    ----------
    ma               : MacroAnalyzer (with load() executed)
    years            : number of years of history (None = all)
    highlight_phase  : if set, dims all other phases.
                       Options: 'early_expansion' | 'late_expansion'
                                'early_contraction' | 'late_contraction'

    Two-panel layout
    ----------------
    Panel 1: Gantt bars per phase with crisis zones
    Panel 2: Phase distribution (stacked bar %)
    """
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

    # Build phase periods
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

    phases_ordered = ["early_expansion", "late_expansion",
                      "early_contraction", "late_contraction"]
    y_pos          = {p: i for i, p in enumerate(phases_ordered)}

    phase_counts   = weekly["cycle_phase"].value_counts()
    total          = phase_counts.sum()
    phase_pct      = {p: (phase_counts.get(p, 0) / total * 100) for p in phases_ordered}

    HL = highlight_phase

    def _color(phase):
        if HL is None or phase == HL:
            return PHASE_COLORS.get(phase, "#7F8C8D")
        return "#444444"

    def _opacity(phase):
        if HL is None or phase == HL:
            return 0.85
        return 0.25

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.78, 0.22],
        subplot_titles=[
            f"Business Cycle Phase Timeline{' — Highlight: ' + PHASE_LABELS.get(HL,'') if HL else ''}",
            "Phase Distribution (% of total time)",
        ],
        vertical_spacing=0.10,
    )

    # ── Panel 1: Gantt bars ──────────────────────────────────────
    # One trace per phase for legend
    for phase in phases_ordered:
        if phase not in y_pos:
            continue
        color   = _color(phase)
        opacity = _opacity(phase)
        label   = PHASE_LABELS.get(phase, phase)

        phase_periods = [(s, e) for s, e, p in periods if p == phase]
        if not phase_periods:
            continue

        for start, end in phase_periods:
            dur_days = (end - start).days
            dur_m    = dur_days / 30.4

            fig.add_trace(go.Bar(
                x=[(end - start).total_seconds() * 1000],  # ms duration
                y=[y_pos[phase]],
                base=[start],
                orientation="h",
                marker=dict(color=color, opacity=opacity, line=dict(width=0)),
                name=label,
                legendgroup=phase,
                showlegend=False,
                hovertemplate=(
                    f"<b>{label}</b><br>"
                    f"Start: {start.strftime('%b %Y')}<br>"
                    f"End: {end.strftime('%b %Y')}<br>"
                    f"Duration: {dur_m:.0f} months<extra></extra>"
                ),
            ), row=1, col=1)

        # One visible trace for the legend
        s, e = phase_periods[0]
        fig.add_trace(go.Bar(
            x=[(e - s).total_seconds() * 1000],
            y=[y_pos[phase]],
            base=[s],
            orientation="h",
            marker=dict(color=color, opacity=opacity),
            name=label,
            legendgroup=phase,
            showlegend=True,
            hoverinfo="skip",
        ), row=1, col=1)

    # Crisis zones
    for name, (s_str, e_str) in CRISIS_EVENTS.items():
        s = pd.Timestamp(s_str)
        e = pd.Timestamp(e_str)
        if s < weekly.index[0]:
            continue
        fig.add_vrect(
            x0=s, x1=e,
            fillcolor="rgba(231,76,60,0.08)",
            layer="below", line_width=0, row=1, col=1,
        )
        fig.add_annotation(
            x=s + (e - s) / 2,
            y=len(phases_ordered) - 0.3,
            text=name,
            showarrow=False,
            font=dict(size=8, color="#E74C3C"),
            row=1, col=1,
        )

    # NOW line -- add_vline with annotation fails on Timestamp x-axis;
    # use add_shape + add_annotation separately instead
    now_str = str(weekly.index[-1].date())
    fig.add_shape(
        type="line",
        x0=now_str, x1=now_str, y0=0, y1=1,
        xref="x", yref="paper",
        line=dict(color="#f0f6fc", width=1.5, dash="dot"),
        row=1, col=1,
    )
    fig.add_annotation(
        x=now_str, y=1.02,
        xref="x", yref="paper",
        text="Now", showarrow=False,
        font=dict(size=9, color="#f0f6fc"),
        xanchor="right",
    )

    fig.update_yaxes(
        tickvals=list(y_pos.values()),
        ticktext=[PHASE_LABELS.get(p, p) for p in phases_ordered],
        row=1, col=1,
    )

    # ── Panel 2: Distribution stacked bar ────────────────────────
    left = 0.0
    for phase in phases_ordered:
        pct     = phase_pct.get(phase, 0)
        color   = _color(phase)
        opacity = _opacity(phase)
        label   = PHASE_LABELS.get(phase, phase)

        fig.add_trace(go.Bar(
            x=[pct],
            y=["Distribution"],
            base=[left],
            orientation="h",
            marker=dict(color=color, opacity=opacity),
            name=label,
            legendgroup=phase,
            showlegend=False,
            hovertemplate=f"<b>{label}</b><br>{pct:.1f}%<extra></extra>",
        ), row=2, col=1)
        left += pct

    fig.update_yaxes(title_text="", row=2, col=1)
    fig.update_xaxes(title_text="% of total time", row=2, col=1)

    try:
        snap      = ma.get_snapshot()
        curr_lbl  = PHASE_LABELS.get(snap.cycle_phase, snap.cycle_phase)
        hl_suffix = f"  |  Highlight: {PHASE_LABELS.get(HL,'')}" if HL else ""
        subtitle  = (f"{weekly.index[0].year}–{weekly.index[-1].year}  |  "
                     f"Current: {curr_lbl}{hl_suffix}")
    except Exception:
        subtitle = ""

    fig.update_layout(
        title=dict(
            text=f"Business Cycle Phase Timeline<br><sup>{subtitle}</sup>",
            font=dict(size=14), x=0.02,
        ),
        template="plotly_dark",
        height=550,
        barmode="overlay",
        hovermode="closest",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01,
            xanchor="left", x=0, font=dict(size=10),
        ),
        margin=dict(t=100, b=40, l=140, r=80),
    )
    fig.update_xaxes(**PLOTLY_AXIS_STYLE)
    fig.update_yaxes(**PLOTLY_AXIS_STYLE)
    return fig