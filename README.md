# Macro Regime Analyzer

A professional-grade macro cycle analysis engine built in Python.  
Identifies the current economic regime, classifies cycle phases, and generates actionable signals for top-down investment frameworks.

---

## What it does

Most quantitative tools focus on price signals. This system works one level above — it reads the macro environment first, then derives investment implications from it.

The analyzer monitors 9 macro signals across commodities, rates, credit, volatility, and currency markets. It classifies the current environment into one of four cycle phases and outputs a structured snapshot that can feed directly into portfolio construction or stock selection frameworks.

**Cycle phases detected:**
- Early Expansion
- Late Expansion
- Early Contraction
- Late Contraction

---

## Signals monitored

| Signal | Proxy | What it captures |
|---|---|---|
| Copper / Gold ratio | HG=F / GC=F | Industrial demand vs safe haven |
| Gold / Silver ratio | GC=F / SI=F | Risk appetite |
| Yield curve | 10Y − 3M | Recession probability |
| Earnings Yield Gap | SPY earnings yield vs 10Y | Equity valuation regime |
| Credit spreads | HYG / LQD | Corporate stress |
| Dollar index | DX-Y.NYB | Global liquidity conditions |
| VIX | ^VIX | Market fear / complacency |
| Real yields | DFII10 (FRED) | True cost of capital |
| Cu/Gold–Yield divergence | Composite | Leading macro divergence signal |

---

## Features

- **Macro Score (0–10)** — weighted composite of all signals into a single regime reading
- **Probit Recession Probability** — Estrella & Mishkin (1998) model, calibrated on US data from 2000
- **VIX Mean Reversion model** — Ornstein-Uhlenbeck process to estimate VIX fair value and half-life
- **Stagflation detection** — 5-state inflation classifier separating deflation, disinflation, normal, inflation, and stagflation
- **Crisis timeline analysis** — pre-crisis signal behavior across 6 historical episodes (GFC 2008, Euro Crisis 2011, China Shock 2015, Q4 2018, COVID 2020, Bear Market 2022)
- **24 interactive charts** (Plotly) and **33 print-ready charts** (Matplotlib 300dpi)
- **Sector rotation guidance** per cycle phase
- **Asset allocation output** per regime

---

## Quickstart

```bash
git clone https://github.com/GkolfosGeorge/macro-compass.git
cd macro-compass
pip install -r requirements.txt
```

```python
from macro_analyzer import MacroAnalyzer
from macro_report import print_macro_snapshot

# Load data (downloads automatically via yfinance)
ma = MacroAnalyzer()
ma.load(start="2000-01-01")

# Current snapshot
snap = ma.get_snapshot()
print_macro_snapshot(snap)
```

**Output:**
```
═══════════════════════════════════════════════════════
  MACRO SNAPSHOT — 2026-06-18
═══════════════════════════════════════════════════════
  Cycle Phase   : Early Contraction
  Risk Mode     : Neutral
  Inflation Env : Low
  Macro Score   : 4.4 / 10
  VIX           : 18.4
  Yield Curve   : +0.87%  (10Y−3M)
  Probit Rec.   : 12.3%
```

---

## Charts

Generate all interactive charts:

```python
from macro_charts import phase_timeline, probit_recession, crisis_heatmap

phase_timeline(ma).show()
probit_recession(ma, snap).show()
crisis_heatmap(ma, snap).show()
```

Generate print-ready static charts (300dpi PNG):

```python
from macro_charts_print import print_all
print_all(ma, snap, save=True, output_dir="charts_print")
```

---

## FRED API (optional)

Some signals (Real Yields via DFII10, TED Spread, SOFR) require a free FRED API key.  
Get one at [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html) — it takes 2 minutes.

```python
ma = MacroAnalyzer(fred_api_key="your_key_here")
```

Without a key, the analyzer runs on the remaining 7 signals with no interruption.

---

## Project structure

```
macro-compass/                     # this folder IS the repo root
├── macro_analyzer.py        # Core engine — data loading, signal computation, phase classification
├── macro_charts.py          # Interactive charts (Plotly)
├── macro_charts_print.py    # Print-ready charts (Matplotlib, 300dpi)
├── macro_report.py          # Snapshot and historical analysis reports
├── Macro_Analyzer.ipynb     # Full interactive workflow (all charts, dev reload)
├── Macro_Demo.ipynb         # End-to-end walkthrough
├── requirements.txt
├── LICENSE
└── README.md
```

All modules use flat, self-contained imports (e.g. `from macro_analyzer import MacroAnalyzer`) — no package structure is required to run this repo standalone. Future integration with sibling projects (trading system, portfolio analyzer — see Roadmap) will revisit this as a proper installable package when that work starts.

---

## Related Projects

`macro-compass` is part of a broader personal quant ecosystem — sibling projects that consume or complement its regime signals:

- **Portfolio Analyzer** — position sizing and portfolio construction with a macro overlay. *(Live — [portfolio-analytics-toolkit](https://github.com/GkolfosGeorge/portfolio-analytics-toolkit))*
- **Trading System — Mean Reversion** — regime-aware mean reversion strategy with backtesting, built on top of `macro-compass` signals. *(Launching shortly)*

## Roadmap

- [ ] Trading System — Momentum/Hype Strategy — a second regime-aware trading script, complementing the mean-reversion system by capturing speculative, hype-driven momentum and herd behavior
- [ ] Commodity Analyzer — dedicated signal engine for agricultural commodities (grains, rice, lumber, fertilizers) — energy and metals are already covered natively within `macro-compass`
- [ ] Sector Rotation Scorer — momentum-weighted sector allocation

---

## License

All rights reserved. This repository is public for portfolio and demonstration purposes only — viewing it does not grant any license to use, copy, modify, or distribute the code. See [LICENSE](LICENSE) for details.

---

## Author

**George Gkolfos**  
Quantitative Investment Systems | Macro-Driven Frameworks  
[LinkedIn](https://linkedin.com/in/giorgos-gkolfos-243122119/) · [GitHub](https://github.com/GkolfosGeorge)

---

*Built for research and educational purposes. Not financial advice.*
