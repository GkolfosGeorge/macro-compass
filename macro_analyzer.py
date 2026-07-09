# macro_analyzer.py
"""
Macro Regime & Economic Cycle Analyzer
───────────────────────────────────────
Analyzes the macroeconomic environment using
exclusively market-based signals from yfinance.

Signals:
  - Commodity ratios   (Copper/Gold, Gold/Silver, Gold/Oil)
  - Yield curve        (10Y-2Y, 10Y-3M)
  - Real yields        (TIP proxy)
  - Credit spreads     (HYG/LQD)
  - Dollar trend       (DXY)
  - Sector rotation    (XL* ETFs)
  - Risk appetite      (SPY, QQQ, IWM, VIX)

Each signal is computed as:
  - Current value
  - Percentile vs history (rolling window)
  - Z-score (rolling window)

Output: MacroSnapshot — an object containing all information
        ready for use in macro_report.py and the trading notebook.

Usage:
    from macro_analyzer import MacroAnalyzer

    ma = MacroAnalyzer()
    ma.load(start="2000-01-01")
    snapshot = ma.get_snapshot()
    history  = ma.get_history()
"""

import json
import time
import warnings
import numpy as np
from scipy.stats import norm as _scipy_norm
try:
    from fredapi import Fred as _Fred
    _FREDAPI_AVAILABLE = True
except ImportError:
    _FREDAPI_AVAILABLE = False
import pandas as pd
import yfinance as yf
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# SYMBOLS
# ─────────────────────────────────────────────────────────────

SYMBOLS = {
    # Commodities (ETF for charts)
    "gold":     "GLD",    # ETF: 1 share ≈ variable oz gold — for charts
    "silver":   "SLV",    # ETF: 1 share ≈ variable oz silver — for charts
    "copper":   "HG=F",   # Futures $/lb
    "oil":      "CL=F",   # Futures $/barrel
    "platinum": "PL=F",   # Futures $/oz
    # Spot prices for ratios (correct units)
    "gold_spot":   "GC=F",  # Gold Futures $/oz — for ratio calculations
    "silver_spot": "SI=F",  # Silver Futures $/oz — for ratio calculations

    # Fixed Income
    "yield_10y": "^TNX",
    "yield_3m":  "^IRX",
    "yield_30y": "^TYX",
    "tips":      "TIP",

    # Credit
    "hyg": "HYG",
    "lqd": "LQD",

    # Dollar
    "dxy": "DX-Y.NYB",

    # Broad Equity
    "spy": "SPY",
    "qqq": "QQQ",
    "iwm": "IWM",

    # Volatility
    "vix": "^VIX",

    # Value vs Growth
    "value":  "IWD",
    "growth": "IWF",

    # Sectors
    "xlk": "XLK",   # Technology
    "xle": "XLE",   # Energy
    "xlf": "XLF",   # Financials
    "xlv": "XLV",   # Healthcare
    "xlu": "XLU",   # Utilities
    "xlp": "XLP",   # Consumer Staples
    "xli": "XLI",   # Industrials
    "xlb": "XLB",   # Materials
    "xlre": "XLRE", # Real Estate
}

SECTOR_NAMES = {
    "xlk":  "Technology",
    "xle":  "Energy",
    "xlf":  "Financials",
    "xlv":  "Healthcare",
    "xlu":  "Utilities",
    "xlp":  "Consumer Staples",
    "xli":  "Industrials",
    "xlb":  "Materials",
    "xlre": "Real Estate",
}

# ─────────────────────────────────────────────────────────────
# THRESHOLDS & CONFIG
# ─────────────────────────────────────────────────────────────

# Rolling window for z-score / percentile (trading days)
ZSCORE_WINDOW    = 252 * 5   # 5 years
PERCENTILE_WINDOW = 252 * 10  # 10 years

# Yield curve
INVERSION_THRESHOLD   = 0.0    # spread < 0 -> inverted
STEEP_THRESHOLD       = 1.5    # spread > 1.5% -> steep (early expansion)

# VIX
VIX_LOW    = 15
VIX_HIGH   = 20
VIX_STRESS = 35

# VIX Mean Reversion (Ornstein-Uhlenbeck proxy)
VIX_LONG_MEAN_WINDOW = 252 * 5   # 5Y rolling mean (long-term anchor)
VIX_MR_WINDOW        = 60        # 60d window for half-life estimation
VIX_MR_HIGH          =  2.0      # z-score > +2: extremely high -> short vol opportunity
VIX_MR_LOW           = -1.0      # z-score < -1: extremely low -> complacency warning

# Credit spreads (HYG/LQD ratio — not absolute spread)
CREDIT_RISK_ON  = 0.98   # ratio high -> spreads tight -> risk-on
CREDIT_RISK_OFF = 0.93   # ratio low  -> spreads wide -> risk-off

# DXY momentum (20-day ROC)
DXY_STRONG =  0.03   # +3% -> strong dollar
DXY_WEAK   = -0.03   # -3% -> weak dollar

# Copper/Gold - growth signal
CG_GROWTH    = 0.0010  # ratio high -> growth optimism
CG_RECESSION = 0.0005  # ratio low  -> slowdown

# Cache
DEFAULT_CACHE_HOURS = 24
DEFAULT_START = "2000-01-01"

# ── FRED Series ──────────────────────────────────────────────
FRED_SERIES = {
    "ism_manufacturing": "MANEMP",    # ISM Manufacturing -- fallback
    "ism_pmi":           "NAPM",      # ISM PMI (legacy, pre-2001)
    "ted_spread":        "TEDRATE",   # TED Spread (historical, until 2023)
    "sofr":              "SOFR",      # SOFR (LIBOR successor from 2022)
    "tbill_3m":          "DTB3",      # 3M T-Bill yield daily
}

# ISM thresholds
ISM_EXPANSION = 50.0   # > 50 = expanding
ISM_STRONG    = 55.0   # > 55 = strongly expanding
ISM_WEAK      = 45.0   # < 45 = strongly contracting

# TED Spread thresholds (basis points)
TED_NORMAL    = 0.50   # < 50bps = normal
TED_ELEVATED  = 1.00   # > 100bps = elevated stress
TED_CRISIS    = 2.00   # > 200bps = crisis

# ── Earnings Yield Gap ───────────────────────────────────────
# EYG = Earnings Yield (1/PE) - 10Y Yield
# Classification based on rolling historical mean:
#   EYG > +3%:              Bullish Equity Regime (equities cheap)
#   EYG ~ 1.5% - 2.5%:      Neutral / Stable (equilibrium)
#   EYG < 0.5% or negative: High Risk / Crisis Signal
EYG_BULLISH   =  0.030   # > +3%: equities cheap vs bonds
EYG_NEUTRAL_H =  0.025   # upper neutral zone
EYG_NEUTRAL_L =  0.015   # lower neutral zone
EYG_WARNING   =  0.005   # < 0.5%: compressed -- warning
EYG_DANGER    =  0.000   # < 0%: negative -- crisis signal

# SPY PE ratio -- fetched via yfinance info
SPY_PE_FALLBACK = 22.0  # fallback if not available

# ── Probit Recession Model (Estrella & Mishkin 1998) ─────────
# P(recession in 12m) = Phi(beta0 + beta1 * spread)
PROBIT_BETA0 = -0.6045
PROBIT_BETA1 = -0.7374
PROBIT_HORIZON = 252  # 12 months forward

# NBER Recession periods (hardcoded — do not change)
NBER_RECESSIONS = [
    ("2001-03-01", "2001-11-01"),  # dot-com
    ("2007-12-01", "2009-06-01"),  # GFC
    ("2020-02-01", "2020-04-01"),  # COVID
]

# ─────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────

@dataclass
class SignalReading:
    """A single signal measurement at a point in time."""
    name:        str
    value:       Optional[float]
    zscore:      Optional[float]
    percentile:  Optional[float]       # 0-100
    direction:   str = "neutral"       # "bullish" | "bearish" | "neutral"
    label:       str = ""              # human-readable description


@dataclass
class MacroSnapshot:
    """
    Complete macro picture for a specific date.
    This is the primary output of MacroAnalyzer.
    """
    date:            pd.Timestamp
    cycle_phase:     str    # "early_expansion" | "late_expansion" |
                            # "early_contraction" | "late_contraction" | "unknown"
    risk_mode:       str    # "risk_on" | "neutral" | "risk_off"
    inflation_env:   str    # "low" | "rising" | "high" | "falling"
    dollar_trend:    str    # "strong" | "neutral" | "weak"

    # Macro score (0-10) - overall environment quality for equities
    macro_score:     float

    # Raw signals
    signals:         dict = field(default_factory=dict)

    # Asset rotation suggestions
    favor_sectors:   list = field(default_factory=list)
    avoid_sectors:   list = field(default_factory=list)
    asset_allocation: dict = field(default_factory=dict)

    # Key ratios (for quick reference)
    copper_gold:     Optional[float] = None
    gold_silver:     Optional[float] = None
    gold_oil:        Optional[float] = None
    yield_curve:     Optional[float] = None   # 10Y - 3M spread
    yield_10y_abs:   Optional[float] = None   # 10Y absolute yield %
    credit_spread:   Optional[float] = None   # HYG/LQD ratio
    vix:             Optional[float] = None
    dxy_roc:         Optional[float] = None   # 20d rate of change

    # Probit Recession Probability (Estrella & Mishkin)
    probit_recession_prob: Optional[float] = None  # 0-100%
    probit_signal:         Optional[str]   = None  # "low"|"elevated"|"high"|"critical"

    # ISM Manufacturing
    ism:            Optional[float] = None  # PMI level
    ism_zscore:     Optional[float] = None
    ism_signal:     Optional[str]   = None  # "strong_expansion"|"expansion"|"contraction"|"strong_contraction"

    # TED Spread / FRA-OIS
    ted_spread:     Optional[float] = None  # TED spread level (%)
    ted_zscore:     Optional[float] = None
    ted_signal:     Optional[str]   = None  # "normal"|"watch"|"elevated"|"crisis"
    fra_ois_spread: Optional[float] = None  # FRA-OIS proxy (post-LIBOR)

    # VIX Mean Reversion
    vix_lt_mean:    Optional[float] = None  # 5Y rolling mean
    vix_mr_zscore:  Optional[float] = None  # distance from LT mean in std devs
    vix_half_life:  Optional[float] = None  # estimated mean reversion half-life (days)
    vix_band_1up:   Optional[float] = None  # mean + 1 std dev above
    vix_band_2up:   Optional[float] = None  # mean + 2 std devs above
    vix_mr_signal:  Optional[str]   = None  # "extended_high"|"extended_low"|"normal"

    # Cu/Gold vs 10Y Yield Divergence
    cg_yield_corr_60:           Optional[float] = None  # 60d rolling correlation
    cg_yield_corr_252:          Optional[float] = None  # 252d rolling correlation
    cg_yield_divergence:        Optional[float] = None  # Cu/Gold z - 10Y z
    cg_yield_divergence_zscore: Optional[float] = None  # normalized divergence
    cg_yield_signal:            Optional[str]   = None  # "bullish_div" | "bearish_div" | "neutral"

    # Real Yields (DFII10 primary / TIP fallback)
    real_yield_10y:         Optional[float] = None
    real_yield_zscore_100d: Optional[float] = None
    real_yield_pressure:    Optional[float] = None

    # Earnings Yield Gap
    spy_pe:         Optional[float] = None
    earnings_yield: Optional[float] = None
    eyg:            Optional[float] = None
    eyg_zscore:     Optional[float] = None
    eyg_signal:     Optional[str]   = None

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """Rolling z-score with minimum periods = window // 2."""
    mean = series.rolling(window, min_periods=window // 2).mean()
    std  = series.rolling(window, min_periods=window // 2).std()
    return ((series - mean) / std.replace(0, np.nan)).round(3)


def _rolling_percentile(series: pd.Series, window: int) -> pd.Series:
    """
    Rolling percentile rank (0-100).
    Resampled to monthly for speed - daily resolution not needed.
    """
    min_p = max(20, window // 4)
    monthly = series.resample("ME").last().dropna()
    def _pct(arr):
        if len(arr) < 5:
            return np.nan
        return round(np.mean(arr[:-1] <= arr[-1]) * 100, 1)
    w = max(3, window // 21)
    mp = max(3, min_p // 21)
    monthly_pct = monthly.rolling(w, min_periods=mp).apply(_pct, raw=True)
    return monthly_pct.reindex(series.index, method="ffill")
def _download_symbol(
    symbol: str,
    start:  str,
    retries: int = 3,
) -> Optional[pd.Series]:
    """
    Downloads Close prices for a symbol.
    Returns pd.Series with DatetimeIndex or None on failure.
    """
    for attempt in range(retries):
        try:
            raw = yf.download(
                symbol,
                start    = start,
                progress = False,
                auto_adjust = True,
            )
            if raw is None or raw.empty:
                time.sleep(1)
                continue

            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)

            raw.index = pd.to_datetime(raw.index).tz_localize(None)

            close = raw["Close"].dropna()
            if len(close) > 10:
                return close

        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1)

    return None


def _roc(series: pd.Series, periods: int = 20) -> pd.Series:
    """Rate of change."""
    return series.pct_change(periods)


# ─────────────────────────────────────────────────────────────
# MAIN CLASS
# ─────────────────────────────────────────────────────────────


def _cg_yield_signal(div, corr):
    """
    Cu/Gold vs 10Y Yield divergence signal classifier.
    Returns: "bullish_div" | "bearish_div" | "neutral"
    """
    if div is None or (isinstance(div, float) and np.isnan(div)):
        return "neutral"
    if div > 1.5:
        return "bullish_div"
    elif div < -1.5:
        return "bearish_div"
    if corr is not None and not (isinstance(corr, float) and np.isnan(corr)):
        if div > 0.8 and corr < 0.3:
            return "bullish_div"
        elif div < -0.8 and corr < 0.3:
            return "bearish_div"
    return "neutral"

class MacroAnalyzer:
    """
    Downloads macro data, computes signals and classifies
    the economic environment into cycle phases.

    Usage:
        ma = MacroAnalyzer()
        ma.load(start="2000-01-01")

        # Current snapshot
        snap = ma.get_snapshot()

        # History (for backtesting)
        hist = ma.get_history()

        # Snapshot at a specific date
        snap_2008 = ma.get_snapshot(pd.Timestamp("2008-09-15"))
    """

    def __init__(
        self,
        cache_folder:      str            = "data/macro_cache",
        cache_hours:       int            = DEFAULT_CACHE_HOURS,
        zscore_window:     int            = ZSCORE_WINDOW,
        percentile_window: int            = PERCENTILE_WINDOW,
        fred_api_key:      Optional[str]  = None,
    ):
        self.cache_folder      = Path(cache_folder)
        self.cache_hours       = cache_hours
        self.zscore_window     = zscore_window
        self.percentile_window = percentile_window
        self.fred_api_key      = fred_api_key

        # Raw price data - dict[name -> pd.Series]
        self._prices:   dict[str, pd.Series] = {}

        # Computed indicators - single DataFrame, daily index
        self._indicators: Optional[pd.DataFrame] = None

        # Computed history - MacroSnapshot per date
        self._history: Optional[pd.DataFrame] = None

        self._start: Optional[str] = None
        self._loaded: bool = False

    # ── Data Loading ──────────────────────────────────────────

    def load(
        self,
        start: str = DEFAULT_START,
        force_refresh: bool = False,
    ) -> "MacroAnalyzer":
        """
        Downloads or loads from cache all macro symbols.
        Then computes indicators and classifies phases.
        """
        self._start = start
        self.cache_folder.mkdir(parents=True, exist_ok=True)

        print(f"📡 Macro Analyzer — loading data from {start}...")

        for name, symbol in SYMBOLS.items():
            series = self._load_symbol(name, symbol, start, force_refresh)
            if series is not None:
                self._prices[name] = series
                print(f"   ✅ {name:<12} ({symbol})  {len(series)} rows")
            else:
                print(f"   ⚠️  {name:<12} ({symbol})  - failed")

        # ── FRED data (optional) ──────────────────────────────
        if self.fred_api_key:
            print(f"\n📊 FRED data download...")
            fred_downloads = [
                ("NAPM",        "ism_pmi",   "ISM Manufacturing PMI"),
                ("TEDRATE", "ted_spread",    "TED Spread"),
                ("SOFR",    "sofr",          "SOFR Rate"),
                ("DTB3",    "tbill_3m_fred", "3M T-Bill (FRED)"),
            ]
            for series_id, name, label in fred_downloads:
                series = self._fetch_fred(series_id, name, start, force_refresh)
                if series is not None and len(series) > 10:
                    self._prices[name] = series
                    print(f"   ✅ {label:<25} {len(series)} rows")
                else:
                    print(f"   ⚠️  {label:<25} - failed or empty")
                time.sleep(0.5)  # rate limit protection
        else:
            print("\n💡 FRED: skipped (no API key provided)")
            print("   For ISM + TED signals: MacroAnalyzer(fred_api_key='your_key')")

        print("\n🔧 Computing indicators...")
        self._indicators = self._compute_indicators()

        print("🗂  Classifying historical phases...")
        self._history = self._build_history()

        self._loaded = True
        print(f"✅ Macro Analyzer ready — {len(self._indicators)} trading days\n")

        return self

    def _load_symbol(
        self,
        name:    str,
        symbol:  str,
        start:   str,
        force:   bool,
    ) -> Optional[pd.Series]:
        """Cache-aware download for a single symbol."""
        cache_path = self.cache_folder / f"{name}.parquet"

        # Check cache
        if not force and cache_path.exists():
            age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
            if age_hours < self.cache_hours:
                try:
                    s = pd.read_parquet(cache_path).squeeze()
                    s.index = pd.to_datetime(s.index).tz_localize(None)
                    # Check if cache covers the requested start date
                    if not s.empty and s.index[0] <= pd.Timestamp(start) + pd.Timedelta(days=30):
                        return s[s.index >= pd.Timestamp(start)]
                except Exception:
                    pass

        # Download
        series = _download_symbol(symbol, start)
        if series is not None:
            try:
                series.to_frame().to_parquet(cache_path)
            except Exception:
                pass

        return series

    def _fetch_fred(
        self,
        series_id: str,
        name:      str,
        start:     str,
        force:     bool = False,
    ) -> Optional[pd.Series]:
        """
        Cache-aware download from FRED.
        Returns None if no API key or on failure.
        """
        if not self.fred_api_key:
            return None
        if not _FREDAPI_AVAILABLE:
            print("   ⚠️  fredapi not installed")
            print(f"      pip install fredapi")
            return None

        cache_path = self.cache_folder / f"fred_{name}.parquet"

        # Check cache
        if not force and cache_path.exists():
            age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
            if age_hours < self.cache_hours * 24:  # FRED data: 24x longer cache
                try:
                    s = pd.read_parquet(cache_path).squeeze()
                    s.index = pd.to_datetime(s.index).tz_localize(None)
                    if not s.empty:
                        return s[s.index >= pd.Timestamp(start)]
                except Exception:
                    pass

        # Download from FRED via fredapi (with retry)
        for attempt in range(3):
            try:
                fred   = _Fred(api_key=self.fred_api_key)
                raw    = fred.get_series(
                    series_id,
                    observation_start=start,
                )
                if raw is None or raw.empty:
                    if attempt < 2:
                        time.sleep(2)
                        continue
                    return None

                series = raw.dropna()
                series.index = pd.to_datetime(series.index).tz_localize(None)
                series.name  = name

                # Cache
                try:
                    series.to_frame().to_parquet(cache_path)
                except Exception:
                    pass

                return series

            except Exception as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                print(f"   ⚠️  FRED {series_id}: {str(e)[:60]}")
                return None
        return None

    # ── Indicator Computation ─────────────────────────────────

    def _compute_indicators(self) -> pd.DataFrame:
        """
        Computes all ratios, spreads, and statistics.
        Returns DataFrame with common index (trading days).
        """
        p = self._prices
        df = pd.DataFrame()

        # ── 1. Commodity Ratios ───────────────────────────────
        #
        # Use spot prices (GC=F, SI=F) for ratios
        # to get correct absolute values ($/oz).
        # Fallback to ETFs (GLD, SLV) if spot prices unavailable.
        #
        # GLD/SLV: ETF with variable multiplier (fees ~0.4-0.5%/year)
        #   → gives incorrect ratio values but correct z-scores internally
        # GC=F/SI=F: Futures continuous series → correct $/oz prices

        gold_p   = p.get("gold_spot",   p.get("gold"))    # GC=F if available, else GLD
        silver_p = p.get("silver_spot", p.get("silver"))  # SI=F if available, else SLV

        if gold_p is not None and silver_p is not None:
            gold_silver = _align_ratio(gold_p, silver_p)
            df["gold_silver"]            = gold_silver
            df["gold_silver_zscore"]     = _rolling_zscore(gold_silver, self.zscore_window)
            df["gold_silver_pct"]        = _rolling_percentile(gold_silver, self.percentile_window)

        if "copper" in p and gold_p is not None:
            copper_gold = _align_ratio(p["copper"], gold_p)
            df["copper_gold"]            = copper_gold
            df["copper_gold_zscore"]     = _rolling_zscore(copper_gold, self.zscore_window)
            df["copper_gold_pct"]        = _rolling_percentile(copper_gold, self.percentile_window)

        if gold_p is not None and "oil" in p:
            gold_oil = _align_ratio(gold_p, p["oil"])
            df["gold_oil"]               = gold_oil
            df["gold_oil_zscore"]        = _rolling_zscore(gold_oil, self.zscore_window)
            df["gold_oil_pct"]           = _rolling_percentile(gold_oil, self.percentile_window)

        # Gold momentum + percentile (for inflation classifier)
        # Uses spot price (GC=F) if available, otherwise GLD
        _gold_for_momentum = p.get("gold_spot", p.get("gold"))
        if _gold_for_momentum is not None:
            df["gold_roc_60"]  = _roc(_gold_for_momentum, 60)
            df["gold_roc_20"]  = _roc(_gold_for_momentum, 20)
            df["gold_zscore"]  = _rolling_zscore(_gold_for_momentum, self.zscore_window)
            df["gold_pct"]     = _rolling_percentile(_gold_for_momentum, self.percentile_window)

        # Copper momentum
        if "copper" in p:
            df["copper_roc_60"] = _roc(p["copper"], 60)

        # ── 2. Yield Curve ────────────────────────────────────

        if "yield_10y" in p and "yield_3m" in p:
            y10, y3m = p["yield_10y"].align(p["yield_3m"], join="inner")
            curve_10_3m = (y10 - y3m).rename("yield_curve_10_3m")
            df["yield_curve"]        = curve_10_3m
            df["yield_curve_zscore"] = _rolling_zscore(curve_10_3m, self.zscore_window)
            df["yield_curve_pct"]    = _rolling_percentile(curve_10_3m, self.percentile_window)
            df["yield_inverted"]     = (curve_10_3m < INVERSION_THRESHOLD).astype(int)

        if "yield_10y" in p:
            df["yield_10y"]        = p["yield_10y"]
            df["yield_10y_roc_60"] = _roc(p["yield_10y"], 60)

        if "yield_3m" in p:
            df["yield_3m"] = p["yield_3m"]

        # ── 2b. Probit Recession Probability (Estrella & Mishkin) ─
        # P(recession in 12m) = Phi(beta0 + beta1 * spread_10y_3m)
        if "yield_curve" in df.columns:
            spread = df["yield_curve"].fillna(0)
            linear_index = PROBIT_BETA0 + PROBIT_BETA1 * spread
            df["probit_recession_prob"] = (
                _scipy_norm.cdf(linear_index) * 100
            ).round(2)  # 0-100%

            # Signal classification
            # > 30%: elevated recession risk (NY Fed threshold)
            # > 50%: high recession risk
            # < 15%: low risk
            df["probit_signal"] = pd.cut(
                df["probit_recession_prob"],
                bins   = [0, 15, 30, 50, 100],
                labels = ["low", "elevated", "high", "critical"],
            ).astype(str)

        # ── 3. Real Yields ────────────────────────────────────
        # Primary: DFII10 from FRED (10Y Real Treasury Rate, direct measurement)
        # Fallback: TIP ETF proxy (when no FRED key available)
        #
        # Signal logic (ChatGPT / Quant approach):
        # Z-score(DFII10, 100d) > 1.5 -> rising real yields -> liquidity tightening
        # -> headwind for growth stocks, gold, leveraged assets

        if "tips" in p:
            df["tips_roc_60"] = _roc(p["tips"], 60)
            df["tips_zscore"] = _rolling_zscore(p["tips"], self.zscore_window)

        if "real_yield_10y" in p:
            # PRIMARY: DFII10 -- actual 10Y real yield (%)
            ry = (p["real_yield_10y"]
                  .reindex(df.index, method="ffill")
                  .fillna(method="bfill"))
            df["real_yield_10y"]        = ry
            df["real_yield_10y_zscore"] = _rolling_zscore(ry, self.zscore_window)
            df["real_yield_10y_pct"]    = _rolling_percentile(ry, self.percentile_window)

            # Z-score vs 100d rolling mean (ChatGPT approach)
            ry_100d_mean = ry.rolling(100, min_periods=30).mean()
            ry_100d_std  = ry.rolling(100, min_periods=30).std()
            df["real_yield_zscore_100d"] = (
                (ry - ry_100d_mean) / ry_100d_std.replace(0, np.nan)
            ).round(3)

            # Pressure signal: z > 1.5 -> tightening, z < -1.5 -> easing
            df["real_yield_pressure"] = df["real_yield_zscore_100d"] / 10.0
            # Scale: z=1.5 -> pressure=0.15 >> threshold 0.02 -> bearish
            #        z=-1.5 -> pressure=-0.15 << -0.02 -> bullish

        elif "tips" in p:
            # FALLBACK: TIP ETF proxy
            df["real_yield_pressure"] = -_roc(p["tips"], 60)

        # ── 4. Credit Spreads (HYG/LQD ratio) ────────────────

        if "hyg" in p and "lqd" in p:
            credit = _align_ratio(p["hyg"], p["lqd"])
            df["credit_spread_ratio"]     = credit
            df["credit_spread_zscore"]    = _rolling_zscore(credit, self.zscore_window)
            df["credit_spread_pct"]       = _rolling_percentile(credit, self.percentile_window)
            df["credit_roc_20"]           = _roc(credit, 20)

        # ── 5. Dollar (DXY) ───────────────────────────────────

        if "dxy" in p:
            df["dxy"]         = p["dxy"]
            df["dxy_roc_20"]  = _roc(p["dxy"], 20)
            df["dxy_roc_60"]  = _roc(p["dxy"], 60)
            df["dxy_zscore"]  = _rolling_zscore(p["dxy"], self.zscore_window)
            df["dxy_pct"]     = _rolling_percentile(p["dxy"], self.percentile_window)

        # ── 6. VIX + Mean Reversion ──────────────────────────

        if "vix" in p:
            vix = p["vix"]
            df["vix"]         = vix
            df["vix_zscore"]  = _rolling_zscore(vix, self.zscore_window)
            df["vix_pct"]     = _rolling_percentile(vix, self.percentile_window)
            df["vix_roc_10"]  = _roc(vix, 10)

            # ── VIX Mean Reversion metrics ──────────────────
            # Long-term rolling mean (5Y) — the "natural" mean
            vix_lt_mean = vix.rolling(
                VIX_LONG_MEAN_WINDOW,
                min_periods=VIX_LONG_MEAN_WINDOW // 4
            ).mean()
            vix_lt_std  = vix.rolling(
                VIX_LONG_MEAN_WINDOW,
                min_periods=VIX_LONG_MEAN_WINDOW // 4
            ).std()

            df["vix_lt_mean"] = vix_lt_mean.round(2)
            df["vix_lt_std"]  = vix_lt_std.round(2)

            # Distance from long-term mean in std devs
            # > +2: extremely high -> mean reversion opportunity (short vol)
            # < -1: extremely low -> complacency warning
            df["vix_mr_zscore"] = (
                (vix - vix_lt_mean) / vix_lt_std.replace(0, np.nan)
            ).round(3)

            # Upper/Lower bands (mean ± 1 std dev, ± 2 std devs)
            df["vix_band_1up"]  = (vix_lt_mean + vix_lt_std).round(2)
            df["vix_band_1dn"]  = (vix_lt_mean - vix_lt_std).round(2)
            df["vix_band_2up"]  = (vix_lt_mean + 2 * vix_lt_std).round(2)
            df["vix_band_2dn"]  = (vix_lt_mean - 2 * vix_lt_std).round(2)

            # Half-life of mean reversion (60d window)
            # Computed via AR(1): lag-1 autocorrelation -> lambda -> HL = -ln(2)/ln(lambda)
            vix_lag = vix.shift(1)
            vix_aligned, vix_lag_aligned = vix.align(vix_lag, join="inner")
            roll_corr = vix_aligned.rolling(
                VIX_MR_WINDOW, min_periods=30
            ).corr(vix_lag_aligned)
            # HL = -ln(2) / ln(|rho|) — in days
            # clip to avoid log(0) or log(>1)
            rho_safe = roll_corr.abs().clip(0.01, 0.999)
            df["vix_half_life"] = (
                -np.log(2) / np.log(rho_safe)
            ).round(1)

            # Mean reversion signal
            # "extended_high": VIX >> mean -> likely to fall
            # "extended_low":  VIX << mean -> complacency
            # "normal": within 1 std dev bands
            mr_z = df["vix_mr_zscore"]
            df["vix_mr_signal"] = np.where(
                mr_z > VIX_MR_HIGH,  "extended_high",
                np.where(mr_z < VIX_MR_LOW, "extended_low", "normal")
            )

        # ── 7. Equity Regime ──────────────────────────────────

        if "spy" in p:
            spy_sma200 = p["spy"].rolling(200).mean()
            df["spy_above_sma200"] = (p["spy"] > spy_sma200).astype(int)
            df["spy_roc_60"]       = _roc(p["spy"], 60)
            df["spy_roc_20"]       = _roc(p["spy"], 20)

        if "qqq" in p and "spy" in p:
            qqq_spy = _align_ratio(p["qqq"], p["spy"])
            df["qqq_spy_ratio"]    = qqq_spy
            df["qqq_spy_roc_20"]   = _roc(qqq_spy, 20)

        if "iwm" in p and "spy" in p:
            iwm_spy = _align_ratio(p["iwm"], p["spy"])
            df["iwm_spy_ratio"]    = iwm_spy
            df["iwm_spy_roc_20"]   = _roc(iwm_spy, 20)

        # ── 8. Value vs Growth ────────────────────────────────

        if "value" in p and "growth" in p:
            val_growth = _align_ratio(p["value"], p["growth"])
            df["value_growth_ratio"]   = val_growth
            df["value_growth_roc_60"]  = _roc(val_growth, 60)

        # ── 9. Sector Momentum (60d ROC vs SPY) ──────────────

        for sec in ["xlk", "xle", "xlf", "xlv", "xlu", "xlp", "xli", "xlb", "xlre"]:
            if sec in p:
                df[f"{sec}_roc_60"] = _roc(p[sec], 60)
                if "spy" in p:
                    sec_rel = _align_ratio(p[sec], p["spy"])
                    df[f"{sec}_rel_spy_20"] = _roc(sec_rel, 20)

        # ── Forward fill and sort ─────────────────────────────


        # ── 10. ISM Manufacturing (FRED) ─────────────────────────

        if "ism_pmi" in p:
            # Monthly data -> reindex to df.index
            ism_raw = (p["ism_pmi"]
                       .reindex(df.index, method="ffill")
                       .fillna(method="bfill"))
            df["ism"]          = ism_raw
            df["ism_zscore"]   = _rolling_zscore(ism_raw, self.zscore_window)
            df["ism_pct"]      = _rolling_percentile(ism_raw, self.percentile_window)
            df["ism_roc_3m"]   = ism_raw.pct_change(63)  # 3-month momentum

            # Signal classification
            # > 55: strongly expanding | 50-55: expanding
            # 45-50: contracting | < 45: strongly contracting
            df["ism_signal"] = np.where(
                ism_raw > ISM_STRONG,   "strong_expansion",
                np.where(ism_raw > ISM_EXPANSION, "expansion",
                np.where(ism_raw > ISM_WEAK,      "contraction",
                                                   "strong_contraction"))
            )

        # ── 11. TED Spread / SOFR Spread (FRED) ──────────────────

        if "ted_spread" in p:
            # Reindex to df.index (avoids length mismatch)
            ted = (p["ted_spread"]
                   .reindex(df.index, method="ffill")
                   .fillna(method="bfill"))
            df["ted_spread"]        = ted
            df["ted_spread_zscore"] = _rolling_zscore(ted, self.zscore_window)
            df["ted_spread_pct"]    = _rolling_percentile(ted, self.percentile_window)

            # Signal: bank systemic risk
            df["ted_signal"] = np.where(
                ted > TED_CRISIS,   "crisis",
                np.where(ted > TED_ELEVATED, "elevated",
                np.where(ted > TED_NORMAL,   "watch", "normal"))
            )

        elif "sofr" in p and "tbill_3m_fred" in p:
            # FRA-OIS proxy: SOFR - 3M T-Bill (post-LIBOR era)
            sofr, tbill = p["sofr"].align(p["tbill_3m_fred"], join="inner")
            sofr_d  = sofr.reindex(df.index, method="ffill")
            tbill_d = tbill.reindex(df.index, method="ffill")
            fra_ois = (sofr_d - tbill_d).dropna()
            df["fra_ois_spread"]        = fra_ois
            df["fra_ois_zscore"]        = _rolling_zscore(fra_ois, self.zscore_window)
            df["fra_ois_pct"]           = _rolling_percentile(fra_ois, self.percentile_window)

            df["ted_signal"] = np.where(
                fra_ois > TED_CRISIS,   "crisis",
                np.where(fra_ois > TED_ELEVATED, "elevated",
                np.where(fra_ois > TED_NORMAL,   "watch", "normal"))
            )

        # -- 12. Cu/Gold vs 10Y Yield Divergence --
        # Measures divergence between growth expectations (Cu/Gold)
        # and bond market (10Y yield) — signal for bond market direction

        if "copper_gold_zscore" in df.columns and "yield_10y" in df.columns:

            # Z-score 10Y yield (5Y rolling) — for comparison with Cu/Gold z-score
            df["yield_10y_zscore"] = _rolling_zscore(
                df["yield_10y"], self.zscore_window
            )

            # Rolling 60-day correlation Cu/Gold vs 10Y yield
            # Normally > 0.6 — when it drops below 0.3 = divergence regime
            cg_al, y10_al = df["copper_gold"].align(df["yield_10y"], join="inner")
            df["cg_yield_corr_60"] = (
                cg_al.rolling(60, min_periods=30).corr(y10_al).round(3)
            )

            # Rolling 252-day correlation (1 year) — smoother context
            df["cg_yield_corr_252"] = (
                cg_al.rolling(252, min_periods=126).corr(y10_al).round(3)
            )

            # Divergence = Cu/Gold z-score minus 10Y yield z-score
            # > +1.5: Bullish div (Cu/Gold running, yields anchored -> yields will rise)
            # < -1.5: Bearish div (Cu/Gold falling, yields holding -> Long duration signal)
            df["cg_yield_divergence"] = (
                df["copper_gold_zscore"] - df["yield_10y_zscore"]
            ).round(3)

            # Z-score of divergence (5Y rolling) — for normalization
            df["cg_yield_divergence_zscore"] = _rolling_zscore(
                df["cg_yield_divergence"], self.zscore_window
            )

        # ── 13. Earnings Yield Gap (EYG) ─────────────────────
        # EYG = Earnings Yield (1/PE_spy) - 10Y Yield
        # Measures whether equities are cheap or expensive vs bonds
        # Uses SPY PE ratio from yfinance + yield_10y

        if "yield_10y" in df.columns:
            # Fetch trailing PE of SPY once
            spy_pe = SPY_PE_FALLBACK
            try:
                import yfinance as _yf
                spy_info = _yf.Ticker("SPY").info
                spy_pe   = (spy_info.get("trailingPE") or
                            spy_info.get("forwardPE") or
                            SPY_PE_FALLBACK)
            except Exception:
                pass

            # Earnings yield = 1 / PE (as %)
            earnings_yield = 1.0 / max(spy_pe, 1.0)

            # EYG = earnings yield - 10Y yield
            # Both as decimals (e.g. 0.045 = 4.5%)
            df["spy_pe"]        = spy_pe
            df["earnings_yield"] = round(earnings_yield, 4)
            df["eyg"] = (
                earnings_yield - df["yield_10y"] / 100.0
            ).round(4)

            # Rolling EYG z-score (5Y) for historical context
            df["eyg_zscore"] = _rolling_zscore(
                df["eyg"], self.zscore_window
            )
            df["eyg_pct"] = _rolling_percentile(
                df["eyg"], self.percentile_window
            )

            # Signal — based on historical mean
            # EYG > 3%:  Bullish Equity Regime
            # 1.5-2.5%:  Neutral / Stable
            # < 0.5%:    High Risk
            # < 0%:      Crisis Signal
            df["eyg_signal"] = np.where(
                df["eyg"] > EYG_BULLISH,                              "bullish",
                np.where(df["eyg"] >= EYG_NEUTRAL_L,                  "neutral",
                np.where(df["eyg"] >= EYG_WARNING,                    "compressed",
                np.where(df["eyg"] >= EYG_DANGER,                     "high_risk",
                                                                       "crisis")))
            )

            # Rolling EYG percentile vs history — shows where we stand vs past
            df["eyg_pct"] = _rolling_percentile(
                df["eyg"], self.percentile_window
            )

        df = df.ffill().sort_index()

        return df

    # ── Cycle Classification ──────────────────────────────────

    def _classify_row(self, row: pd.Series) -> dict:
        """
        Classifies a day into cycle phase + risk mode.
        Returns dict with classification and scores.
        """
        bull_score = 0
        bear_score = 0
        signals    = {}

        # ── Copper/Gold: growth signal ────────────────────────
        cg_z = row.get("copper_gold_zscore")
        cg_roc = row.get("copper_roc_60")
        if pd.notna(cg_z):
            if cg_z > 0.3:  # relaxed threshold: from 13% to ~25% of bullish days
                bull_score += 2
                signals["copper_gold"] = SignalReading("Copper/Gold", row.get("copper_gold"), cg_z, row.get("copper_gold_pct"), "bullish", "Growth expectations rising")
            elif cg_z < -0.5:
                bear_score += 2
                signals["copper_gold"] = SignalReading("Copper/Gold", row.get("copper_gold"), cg_z, row.get("copper_gold_pct"), "bearish", "Growth expectations falling")
            else:
                signals["copper_gold"] = SignalReading("Copper/Gold", row.get("copper_gold"), cg_z, row.get("copper_gold_pct"), "neutral", "Neutral growth signal")

        # ── Yield Curve ───────────────────────────────────────
        curve = row.get("yield_curve")
        inverted = row.get("yield_inverted", 0)
        if pd.notna(curve):
            if inverted:
                bear_score += 2
                signals["yield_curve"] = SignalReading("Yield Curve (10Y-3M)", curve, row.get("yield_curve_zscore"), row.get("yield_curve_pct"), "bearish", f"INVERTED ({curve:.2f}%) - recession risk")
            elif curve > STEEP_THRESHOLD:
                bull_score += 1
                signals["yield_curve"] = SignalReading("Yield Curve (10Y-3M)", curve, row.get("yield_curve_zscore"), row.get("yield_curve_pct"), "bullish", f"Steep ({curve:.2f}%) - early expansion")
            else:
                signals["yield_curve"] = SignalReading("Yield Curve (10Y-3M)", curve, row.get("yield_curve_zscore"), row.get("yield_curve_pct"), "neutral", f"Flat ({curve:.2f}%) - late cycle")

        # ── Credit Spreads ────────────────────────────────────
        credit_z = row.get("credit_spread_zscore")
        credit_roc = row.get("credit_roc_20")
        if pd.notna(credit_z):
            if credit_z > 0.5:
                bull_score += 1   # HYG/LQD high -> spreads tight -> risk-on
                signals["credit"] = SignalReading("Credit Spreads", row.get("credit_spread_ratio"), credit_z, row.get("credit_spread_pct"), "bullish", "Tight spreads - risk appetite high")
            elif credit_z < -0.5:
                bear_score += 2
                signals["credit"] = SignalReading("Credit Spreads", row.get("credit_spread_ratio"), credit_z, row.get("credit_spread_pct"), "bearish", "Wide spreads - stress in system")
            else:
                signals["credit"] = SignalReading("Credit Spreads", row.get("credit_spread_ratio"), credit_z, row.get("credit_spread_pct"), "neutral", "Normal credit conditions")

        # ── VIX ───────────────────────────────────────────────
        vix = row.get("vix")
        if pd.notna(vix):
            if vix < VIX_HIGH:  # < 20: below elevated threshold = constructive
                bull_score += 1
                signals["vix"] = SignalReading("VIX", vix, row.get("vix_zscore"), row.get("vix_pct"), "bullish", f"Low fear ({vix:.1f}) - constructive")
            elif vix > VIX_STRESS:
                bear_score += 2
                signals["vix"] = SignalReading("VIX", vix, row.get("vix_zscore"), row.get("vix_pct"), "bearish", f"High stress ({vix:.1f}) - fear elevated")
            elif vix > VIX_HIGH:
                bear_score += 1
                signals["vix"] = SignalReading("VIX", vix, row.get("vix_zscore"), row.get("vix_pct"), "bearish", f"Elevated ({vix:.1f}) - caution")
            else:
                signals["vix"] = SignalReading("VIX", vix, row.get("vix_zscore"), row.get("vix_pct"), "neutral", f"Normal ({vix:.1f})")

        # ── Gold/Silver: risk appetite ────────────────────────
        gs_z = row.get("gold_silver_zscore")
        gs_pct = row.get("gold_silver_pct")
        if pd.notna(gs_z):
            if gs_z > 1.0:
                bear_score += 1
                signals["gold_silver"] = SignalReading("Gold/Silver", row.get("gold_silver"), gs_z, gs_pct, "bearish", "High ratio - fear/defensive")
            elif gs_z < -0.3:  # relaxed: from 14.6% to ~25% of bullish days
                bull_score += 1
                signals["gold_silver"] = SignalReading("Gold/Silver", row.get("gold_silver"), gs_z, gs_pct, "bullish", "Low ratio - risk appetite / industrial demand")
            else:
                signals["gold_silver"] = SignalReading("Gold/Silver", row.get("gold_silver"), gs_z, gs_pct, "neutral", "Normal ratio")

        # ── Gold/Oil: stagflation / stress ───────────────────
        go_z = row.get("gold_oil_zscore")
        if pd.notna(go_z):
            if go_z > 1.5:
                bear_score += 1
                signals["gold_oil"] = SignalReading("Gold/Oil", row.get("gold_oil"), go_z, row.get("gold_oil_pct"), "bearish", "Very high - demand collapse / deflation scare")
            elif go_z > 0.5:
                signals["gold_oil"] = SignalReading("Gold/Oil", row.get("gold_oil"), go_z, row.get("gold_oil_pct"), "neutral", "Elevated - some stress")
            else:
                signals["gold_oil"] = SignalReading("Gold/Oil", row.get("gold_oil"), go_z, row.get("gold_oil_pct"), "bullish", "Normal - healthy demand")

        # ── DXY ───────────────────────────────────────────────
        dxy_roc = row.get("dxy_roc_20")
        dxy_z   = row.get("dxy_zscore")
        if pd.notna(dxy_roc):
            if dxy_roc > DXY_STRONG:
                signals["dxy"] = SignalReading("DXY", row.get("dxy"), dxy_z, row.get("dxy_pct"), "bearish", f"Strong dollar ({dxy_roc*100:+.1f}%) - pressure on commodities/EM")
            elif dxy_roc < DXY_WEAK:
                signals["dxy"] = SignalReading("DXY", row.get("dxy"), dxy_z, row.get("dxy_pct"), "bullish", f"Weak dollar ({dxy_roc*100:+.1f}%) - tailwind for commodities")
            else:
                signals["dxy"] = SignalReading("DXY", row.get("dxy"), dxy_z, row.get("dxy_pct"), "neutral", f"Stable dollar ({dxy_roc*100:+.1f}%)")

        # ── Real Yield Pressure ───────────────────────────────
        ryp = row.get("real_yield_pressure")
        if pd.notna(ryp):
            if ryp > 0.02:
                bear_score += 1
                signals["real_yields"] = SignalReading("Real Yields (TIP proxy)", ryp, None, None, "bearish", "Rising real yields - growth/gold headwind")
            elif ryp < -0.02:
                bull_score += 1
                signals["real_yields"] = SignalReading("Real Yields (TIP proxy)", ryp, None, None, "bullish", "Falling real yields - growth/gold tailwind")
            else:
                signals["real_yields"] = SignalReading("Real Yields (TIP proxy)", ryp, None, None, "neutral", "Stable real yields")

        # ── SPY vs SMA200 ─────────────────────────────────────
        spy_above = row.get("spy_above_sma200")
        spy_roc   = row.get("spy_roc_60")
        if pd.notna(spy_above):
            if spy_above == 1 and pd.notna(spy_roc) and spy_roc > 0:
                bull_score += 1
                signals["equity_trend"] = SignalReading("Equity Trend (SPY)", spy_roc, None, None, "bullish", "SPY above SMA200, momentum positive")
            elif spy_above == 0:
                bear_score += 1
                signals["equity_trend"] = SignalReading("Equity Trend (SPY)", spy_roc, None, None, "bearish", "SPY below SMA200 - downtrend")
            else:
                signals["equity_trend"] = SignalReading("Equity Trend (SPY)", spy_roc, None, None, "neutral", "SPY above SMA200, momentum fading")

        # ── Earnings Yield Gap ────────────────────────────────
        eyg_val = row.get("eyg")
        if eyg_val is not None and pd.notna(eyg_val):
            eyg_pct = eyg_val * 100  # as %
            if eyg_pct > 1.0:  # > 1%: equity premium vs bonds (from 2.9% to ~35% of days)
                bull_score += 1
                signals["eyg"] = SignalReading(
                    "Earnings Yield Gap", eyg_val,
                    row.get("eyg_zscore"), None,
                    "bullish",
                    f"Equity premium vs bonds ({eyg_pct:+.1f}%)"
                )
            elif eyg_pct < 0.0:
                bear_score += 2
                signals["eyg"] = SignalReading(
                    "Earnings Yield Gap", eyg_val,
                    row.get("eyg_zscore"), None,
                    "bearish",
                    f"Crisis Signal — bonds dominating equities ({eyg_pct:+.1f}%)"
                )
            elif eyg_pct < 0.5:
                bear_score += 1
                signals["eyg"] = SignalReading(
                    "Earnings Yield Gap", eyg_val,
                    row.get("eyg_zscore"), None,
                    "bearish",
                    f"High Risk — bonds more attractive ({eyg_pct:+.1f}%)"
                )
            else:
                signals["eyg"] = SignalReading(
                    "Earnings Yield Gap", eyg_val,
                    row.get("eyg_zscore"), None,
                    "neutral",
                    f"Neutral/Compressed ({eyg_pct:+.1f}%)"
                )

        # ── Composite Classification ──────────────────────────
        total = bull_score + bear_score
        if total == 0:
            risk_mode = "neutral"
        elif bull_score / (total) >= 0.65:
            risk_mode = "risk_on"
        elif bear_score / (total) >= 0.65:
            risk_mode = "risk_off"
        else:
            risk_mode = "neutral"

        # ── Cycle Phase ───────────────────────────────────────
        curve     = row.get("yield_curve")
        cg_z_val  = row.get("copper_gold_zscore")
        credit_z_val = row.get("credit_spread_zscore")

        cycle_phase = _determine_cycle_phase(
            yield_curve         = curve,
            copper_gold_z       = cg_z_val,
            credit_z            = credit_z_val,
            vix                 = row.get("vix"),
            spy_above_200       = row.get("spy_above_sma200"),
            gold_roc            = row.get("gold_roc_60"),
            real_yield_pressure = row.get("real_yield_pressure"),
        )

        # ── Inflation Environment ─────────────────────────────
        inflation_env = _determine_inflation_env(
            gold_roc      = row.get("gold_roc_60"),
            gold_oil_z    = row.get("gold_oil_zscore"),
            tips_roc      = row.get("tips_roc_60"),
            yield_10y_roc = row.get("yield_10y_roc_60"),
            gold_pct      = row.get("gold_pct"),         # gold percentile
            copper_gold_z = row.get("copper_gold_zscore"), # growth signal
        )

        # ── Dollar Trend ──────────────────────────────────────
        dxy_roc_val = row.get("dxy_roc_60")
        if pd.isna(dxy_roc_val):
            dollar_trend = "neutral"
        elif dxy_roc_val > 0.03:
            dollar_trend = "strong"
        elif dxy_roc_val < -0.03:
            dollar_trend = "weak"
        else:
            dollar_trend = "neutral"

        # ── Macro Score (0-10) ────────────────────────────────
        # Scores how favorable the environment is for equities
        # CG=2(z>0.3), YC=1(>1.5%), CR=1(z>0.5), VIX=1(<20),
        # GS=1(z<-0.3), GO=0, RY=1(<-0.02), SPY=1, EYG=1(>1%) → max=9
        max_possible = 9
        macro_score  = round(min(10, (bull_score / max_possible) * 10), 2)

        # ── Sector Rotation ───────────────────────────────────
        favor, avoid, allocation = _get_asset_rotation(
            cycle_phase   = cycle_phase,
            risk_mode     = risk_mode,
            inflation_env = inflation_env,
            dollar_trend  = dollar_trend,
        )

        return {
            "bull_score":    bull_score,
            "bear_score":    bear_score,
            "risk_mode":     risk_mode,
            "cycle_phase":   cycle_phase,
            "inflation_env": inflation_env,
            "dollar_trend":  dollar_trend,
            "macro_score":   macro_score,
            "signals":       signals,
            "favor_sectors": favor,
            "avoid_sectors": avoid,
            "asset_allocation": allocation,
        }

    def _build_history(self) -> pd.DataFrame:
        """
        Builds history DataFrame efficiently.
        Resamples to WEEKLY before classification: 5000 rows -> 250 rows = 20x speedup.
        Cycle phases do not change day-to-day so weekly resolution is sufficient.
        """
        if self._indicators is None or self._indicators.empty:
            return pd.DataFrame()

        # Weekly resample - massive speedup vs daily iteration
        weekly = self._indicators.resample("W").last().dropna(how="all")
        print(f"   Building history: {len(weekly)} weekly points (from {len(self._indicators)} daily)...")

        records = []
        for date, row in weekly.iterrows():
            result = self._classify_row(row)
            records.append({
                "date":          date,
                "cycle_phase":   result["cycle_phase"],
                "risk_mode":     result["risk_mode"],
                "inflation_env": result["inflation_env"],
                "dollar_trend":  result["dollar_trend"],
                "macro_score":   result["macro_score"],
                "bull_score":    result["bull_score"],
                "bear_score":    result["bear_score"],
                "copper_gold":   row.get("copper_gold"),
                "gold_silver":   row.get("gold_silver"),
                "gold_oil":      row.get("gold_oil"),
                "yield_curve":   row.get("yield_curve"),
                "vix":           row.get("vix"),
                "credit_ratio":  row.get("credit_spread_ratio"),
                "dxy_roc_20":    row.get("dxy_roc_20"),
            })

        weekly_df = pd.DataFrame(records).set_index("date")

        # ── Phase Smoothing (iterative) ───────────────────────
        # If a phase lasts less than MIN_PHASE_WEEKS weeks,
        # it is replaced by the previous phase.
        # Iterative: runs until nothing changes (converges in 2-3 passes).
        MIN_PHASE_WEEKS = 8
        MAX_PASSES      = 10  # safety limit

        phases = weekly_df["cycle_phase"].tolist()

        for _pass in range(MAX_PASSES):
            smoothed = phases.copy()
            changed  = False

            i = 0
            while i < len(phases):
                current = phases[i]
                j = i
                while j < len(phases) and phases[j] == current:
                    j += 1
                duration = j - i

                if duration < MIN_PHASE_WEEKS and i > 0:
                    prev_phase = smoothed[i - 1]
                    for k in range(i, j):
                        smoothed[k] = prev_phase
                    changed = True

                i = j

            phases = smoothed
            if not changed:
                break  # converged

        weekly_df["cycle_phase"] = phases

        # Forward-fill back to daily index for full compatibility
        daily_idx = self._indicators.index
        return weekly_df.reindex(daily_idx, method="ffill")

    # ── Public API ────────────────────────────────────────────


    def get_snapshot(
        self,
        date: Optional[pd.Timestamp] = None,
    ) -> MacroSnapshot:
        """
        Returns MacroSnapshot for a given date.
        If date=None -> last available day (today).
        """
        self._check_loaded()

        if date is None:
            date = self._indicators.index[-1]

        # Find the nearest available day
        available = self._indicators.index[self._indicators.index <= date]
        if available.empty:
            raise ValueError(f"No data available before {date}")
        actual_date = available[-1]
        row = self._indicators.loc[actual_date]

        result = self._classify_row(row)

        return MacroSnapshot(
            date          = actual_date,
            cycle_phase   = result["cycle_phase"],
            risk_mode     = result["risk_mode"],
            inflation_env = result["inflation_env"],
            dollar_trend  = result["dollar_trend"],
            macro_score   = result["macro_score"],
            signals       = result["signals"],
            favor_sectors = result["favor_sectors"],
            avoid_sectors = result["avoid_sectors"],
            asset_allocation = result["asset_allocation"],
            copper_gold   = row.get("copper_gold"),
            gold_silver   = row.get("gold_silver"),
            gold_oil      = row.get("gold_oil"),
            yield_curve   = row.get("yield_curve"),
            yield_10y_abs = row.get("yield_10y"),
            credit_spread = row.get("credit_spread_ratio"),
            vix           = row.get("vix"),
            dxy_roc       = row.get("dxy_roc_20"),
            probit_recession_prob = row.get("probit_recession_prob"),
            probit_signal         = row.get("probit_signal"),
            # Real Yields (DFII10)
            real_yield_10y         = row.get("real_yield_10y"),
            real_yield_zscore_100d = row.get("real_yield_zscore_100d"),
            real_yield_pressure    = row.get("real_yield_pressure"),
            # Earnings Yield Gap
            spy_pe         = row.get("spy_pe"),
            earnings_yield = row.get("earnings_yield"),
            eyg            = row.get("eyg"),
            eyg_zscore     = row.get("eyg_zscore"),
            eyg_signal     = row.get("eyg_signal"),
            # ISM Manufacturing
            ism           = row.get("ism"),
            ism_zscore    = row.get("ism_zscore"),
            ism_signal    = row.get("ism_signal"),
            # TED Spread / FRA-OIS
            ted_spread    = row.get("ted_spread"),
            ted_zscore    = row.get("ted_spread_zscore"),
            ted_signal    = row.get("ted_signal"),
            fra_ois_spread = row.get("fra_ois_spread"),
            # VIX Mean Reversion
            vix_lt_mean   = row.get("vix_lt_mean"),
            vix_mr_zscore = row.get("vix_mr_zscore"),
            vix_half_life = row.get("vix_half_life"),
            vix_band_1up  = row.get("vix_band_1up"),
            vix_band_2up  = row.get("vix_band_2up"),
            vix_mr_signal = row.get("vix_mr_signal"),
            # Cu/Gold vs 10Y Yield Divergence
            cg_yield_corr_60           = row.get("cg_yield_corr_60"),
            cg_yield_corr_252          = row.get("cg_yield_corr_252"),
            cg_yield_divergence        = row.get("cg_yield_divergence"),
            cg_yield_divergence_zscore = row.get("cg_yield_divergence_zscore"),
            cg_yield_signal            = _cg_yield_signal(
                div   = row.get("cg_yield_divergence"),
                corr  = row.get("cg_yield_corr_60"),
            ),
        )

    def get_history(self) -> pd.DataFrame:
        """
        Returns the historical classification.
        Useful for backtesting and visualization.
        """
        self._check_loaded()
        return self._history.copy()

    def get_indicators(self) -> pd.DataFrame:
        """
        Raw indicators DataFrame — for advanced use / plotting.
        """
        self._check_loaded()
        return self._indicators.copy()

    def get_phase_stats(self) -> pd.DataFrame:
        """
        Statistics per cycle phase — number of days, % of total.
        """
        self._check_loaded()
        if self._history.empty:
            return pd.DataFrame()

        phases = self._history["cycle_phase"].value_counts()
        total  = len(self._history)
        stats  = []
        for phase, count in phases.items():
            stats.append({
                "phase":   phase,
                "days":    count,
                "pct":     round(count / total * 100, 1),
            })
        return pd.DataFrame(stats).set_index("phase")

    def _check_loaded(self):
        if not self._loaded:
            raise RuntimeError("Call .load() first")


# ─────────────────────────────────────────────────────────────
# CLASSIFICATION HELPERS
# ─────────────────────────────────────────────────────────────

def _align_ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    """Align two series and compute ratio."""
    a2, b2 = a.align(b, join="inner")
    return (a2 / b2.replace(0, np.nan)).dropna()


def _determine_cycle_phase(
    yield_curve:          Optional[float],
    copper_gold_z:        Optional[float],
    credit_z:             Optional[float],
    vix:                  Optional[float],
    spy_above_200:        Optional[float],
    gold_roc:             Optional[float],
    real_yield_pressure:  Optional[float] = None,
) -> str:
    """
    4-phase business cycle classification.

    Early Expansion:  curve steepening, copper/gold rising, credit tight, equities recovering
    Late Expansion:   curve flattening, inflation signals, momentum still positive
    Early Contraction: curve inverted or flat, copper/gold falling, credit widening
    Late Contraction:  credit at extremes, VIX high, equities below SMA200, gold rising
    """
    # Defaults
    cg_z   = copper_gold_z if pd.notna(copper_gold_z) else 0
    cr_z   = credit_z      if pd.notna(credit_z)      else 0
    curve  = yield_curve   if pd.notna(yield_curve)   else 0
    vix_v  = vix           if pd.notna(vix)           else 20
    above  = spy_above_200 if pd.notna(spy_above_200) else 1
    g_roc  = gold_roc      if pd.notna(gold_roc)      else 0

    inverted   = curve < INVERSION_THRESHOLD
    steep      = curve > STEEP_THRESHOLD
    growth_up  = cg_z > 0.3
    growth_dn  = cg_z < -0.3
    credit_ok  = cr_z > -0.3
    credit_bad = cr_z < -0.5
    fear_high  = vix_v > VIX_HIGH
    fear_low   = vix_v < VIX_LOW
    gold_up    = g_roc > 0.05

    # ── Late Contraction ─────────────────────────────────────
    # Type 1 — Credit Crisis (2008, 2020): spreads widen extremely
    credit_crisis = (above == 0) and fear_high and credit_bad
    # Type 2 — Macro/Rate Crisis (2022, 2000): growth collapse without credit crunch
    macro_crisis  = (above == 0) and fear_high and growth_dn
    # Type 3 — Rate Crisis (2022): real yields spike, equities pressured
    ryp = real_yield_pressure if (real_yield_pressure is not None and pd.notna(real_yield_pressure)) else 0.0
    rate_crisis   = (above == 0) and fear_high and (ryp > 0.02)
    if credit_crisis or macro_crisis or rate_crisis:
        return "late_contraction"

    # ── Early Contraction ────────────────────────────────────
    # Characteristics: curve inverted or flat, copper/gold falling, credit starting to widen
    if (inverted or growth_dn) and not (above == 1 and fear_low):
        return "early_contraction"

    # ── Late Expansion ───────────────────────────────────────
    # Characteristics: curve flat, still risk-on, but growth momentum fading
    if above == 1 and not steep and not growth_up and credit_ok:
        return "late_expansion"

    # ── Early Expansion ──────────────────────────────────────
    # Characteristics: curve steepening, copper/gold rising, credit tight, equities bullish
    if (steep or growth_up) and above == 1 and credit_ok:
        return "early_expansion"

    return "late_expansion"   # default — most common phase


def _determine_inflation_env(
    gold_roc:      Optional[float],
    gold_oil_z:    Optional[float],
    tips_roc:      Optional[float],
    yield_10y_roc: Optional[float],
    gold_pct:      Optional[float] = None,   # gold percentile (10Y)
    copper_gold_z: Optional[float] = None,   # Cu/Gold z-score
) -> str:
    """
    Classifies the inflation environment.
    Uses market-based signals (Gold, Oil, TIPS, yields).

    Improvement vs initial version:
    Adds gold_pct + copper_gold_z to capture
    stagflation regimes where gold is at historical highs
    but 60d ROC is flat (momentum vs level problem).
    """
    g_roc  = gold_roc      if pd.notna(gold_roc)      else 0
    go_z   = gold_oil_z    if pd.notna(gold_oil_z)    else 0
    t_roc  = tips_roc      if pd.notna(tips_roc)      else 0
    y_roc  = yield_10y_roc if pd.notna(yield_10y_roc) else 0
    g_pct  = gold_pct      if (gold_pct is not None and pd.notna(gold_pct)) else 50
    cg_z   = copper_gold_z if (copper_gold_z is not None and pd.notna(copper_gold_z)) else 0

    # ── Stagflation (4 simultaneous conditions) ──────────────
    # Uses gold_oil_z instead of TIPS ROC as inflation proxy.
    # Reason: TIPS are affected by rate-cut expectations.
    # Exclusions:
    #   2011 Euro Crisis: y_roc=-0.10 < -0.08 → flight to safety
    #   2020 COVID:       go_z=2.5 > 1.0      → oil collapse
    #   2024 Gold Rally:  cg_z=-0.2 > -0.5    → growth neutral
    #   QE 2009-2011:     cg_z > -0.5          → growth weak but recovering
    stag = (
        g_pct > 88        # gold at historical highs (10Y percentile)
        and cg_z < -0.5   # growth falling strongly
        and go_z < 1.0    # oil has not collapsed vs gold
        and y_roc > -0.08 # yields not dropping sharply
    )
    if stag:
        return "stagflation"

    # ── Deflationary ─────────────────────────────────────────
    if go_z > 2.0 and y_roc < 0:
        return "falling"

    # ── High inflation ────────────────────────────────────────
    if g_roc > 0.08 and y_roc > 0.01 and t_roc > 0:
        return "high"

    # ── Rising — Path A (classic reflationary) ────────────────
    if g_roc > 0.05 and y_roc > 0 and go_z < 1.0:
        return "rising"

    # ── Rising — Path B (2021-style: inflation without gold momentum) ──
    if y_roc > 0.06 and go_z < 0:
        return "rising"

    # ── Low inflation ─────────────────────────────────────────
    if g_roc < 0 and y_roc <= 0:
        return "low"

    return "rising" if (g_roc > 0 and y_roc > 0) else "low"


def _get_asset_rotation(
    cycle_phase:   str,
    risk_mode:     str,
    inflation_env: str,
    dollar_trend:  str,
) -> tuple[list, list, dict]:
    """
    Returns (favor_sectors, avoid_sectors, asset_allocation)
    based on cycle phase + macro environment.

    Based on the classic sector rotation model.
    """

    ROTATION = {
        "early_expansion": {
            "favor":  ["Financials", "Industrials", "Consumer Cyclical", "Technology"],
            "avoid":  ["Utilities", "Consumer Staples", "Real Estate"],
            "alloc":  {"equities": 0.80, "bonds": 0.10, "gold": 0.05, "cash": 0.05},
        },
        "late_expansion": {
            "favor":  ["Energy", "Materials", "Industrials", "Financials"],
            "avoid":  ["Technology", "Real Estate", "Utilities"],
            "alloc":  {"equities": 0.70, "bonds": 0.10, "gold": 0.10, "cash": 0.10},
        },
        "early_contraction": {
            "favor":  ["Healthcare", "Consumer Staples", "Utilities", "Gold"],
            "avoid":  ["Financials", "Industrials", "Consumer Cyclical"],
            "alloc":  {"equities": 0.45, "bonds": 0.30, "gold": 0.15, "cash": 0.10},
        },
        "late_contraction": {
            "favor":  ["Consumer Staples", "Utilities", "Gold", "Bonds"],
            "avoid":  ["Energy", "Materials", "Technology"],
            "alloc":  {"equities": 0.25, "bonds": 0.40, "gold": 0.20, "cash": 0.15},
        },
    }

    base = ROTATION.get(cycle_phase, ROTATION["late_expansion"])
    favor  = list(base["favor"])
    avoid  = list(base["avoid"])
    alloc  = dict(base["alloc"])

    # Inflation overlay
    if inflation_env in ("rising", "high", "stagflation"):
        if "Energy" not in favor:
            favor.append("Energy")
        if "Materials" not in favor:
            favor.append("Materials")
        # Increase gold allocation in inflationary environments
        alloc["gold"]     = min(0.20, alloc["gold"] + 0.05)
        alloc["equities"] = max(0.20, alloc["equities"] - 0.05)

    # Dollar overlay
    if dollar_trend == "strong":
        # Strong dollar -> pressure on commodities / EM
        if "Materials" in favor:
            favor.remove("Materials")
        if "Energy" in favor:
            favor.remove("Energy")
    elif dollar_trend == "weak":
        # Weak dollar -> tailwind for Materials, Energy, Gold
        if "Materials" not in favor:
            favor.append("Materials")

    # Risk mode overlay
    if risk_mode == "risk_off":
        alloc["equities"] = max(0.20, alloc["equities"] - 0.10)
        alloc["cash"]     = min(0.30, alloc["cash"]     + 0.05)
        alloc["gold"]     = min(0.25, alloc["gold"]     + 0.05)
    elif risk_mode == "risk_on":
        alloc["equities"] = min(0.90, alloc["equities"] + 0.05)
        alloc["cash"]     = max(0.00, alloc["cash"]     - 0.05)

    return favor, avoid, alloc