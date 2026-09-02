"""Demand: national hourly series, timing archetypes, per-bus disaggregation.

Timing is deliberately NOT a Sobol factor. Every other factor in this study is
epistemic - a modelling choice someone could in principle get right. When an
event happens is aleatory, and mixing the two into one variance decomposition
makes the resulting indices hard to read. Instead the whole sensitivity
analysis is run once per archetype, and the archetypes are compared. That
answers both questions the single-factor version would have blurred: how much
does timing shift EAENS, and does timing change WHICH assumptions dominate.

The extremes are taken from the affected country's own series rather than from
a European aggregate, because a continental minimum is not a minimum anywhere
in particular.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .network import ISO2_TO_ISO3, NO_OPSD_DEMAND, OPSD_DEMAND_ALIAS

DEMAND_COL = "_load_actual_entsoe_transparency"

# Hour-of-day and month windows each archetype searches within, and whether it
# takes the busiest or quietest hour in that window.
ARCHETYPES = {
    "winter_peak":   {"months": (12, 1, 2), "hours": (17, 18, 19, 20), "how": "max"},
    "winter_night":  {"months": (12, 1, 2), "hours": (2, 3, 4, 5), "how": "min"},
    "summer_peak":   {"months": (6, 7, 8), "hours": (13, 14, 15, 16), "how": "max"},
    "summer_night":  {"months": (6, 7, 8), "hours": (2, 3, 4, 5), "how": "min"},
    "annual_max":    {"months": None, "hours": None, "how": "max"},
    "annual_min":    {"months": None, "hours": None, "how": "min"},
}
DEFAULT_ARCHETYPE = "winter_peak"


@lru_cache(maxsize=4)
def _demand_frame(em_dir: str) -> pd.DataFrame:
    """Hourly national demand (MW), one column per ISO2, indexed by UTC time.

    Cached because the OPSD single-index CSV is ~100 MB and every worker would
    otherwise re-read it. Only country-level series are kept: TSO sub-zones
    (DE_50hertz, DK_1, ...) would double-count against their parent.
    """
    path = Path(em_dir) / "time_series_60min_singleindex.csv"
    head = pd.read_csv(path, nrows=1)
    wanted = {OPSD_DEMAND_ALIAS.get(k, k): k for k in ISO2_TO_ISO3}
    cols = [c for c in head.columns
            if c.endswith(DEMAND_COL) and c[: -len(DEMAND_COL)] in wanted]
    ts = pd.read_csv(path, usecols=["utc_timestamp"] + cols,
                     parse_dates=["utc_timestamp"])
    ts = ts.set_index("utc_timestamp")
    ts.columns = [wanted[c[: -len(DEMAND_COL)]] for c in ts.columns]
    return ts


def national_demand_at(em_dir: Path, timestamp: str) -> dict[str, float]:
    """National demand (MW) per ISO2 at one hour."""
    df = _demand_frame(str(em_dir))
    stamp = pd.Timestamp(timestamp)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    if stamp not in df.index:
        raise KeyError(f"timestamp {timestamp} not in the OPSD series")
    row = df.loc[stamp]
    return {c: float(row[c]) for c in df.columns if np.isfinite(row[c])}


def archetype_timestamp(em_dir: Path, iso2: str, archetype: str) -> pd.Timestamp:
    """The hour representing `archetype` in `iso2`'s own demand series."""
    if archetype not in ARCHETYPES:
        raise ValueError(f"unknown archetype '{archetype}'; choose from {list(ARCHETYPES)}")
    df = _demand_frame(str(em_dir))
    if iso2 not in df.columns:
        raise KeyError(
            f"{iso2} has no OPSD demand series "
            f"(known gaps: {', '.join(NO_OPSD_DEMAND)}); it cannot be a study country"
        )
    s = df[iso2].dropna()
    if s.empty:
        raise KeyError(f"{iso2}'s OPSD demand series is entirely missing")
    spec = ARCHETYPES[archetype]
    if spec["months"] is not None:
        s = s[s.index.month.isin(spec["months"]) & s.index.hour.isin(spec["hours"])]
    if s.empty:
        raise ValueError(f"{iso2} has no hours matching archetype '{archetype}'")
    return s.idxmax() if spec["how"] == "max" else s.idxmin()


def bus_load_mw(net, demand: dict[str, float], mode: str = "equal",
                weights: np.ndarray | None = None) -> tuple[np.ndarray, list[str]]:
    """Split national demand across buses. Returns (load per bus, missing ISO2).

    "equal"  splits a country's demand evenly across its buses - what the D3.3
             model does, kept as the default so results stay comparable.
    "weights" uses a caller-supplied per-bus weight (e.g. NUTS3 population or
             GDP), renormalised within each country.

    Countries absent from `demand` are left at ZERO and reported. They are never
    given an invented default: D3.3's `.fillna(500.0)` MW-per-bus fallback is
    the traceable origin of its 245,864 MW artefact.
    """
    zone = net.bus["zone"].fillna("").to_numpy()
    load = np.zeros(len(net.bus))
    missing: list[str] = []
    for iso2 in sorted(set(zone) - {""}):
        mask = zone == iso2
        total = demand.get(iso2)
        if total is None or not np.isfinite(total):
            missing.append(iso2)
            continue
        if mode == "equal" or weights is None:
            load[mask] = total / mask.sum()
        else:
            w = np.asarray(weights, np.float64)[mask]
            s = w.sum()
            load[mask] = total * (w / s if s > 0 else 1.0 / mask.sum())
    return load, missing


def country_series(em_dir: Path, iso2: str) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """One country's hourly demand as a bare array, plus its time index."""
    df = _demand_frame(str(em_dir))
    if iso2 not in df.columns:
        raise KeyError(f"{iso2} has no OPSD demand series")
    s = df[iso2].dropna()
    return s.to_numpy(np.float64), s.index


def load_shape_factor(series: np.ndarray, t0_idx: int, hours: float) -> float:
    """Sum of d(t0+h)/d(t0) over the outage window - the correction to `shed x T`.

    Energy not supplied is the INTEGRAL of unserved power over the outage, not
    the power at the event hour times the duration. Demand moves underneath a
    multi-hour outage, so holding it at its t0 value is wrong by a factor that
    depends on when the event started:

        LUX winter_peak,  T=24 h  ->  0.83  (constant-power OVERstates by ~20%)
        LUX winter_night, T=24 h  ->  4.82  (constant-power UNDERstates ~5x)

    Because the two errors run in opposite directions, the naive form
    exaggerates the peak-vs-night contrast - precisely the comparison the
    timing archetypes exist to make.

    The shed power is scaled by the national profile rather than re-solving the
    OPF each hour: the failed set and hence the topology is fixed over the
    window, and bus demand tracks the national shape, so re-solving 24 times
    would cost 24x for a second-order correction.

    A window running off the end of the series is clamped to what exists; the
    OPSD series spans years, so this only ever bites at the very last hours.
    """
    n = int(round(hours))
    if n <= 0 or t0_idx >= len(series):
        return 0.0
    d0 = series[t0_idx]
    if not np.isfinite(d0) or d0 <= 0:
        return float(n)
    window = series[t0_idx: t0_idx + n]
    window = window[np.isfinite(window)]
    if len(window) == 0:
        return float(n)
    # Clamped windows are rescaled to the full duration so a truncated series
    # does not silently shorten the outage.
    return float(window.sum() / d0) * (n / len(window))


# ---------------------------------------------------------------------------
# Outage duration
# ---------------------------------------------------------------------------
#
# Bracketing range [0.25, 14] days, anchored on the documented European
# transmission record rather than on a fitted distribution:
#
#   lower ~6 h    Nothing plausible restores faster. The Iberian peninsular
#                 collapse with black-start took ~10 h; North Hyde, a destroyed
#                 275 kV transformer, ~13 h for domestic customers.
#   median ~2 d   Carlisle (36 h), Castle Meads (~2 d), Lancaster on mobile
#                 generation (~2 d), Valencia (<72 h) and Turkish transmission
#                 after Kahramanmaras (3-5 d) all cluster here - the "damaged
#                 HV asset, reconfigure, emergency repair, re-serve" regime.
#   upper ~14 d   Kristin's transmission repairs bound restoration for about
#                 two weeks in the worst district; about the most severe
#                 transmission-caused case in the recent European record.
#
# Sampling is expressed as a unit draw mapped by DURATION_SAMPLING, so the
# log-uniform and uniform variants can be run on an IDENTICAL experimental
# design - same LHS/Sobol points, only the mapping differs. That makes the
# comparison paired rather than two independent samples.
DURATION_RANGE_H = (6.0, 336.0)
DURATION_SAMPLING = ["loguniform", "uniform", "fixed"]


def draw_to_hours(u: float, mode: str = "loguniform",
                  fixed_hours: float = 24.0) -> float:
    """Map a unit draw to an outage duration in hours."""
    lo, hi = DURATION_RANGE_H
    u = float(np.clip(u, 0.0, 1.0))
    if mode == "loguniform":
        return float(lo * (hi / lo) ** u)
    if mode == "uniform":
        return float(lo + u * (hi - lo))
    if mode == "fixed":
        return float(fixed_hours)
    raise ValueError(f"unknown duration_sampling '{mode}'; choose from {DURATION_SAMPLING}")


def hourly_scales(series: np.ndarray, t0_idx: int, hours: float,
                  quantise: float = 0.02) -> np.ndarray:
    """d(t0+h)/d(t0) for each hour of the window, rounded for memo reuse.

    Rounding to ~2% bins is what makes hourly re-solving affordable: a 336-hour
    window contains at most ~50 distinct scale values after quantisation, so
    the solve cache is hit for the overwhelming majority of hours instead of
    running one optimisation per hour.
    """
    n = int(round(hours))
    if n <= 0 or t0_idx >= len(series):
        return np.zeros(0)
    d0 = series[t0_idx]
    if not np.isfinite(d0) or d0 <= 0:
        return np.ones(n)
    w = series[t0_idx: t0_idx + n].astype(np.float64)
    if len(w) < n:  # clamped at the end of the series - tile what we have
        w = np.resize(w, n) if len(w) else np.full(n, d0)
    k = np.where(np.isfinite(w), w / d0, 1.0)
    return np.round(k / quantise) * quantise
