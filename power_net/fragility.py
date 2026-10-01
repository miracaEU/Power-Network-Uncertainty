"""Fragility: hazard intensity -> probability that a grid component fails.

The direct-damage study asks "what fraction of this asset's value was lost?".
A network model asks a different question - "is this component in or out?" - so
damage ratios are replaced by failure probabilities wherever real fragility
curves exist, and by an explicit, sampled mapping where they do not.

What the MIRACA Table D2 workbook actually contains (checked, not assumed):

  W_Frag_V10m   164 wind fragility curves, incl. transmission towers, power
                poles and overhead lines. Axis is 10 m sustained wind; the
                hazard rasters are 3-second gusts, so the axis is scaled by
                GUST_3SEC_FACTOR (the workbook's own "_3sec calculation").
  E_Frag_PGA    162 earthquake fragility curves, incl. E2.x substations,
                E3.x towers, E4.x poles, E6.x lines/cables. Some are tabulated
                exceedance probabilities, some parametric (median/beta).
  F_Frag_Depth  Flood fragility on a depth axis. In V1.1.0 this sheet did not
                exist and the only flood curves for power were damage ratios, so
                the study converted them to failure probabilities under a
                sampled, invented mapping. **V3.0.0 adds F2.4 and F2.5, real
                substation flood fragility curves**, and the invention is gone.

What is still missing, and is now stated rather than papered over: V3.0.0 has NO
flood fragility for lines, cables, towers or poles - only damage ratios. Rather
than invent a depth -> failure mapping for them, **lines are flood-immune**.
That is defensible for a high-voltage tower under slow-onset flooding, and it is
a scope statement, not an oversight. Flood and coastal ENS therefore arise from
substation failure alone.

FLOOD_MAPPINGS and load_flood_damage_curves are retained only so the superseded
construction can be reproduced for comparison; nothing in the live model path
uses them.

Damage states are ordinal; which one counts as "failed" is itself a modelling
choice (failure_state), not a constant.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

# Workbook "_3sec calculation": 10 m sustained wind -> 3-second gust.
GUST_3SEC_FACTOR = 1.261364

# Ordinal damage states, weakest first. Wind sheets use Severe as their top
# state; earthquake sheets use Extensive/Complete.
STATE_ORDER = ["slight", "moderate", "extensive", "severe", "complete"]

FAILURE_STATES = ["moderate", "extensive", "complete"]

FLOOD_MAPPINGS = ["lognormal", "threshold_on_ratio", "ratio_as_prob"]

# The two sheets do not share a vocabulary, and most of the wind table does not
# use damage states at all: 133 of its 164 curves carry a single "Collapse"
# column, which IS the failure probability this model wants. Mapping those to
# the top of the ladder is what makes them usable - reading the sheet with a
# HAZUS-only vocabulary silently drops 90% of the wind curves.
STATE_ALIASES = {
    # Direct failure probabilities - the top of the ladder.
    "collapse": "complete",
    "probability failure": "complete",
    "failure probability": "complete",
    "probability of failure": "complete",
    "probability of damage": "complete",
    "physical damage probability": "complete",
    "sufficient damage": "complete",
    "structural failure": "complete",
    # Alternative ordinal vocabularies.
    "minor": "slight",
    "low": "slight",
    "little or no repair": "slight",
    "medium": "moderate",
    "partial": "moderate",
    "major": "extensive",
    "extensive/complete": "extensive",
}

# HAZUS-style prose descriptions used instead of a state name on some
# earthquake curves; matched on a distinguishing fragment.
STATE_PHRASES = [
    ("reasonable safety margin against collapse", "extensive"),
    ("no margin against collapse", "complete"),
]

# Columns that are not failure probabilities at all - they report a physical
# response quantity. Consuming them as P(fail) would be a silent unit error.
STATE_SKIP = {"residual displacement", "yielding of tower"}


def _canonical_state(label: str) -> str:
    """Normalise a sheet's damage-state label onto STATE_ORDER.

    Returns "" for anything that is not a failure probability, so the caller
    skips the column rather than misreading it.
    """
    s = str(label).strip().lower().rstrip(".")
    if not s or s == "nan" or s in STATE_SKIP:
        return ""
    if s in STATE_ALIASES:
        return STATE_ALIASES[s]
    for st in STATE_ORDER:
        if s.startswith(st):
            return st
    for fragment, st in STATE_PHRASES:
        if fragment in s:
            return st
    return ""


def load_fragility_sheet(path, sheet: str) -> dict[str, dict[str, tuple]]:
    """Parse one fragility sheet into {curve_id: {state: (x, P_exceed)}}.

    Layout (shared by W_Frag_V10m and E_Frag_PGA): a two-row header whose top
    level is the curve ID; row 2 of the body holds the damage-state labels and
    the numeric axis/data start at row 3. Parametric curves store "median" and
    "beta" in labelled cells instead of a tabulated column.
    """
    raw = pd.read_excel(path, sheet_name=sheet, header=[0, 1])
    body = raw.iloc[3:]
    axis = pd.to_numeric(body.iloc[:, 0], errors="coerce")
    valid = ~axis.isna()
    x = axis[valid].to_numpy(np.float64)
    state_row = raw.iloc[2]

    out: dict[str, dict[str, tuple]] = {}
    for col in raw.columns[1:]:
        cid = str(col[0])
        if cid.startswith("Unnamed") or cid == "nan":
            continue
        state = _canonical_state(state_row[col])
        if state not in STATE_ORDER:
            continue
        series = raw[col]
        median = beta = None
        for i in range(1, len(series)):
            prev = str(series.iloc[i - 1]).lower()
            cell = series.iloc[i]
            if pd.isna(cell):
                continue
            try:
                val = float(str(cell).replace(",", "."))
            except ValueError:
                continue
            if "median" in prev and median is None:
                median = val
            elif "beta" in prev and beta is None:
                beta = val
        if median is not None and beta is not None and median > 0 and beta > 0:
            xs = x if len(x) else np.arange(0.0, 3.35, 0.05)
            y = norm.cdf((np.log(np.maximum(xs, 1e-9)) - np.log(median)) / beta)
        else:
            ys = pd.to_numeric(body[col], errors="coerce")[valid].ffill().fillna(0.0)
            y = np.clip(ys.to_numpy(np.float64), 0.0, 1.0)
            xs = x
        if len(xs) == 0 or len(xs) != len(y):
            continue
        out.setdefault(cid, {})[state] = (xs, y)
    return out


def failure_curve(states: dict[str, tuple], failure_state: str) -> tuple:
    """P(fail) = P(exceed failure_state), falling back to the nearest state.

    Not every curve is tabulated for every state (wind sheets stop at Severe;
    some earthquake curves omit Complete). Rather than drop those curves or
    silently return zeros, fall back to the most severe state at or below the
    requested one - and if none exists, the least severe available.
    """
    want = STATE_ORDER.index(failure_state)
    below = [s for s in states if STATE_ORDER.index(s) <= want]
    if below:
        pick = max(below, key=lambda s: STATE_ORDER.index(s))
    else:
        pick = min(states, key=lambda s: STATE_ORDER.index(s))
    return states[pick]


def p_fail_from_curve(intensity: np.ndarray, curve: tuple,
                      scale: float = 1.0) -> np.ndarray:
    """Interpolate a fragility curve, with the axis optionally rescaled."""
    x, y = curve
    return np.clip(
        np.interp(intensity, np.asarray(x) * scale, y, left=0.0, right=float(y[-1])),
        0.0,
        1.0,
    )


def flood_p_fail(depth: np.ndarray, mapping: str, *,
                 damage_ratio: np.ndarray | None,
                 median: float, beta: float, threshold: float) -> np.ndarray:
    """Depth -> P(fail) under one of the three readings of the flood gap.

    lognormal            A genuine fragility: P(fail) = Phi((ln d - ln mu)/beta).
                         Treats the missing curve as a parameter to be sampled.
    threshold_on_ratio   The direct-damage depth-damage curve is believed, and a
                         component fails once its damage ratio crosses a
                         threshold. Deterministic given the ratio.
    ratio_as_prob        The damage ratio is read directly as a failure
                         probability - the implicit assumption in much of the
                         literature, stated here so it can be tested.
    """
    if mapping == "lognormal":
        with np.errstate(divide="ignore", invalid="ignore"):
            z = (np.log(np.maximum(depth, 1e-9)) - np.log(median)) / beta
        return np.where(depth > 0, np.clip(norm.cdf(z), 0.0, 1.0), 0.0)
    if damage_ratio is None:
        raise ValueError(f"mapping {mapping} needs a damage ratio")
    if mapping == "threshold_on_ratio":
        return (damage_ratio >= threshold).astype(np.float64)
    if mapping == "ratio_as_prob":
        return np.clip(damage_ratio, 0.0, 1.0)
    raise ValueError(f"unknown flood mapping {mapping}")


def correlated_uniforms(n: int, rho: float, seed: int) -> np.ndarray:
    """Equicorrelated U(0,1) draws for one event footprint.

    A Gaussian copula with a single common factor: z_i = sqrt(rho) * z0 +
    sqrt(1-rho) * e_i. rho = 0 gives independent component failures, rho -> 1
    gives an all-or-nothing footprint.

    These are COMMON RANDOM NUMBERS: the draw depends only on (seed, n, rho),
    never on the factor values being varied, so the model is deterministic
    given (theta, seed). That is what keeps Sobol indices meaningful without a
    nested Monte Carlo - and it is why failure_seed is itself a factor, so
    ST(failure_seed) measures how much of the variance is realisation noise.
    """
    rng = np.random.default_rng(seed)
    rho = float(np.clip(rho, 0.0, 0.999))
    z0 = rng.standard_normal()
    e = rng.standard_normal(n)
    z = np.sqrt(rho) * z0 + np.sqrt(1.0 - rho) * e
    return norm.cdf(z)


# ---------------------------------------------------------------------------
# Which curves apply to which grid component
# ---------------------------------------------------------------------------
#
# Taken from the workbook's own infrastructure descriptions (level-1 header of
# each sheet), NOT inherited from the direct-damage study's assignments. That
# matters: the direct study assigns power towers the W3.5-W3.14 curves, which
# exist only in the VULNERABILITY table - they have no fragility counterpart -
# and it assigns substations the identically-zero W7.2 placeholder. Reusing
# either would have made wind failure impossible by construction.
#
# For lines the group deliberately spans two mechanisms. A transmission line
# fails under wind either through its conductors (overhead-line curves) or,
# far more often in practice, through tower collapse (power-tower curves,
# W3.51/52/53 being a standard / hardened / no-design triple). Practitioners
# use both, so the categorical spans both: that spread IS the epistemic
# uncertainty this study is trying to measure, not something to average away.

WIND_FRAGILITY = {
    # "Substation" in W_Frag_V10m: 400 kV double-switch, varying design standard.
    "bus": ["W2.1", "W2.2", "W2.3", "W2.4", "W2.5", "W2.6", "W2.7"],
    # "Power (minor) line" / "Overhead line(s)" + "Power tower".
    "line": ["W6.1", "W6.2", "W6.3",
             "W3.15", "W3.44", "W3.45", "W3.51", "W3.52", "W3.53"],
}

EQ_FRAGILITY = {
    # E2.x substations (anchored/unanchored, low/medium/high voltage).
    "bus": ["E2.1", "E2.2", "E2.3", "E2.4", "E2.5", "E2.6", "E2.7", "E2.8", "E2.9"],
    # E6.x distribution/transmission circuits + E3.x towers.
    "line": ["E6.1", "E6.2", "E6.3", "E6.4", "E3.1", "E3.2"],
}

# Flood has no power fragility at all, so failure is derived from the
# direct-damage depth-damage curves under a sampled mapping. These are the
# same IDs the direct study uses for the corresponding object types.
FLOOD_CURVES = {
    "bus": ["F2.1", "F2.2", "F2.3"],
    "line": ["F6.1", "F6.2"],
}

# Real flood FRAGILITY, from V3.0.0's F_Frag_Depth sheet. These did not exist in
# V1.1.0, which is why this study used to invent one.
#
#   F2.4  Substation, intense short event (<3 h)   median 0.111 m, beta 0.208
#   F2.5  Substation, prolonged event   (<10 h)    median 0.226 m, beta 0.208
#
# Both are 3-5x MORE fragile than the invented lognormal they replace (which
# defaulted to median 0.6 m, beta 0.4), so adopting them raises substation flood
# failure substantially.
#
# There is NO entry for "line". V3.0.0 has no flood fragility for lines, cables,
# towers or poles - only damage-ratio curves, and converting those to failure
# probabilities is exactly the invention this study no longer makes. Lines are
# therefore flood-immune, which is physically defensible for a high-voltage
# tower under slow-onset flooding but must be stated, not assumed.
# Chosen per hazard rather than sampled, because at the depths that actually
# occur the two curves are indistinguishable: measured across ITA and NLD, 80-82%
# of flood-exposed transmission buses sit above 0.5 m and median depth is
# 1.0-1.4 m, while F2.4 saturates at ~0.30 m and F2.5 at ~0.50 m. Both return
# P(fail) = 1.0 over essentially the whole exposed population - sampling between
# them would spend a Sobol dimension on a constant, as `slack_placement` did.
#
# The split follows event duration, which is what actually distinguishes the two
# curves: river floods are prolonged (F2.5, <10 h), coastal surge is short-lived
# (F2.4, <3 h).
FLOOD_FRAGILITY_RIVER = {"bus": ["F2.5"]}
FLOOD_FRAGILITY_COASTAL = {"bus": ["F2.4"]}

# Both, for the sensitivity run that demonstrates the choice does not matter.
FLOOD_FRAGILITY = {"bus": ["F2.4", "F2.5"]}

SHEET_FOR_HAZARD = {
    "windstorm": "W_Frag_V10m",
    "earthquake": "E_Frag_PGA",
    "river": "F_Frag_Depth",
    "coastal": "F_Frag_Depth",   # no surge-specific curve exists in V3.0.0
}
GROUPS_FOR_HAZARD = {
    "windstorm": WIND_FRAGILITY,
    "earthquake": EQ_FRAGILITY,
    "river": FLOOD_FRAGILITY_RIVER,
    "coastal": FLOOD_FRAGILITY_COASTAL,
}

# The wind sheet's axis is 10 m sustained wind; hazard rasters are 3-sec gusts.
# Flood axes are depth in metres on both sides, so no conversion.
AXIS_SCALE = {"windstorm": GUST_3SEC_FACTOR, "earthquake": 1.0,
              "river": 1.0, "coastal": 1.0}


def load_component_fragility(vulnerability_path, hazard: str,
                             failure_state: str = "extensive") -> dict[str, dict[str, tuple]]:
    """{component kind: {curve_id: (x, P_fail)}} for one hazard.

    Raises if a declared curve is absent from the sheet, rather than quietly
    yielding a smaller factor domain - a shrinking categorical would change the
    Sobol design without anything failing.
    """
    if hazard not in SHEET_FOR_HAZARD:
        raise ValueError(f"no fragility sheet for hazard '{hazard}'")
    table = load_fragility_sheet(vulnerability_path, SHEET_FOR_HAZARD[hazard])
    scale = AXIS_SCALE[hazard]
    out: dict[str, dict[str, tuple]] = {}
    for kind, ids in GROUPS_FOR_HAZARD[hazard].items():
        missing = [c for c in ids if c not in table]
        if missing:
            raise KeyError(
                f"{SHEET_FOR_HAZARD[hazard]}: curves {missing} declared for "
                f"'{kind}' but not present in the sheet"
            )
        out[kind] = {}
        for cid in ids:
            x, y = failure_curve(table[cid], failure_state)
            out[kind][cid] = (np.asarray(x) * scale, y)
    return out


def load_flood_damage_curves(vulnerability_path) -> dict[str, dict[str, tuple]]:
    """Depth -> damage ratio, for the two flood mappings that need a ratio."""
    from miraca_uq.curves import load_flood_curves

    out: dict[str, dict[str, tuple]] = {}
    for kind, ids in FLOOD_CURVES.items():
        curves = load_flood_curves(vulnerability_path, ids)
        depth = curves.index.to_numpy(np.float64)
        out[kind] = {cid: (depth, curves[cid].to_numpy(np.float64)) for cid in ids}
    return out
