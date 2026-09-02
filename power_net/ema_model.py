"""EMA Workbench model definition for the network-level study.

Mirrors miraca_uq.ema_model deliberately - one hazard per scenario, factors
declared as Parameters or Constants, a module-level model function and a
per-process data cache - so the two studies can be analysed by the same
machinery and compared without translation.

Scenarios (one hazard each):

  net_flood_baseline / _absprot / _noprot      river flood
  net_coastal_baseline / _absprot / _noprot    coastal flood
  net_earthquake
  net_windstorm / net_windstorm_absprot

The three protection treatments carry over unchanged from the direct-damage
study: `baseline` scales the FLOPROS design standard by a multiplier,
`absprot` replaces it with a sampled absolute return period (which, unlike a
multiplier, actually varies protection for features FLOPROS marks unprotected),
and `noprot` holds it at FLOPROS.

Factors specific to the network study
-------------------------------------
event_footprint     Which components a single event can hit at once, from one
                    HydroBASINS sub-basin to the whole country coherently to
                    every component independently. Expected top driver: damage
                    is additive so correlation does not bias EAD, but islanding
                    is super-additive, so it can dominate ENS.
failure_seed        The realisation index. Because the model uses common random
                    numbers it is deterministic given (theta, seed), so
                    ST(failure_seed) reads directly as "how much of the variance
                    is realisation noise rather than assumptions".
failure_correlation Gaussian-copula rho within a footprint, 0 (independent) to
                    0.8 (near all-or-nothing).
network_model       connectivity vs DC OPF - model fidelity as a factor rather
                    than a fixed choice.
flood_mapping       The three readings of the missing flood fragility.
curve_bus/_line     Which real fragility curve to believe, per component kind.

Timing is NOT a factor here - see power_net/demand.py for why. It is a
conditioning dimension: the whole analysis is run once per archetype and the
archetypes compared.
"""

from __future__ import annotations

from ema_workbench import (
    CategoricalParameter,
    Constant,
    IntegerParameter,
    Model,
    RealParameter,
    ScalarOutcome,
)

from miraca_uq.risk_model import WARMING_LEVELS

from .ens_model import FOOTPRINT_LEVELS, NETWORK_MODELS, NetData, compute_ens, load_net_data
from .demand import DURATION_RANGE_H
from .fragility import EQ_FRAGILITY, FLOOD_CURVES, FLOOD_MAPPINGS, WIND_FRAGILITY
from .paths import load_config

SCENARIOS = [
    "net_flood_baseline", "net_flood_absprot", "net_flood_noprot",
    "net_coastal_baseline", "net_coastal_absprot", "net_coastal_noprot",
    "net_earthquake",
    "net_windstorm", "net_windstorm_absprot",
]

# The subset the orchestrator runs by default; the rest stay available via
# --scenarios. Mirrors the direct-damage study's choice of the absolute- and
# no-protection treatments plus the two non-flood hazards.
DEFAULT_SCENARIOS = [
    "net_flood_absprot",
    "net_flood_noprot",
    "net_earthquake",
    "net_windstorm",
]

SCENARIO_HAZARD = {
    "net_flood_baseline": "river", "net_flood_absprot": "river",
    "net_flood_noprot": "river",
    "net_coastal_baseline": "coastal", "net_coastal_absprot": "coastal",
    "net_coastal_noprot": "coastal",
    "net_earthquake": "earthquake",
    "net_windstorm": "windstorm", "net_windstorm_absprot": "windstorm",
}

_PROT_TREATMENT = {
    "net_flood_baseline": "scale", "net_flood_absprot": "abs",
    "net_flood_noprot": "fixed",
    "net_coastal_baseline": "scale", "net_coastal_absprot": "abs",
    "net_coastal_noprot": "fixed",
}

OUTCOMES = [
    "EAENS_MWh", "EAENS_GWh", "peak_unserved_MW_RP100",
    "n_failed_RP100", "n_solves", "n_footprints", "structural_floor_MW",
    "load_shape_hours", "outage_hours",
]

_DATA_CACHE: dict[tuple[str, str], NetData] = {}


def _get_data() -> NetData:
    """Per-process cache; keyed on (country, timing archetype).

    Reads the key from config/env rather than a captured argument so the model
    function stays a plain module-level function. MultiprocessingEvaluator has
    to pickle it for spawned workers on Windows, and closures are not picklable.
    """
    cfg = load_config()
    archetype = cfg["network"].get("timing_archetype", "winter_peak")
    key = (cfg["country"], archetype)
    if key not in _DATA_CACHE:
        _DATA_CACHE[key] = load_net_data(cfg, archetype=archetype)
    return _DATA_CACHE[key]


def ens_model(**kwargs) -> dict:
    """The EMA Workbench model function - one for every scenario."""
    return compute_ens(_get_data(), **kwargs)


def scenario_applies(scenario: str, country: str | None = None) -> bool:
    """True iff Stage 1 produced this scenario's hazard for this country."""
    from miraca_uq.curves import applicable_hazards

    return SCENARIO_HAZARD[scenario] in applicable_hazards("power", country)


def applicable_scenarios(country: str | None = None) -> list[str]:
    return [s for s in SCENARIOS if scenario_applies(s, country)]


def _fragility_params(hazard: str) -> list:
    """One curve-choice factor per component kind, for whichever table applies."""
    groups = {"windstorm": WIND_FRAGILITY, "earthquake": EQ_FRAGILITY}.get(
        hazard, FLOOD_CURVES
    )
    out = []
    for kind, ids in sorted(groups.items()):
        if len(ids) > 1:
            out.append(CategoricalParameter(f"curve_{kind}", list(ids)))
        else:
            out.append(Constant(f"curve_{kind}", ids[0]))
    return out


def _shared_params() -> list:
    """Factors every scenario carries, regardless of hazard."""
    return [
        CategoricalParameter("event_footprint", FOOTPRINT_LEVELS),
        CategoricalParameter("network_model", NETWORK_MODELS),
        CategoricalParameter("slack_placement",
                             ["per_country", "per_synchronous_area"]),
        RealParameter("failure_correlation", 0.0, 0.8),
        RealParameter("gen_availability", 0.5, 1.0),
        IntegerParameter("failure_seed", 0, 999),
    ]


def build_model(cfg: dict | None = None) -> Model:
    if cfg is None:
        cfg = load_config()
    scenario = cfg["scenario"]
    if scenario not in SCENARIO_HAZARD:
        raise ValueError(f"Unknown scenario '{scenario}'; choose from {SCENARIOS}")
    hazard = SCENARIO_HAZARD[scenario]
    netcfg = cfg["network"]

    # Outage duration as a unit draw, mapped onto [6 h, 14 d] by
    # `duration_sampling`. Identical draws feed the log-uniform and uniform
    # mappings so the two can be compared on the same design. "fixed" keeps
    # the earlier conditioning-constant behaviour.
    dur_mode = netcfg.get("duration_sampling", "fixed")
    uncertainties: list = list(_shared_params())
    if dur_mode != "fixed":
        uncertainties.append(RealParameter("outage_draw", 0.0, 1.0))
    constants: list = []
    for p in _fragility_params(hazard):
        (uncertainties if isinstance(p, CategoricalParameter) else constants).append(p)

    if hazard in ("river", "coastal"):
        uncertainties += [
            CategoricalParameter("flood_mapping", FLOOD_MAPPINGS),
            RealParameter("flood_frag_median", 0.3, 1.0),
            RealParameter("flood_frag_beta", 0.2, 0.6),
            RealParameter("flood_threshold", 0.1, 0.6),
            RealParameter("depth_scale", 0.9, 1.1),
        ]
        treatment = _PROT_TREATMENT[scenario]
        if treatment == "scale":
            uncertainties.append(RealParameter("protection_scale", 0.0, 2.0))
        elif treatment == "abs":
            uncertainties.append(RealParameter("protection_abs_rp", 5.0, 200.0))
        else:
            constants.append(Constant("protection_scale", 1.0))
        if hazard == "river":
            # The climate RP shift is river-basin-anchor based; coastal
            # sea-level rise is a separate mechanism, not modelled here.
            uncertainties.append(
                CategoricalParameter("warming", list(WARMING_LEVELS.keys()))
            )
    elif hazard == "earthquake":
        uncertainties.append(RealParameter("pga_scale", 0.8, 1.2))
    elif hazard == "windstorm":
        uncertainties.append(RealParameter("gust_scale", 0.9, 1.1))
        if scenario == "net_windstorm_absprot":
            uncertainties.append(RealParameter("protection_abs_rp", 25.0, 200.0))

    constants += [
        Constant("include_river", hazard == "river"),
        Constant("include_coastal", hazard == "coastal"),
        Constant("include_earthquake", hazard == "earthquake"),
        Constant("include_windstorm", hazard == "windstorm"),
        # A conditioning constant, not a factor: ENS is linear in the outage
        # window, so alternative windows are exact post-processing rescales.
        Constant("outage_window_hours",
                 float(netcfg.get("outage_window_hours", 24.0))),
        Constant("duration_sampling", dur_mode),
        # Whether unserved power is re-solved at every hour of the outage or
        # scaled by the demand profile from a single solve at t0. A switch, not
        # a factor: it is a question about model fidelity to be answered by
        # paired runs, not a quantity to integrate over.
        Constant("hourly_resolve", bool(netcfg.get("hourly_resolve", False))),
    ]

    model = Model(f"{cfg['country']}_powergrid_{scenario}", function=ens_model)
    model.uncertainties = uncertainties
    model.constants = constants
    model.outcomes = [ScalarOutcome(n) for n in OUTCOMES]
    return model
