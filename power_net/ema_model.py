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
from . import fragility as frag
from .paths import load_config

# One scenario per hazard. The flood and coastal triplets
# (baseline/absprot/noprot) existed only to vary flood protection three ways
# INSIDE the model; protection is now a post-hoc factor applied to the stored
# event table (scripts/postprocess_factors.py), so a single run per hazard
# serves every protection assumption. `net_windstorm_absprot` is gone for the
# same reason - and because it never did anything: it sampled
# `protection_abs_rp`, which ens_model only ever read for river and coastal, so
# it was numerically identical to `net_windstorm`.
SCENARIOS = [
    "net_flood", "net_coastal", "net_earthquake", "net_windstorm",
]

DEFAULT_SCENARIOS = list(SCENARIOS)

SCENARIO_HAZARD = {
    "net_flood": "river",
    "net_coastal": "coastal",
    "net_earthquake": "earthquake",
    "net_windstorm": "windstorm",
}

OUTCOMES = [
    "EAENS_MWh", "EAENS_GWh", "peak_unserved_MW_RP100",
    "n_failed_RP100", "n_solves", "n_footprints", "structural_floor_MW",
    "load_shape_hours", "outage_hours",
    # Both were computed by compute_ens and silently discarded: EMA binds
    # outcomes by name, so a key absent from this list never reaches disk.
    # n_opf_fallback is the ONLY signal that a result came from the connectivity
    # model after an LP failure rather than from the DC OPF; n_unbasined is the
    # number of the study country's components that carry no HydroBASINS id and
    # therefore sit in the residual footprint.
    "n_opf_fallback", "n_unbasined",
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
    """One curve-choice factor per component kind that HAS curves for this hazard.

    A kind absent from the hazard's group table gets no parameter at all, not a
    placeholder: under flood only `bus` has published fragility (F2.4 / F2.5),
    so `curve_line` simply does not exist for flood scenarios and `_p_fail`
    returns zeros for lines.
    """
    groups = frag.GROUPS_FOR_HAZARD[hazard]
    out = []
    for kind, ids in sorted(groups.items()):
        if len(ids) > 1:
            out.append(CategoricalParameter(f"curve_{kind}", list(ids)))
        else:
            out.append(Constant(f"curve_{kind}", ids[0]))
    return out


def _shared_params() -> list:
    """Factors every scenario carries, regardless of hazard.

    `event_footprint`, `network_model` and `slack_placement` were all sampled
    here and are now constants or gone - see build_model for why.
    """
    return [
        RealParameter("failure_correlation", 0.0, 0.8),
        RealParameter("gen_availability", 0.5, 1.0),
        IntegerParameter("failure_seed", 0, 999),
        # Where demand sits, as a continuous blend: 0 = pure NUTS3 population,
        # 1 = pure NUTS3 GDP. Measured spread at winter peak with thermal limits
        # on was 3,411 MW (pop+GDP) to 5,267 MW (GDP-only) of shed, larger than
        # several factors already here, and pop-only shed zero at every hour
        # tested while GDP-only was the only mode with material shed.
        RealParameter("nuts3_gdp_weight", 0.0, 1.0),
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
        # flood_mapping / flood_frag_median / flood_frag_beta / flood_threshold
        # are gone: they parameterised an INVENTED depth -> failure mapping that
        # existed only because V1.1.0 had no flood fragility for power. V3.0.0
        # has F2.4/F2.5 for substations, which `_fragility_params` now supplies
        # as `curve_bus`, and nothing for lines, which are therefore immune.
        #
        # Protection is gone too: it is applied post-hoc to the stored event
        # table, in the PRESENT-DAY frame. Applying it here against the
        # climate-shifted return periods made warming appear to strengthen
        # defences - at a 200-year standard flood EAENS fell to exactly zero
        # from 2.0C upward.
        uncertainties.append(RealParameter("depth_scale", 0.9, 1.1))
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

    constants += [
        # Fixed to HydroBASINS level 07 for every hazard, matching the partner
        # analyses. FOOTPRINT_LEVELS and _footprint_labels keep the other levels
        # so a sensitivity run is a config change, not a code change.
        Constant("event_footprint", netcfg.get("event_footprint", "basin_lev07")),
        # DC OPF always. Sampling {connectivity, dc_opf} meant half the draws
        # never ran an OPF - and connectivity cannot see capacity-constrained
        # shedding, which is the whole reason the OPF is here now that thermal
        # limits are enforced. Mixing them also made every aggregate a blend of
        # two different models.
        Constant("network_model", "dc_opf"),
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
