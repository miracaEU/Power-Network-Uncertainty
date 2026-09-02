"""Stage 2: energy not supplied, and its integration to EAENS.

    per event footprint f, per return period r:
        intensity -> fragility -> P(fail) -> correlated CRN threshold -> failed set
        -> screen (nothing failed, or nothing that changes the network -> ENS 0)
        -> memoised solve on the failed-set hash -> unserved MW
        -> ENS_f(r) = unserved_MW * load_shape_factor(t0, window)
    EAENS = trapezoid over p = 1/r, summed across footprints

The integration reuses miraca_uq.risk_model._integrate_ead unchanged, so the
network metric is built by the same quadrature as the direct-damage EAD it will
be compared against. Footprints take the place of features: each is an
independent event source, and at most one event per footprint per year is
assumed (which is what the return-period bookkeeping means).

Two things this module is careful about, both learned from the benchmark:

  * A structural floor. 42 of 76 islands are short of generation with NO hazard
    applied - about 3.3 GW of unserved load that is a property of the network
    data, not of any event. Every outcome is reported both raw and net of that
    floor, because reporting only the raw number would attribute it to hazard.
  * Solve count. `n_solves` is an explicit outcome because it is the quantity
    that decides whether the pan-European Sobol design is affordable at all
    (BENCHMARK.md: at ~0.6 s a solve, S=10 is 44 core-hours and S=1000 is
    4,400). Measuring it is the point of the first runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from miraca_uq.risk_model import WARMING_LEVELS, _integrate_ead, _shift_rps

from . import fragility as frag
from .demand import (
    archetype_timestamp,
    bus_load_mw,
    country_series,
    draw_to_hours,
    hourly_scales,
    load_shape_factor,
    national_demand_at,
)
from .hazard import build_component_hazard, load_segments
from .linkage import BUS, LINE, load_linkage
from .injections import (
    add_shedding_units,
    add_slack,
    assign_generation,
    assign_loads,
    snap_plants_to_buses,
)
from .network import (
    ISO3_TO_ISO2,
    VOLL_EUR_MWH,
    build_base_network,
    bus_coords_3035,
)

# Event footprints, from fully coherent to fully independent. The factor spans
# both extremes on purpose: for additive damage the correlation structure does
# not bias the expectation, but islanding is super-additive, so for ENS it can
# dominate - and no single choice is defensible enough to fix.
FOOTPRINT_LEVELS = ["basin", "basin_coarse", "main_basin", "country", "independent"]

NETWORK_MODELS = ["connectivity", "dc_opf"]

# Warming code -> column of the (n_comp, 3, 4) anchor array.
_WARM_COL = {"15": 0, "20": 1, "30": 2, "40": 3}

HAZARD_FLAGS = {
    "river": "include_river",
    "coastal": "include_coastal",
    "earthquake": "include_earthquake",
    "windstorm": "include_windstorm",
}


@dataclass
class NetData:
    """Everything one worker process needs, loaded once."""

    net: Any
    n_bus: int
    n_line: int
    line_from: np.ndarray
    line_to: np.ndarray
    trafo_hv: np.ndarray
    trafo_lv: np.ndarray
    bus_load: np.ndarray            # MW at the chosen timing archetype
    bus_gen: np.ndarray             # MW nameplate at 100% availability
    shed_idx: np.ndarray            # net.sgen rows that are load-shedding units
    country_bus: np.ndarray         # bool, buses in the study country
    country_line: np.ndarray        # bool, lines touching the study country
    hazards: dict                   # hazard -> component hazard tables
    frag_tables: dict               # hazard -> {kind: {curve_id: (x, p)}}
    flood_curves: dict              # kind -> {curve_id: (depth, ratio)}
    footprints: dict                # level -> {kind: labels per component}
    demand_series: np.ndarray = field(default_factory=lambda: np.zeros(0))
    t0_idx: int = 0
    timestamp: str = ""
    missing_demand: list = field(default_factory=list)
    structural_floor_mw: float = 0.0
    _memo: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def _footprint_labels(cfg: dict, hz_tables: dict, link) -> dict:
    """Component -> event-footprint label, at every nesting level.

    HydroBASINS gives the hydrological levels; `country` makes the whole study
    area one coherent event; `independent` makes every component its own. Label
    -1 means "not assigned to any footprint" (no basin), and such components
    only ever fail under the coherent levels.
    """
    basins = pd.read_parquet(cfg["basin_data_path"],
                             columns=["HYBAS_ID", "MAIN_BAS", "PFAF_ID"])
    hyb_to_main = dict(zip(basins["HYBAS_ID"], basins["MAIN_BAS"]))
    # Pfafstetter codes nest by prefix: dropping the last digit is exactly one
    # level coarser, which is what "Pfafstetter-prefix truncation" means.
    hyb_to_pfaf = dict(zip(basins["HYBAS_ID"], basins["PFAF_ID"] // 10))

    out: dict[str, dict[str, np.ndarray]] = {lv: {} for lv in FOOTPRINT_LEVELS}
    any_hz = next(iter(hz_tables.values()))
    for kind in (BUS, LINE):
        hyb = any_hz[kind]["hybas"]
        n = len(hyb)
        out["basin"][kind] = np.where(hyb > 0, hyb, -1).astype(np.int64)
        out["basin_coarse"][kind] = np.array(
            [hyb_to_pfaf.get(h, -1) if h > 0 else -1 for h in hyb], dtype=np.int64
        )
        out["main_basin"][kind] = np.array(
            [hyb_to_main.get(h, -1) if h > 0 else -1 for h in hyb], dtype=np.int64
        )
        out["country"][kind] = np.zeros(n, dtype=np.int64)
        # Offset the line ids so a bus and a line never share an "independent"
        # footprint label.
        out["independent"][kind] = (
            np.arange(n, dtype=np.int64) + (0 if kind == BUS else 10**7)
        )
    return out


def load_net_data(cfg: dict, archetype: str | None = None) -> NetData:
    """Stage 1b: build the network, attach demand and hazard, once per process."""
    import pandapower as pp

    em = cfg["energy_model_dir"]
    netcfg = cfg["network"]
    iso2 = ISO3_TO_ISO2[cfg["country"]]

    net, geo = build_base_network(em)
    xy = bus_coords_3035(geo)
    link = load_linkage(em)

    archetype = archetype or netcfg.get("timing_archetype", "winter_peak")
    stamp = archetype_timestamp(em, iso2, archetype)
    dem_series, dem_index = country_series(em, iso2)
    t0_idx = int(dem_index.get_loc(stamp))
    demand = national_demand_at(em, str(stamp))
    load, missing = bus_load_mw(net, demand,
                                mode=netcfg.get("load_disaggregation", "equal"))

    plants = pd.read_csv(em / "powerplants.csv")
    bus_of_plant = snap_plants_to_buses(plants, xy)
    gen = np.zeros(len(net.bus))
    np.add.at(gen, bus_of_plant, plants["Capacity"].to_numpy(np.float64))

    # Make the net OPF-ready once, so the `dc_opf` network model can simply
    # toggle in_service rather than rebuild. Costs about a second here and
    # nothing per solve.
    assign_loads(net, load)
    assign_generation(net, plants, xy, availability=1.0)
    shed_idx = add_shedding_units(net, VOLL_EUR_MWH)
    add_slack(net, mode=netcfg.get("slack_placement", "per_synchronous_area"))
    # Thermal limits are OFF by default and this is a real limitation, not an
    # oversight: with the equal-split load disaggregation 12.9% of lines exceed
    # 100% loading before any hazard (BENCHMARK.md), so a limited OPF does not
    # converge even in the base case. Leaving them off keeps the OPF solvable
    # but blinds it to capacity-constrained shedding - the one failure mode a
    # connectivity model also cannot see. Turn on only with a load
    # disaggregation that makes the base case feasible.
    if netcfg.get("thermal_limits", False):
        net.line["max_loading_percent"] = 100.0
        net.trafo["max_loading_percent"] = 100.0

    seg = load_segments(cfg)
    from .hazard import available_hazards
    hazards = {hz: build_component_hazard(cfg, link, hz, seg)
               for hz in available_hazards(cfg)}

    frag_tables = {}
    for hz in hazards:
        if hz in frag.SHEET_FOR_HAZARD:
            frag_tables[hz] = frag.load_component_fragility(
                cfg["vulnerability_path"], hz, "extensive"
            )
    flood_curves = frag.load_flood_damage_curves(cfg["vulnerability_path"])

    line_from = net.line.from_bus.to_numpy(np.int64)
    line_to = net.line.to_bus.to_numpy(np.int64)
    cbus = link.bus_zone == iso2
    cline = cbus[line_from] | cbus[line_to]

    data = NetData(
        net=net,
        n_bus=len(net.bus),
        n_line=len(net.line),
        line_from=line_from,
        line_to=line_to,
        trafo_hv=net.trafo.hv_bus.to_numpy(np.int64),
        trafo_lv=net.trafo.lv_bus.to_numpy(np.int64),
        bus_load=load,
        bus_gen=gen,
        shed_idx=shed_idx,
        country_bus=cbus,
        country_line=cline,
        hazards=hazards,
        frag_tables=frag_tables,
        flood_curves=flood_curves,
        footprints=_footprint_labels(cfg, hazards, link),
        demand_series=dem_series,
        t0_idx=t0_idx,
        timestamp=str(stamp),
        missing_demand=missing,
    )
    # The no-hazard baseline, so every reported number can be stated net of it.
    data.structural_floor_mw = _unserved_connectivity(
        data, np.zeros(data.n_line, bool), np.zeros(data.n_bus, bool), 1.0
    )
    return data


# ---------------------------------------------------------------------------
# Network solves
# ---------------------------------------------------------------------------


def _islands(data: NetData, line_out: np.ndarray, bus_out: np.ndarray) -> np.ndarray:
    """Connected-component label per bus, given failed lines and buses."""
    keep = ~line_out & ~bus_out[data.line_from] & ~bus_out[data.line_to]
    tkeep = ~bus_out[data.trafo_hv] & ~bus_out[data.trafo_lv]
    rows = np.concatenate([data.line_from[keep], data.trafo_hv[tkeep]])
    cols = np.concatenate([data.line_to[keep], data.trafo_lv[tkeep]])
    adj = coo_matrix((np.ones(len(rows)), (rows, cols)),
                     shape=(data.n_bus, data.n_bus)).tocsr()
    _, labels = connected_components(adj, directed=False)
    return labels


def _unserved_connectivity(data: NetData, line_out: np.ndarray,
                           bus_out: np.ndarray, gen_availability: float,
                           load_scale: float = 1.0) -> float:
    """Unserved MW from topology alone: per island, demand minus generation.

    Cruder than an OPF - it cannot see capacity-constrained shedding, where the
    network stays connected but lines bind - but it costs milliseconds, and it
    is the screen that decides whether an OPF is worth running at all.
    """
    labels = _islands(data, line_out, bus_out)
    load = np.where(bus_out, 0.0, data.bus_load) * load_scale
    gen = np.where(bus_out, 0.0, data.bus_gen) * gen_availability
    n = labels.max() + 1
    isl_load = np.bincount(labels, weights=load, minlength=n)
    isl_gen = np.bincount(labels, weights=gen, minlength=n)
    # Load at a de-energised bus is unserved outright; the rest is the
    # generation shortfall in each surviving island.
    shed_at_failed = float(data.bus_load[bus_out].sum()) * load_scale
    return shed_at_failed + float(np.maximum(isl_load - isl_gen, 0.0).sum())


def _unserved_dc_opf(data: NetData, line_out: np.ndarray, bus_out: np.ndarray,
                     gen_availability: float, slack_mode: str,
                     load_scale: float = 1.0) -> float:
    """Unserved MW from a DC OPF with explicit per-bus load shedding.

    pandapower's `rundcopp` reports *infeasible* when demand cannot be met - it
    does not shed - so shedding has to be a decision variable: a dispatchable
    injector at every load bus, capped at that bus's demand and priced at VoLL
    far above any generator, so the optimiser uses it only as a last resort.

    Falls back to the connectivity answer if the OPF does not converge, and
    says so through the returned flag rather than silently reporting zero.
    """
    import pandapower as pp

    from .injections import add_shedding_units, add_slack, assign_generation

    net = data.net
    saved_line = net.line["in_service"].to_numpy().copy()
    saved_bus = net.bus["in_service"].to_numpy().copy()
    saved_genmax = net.gen["max_p_mw"].to_numpy().copy()
    saved_load = net.load["p_mw"].to_numpy().copy()
    saved_shedmax = net.sgen["max_p_mw"].to_numpy().copy()
    try:
        net.line["in_service"] = ~line_out & saved_line
        net.bus["in_service"] = ~bus_out & saved_bus
        net.gen["max_p_mw"] = saved_genmax * gen_availability
        # Demand and the shedding-unit caps move together: a shedding injector
        # is capped at ITS bus's demand, so scaling load without scaling the
        # caps would silently forbid shedding more than the t0 demand.
        net.load["p_mw"] = saved_load * load_scale
        net.sgen["max_p_mw"] = saved_shedmax * load_scale
        pp.rundcopp(net)
        # Only the shedding injectors count as unserved energy; other sgens
        # (if any) are real generation.
        shed = float(net.res_sgen.p_mw.iloc[data.shed_idx].sum())
        return shed
    except Exception:
        return _unserved_connectivity(data, line_out, bus_out, gen_availability,
                                      load_scale)
    finally:
        net.line["in_service"] = saved_line
        net.bus["in_service"] = saved_bus
        net.gen["max_p_mw"] = saved_genmax
        net.load["p_mw"] = saved_load
        net.sgen["max_p_mw"] = saved_shedmax


def _solve(data: NetData, line_out: np.ndarray, bus_out: np.ndarray,
           model: str, gen_availability: float, slack_mode: str,
           load_scale: float = 1.0) -> tuple[float, bool]:
    """Unserved MW, memoised on (failed set, load scale).

    Returns (MW, was_a_fresh_solve). load_scale is part of the key, and
    `hourly_scales` quantises it to ~2% bins so an hourly re-solve over a long
    window still hits the cache for most hours.
    """
    key = (model, round(gen_availability, 4), slack_mode, round(load_scale, 3),
           line_out.tobytes(), bus_out.tobytes())
    if key in data._memo:
        return data._memo[key], False
    if model == "connectivity":
        mw = _unserved_connectivity(data, line_out, bus_out, gen_availability,
                                    load_scale)
    elif model == "dc_opf":
        mw = _unserved_dc_opf(data, line_out, bus_out, gen_availability,
                              slack_mode, load_scale)
    else:
        raise ValueError(f"unknown network_model '{model}'")
    data._memo[key] = mw
    return mw, True


def _event_energy(data: NetData, line_out: np.ndarray, bus_out: np.ndarray,
                  model: str, gen_av: float, slack: str,
                  scales: np.ndarray, hourly: bool) -> tuple[float, int, float]:
    """Unserved ENERGY (MWh) over the outage window.

    Two modes, and the whole point of offering both is that they are not
    obviously equivalent:

    hourly=False  Solve once at the event hour and scale the resulting unserved
                  power by the demand profile. Cheap, and exact only if
                  unserved power were proportional to demand.
    hourly=True   Re-solve at every hour with demand scaled to that hour.

    They differ because unserved power is NOT proportional to demand: in an
    island it is max(0, k*L - G), which is piecewise linear with a kink where
    load crosses generation. Below the kink an island is self-sufficient and
    sheds nothing however demand moves; above it, unserved power rises FASTER
    than demand. So profile scaling can err in either direction, and the size
    of the error is an empirical question - which is what the hourly mode is
    for.

    The structural floor is subtracted at each hour's own scale, not once at
    t0, because the no-hazard shortfall itself moves with demand.
    """
    empty_l = np.zeros(data.n_line, bool)
    empty_b = np.zeros(data.n_bus, bool)
    if not hourly:
        mw, f1 = _solve(data, line_out, bus_out, model, gen_av, slack, 1.0)
        base, f2 = _solve(data, empty_l, empty_b, model, gen_av, slack, 1.0)
        net = max(mw - base, 0.0)
        return net * float(scales.sum()), int(f1) + int(f2), net
    total = 0.0
    solves = 0
    mw0 = 0.0
    for h, k in enumerate(scales):
        mw, f1 = _solve(data, line_out, bus_out, model, gen_av, slack, float(k))
        base, f2 = _solve(data, empty_l, empty_b, model, gen_av, slack, float(k))
        solves += int(f1) + int(f2)
        net = max(mw - base, 0.0)
        total += net
        if h == 0:
            mw0 = net
    return total, solves, mw0


# ---------------------------------------------------------------------------
# Failure probabilities
# ---------------------------------------------------------------------------


def _p_fail(data: NetData, hazard: str, kind: str, intensity: np.ndarray,
            kwargs: dict) -> np.ndarray:
    """(n_comp, n_rp) failure probability for one component kind."""
    if hazard in ("river", "coastal"):
        mapping = kwargs.get("flood_mapping", "lognormal")
        ratio = None
        if mapping != "lognormal":
            cid = kwargs.get(f"curve_{kind}")
            depth, dmg = data.flood_curves[kind][cid]
            ratio = np.interp(intensity, depth, dmg, left=0.0, right=float(dmg[-1]))
        return frag.flood_p_fail(
            intensity, mapping, damage_ratio=ratio,
            median=float(kwargs.get("flood_frag_median", 0.6)),
            beta=float(kwargs.get("flood_frag_beta", 0.4)),
            threshold=float(kwargs.get("flood_threshold", 0.3)),
        )
    cid = kwargs.get(f"curve_{kind}")
    curve = data.frag_tables[hazard][kind][cid]
    return frag.p_fail_from_curve(intensity, curve)


def _scaled_intensity(hazard: str, intensity: np.ndarray, kwargs: dict) -> np.ndarray:
    """Apply the hazard-side error factors reused from the direct-damage study."""
    if hazard in ("river", "coastal"):
        out = intensity * float(kwargs.get("depth_scale", 1.0))
        out = out + float(kwargs.get("depth_offset", 0.0))
        return np.where(intensity > 0, np.maximum(out, 0.0), 0.0)
    if hazard == "earthquake":
        return intensity * float(kwargs.get("pga_scale", 1.0))
    if hazard == "windstorm":
        return intensity * float(kwargs.get("gust_scale", 1.0))
    return intensity


def _protection_rp(base: np.ndarray, kwargs: dict) -> np.ndarray:
    if "protection_abs_rp" in kwargs:
        return np.full_like(base, float(kwargs["protection_abs_rp"]))
    return base * float(kwargs.get("protection_scale", 1.0))


def _shifted_rps(rps: np.ndarray, anchors: np.ndarray, warming: str,
                 n_comp: int) -> np.ndarray:
    """(n_comp, n_rp) return periods after the climate shift."""
    code = WARMING_LEVELS.get(warming)
    base = np.tile(rps, (n_comp, 1))
    if code is None:
        return base
    return _shift_rps(base, anchors[:, :, _WARM_COL[code]])


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------


def compute_ens(data: NetData, **kwargs) -> dict:
    """One model evaluation: factor settings in, EAENS and diagnostics out."""
    hazard = next(hz for hz, flag in HAZARD_FLAGS.items() if kwargs.get(flag))
    if hazard not in data.hazards:
        raise ValueError(
            f"no Stage 1 profile for hazard '{hazard}' in {data.timestamp[:4]} "
            "- it was not preprocessed for this country"
        )
    hz = data.hazards[hazard]
    rps = hz["rps"]
    n_rp = len(rps)
    # Outage duration: either a fixed conditioning constant, or a unit draw
    # mapped onto [6 h, 14 d]. The same draw feeds the log-uniform and uniform
    # mappings, so the two variants can be compared on an identical design.
    dur_mode = kwargs.get("duration_sampling", "fixed")
    window = draw_to_hours(
        float(kwargs.get("outage_draw", 0.5)), dur_mode,
        fixed_hours=float(kwargs.get("outage_window_hours", 24.0)),
    )
    hourly = bool(kwargs.get("hourly_resolve", False))
    # Energy, not power x duration. Demand moves underneath a multi-hour
    # outage, and the naive form is wrong by an archetype-dependent factor
    # (LUX winter_peak 0.83x, LUX winter_night 4.82x at T=24 h) - errors in
    # opposite directions, so it exaggerates the very peak-vs-night contrast
    # the archetypes exist to measure. See demand.load_shape_factor.
    scales = hourly_scales(data.demand_series, data.t0_idx, window)
    lsf = float(scales.sum())
    level = kwargs.get("event_footprint", "basin")
    model = kwargs.get("network_model", "connectivity")
    rho = float(kwargs.get("failure_correlation", 0.0))
    seed = int(kwargs.get("failure_seed", 0))
    gen_av = float(kwargs.get("gen_availability", 1.0))
    slack_mode = kwargs.get("slack_placement", "per_synchronous_area")

    # Per-kind failure probability, with protection and the climate shift.
    pf: dict[str, np.ndarray] = {}
    rp_adj: dict[str, np.ndarray] = {}
    for kind, n_comp in ((BUS, data.n_bus), (LINE, data.n_line)):
        d = hz[kind]
        inten = _scaled_intensity(hazard, d["intensity"], kwargs)
        p = _p_fail(data, hazard, kind, inten, kwargs)
        r = _shifted_rps(rps, d["anchors"], kwargs.get("warming", "current"), n_comp)
        if hazard in ("river", "coastal"):
            prot = _protection_rp(d["prot_rp"], kwargs)
            p = np.where(r <= prot[:, None], 0.0, p)
        # Only the study country's own components can fail: the rest of the
        # continent stays intact so the surrounding network still behaves.
        mask = data.country_bus if kind == BUS else data.country_line
        p = np.where(mask[:, None], p, 0.0)
        pf[kind] = p
        rp_adj[kind] = r

    labels = data.footprints[level]
    # Only footprints that contain a component which can actually fail. Without
    # this, `independent` would enumerate all 15,731 pan-European components as
    # separate footprints - 99.8% of them incapable of failing, because only the
    # study country's components carry a non-zero probability - which both
    # wastes the loop and reports a meaningless n_footprints.
    cand = np.concatenate([labels[BUS][data.country_bus],
                           labels[LINE][data.country_line]])
    fp_ids = np.unique(cand)
    fp_ids = fp_ids[fp_ids >= 0]

    ens = np.zeros((max(len(fp_ids), 1), n_rp))
    rp_fp = np.tile(rps, (max(len(fp_ids), 1), 1))
    n_solves = 0
    n_failed_by_rp = np.zeros(n_rp)
    peak_mw = np.zeros(n_rp)

    for fi, fid in enumerate(fp_ids):
        in_bus = (labels[BUS] == fid) & data.country_bus
        in_line = (labels[LINE] == fid) & data.country_line
        nb, nl = int(in_bus.sum()), int(in_line.sum())
        if nb + nl == 0:
            continue
        # One CRN draw per footprint, shared across every return period so the
        # same components are "weak" throughout - and independent of the factor
        # values, which is what keeps the model deterministic given (theta, seed).
        u = frag.correlated_uniforms(nb + nl, rho, seed + int(fid) % 100_000)
        ub, ul = u[:nb], u[nb:]

        # Footprint-mean shifted RP: the integration grid must be per row, and
        # a footprint's components share a basin, so their shifts agree closely.
        if len(fp_ids):
            rr = []
            if nb:
                rr.append(rp_adj[BUS][in_bus])
            if nl:
                rr.append(rp_adj[LINE][in_line])
            rp_fp[fi] = np.concatenate(rr, axis=0).mean(axis=0)

        for ri in range(n_rp):
            bus_out = np.zeros(data.n_bus, bool)
            line_out = np.zeros(data.n_line, bool)
            if nb:
                bus_out[np.flatnonzero(in_bus)] = ub < pf[BUS][in_bus, ri]
            if nl:
                line_out[np.flatnonzero(in_line)] = ul < pf[LINE][in_line, ri]
            nf = int(bus_out.sum() + line_out.sum())
            n_failed_by_rp[ri] += nf
            if nf == 0:
                continue          # the screen: nothing failed, nothing to solve
            energy, fresh, net_mw = _event_energy(
                data, line_out, bus_out, model, gen_av, slack_mode, scales, hourly
            )
            n_solves += fresh
            ens[fi, ri] = energy
            peak_mw[ri] += net_mw

    eaens = float(_integrate_ead(ens, rp_fp, np.zeros(len(ens))).sum())

    # RP100 diagnostics, on whichever tabulated RP is closest to 100.
    i100 = int(np.argmin(np.abs(rps - 100.0)))
    return {
        "EAENS_MWh": eaens,
        "EAENS_GWh": eaens / 1000.0,
        "peak_unserved_MW_RP100": float(peak_mw[i100]),
        "n_failed_RP100": float(n_failed_by_rp[i100]),
        "n_solves": float(n_solves),
        "n_footprints": float(len(fp_ids)),
        "structural_floor_MW": float(data.structural_floor_mw),
        # Reported so the correction is auditable rather than buried: the
        # naive constant-power figure is EAENS_MWh * window / load_shape_hours.
        "load_shape_hours": float(lsf),
        "outage_hours": float(window),
    }
