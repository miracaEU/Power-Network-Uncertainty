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
from pathlib import Path
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
from . import eventlog
from .dcopf_lp import build_lp_net, solve_dc_opf
from .disaggregation import DEFAULT_STATS_YEAR, blend_shares, nuts3_bus_weights
from .hazard import build_component_hazard, load_segments
from .hvdc import dc_terminal_mask, load_hvdc
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
    NO_OPSD_DEMAND,
    VOLL_EUR_MWH,
    build_base_network,
    bus_coords_3035,
)

# Event footprints, from fully coherent to fully independent. The factor spans
# both extremes on purpose: for additive damage the correlation structure does
# not bias the expectation, but islanding is super-additive, so for ENS it can
# dominate - and no single choice is defensible enough to fix.
# "basin" is HydroBASINS level 08 (raw HYBAS_ID); "basin_lev07" is level 07,
# which is the study's event grain. "basin_coarse" is an alias of lev07 kept so
# older configs still resolve.
FOOTPRINT_LEVELS = ["basin", "basin_lev07", "basin_coarse", "main_basin",
                    "country", "independent"]

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
    disagg_report: dict = field(default_factory=dict)
    # Kept so `apply_load` can re-place demand per evaluation without a rebuild:
    # the national totals at the archetype hour, and the converter-terminal mask
    # that keeps load off DC nodes.
    demand_at_t0: dict = field(default_factory=dict)
    dc_terminal_mask: Any = None
    country: str = ""               # ISO3 of the run, for labelling event rows
    # lev07 basins this run is responsible for evaluating. None means "every
    # basin holding one of this country's components", which is the
    # single-country behaviour. The partition matters once events are
    # Europe-wide: without it, each run would evaluate every basin its network
    # reaches and a shared basin would be counted once per run.
    owned_basins: Any = None
    structural_floor_mw: float = 0.0
    # HVDC transfer elements (links + converters), empty when include_hvdc is
    # off. The connectivity model treats them as plain edges, which overstates
    # what a finite link can move; the LP bounds them at p_nom.
    hvdc_from: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    hvdc_to: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    lp: Any = None                  # dcopf_lp.LPNet; None -> pandapower backend
    opf_objective: str = "min_shed"
    thermal_limits: bool = False
    opf_fallbacks: int = 0          # solves that fell back to connectivity
    # (bus indices, shed MW) from the most recent solve, restored on a memo hit.
    # Feeds the attribution table; None when the solve produced no per-bus
    # detail, i.e. after an OPF failure.
    _last_shed: Any = None
    _memo: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def _owned_basins(cfg: dict) -> np.ndarray | None:
    """This country's share of the lev07 basin partition, or None.

    Built by `scripts/build_basin_owner.py`. Absent means "not partitioned" -
    single-country runs then fall back to every basin holding one of this
    country's components, which is what the study did before events went
    Europe-wide. That fallback is safe alone and WRONG across several countries,
    because a basin straddling a border would be evaluated by both runs and
    counted twice in the European total.
    """
    path = Path(cfg["cache_dir"]) / "basin_owner.parquet"
    if not path.is_file():
        return None
    tab = pd.read_parquet(path)
    own = tab.loc[tab["country"] == cfg["country"], "event_basin"].to_numpy()
    return own if len(own) else None


def apply_load(data: NetData, gdp_weight: float) -> None:
    """Re-place demand for a sampled `nuts3_gdp_weight`, in place.

    `nuts3_gdp_weight` is a sampled factor, but load used to enter only at build
    time and NetData is cached per (country, archetype) - a continuous weight
    would have forced a ~30-160 s rebuild per experiment. Nothing expensive
    depends on load: topology, linkage, hazard tables, fragility tables,
    footprint labels and the LP's B-matrix are all load-independent. Only these
    five are, and all are cheap.

    The memo clear is essential and easy to miss: `_solve` keys its cache on
    (model, gen_availability, slack_mode, load_scale, failed set) and NOT on the
    load vector, so a stale entry would be reused under different demand and
    silently return the wrong answer.
    """
    shares = data.disagg_report.get("pop_share_bus")
    if shares is None:
        return                      # equal split, or a build with no NUTS3 join
    w = blend_shares(shares, data.disagg_report["gdp_share_bus"], gdp_weight)
    load, _ = bus_load_mw(data.net, data.demand_at_t0, mode="nuts3", weights=w,
                          exclude=data.dc_terminal_mask)

    net = data.net
    bus_of_load = net.load["bus"].to_numpy()
    net.load["p_mw"] = load[bus_of_load]
    # Shedding injectors are capped at their own bus's demand; the sgen table is
    # row-aligned with net.load by construction (add_shedding_units).
    net.sgen["max_p_mw"] = load[net.sgen["bus"].to_numpy()]
    data.bus_load = load
    if data.lp is not None:
        data.lp.shed_max = load[data.lp.shed_bus]
    data._memo.clear()
    data.structural_floor_mw = _unserved_connectivity(
        data, np.zeros(data.n_line, bool), np.zeros(data.n_bus, bool), 1.0
    )


def _apply_zero_load_policy(net, netcfg: dict) -> None:
    """How to treat a country with generation but no demand series.

    AL, BA, MD, MK and XK have no OPSD demand, so their 85 buses carry zero
    load while keeping 11.53 GW of generation. In the base case they dispatch
    6.7 GW with nothing to serve, which makes **Bosnia the fourth largest
    exporter in the model** - ahead of Lithuania, Sweden and Greece. That is an
    artefact of missing demand data, not a feature of the European grid.

    `keep_generation` (default) leaves it alone. `neutral` also switches their
    generation off, so the country is neither source nor sink and its lines
    remain only as transit corridors - the honest reading of "this country's
    energy balance is not modelled".

    **EAENS is insensitive to the choice**, measured one factor draw at a time:
    HRV flood +0.004 MWh, HUN flood -0.959, ROU flood -0.037, HRV windstorm
    +0.002 - all under 0.03%, and of both signs, i.e. re-dispatch noise rather
    than bias. Two reasons: ENS is a difference of two solves that both contain
    the artefact, and what remains unserved is delivery-limited, so supply
    hundreds of kilometres away does not relieve it. The switch exists so that
    insensitivity stays checkable, and for base-case diagnostics where the
    absolute flows are the point.
    """
    policy = netcfg.get("zero_load_country_policy", "keep_generation")
    if policy == "keep_generation":
        return
    if policy != "neutral":
        raise ValueError(
            f"unknown zero_load_country_policy '{policy}'; "
            "choose keep_generation or neutral"
        )
    zone = net.bus["zone"].astype(str).to_numpy()
    hit = np.isin(zone[net.gen["bus"].to_numpy(np.int64)], list(NO_OPSD_DEMAND))
    net.gen.loc[hit, "max_p_mw"] = 0.0
    net.gen.loc[hit, "p_mw"] = 0.0


def _paneu_hazard(cfg: dict, hazard: str, rps: np.ndarray) -> dict | None:
    """Pan-European per-component hazard on THIS run's return-period grid.

    Built by `scripts/build_paneu_hazard.py`. Without it, `hazard.py` supplies
    only the study country's exposure, so every component abroad is unexposed
    and immune - which silently reduced a Europe-wide basin event to its
    study-country half, whatever the basin labels said.

    The run keeps its own return-period grid. The cache is stored on the UNION
    of all countries' grids, and every country's grid is a subset of that
    union, so this is a column selection and not a second interpolation: the
    study country's own components come back bit-identical to the
    country-only build, and only foreign components whose country has a
    coarser native grid carry interpolated values.
    """
    path = Path(cfg["cache_dir"]) / f"hazard_paneu_{hazard}.npz"
    if not path.is_file():
        return None
    if not len(rps):
        # A structurally empty profile - the file exists but holds no rows, so
        # no asset of this country is exposed to this hazard (ROU, BGR, BEL and
        # ISL coastal). There is no grid to select columns on; the country-only
        # tables are equally empty and the scenario integrates to zero.
        return None
    z = np.load(path)
    union = z["rps"]
    cols = []
    for r in rps:
        hit = np.flatnonzero(np.isclose(union, r))
        if not len(hit):
            # The union is built from every country's grid, so a run-time RP
            # missing from it means the cache predates a Stage 1 change.
            raise ValueError(
                f"{path.name} has no return period {r} (union {union.tolist()}). "
                f"The cache is stale - re-run scripts/build_paneu_hazard.py."
            )
        cols.append(int(hit[0]))
    # Explicit dtype: an empty list makes np.array float64, which is not a
    # legal index. The empty case is guarded above, but the dtype is stated
    # here so it cannot come back.
    cols = np.array(cols, dtype=np.int64)
    out = {"rps": np.asarray(rps, float),
           "aggregation": cfg["network"].get("intensity_aggregation", "max")}
    for kind in (BUS, LINE):
        out[kind] = {
            "intensity": z[f"{kind}_intensity"][:, cols],
            "prot_rp": z[f"{kind}_prot_rp"],
            "hybas": z[f"{kind}_hybas"],
            "anchors": z[f"{kind}_anchors"],
            "matched": z[f"{kind}_matched"],
        }
    return out


def _component_basins(cfg: dict) -> dict[str, np.ndarray] | None:
    """Pan-European lev07 label per component, or None if not precomputed.

    Built by `scripts/build_component_basins.py`. Absent means the labels fall
    back to Stage 1's `HYBAS_ID`, which covers the study country alone - usable
    for a single-country run's event side, but it leaves the attribution table
    blank and cannot express a cross-border basin.
    """
    path = Path(cfg["cache_dir"]) / "component_basin.parquet"
    if not path.is_file():
        return None
    tab = pd.read_parquet(path)
    out = {}
    for kind in (BUS, LINE):
        part = tab[tab["kind"] == kind].sort_values("idx")
        if len(part):
            out[kind] = part["event_basin"].to_numpy(np.int64)
    return out or None


def _footprint_labels(cfg: dict, hz_tables: dict) -> dict:
    """Component -> event-footprint label, at every nesting level.

    HydroBASINS gives the hydrological levels; `country` makes the whole study
    area one coherent event; `independent` makes every component its own. Label
    -1 means "not assigned to any footprint" (no basin); those components form a
    single residual footprint rather than being dropped from the risk sum.

    NOTE the level names. `basin` is the raw `HYBAS_ID`, which is HydroBASINS
    **level 08** (10,309 basins pan-EU, median 499 km2). The study's event grain
    is **level 07** - coarser, 3,335 basins - which is `basin_lev07` here, and is
    what `Pfafstetter-prefix truncation` produces: dropping the last digit of an
    8-digit PFAF_ID is exactly one level up. `basin_coarse` is retained as an
    alias for backward comparability with earlier runs.

    `basin_lev07` comes from `cache/component_basin.parquet` when that exists -
    a pan-European point-in-polygon assignment covering all 6,737 buses, not
    just the study country's. The Stage 1 `HYBAS_ID` route below covers only the
    country whose exposure file `hazard.py` loaded (measured: 20 of 6,737 buses
    for HRV), which left every `affected_basin` in the attribution table at -1
    and made the Europe-wide event definition unreachable. The two routes were
    checked against each other and agree on 20 of 20 buses, so this is a
    coverage fix, not a change of grain.
    """
    basins = pd.read_parquet(cfg["basin_data_path"],
                             columns=["HYBAS_ID", "MAIN_BAS", "PFAF_ID"])
    hyb_to_main = dict(zip(basins["HYBAS_ID"], basins["MAIN_BAS"]))
    # Pfafstetter codes nest by prefix: dropping the last digit is exactly one
    # level coarser, which is what "Pfafstetter-prefix truncation" means.
    hyb_to_pfaf = dict(zip(basins["HYBAS_ID"], basins["PFAF_ID"] // 10))

    paneu = _component_basins(cfg)

    out: dict[str, dict[str, np.ndarray]] = {lv: {} for lv in FOOTPRINT_LEVELS}
    any_hz = next(iter(hz_tables.values()))
    for kind in (BUS, LINE):
        hyb = any_hz[kind]["hybas"]
        n = len(hyb)
        out["basin"][kind] = np.where(hyb > 0, hyb, -1).astype(np.int64)
        lev07 = np.array(
            [hyb_to_pfaf.get(h, -1) if h > 0 else -1 for h in hyb], dtype=np.int64
        )
        if paneu is not None and kind in paneu:
            cached = paneu[kind]
            if len(cached) != n:
                raise ValueError(
                    f"cache/component_basin.parquet has {len(cached)} {kind} "
                    f"rows but the network has {n}. The cache is stale - "
                    f"re-run scripts/build_component_basins.py."
                )
            # Where Stage 1 also has an opinion the two agree (verified 20/20),
            # so preferring the cache changes coverage only.
            lev07 = cached
        out["basin_lev07"][kind] = lev07
        out["basin_coarse"][kind] = lev07
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
    # The spatial join behind the NUTS3 weights runs once here, not per
    # evaluation: the grid topology and the regions are both fixed for the
    # whole study, only the national totals move with the archetype hour.
    # HVDC converter terminals are stations, not demand centres. Excluding them
    # is independent of whether the DC layer is modelled: assigning a share of
    # national demand to a DC node is wrong either way, and it is what puts
    # 6,766 MW of phantom load on 70 buses that then shed 100% of it.
    exclude = (dc_terminal_mask(em, net)
               if netcfg.get("exclude_dc_terminal_load", False) else None)
    disagg = netcfg.get("load_disaggregation", "equal")
    weights, disagg_report = None, {"mode": "equal"}
    if disagg != "equal":
        weights, disagg_report = nuts3_bus_weights(
            net, geo, em,
            mode=disagg,
            year=int(netcfg.get("nuts3_year", DEFAULT_STATS_YEAR)),
            gdp_weight=float(netcfg.get("nuts3_gdp_weight", 0.5)),
            snap_max_m=float(netcfg.get("snap_max_m", 25_000.0)),
        )
    load, missing = bus_load_mw(net, demand, mode=disagg, weights=weights,
                                exclude=exclude)
    if exclude is not None:
        disagg_report["dc_terminals_excluded"] = int(exclude.sum())

    plants = pd.read_csv(em / "powerplants.csv")
    bus_of_plant = snap_plants_to_buses(plants, xy)
    gen = np.zeros(len(net.bus))
    np.add.at(gen, bus_of_plant, plants["Capacity"].to_numpy(np.float64))

    # Make the net OPF-ready once, so the `dc_opf` network model can simply
    # toggle in_service rather than rebuild. Costs about a second here and
    # nothing per solve.
    assign_loads(net, load)
    assign_generation(net, plants, xy, availability=1.0)
    _apply_zero_load_policy(net, netcfg)
    shed_idx = add_shedding_units(net, VOLL_EUR_MWH)
    add_slack(net, mode=netcfg.get("slack_placement", "per_synchronous_area"))
    # Thermal limits are OFF by default and this is a real limitation, not an
    # oversight: 12.9% of lines exceed 100% loading before any hazard is
    # applied (BENCHMARK.md), so a limited OPF does not converge even in the
    # base case. Leaving them off keeps the OPF solvable but blinds it to
    # capacity-constrained shedding - the one failure mode a connectivity
    # model also cannot see.
    #
    # This was long assumed to be the equal-split load disaggregation's fault.
    # It is not: measured 2026-09-09, `nuts3` moves it to 14.3%, slightly
    # WORSE. The open suspect is the missing HVDC layer - links.csv holds
    # 32,530 MW of DC that build_base_network never reads, and adding it
    # merges 36 of the 76 connected components. Do not re-enable this without
    # re-measuring; a converging limited OPF is the evidence, not a config
    # change.
    # Per-basin event rows go beside the EMA archive, one file per worker.
    # Configured here rather than in run_experiments because spawned workers
    # re-enter through load_net_data and never see the parent's setup.
    if netcfg.get("write_event_log", True):
        from .paths import country_results_dir, result_stem
        eventlog.configure(country_results_dir(cfg) / "events", result_stem(cfg))

    thermal = bool(netcfg.get("thermal_limits", False))
    if thermal:
        net.line["max_loading_percent"] = 100.0
        net.trafo["max_loading_percent"] = 100.0

    # The LP backend is built once here. It is a different formulation of the
    # same DC OPF, not a different model: with the dispatch held fixed at
    # pandapower's own optimum the two agree on every line flow to 4e-5 MW
    # against a 622 MW mean. What it buys is the three things pandapower's path
    # could not do - count stranded load, keep angle references fresh, and
    # actually converge with thermal limits on - plus ~25x the speed.
    backend = netcfg.get("opf_backend", "highs")
    lp = build_lp_net(net, voll=VOLL_EUR_MWH) if backend == "highs" else None

    hv_from = hv_to = np.zeros(0, np.int64)
    if netcfg.get("include_hvdc", False):
        hv = load_hvdc(em, net)
        hv_from = hv["from_bus"].to_numpy(np.int64)
        hv_to = hv["to_bus"].to_numpy(np.int64)
        if lp is not None:
            lp.dc_from, lp.dc_to = hv_from, hv_to
            lp.dc_cap = hv["p_nom"].to_numpy(float)

    seg = load_segments(cfg)
    from .hazard import available_hazards
    hazards = {}
    for hz in available_hazards(cfg):
        own = build_component_hazard(cfg, link, hz, seg)
        paneu = _paneu_hazard(cfg, hz, own["rps"])
        hazards[hz] = paneu if paneu is not None else own

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
        footprints=_footprint_labels(cfg, hazards),
        demand_series=dem_series,
        t0_idx=t0_idx,
        timestamp=str(stamp),
        missing_demand=missing,
        disagg_report=disagg_report,
        demand_at_t0=demand,
        dc_terminal_mask=exclude,
        country=str(cfg["country"]),
        owned_basins=_owned_basins(cfg),
        lp=lp,
        hvdc_from=hv_from,
        hvdc_to=hv_to,
        opf_objective=netcfg.get("dispatch_objective", "min_shed"),
        thermal_limits=thermal,
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
    rows = [data.line_from[keep], data.trafo_hv[tkeep]]
    cols = [data.line_to[keep], data.trafo_lv[tkeep]]
    if len(data.hvdc_from):
        # A DC link joins two areas for the purpose of "can this load be
        # served at all". Capacity is NOT represented here - connectivity has
        # no flows - so this is optimistic by exactly the amount a link's
        # p_nom falls short of the transfer the merged island implies. The LP
        # bounds the same elements properly.
        hk = ~bus_out[data.hvdc_from] & ~bus_out[data.hvdc_to]
        rows.append(data.hvdc_from[hk])
        cols.append(data.hvdc_to[hk])
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
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

    Shedding is a decision variable rather than an outcome: pandapower's
    `rundcopp` reports *infeasible* when demand cannot be met, it does not shed,
    so there is a dispatchable injector at every load bus capped at that bus's
    demand.

    Two backends. `highs` (the default) solves the LP in `dcopf_lp`; `pandapower`
    keeps the original `rundcopp` path as a cross-check. They are the same DC
    OPF - flows agree to 4e-5 MW with the dispatch held fixed - but only the LP
    counts load stranded at a de-energised bus, and that omission is what made
    the pandapower path return exactly 0.000 MWh once `_event_energy` began
    subtracting the no-hazard baseline.

    A failed solve falls back to the connectivity answer and increments
    `data.opf_fallbacks`, so a run can report how often that happened instead of
    silently reporting a number from a different model.
    """
    if data.lp is not None:
        r = solve_dc_opf(data.lp, line_out, bus_out, gen_availability, load_scale,
                         thermal_limits=data.thermal_limits,
                         objective=data.opf_objective)
        if r["success"]:
            # Per-bus shed is stashed for the attribution table. It rides
            # through the memo alongside the MW total, because a memoised solve
            # is not recomputed and a bare attribute would leave a stale vector
            # behind - attribution would then be credited to whichever event
            # last missed the cache.
            data._last_shed = (r["shed_bus"], r["shed_unit"])
            return r["shed"]
        data.opf_fallbacks += 1
        data._last_shed = None
        return _unserved_connectivity(data, line_out, bus_out, gen_availability,
                                      load_scale)

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
        data._last_shed = (net.sgen["bus"].to_numpy()[data.shed_idx],
                           net.res_sgen.p_mw.to_numpy()[data.shed_idx])
        return shed
    except Exception:
        data.opf_fallbacks += 1
        data._last_shed = None
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
        # Restore the per-bus shed that went with this solve. Without this, a
        # cache hit would leave `_last_shed` pointing at some earlier event and
        # the attribution table would credit the energy to the wrong basins.
        mw, shed = data._memo[key]
        data._last_shed = shed
        return mw, False
    data._last_shed = None
    if model == "connectivity":
        mw = _unserved_connectivity(data, line_out, bus_out, gen_availability,
                                    load_scale)
    elif model == "dc_opf":
        mw = _unserved_dc_opf(data, line_out, bus_out, gen_availability,
                              slack_mode, load_scale)
    else:
        raise ValueError(f"unknown network_model '{model}'")
    data._memo[key] = (mw, data._last_shed)
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

    **Per-bus attribution is the INCREMENT, and is returned explicitly.** The
    energy is a difference of two solves, so the place the energy was lost is a
    difference of two shed vectors - the buses that shed *because of* the event,
    not the buses that shed anyway. It has to be returned rather than read off
    `data._last_shed` afterwards, because the baseline solve runs second and
    leaves `_last_shed` pointing at the NO-HAZARD vector. Attribution built on
    that credited every event to the same handful of chronically-short buses
    (measured on HRV: the identical three affected basins for all five event
    basins, at every return period).
    """
    empty_l = np.zeros(data.n_line, bool)
    empty_b = np.zeros(data.n_bus, bool)

    def _inc(ev, bs):
        """Per-bus shed caused by the event: max(event - baseline, 0)."""
        if ev is None:
            return None            # OPF failed; no per-bus detail to attribute
        if bs is None:
            return ev
        # Both vectors are `lp.shed_bus`-aligned by construction, so they
        # subtract elementwise; assert rather than assume.
        if len(ev[1]) != len(bs[1]):
            return ev
        return (ev[0], np.maximum(ev[1] - bs[1], 0.0))

    if not hourly:
        mw, f1 = _solve(data, line_out, bus_out, model, gen_av, slack, 1.0)
        shed_ev = data._last_shed
        base, f2 = _solve(data, empty_l, empty_b, model, gen_av, slack, 1.0)
        net = max(mw - base, 0.0)
        return (net * float(scales.sum()), int(f1) + int(f2), net,
                _inc(shed_ev, data._last_shed))
    total = 0.0
    solves = 0
    mw0 = 0.0
    acc_bus, acc_mw = None, None
    for h, k in enumerate(scales):
        mw, f1 = _solve(data, line_out, bus_out, model, gen_av, slack, float(k))
        shed_ev = data._last_shed
        base, f2 = _solve(data, empty_l, empty_b, model, gen_av, slack, float(k))
        solves += int(f1) + int(f2)
        net = max(mw - base, 0.0)
        total += net
        inc = _inc(shed_ev, data._last_shed)
        if inc is not None:
            # Accumulated over the window, so a bus that is unserved for longer
            # carries proportionally more of the event's energy.
            acc_bus = inc[0]
            acc_mw = inc[1] if acc_mw is None else acc_mw + inc[1]
        if h == 0:
            mw0 = net
    return total, solves, mw0, (None if acc_mw is None else (acc_bus, acc_mw))


# ---------------------------------------------------------------------------
# Failure probabilities
# ---------------------------------------------------------------------------


def _p_fail(data: NetData, hazard: str, kind: str, intensity: np.ndarray,
            kwargs: dict) -> np.ndarray:
    """(n_comp, n_rp) failure probability for one component kind.

    A kind with no published fragility curve for this hazard cannot fail. That
    is the case for `line` under flood: V3.0.0 has substation curves (F2.4,
    F2.5) but nothing for lines, cables, towers or poles, and converting their
    damage ratios into failure probabilities would be inventing the curve this
    study has stopped inventing. Returning zeros is the honest encoding of "no
    evidence that this fails", and it is reported as a scope limitation rather
    than left for a reader to infer from the numbers.
    """
    kinds = frag.GROUPS_FOR_HAZARD.get(hazard, {})
    if kind not in kinds:
        return np.zeros_like(intensity)
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
    """(n_comp, n_rp) return periods after the climate shift.

    A component that matched no Stage 1 exposure row carries all-zero anchors.
    `_shift_rps` maps those to 1.0 for every return period, which is not a
    climate signal - it is missing data claiming every hazard is annual. Those
    rows are left unshifted instead. Averaging them into a footprint's
    integration grid was biasing it ~14% low for LUX at `event_footprint=
    "country"`, a confound between the warming and footprint factors.
    """
    code = WARMING_LEVELS.get(warming)
    base = np.tile(rps, (n_comp, 1))
    if code is None:
        return base
    a = anchors[:, :, _WARM_COL[code]]
    shifted = _shift_rps(base, a)
    has_anchor = np.isfinite(a).all(axis=1) & (a > 0).any(axis=1)
    return np.where(has_anchor[:, None], shifted, base)


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------


def compute_ens(data: NetData, **kwargs) -> dict:
    """One model evaluation: factor settings in, EAENS and diagnostics out."""
    # Where demand sits is a sampled factor, so it is re-placed per evaluation
    # rather than baked in at build time. No-op when the weight has not moved.
    if "nuts3_gdp_weight" in kwargs:
        apply_load(data, float(kwargs["nuts3_gdp_weight"]))
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
        # Protection is NOT applied here any more - it is a post-hoc factor
        # (scripts/postprocess_factors.py) applied to the stored event table in
        # the present-day frame. `prot_rp` is still carried on the event rows so
        # stage 2 can use the per-component standard where it wants to, and
        # `defence_rp` sweeps a uniform override. Set `apply_protection_inline`
        # only to reproduce a pre-2026-09-17 run.
        if hazard in ("river", "coastal") and kwargs.get("apply_protection_inline"):
            prot = _protection_rp(d["prot_rp"], kwargs)
            # The protection test happens in the PRESENT-DAY frame, against the
            # nominal return periods - not against `r`, the climate-shifted
            # ones. A defence is a physical structure whose design standard is
            # labelled by a present-day return period; warming does not move the
            # wall, it changes how often the water reaches it.
            #
            # Comparing shifted RPs against a flat standard made warming appear
            # to STRENGTHEN the defences: on LUX at a 100-year standard the
            # protected (component, RP) cell count rose 84 -> 84 -> 95 -> 97 ->
            # 98 of 108 across the warming levels, and at a 200-year standard
            # flood EAENS fell to exactly 0.0 from 2.0C upward. With protection
            # off the response is monotonic (316 -> 424 -> 490 -> 559 -> 706
            # MWh), so the model was sound and only this comparison was wrong.
            #
            # Equivalent to shifting hazard and protection together, as
            # miraca_uq.risk_model does, because `_shift_rps` is monotone once
            # the anchors are (hazard.py enforces that). Testing here needs no
            # shift at all, and `defence_rp` stays in the units an engineer
            # states it in.
            p = np.where(rps[None, :] <= prot[:, None], 0.0, p)
        # NOT masked to the study country. A basin event fails every component
        # in the basin, wherever it is. Masking made a cross-border basin fail
        # only its French half in the FR run and only its German half in the DE
        # run, so neither run modelled the actual event and summing them both
        # double-counted the basin and never represented it whole. Ownership is
        # resolved instead by giving each basin to exactly one run
        # (`owned_basins`), which keeps European totals additive.
        pf[kind] = p
        rp_adj[kind] = r

    labels = data.footprints[level]
    # Which basins this run is responsible for. `owned_basins` is the partition
    # that prevents double counting; without it, every run would evaluate every
    # basin its network touches. Falling back to the study country's own basins
    # keeps single-country runs working unchanged.
    owned = data.owned_basins
    if owned is not None:
        fp_ids = np.asarray(sorted(owned), dtype=np.int64)
    else:
        cand = np.concatenate([labels[BUS][data.country_bus],
                               labels[LINE][data.country_line]])
        fp_ids = np.unique(cand)
    # -1 means "no basin" - a component whose Stage 1 rows carry no HYBAS_ID.
    # These used to be dropped here, which deleted them from the risk sum
    # silently (2 of 14 buses and 1 line for LUX). They are now kept as a single
    # residual footprint so their risk is counted, and the count is reported as
    # an outcome so the size of the residual is visible rather than inferred.
    n_unbasined = int((labels[BUS][data.country_bus] < 0).sum()
                      + (labels[LINE][data.country_line] < 0).sum())

    ens = np.zeros((max(len(fp_ids), 1), n_rp))
    rp_fp = np.tile(rps, (max(len(fp_ids), 1), 1))
    n_solves = 0
    n_failed_by_rp = np.zeros(n_rp)
    peak_mw = np.zeros(n_rp)
    # Per-(basin, RP) detail for the event log. `n_failed_by_rp` and `peak_mw`
    # above accumulate ACROSS basins and so cannot serve: the post-hoc N-1 rule
    # needs the failure count of each individual event.
    nfail_cell = np.zeros_like(ens, dtype=np.int32)
    peak_cell = np.zeros_like(ens)
    fb_cell = np.zeros_like(ens, dtype=bool)
    key = eventlog.exp_key(kwargs)
    basin_of_bus = labels[BUS]

    for fi, fid in enumerate(fp_ids):
        # Europe-wide membership: every component in this basin, not just the
        # study country's share of it.
        in_bus = labels[BUS] == fid
        in_line = labels[LINE] == fid
        nb, nl = int(in_bus.sum()), int(in_line.sum())
        if nb + nl == 0:
            continue
        # One CRN draw per footprint, shared across every return period so the
        # same components are "weak" throughout - and independent of the factor
        # values, which is what keeps the model deterministic given (theta, seed).
        #
        # The seed was `seed + int(fid) % 100_000`, which binds as
        # `seed + (fid % 100_000)`: HYBAS_IDs are 10 digits, so any two basins
        # sharing their last five got IDENTICAL failure draws. SeedSequence
        # mixes both values properly and cannot collide that way.
        fp_seed = int(np.random.SeedSequence([int(seed), int(fid) & 0x7FFFFFFF]
                                             ).generate_state(1)[0])
        u = frag.correlated_uniforms(nb + nl, rho, fp_seed)
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
            energy, fresh, net_mw, shed_inc = _event_energy(
                data, line_out, bus_out, model, gen_av, slack_mode, scales, hourly
            )
            n_solves += fresh
            ens[fi, ri] = energy
            peak_mw[ri] += net_mw
            nfail_cell[fi, ri] = nf
            peak_cell[fi, ri] = net_mw
            fb_cell[fi, ri] = shed_inc is None
            # Where the energy was actually lost, by basin of the shedding bus.
            # The event happened in `fid`; this records the A -> B split so the
            # attribution can be re-keyed later without re-running anything.
            # `shed_inc` is the event-minus-baseline increment - see
            # `_event_energy`; `data._last_shed` here would be the baseline.
            if energy > 0 and shed_inc is not None:
                sb, su = shed_inc
                hit = su > 1e-9
                if hit.any():
                    aff = basin_of_bus[np.asarray(sb)[hit]]
                    # Scale the instantaneous MW split up to the event's energy,
                    # so the parts sum to `ens[fi, ri]` by construction.
                    mw = np.asarray(su)[hit]
                    share = mw / mw.sum()
                    tab = pd.Series(share * energy).groupby(aff).sum()
                    eventlog.record_attribution(
                        key, hazard, int(fid), float(rps[ri]),
                        tab.index.to_numpy(), tab.to_numpy(),
                    )

    # The climate shift is a piecewise-linear map through per-component anchors,
    # and a non-monotonic anchor set can leave a row's return periods unsorted -
    # which makes the trapezoid in _integrate_ead accumulate negative-width
    # strips. Sort each row and carry the energies with it, as the direct-damage
    # original does (risk_model.py:462-466).
    order = np.argsort(rp_fp, axis=1)
    rp_fp = np.take_along_axis(rp_fp, order, axis=1)
    ens = np.take_along_axis(ens, order, axis=1)
    nfail_cell = np.take_along_axis(nfail_cell, order, axis=1)
    peak_cell = np.take_along_axis(peak_cell, order, axis=1)
    fb_cell = np.take_along_axis(fb_cell, order, axis=1)
    rp_nom = np.take_along_axis(np.tile(rps, (len(ens), 1)), order, axis=1)

    # Persist the per-basin, per-return-period detail before it is collapsed.
    # `rp_nominal` is the present-day label of the event and `rp_shifted` how
    # often it occurs under this warming level: the post-hoc defence cutoff
    # tests the former and the integration uses the latter, and conflating them
    # is what made warming appear to strengthen defences.
    ids_out = fp_ids if len(fp_ids) else np.array([-1])
    nominal_out = rp_nom[0] if len(rp_nom) else rps
    warm = str(kwargs.get("warming", "current"))
    eventlog.record_events(
        key, hazard, data.country, warm, ids_out, nominal_out, rp_fp, ens,
        nfail_cell, peak_cell, fb_cell,
    )
    # The dense grid, so stage 2 can re-integrate over cells that `events`
    # legitimately omits.
    eventlog.record_rp_grid(hazard, warm, ids_out, nominal_out, rp_fp)

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
        # Cumulative over the worker's life, so a rising value across a run means
        # the OPF is degrading to the connectivity model somewhere. It used to do
        # that silently, which made a dc_opf result indistinguishable from a
        # connectivity one.
        "n_opf_fallback": float(data.opf_fallbacks),
        "n_unbasined": float(n_unbasined),
        # Reported so the correction is auditable rather than buried: the
        # naive constant-power figure is EAENS_MWh * window / load_shape_hours.
        "load_shape_hours": float(lsf),
        "outage_hours": float(window),
    }
