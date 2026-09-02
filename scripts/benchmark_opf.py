"""Measure DC OPF wall time on the pan-European network.

This is the number every downstream feasibility decision depends on: Sobol at
N=1024 with k~12 needs ~26,600 model evaluations, and each evaluation solves
one OPF per (footprint, return period) pair that survives screening. If a solve
costs ~1 s and a few hundred survive per evaluation, full-fidelity pan-EU Sobol
is out of reach and the plan's fallback levers (memoisation, localised OPF,
tiered fidelity) decide the design instead.

Load disaggregation here is a deliberate placeholder - see LOAD NOTE below.
Solve time depends on problem *size* and structure, not on how accurately
demand was split between buses, so this does not affect the timing result.

    python scripts/benchmark_opf.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from power_net.injections import (  # noqa: E402
    add_shedding_units,
    add_slack,
    assign_generation,
    assign_loads,
    island_labels,
)
from power_net.network import (  # noqa: E402
    ENERGY_MODEL_DIR,
    ISO2_TO_ISO3,
    OPSD_DEMAND_ALIAS,
    VOLL_EUR_MWH,
    build_base_network,
    bus_coords_3035,
)

DEMAND_COL = "_load_actual_entsoe_transparency"


def national_demand(em_dir: Path, timestamp: str) -> dict[str, float]:
    """National demand (MW) per ISO2 at one hour of the OPSD series."""
    path = em_dir / "time_series_60min_singleindex.csv"
    head = pd.read_csv(path, nrows=1)
    cols = [c for c in head.columns if c.endswith(DEMAND_COL)]
    wanted = {OPSD_DEMAND_ALIAS.get(k, k): k for k in ISO2_TO_ISO3}
    # Country-level series only: TSO sub-zones (DE_50hertz, DK_1, ...) would
    # double-count against their parent country.
    cols = [c for c in cols if c[: -len(DEMAND_COL)] in wanted]
    ts = pd.read_csv(path, usecols=["utc_timestamp"] + cols)
    row = ts.loc[ts["utc_timestamp"] == timestamp]
    if row.empty:
        raise SystemExit(f"timestamp {timestamp} not in the OPSD series")
    return {wanted[c[: -len(DEMAND_COL)]]: float(row[c].iloc[0]) for c in cols}


def benchmark_load_vector(net, demand: dict[str, float]) -> tuple[np.ndarray, list[str]]:
    """LOAD NOTE - benchmark only, NOT the model's disaggregation.

    Splits each country's demand equally across its buses. The real model uses
    NUTS3 population/GDP weights (Eurostat files ship with the Energy Model),
    which changes *where* load sits but not the OPF's size or sparsity, so the
    timing measured here carries over. Crucially, countries with no OPSD series
    are left at ZERO and reported - never given an invented default, which is
    what produces D3.3's implausible load-loss figures.
    """
    zone = net.bus["zone"].fillna("").to_numpy()
    load = np.zeros(len(net.bus))
    missing = []
    for iso2 in sorted(set(zone) - {""}):
        mask = zone == iso2
        if iso2 not in demand or not np.isfinite(demand[iso2]):
            missing.append(iso2)
            continue
        load[mask] = demand[iso2] / mask.sum()
    return load, missing


def main() -> None:
    import pandapower as pp

    em = ENERGY_MODEL_DIR
    stamp = "2018-01-24T17:00:00Z"          # a winter evening peak

    t = time.perf_counter()
    net, geo = build_base_network(em)
    xy = bus_coords_3035(geo)
    print(f"build base network      {time.perf_counter()-t:7.2f}s  "
          f"{len(net.bus)} buses / {len(net.line)} lines / {len(net.trafo)} trafos")

    t = time.perf_counter()
    demand = national_demand(em, stamp)
    load, missing = benchmark_load_vector(net, demand)
    assign_loads(net, load)
    print(f"loads ({stamp})         {time.perf_counter()-t:7.2f}s  "
          f"{len(net.load)} loads, {load.sum()/1000:.1f} GW total")
    if missing:
        print(f"  no OPSD demand series (left at ZERO, not defaulted): {missing}")

    t = time.perf_counter()
    plants = pd.read_csv(em / "powerplants.csv")
    assign_generation(net, plants, xy, availability=1.0)
    print(f"generation              {time.perf_counter()-t:7.2f}s  "
          f"{len(net.gen)} (bus,fuel) units, "
          f"{net.gen.max_p_mw.sum()/1000:.1f} GW nameplate")

    shed_idx = add_shedding_units(net, VOLL_EUR_MWH)
    slack = add_slack(net, mode="per_synchronous_area")
    print(f"shedding units          {len(shed_idx)}   slack references {len(slack)}")

    labels = island_labels(net)
    print(f"islands                 {labels.max()+1}")

    # Line/transformer thermal limits, so the OPF can actually bind on capacity.
    net.line["max_loading_percent"] = 100.0
    net.trafo["max_loading_percent"] = 100.0

    print("\n--- DC OPF ---")
    t = time.perf_counter()
    try:
        pp.rundcopp(net)
        dt = time.perf_counter() - t
        ens = float(net.res_sgen.p_mw.iloc[shed_idx].sum())
        print(f"SOLVED in {dt:.2f}s")
        print(f"  generation dispatched {net.res_gen.p_mw.sum()/1000:9.1f} GW")
        print(f"  load shed             {ens:9.1f} MW  "
              f"({100*ens/max(load.sum(),1):.2f}% of demand)")
        print(f"  max line loading      {net.res_line.loading_percent.max():9.1f} %")
    except Exception as e:
        dt = time.perf_counter() - t
        print(f"FAILED after {dt:.2f}s -> {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()