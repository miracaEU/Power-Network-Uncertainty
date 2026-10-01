"""Are the same buses always shed, and if so, why?

Two explanations look identical on a map and have opposite consequences:

  * **Topology artefact** - the bus sits in a small island with no generation,
    because the network data is incomplete or a transformer is missing. It sheds
    in every scenario because it is disconnected from supply by construction. No
    rating change, load reweighting or HVDC link will ever fix it, and its
    contribution to EAENS is noise dressed as risk.
  * **Genuine bottleneck** - the bus sits in a well-supplied island but the
    corridors serving it are at their thermal limit. It sheds because the
    network really cannot deliver, which is a result worth reporting.

The discriminator is the island the bus lands in: its size, and whether it
contains enough generation to serve its own load. A bus that is short because
its island holds three buses and no plant is an artefact; a bus that is short
inside a 5,000-bus island with surplus generation is congested.

Runs a grid of scenarios (hours x disaggregation x HVDC), records the per-bus
shed vector for each, and reports how much of the shed is the same buses every
time.

Run:
    python scripts/shed_persistence.py [--outdir results/diagnostics]
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from power_net.dcopf_lp import build_lp_net, solve_dc_opf
from power_net.demand import bus_load_mw, national_demand_at
from power_net.disaggregation import nuts3_bus_weights
from power_net.injections import (add_shedding_units, add_slack,
                                  assign_generation, assign_loads)
from power_net.network import VOLL_EUR_MWH, build_base_network, bus_coords_3035

HOURS = [("2018-01-24T17:00:00Z", "winter peak"),
         ("2018-07-19T03:00:00Z", "summer night"),
         ("2020-06-21T03:00:00Z", "series minimum")]
DISAGG = ["equal", "nuts3", "nuts3_pop", "nuts3_gdp"]
SHED_EPS = 1.0        # MW; below this a "shed" is numerical dust


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results/diagnostics")
    args = ap.parse_args()
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    from power_net.paths import load_config
    em = load_config()["energy_model_dir"]
    print("building...", flush=True)
    net0, geo = build_base_network(em)
    xy = bus_coords_3035(geo)
    plants = pd.read_csv(em / "powerplants.csv")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from map_network_state import hvdc_link_table

    weights = {"equal": None}
    for m in DISAGG[1:]:
        weights[m], _ = nuts3_bus_weights(net0, geo, em, mode=m)

    net = copy.deepcopy(net0)
    dummy, _ = bus_load_mw(net0, national_demand_at(em, HOURS[0][0]), mode="equal")
    assign_loads(net, dummy)
    assign_generation(net, plants, xy, availability=1.0)
    add_shedding_units(net, VOLL_EUR_MWH)
    add_slack(net, mode="per_synchronous_area")
    links = hvdc_link_table(net, em)
    lp = build_lp_net(net, voll=VOLL_EUR_MWH)
    n_bus = lp.n_bus

    runs, names = [], []
    for stamp, hlabel in HOURS:
        dem = national_demand_at(em, stamp)
        for mode in DISAGG:
            load, _ = bus_load_mw(net0, dem, mode=mode, weights=weights[mode])
            for dc_on in (False, True):
                l2 = copy.copy(lp)
                # shed caps follow the load vector for this disaggregation
                l2.shed_max = load[lp.shed_bus]
                if dc_on:
                    l2.dc_from = links.from_bus.to_numpy(np.int64)
                    l2.dc_to = links.to_bus.to_numpy(np.int64)
                    l2.dc_cap = links.p_nom.to_numpy(float)
                else:
                    l2.dc_from = l2.dc_to = l2.dc_cap = None
                r = solve_dc_opf(l2, np.zeros(lp.n_line, bool), np.zeros(n_bus, bool),
                                 1.0, 1.0, thermal_limits=True, objective="min_shed")
                v = np.zeros(n_bus)
                if r["success"]:
                    np.add.at(v, lp.shed_bus, r["shed_unit"])
                runs.append(v)
                names.append(f"{hlabel}|{mode}|{'DC' if dc_on else 'AC'}")
                print(f"  {names[-1]:36s} shed {v.sum():9,.1f} MW at "
                      f"{int((v > SHED_EPS).sum()):5d} buses", flush=True)

    S = np.vstack(runs)                       # (n_runs, n_bus)
    hit = S > SHED_EPS
    n_runs = len(runs)
    times = hit.sum(axis=0)

    print(f"\n=== persistence across {n_runs} scenarios ===")
    print(f"  buses shed in at least one : {int((times > 0).sum()):6d}")
    print(f"  buses shed in ALL          : {int((times == n_runs).sum()):6d}")
    core = times == n_runs
    print(f"  share of total shed MW that sits on the always-shed buses: "
          f"{100 * S[:, core].sum() / S.sum():.1f}%")
    for k in (n_runs, int(n_runs * .75), int(n_runs * .5), 1):
        m = times >= k
        print(f"    shed in >= {k:2d}/{n_runs} scenarios: {int(m.sum()):5d} buses, "
              f"{100 * S[:, m].sum() / S.sum():5.1f}% of shed MW")

    # --- why are the persistent ones shed? ---------------------------------
    f, t = lp.br_from, lp.br_to
    adj = sp.coo_matrix((np.ones(len(f)), (f, t)), shape=(n_bus, n_bus)).tocsr()
    ncomp, lab = connected_components(adj, directed=False)
    isl_size = np.bincount(lab, minlength=ncomp)
    gen = np.zeros(n_bus)
    np.add.at(gen, lp.gen_bus, lp.gen_max)
    isl_gen = np.bincount(lab, weights=gen, minlength=ncomp)
    load_ref = runs[0] * 0
    np.add.at(load_ref, lp.shed_bus, lp.shed_max)
    isl_load = np.bincount(lab, weights=load_ref, minlength=ncomp)

    # A real OSM-derived bus is named way/... , relation/... or
    # merged_relation/... ; anything else is a synthetic PyPSA-Eur node with no
    # OSM identifier, and therefore one of the 27% that cannot be joined to any
    # hazard. pandas string methods rather than np.char: the name column is
    # object dtype, which np.char.startswith has no loop for.
    bus_names = net0.bus["name"].astype(str)      # NOT `names` - that is the
    name = bus_names.to_numpy()                   # scenario label list above
    synthetic = ~bus_names.str.startswith(
        ("way/", "relation/", "merged_relation/")).to_numpy()
    zone = net0.bus["zone"].fillna("").astype(str).to_numpy()

    idx = np.flatnonzero(core)
    rows = []
    for b in idx:
        k = lab[b]
        rows.append({"bus": int(b), "name": name[b], "zone": zone[b],
                     "synthetic": bool(synthetic[b]),
                     "island_buses": int(isl_size[k]),
                     "island_gen_MW": float(isl_gen[k]),
                     "island_load_MW": float(isl_load[k]),
                     "island_surplus_MW": float(isl_gen[k] - isl_load[k]),
                     "mean_shed_MW": float(S[:, b].mean())})
    df = pd.DataFrame(rows).sort_values("mean_shed_MW", ascending=False)
    if len(df):
        short = df.island_surplus_MW < 0
        tiny = df.island_buses <= 5
        print(f"\n=== the {len(df)} always-shed buses ===")
        print(f"  in a generation-SHORT island        : {int(short.sum()):5d} "
              f"({100*df.mean_shed_MW[short].sum()/df.mean_shed_MW.sum():.1f}% of their MW)")
        print(f"  in an island of <= 5 buses          : {int(tiny.sum()):5d}")
        print(f"  in a surplus island (-> congestion) : {int((~short).sum()):5d} "
              f"({100*df.mean_shed_MW[~short].sum()/df.mean_shed_MW.sum():.1f}% of their MW)")
        print(f"  synthetic PyPSA-Eur nodes (no OSM id): {int(df.synthetic.sum()):5d}")
        print(f"  countries: {df.zone.value_counts().head(8).to_dict()}")
        print("\n  top 15 by mean shed:")
        print(df.head(15).to_string(index=False))
        df.to_csv(out / "shed_persistence_core.csv", index=False)
        print(f"\nwrote {out / 'shed_persistence_core.csv'}")

    pd.DataFrame({"scenario": names, "shed_MW": S.sum(axis=1),
                  "n_buses": hit.sum(axis=1)}).to_csv(
        out / "shed_persistence_runs.csv", index=False)


if __name__ == "__main__":
    main()
