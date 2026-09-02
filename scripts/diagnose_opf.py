"""Isolate why the pan-European DC OPF does not converge.

Works from the cheapest check upward: does the network solve at all (plain DC
power flow), does it solve without thermal limits, and does it solve on a single
country. The answer decides whether the plan's localised-OPF lever is optional
or mandatory.
"""
from __future__ import annotations

import sys, time
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from power_net.injections import (add_shedding_units, add_slack, assign_generation,
                                  assign_loads, island_labels)
from power_net.network import ENERGY_MODEL_DIR, VOLL_EUR_MWH, build_base_network, bus_coords_3035
from benchmark_opf import benchmark_load_vector, national_demand


def build(stamp="2018-01-24T17:00:00Z", limits=True, slack_mode="per_synchronous_area"):
    em = ENERGY_MODEL_DIR
    net, geo = build_base_network(em)
    xy = bus_coords_3035(geo)
    load, _ = benchmark_load_vector(net, national_demand(em, stamp))
    assign_loads(net, load)
    assign_generation(net, pd.read_csv(em / "powerplants.csv"), xy, availability=1.0)
    idx = add_shedding_units(net, VOLL_EUR_MWH)
    if slack_mode is not None:
        add_slack(net, mode=slack_mode)
    if limits:
        net.line["max_loading_percent"] = 100.0
        net.trafo["max_loading_percent"] = 100.0
    return net, idx, load


def data_health(net):
    print("--- data health ---")
    print("  bus in_service False   :", int((~net.bus.in_service).sum()))
    print("  line in_service False  :", int((~net.line.in_service).sum()))
    print("  line x_ohm_per_km <= 0 :", int((net.line.x_ohm_per_km <= 0).sum()))
    print("  line max_i_ka <= 0     :", int((net.line.max_i_ka <= 0).sum()))
    print("  line parallel <= 0     :", int((net.line.parallel <= 0).sum()))
    print("  line df <= 0           :", int((net.line.df <= 0).sum()))
    print("  trafo sn_mva <= 0      :", int((net.trafo.sn_mva <= 0).sum()))
    print("  trafo vk_percent <= 0  :", int((net.trafo.vk_percent <= 0).sum()))
    nan_cols = [c for c in ["x_ohm_per_km", "max_i_ka", "length_km", "parallel", "df"]
                if net.line[c].isna().any()]
    print("  line NaN columns       :", nan_cols or "none")

    lbl = island_labels(net)
    load_by = np.bincount(lbl[net.load.bus.to_numpy()], weights=net.load.p_mw.to_numpy(),
                          minlength=lbl.max() + 1)
    gen_by = np.bincount(lbl[net.gen.bus.to_numpy()], weights=net.gen.max_p_mw.to_numpy(),
                         minlength=lbl.max() + 1)
    short = np.flatnonzero(load_by > gen_by + 1e-6)
    print(f"  islands                : {lbl.max()+1}")
    print(f"  islands short of gen   : {len(short)} "
          f"(unserved floor {max(0.0,(load_by[short]-gen_by[short]).sum())/1000:.1f} GW)")


def timed(label, fn):
    t = time.perf_counter()
    try:
        fn()
        print(f"  {label:<38s} OK      {time.perf_counter()-t:7.2f}s")
        return True
    except Exception as e:
        print(f"  {label:<38s} FAILED  {time.perf_counter()-t:7.2f}s  {type(e).__name__}")
        return False


def main():
    import pandapower as pp

    net, shed_idx, load = build(limits=True)
    data_health(net)

    print("\n--- solver ladder (full continent) ---")
    timed("rundcpp (plain DC power flow)", lambda: pp.rundcpp(net))

    net2, idx2, _ = build(limits=False)
    ok = timed("rundcopp, NO thermal limits", lambda: pp.rundcopp(net2))
    if ok:
        ens = float(net2.res_sgen.p_mw.iloc[idx2].sum())
        print(f"      -> shed {ens:.1f} MW of {net2.load.p_mw.sum():.0f} MW")

    timed("rundcopp, thermal limits", lambda: pp.rundcopp(net))

    print("\n--- single country (PRT), full continent removed ---")
    for iso2 in ("LU", "PT"):
        n3, i3, _ = build(limits=False)
        keep = (n3.bus.zone == iso2).to_numpy()
        drop = np.flatnonzero(~keep)
        n3.bus.loc[drop, "in_service"] = False
        bad_l = n3.line.from_bus.isin(drop) | n3.line.to_bus.isin(drop)
        n3.line.loc[bad_l, "in_service"] = False
        bad_t = n3.trafo.hv_bus.isin(drop) | n3.trafo.lv_bus.isin(drop)
        n3.trafo.loc[bad_t, "in_service"] = False
        n3.load = n3.load[n3.load.bus.isin(np.flatnonzero(keep))]
        n3.gen = n3.gen[n3.gen.bus.isin(np.flatnonzero(keep))]
        n3.sgen = n3.sgen[n3.sgen.bus.isin(np.flatnonzero(keep))]
        add_slack(n3, mode="per_synchronous_area")
        print(f"  [{iso2}] {keep.sum()} buses, {len(n3.load)} loads, {len(n3.gen)} gens, "
              f"{len(n3.ext_grid)} refs")
        timed(f"  [{iso2}] rundcopp", lambda n=n3: pp.rundcopp(n))


if __name__ == "__main__":
    main()
