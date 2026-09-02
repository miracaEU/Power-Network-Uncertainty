"""Loads, generation, slack placement and load-shedding units.

The load-shedding units are the point of this module. pandapower's `rundcopp`
returns *infeasible* when demand cannot be met - it does not shed - so energy
not supplied has to be an explicit decision variable. Each load bus gets a
dispatchable injector capped at that bus's demand and priced at VoLL, far above
any generator's marginal cost. The optimiser therefore uses it only as a last
resort, and

    ENS = sum(shedding unit dispatch)

comes straight out of the optimisation. This also captures *capacity-constrained*
shedding - the network stays connected but lines bind - which a pure
connectivity model cannot see at all.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .network import (
    DEFAULT_MARGINAL_COST,
    conform_to_schema,
    ISO2_TO_ISO3,
    MARGINAL_COST,
    VOLL_EUR_MWH,
)


def _assign_table(net, name: str, data: dict, n: int) -> None:
    """Assign a whole element table, checked against pandapower's own schema.

    Wholesale assignment is what keeps network construction sub-second instead
    of ~200 s, but it bypasses `pp.create_*` and therefore its column handling -
    a single missing column surfaces much later as an opaque KeyError from deep
    inside the solver. Filling from the installed version's own schema removes
    that failure mode and keeps the code working across pandapower generations,
    which differ in exactly these columns. Extra columns are preserved: they are
    how pandapower carries the optional OPF fields (min_p_mw, controllable, ...).
    """
    frame = pd.DataFrame(data, index=np.arange(n, dtype=np.int64))
    setattr(net, name, conform_to_schema(name, frame))


def snap_plants_to_buses(plants: pd.DataFrame, bus_xy_3035: np.ndarray) -> np.ndarray:
    """Nearest bus for each power plant, in EPSG:3035 metres.

    The D3.3 script builds its KDTree on raw (lon, lat) degrees, where a degree
    of longitude is cos(latitude) shorter than a degree of latitude - so
    "nearest" is systematically biased east-west, increasingly so towards the
    north. Projecting first removes that.
    """
    from pyproj import Transformer
    from scipy.spatial import KDTree

    tf = Transformer.from_crs(4326, 3035, always_xy=True)
    px, py = tf.transform(plants["lon"].to_numpy(), plants["lat"].to_numpy())
    _, idx = KDTree(bus_xy_3035).query(np.column_stack([px, py]))
    return idx.astype(np.int64)


def assign_generation(
    net,
    plants: pd.DataFrame,
    bus_xy_3035: np.ndarray,
    availability: float = 1.0,
) -> None:
    """One controllable generator per (bus, fuel), with a real merit order.

    Aggregating by (bus, fuel) rather than creating 29,565 individual units
    keeps the OPF variable count sane while preserving the cost structure that
    actually drives dispatch. `availability` derates nameplate capacity and is
    the `gen_availability` uncertainty factor - explicit, rather than the D3.3
    script's hidden flat 0.9x-of-load scale factor.
    """
    plants = plants.dropna(subset=["lat", "lon", "Capacity"]).copy()
    plants = plants[plants["Capacity"] > 0]
    plants["bus"] = snap_plants_to_buses(plants, bus_xy_3035)

    agg = (
        plants.groupby(["bus", "Fueltype"], as_index=False)["Capacity"]
        .sum()
        .rename(columns={"Capacity": "p_max"})
    )
    n = len(agg)
    _assign_table(net, "gen", {
        "name": (agg["bus"].astype(str) + "_" + agg["Fueltype"].astype(str)).to_numpy(),
        "bus": agg["bus"].to_numpy(),
        "p_mw": 0.0,
        "vm_pu": 1.0,
        "sn_mva": np.nan,
        "min_q_mvar": np.nan,
        "max_q_mvar": np.nan,
        "scaling": 1.0,
        "slack": False,
        "slack_weight": 0.0,
        "in_service": True,
        "type": None,
        "controllable": True,
        "min_p_mw": 0.0,
        "max_p_mw": agg["p_max"].to_numpy() * availability,
    }, n)
    costs = agg["Fueltype"].map(MARGINAL_COST).fillna(DEFAULT_MARGINAL_COST)
    _set_poly_cost(net, "gen", np.arange(n), costs.to_numpy())


def assign_loads(net, bus_load_mw: np.ndarray) -> None:
    """Fixed demand per bus. Zero-load buses are simply omitted."""
    active = np.flatnonzero(bus_load_mw > 0)
    _assign_table(net, "load", {
        "name": np.array([f"load_{b}" for b in active]),
        "bus": active,
        "p_mw": bus_load_mw[active],
        "q_mvar": 0.0,
        "const_z_percent": 0.0,
        "const_i_percent": 0.0,
        "sn_mva": np.nan,
        "scaling": 1.0,
        "in_service": True,
        "type": "wye",
        "controllable": False,
    }, len(active))


def add_shedding_units(net, voll: float = VOLL_EUR_MWH) -> np.ndarray:
    """A dispatchable injector at every load bus, capped at its demand.

    Returns the sgen indices, so ENS can be read back as
    `net.res_sgen.p_mw[idx].sum()` after the solve.
    """
    load_bus = net.load["bus"].to_numpy()
    load_mw = net.load["p_mw"].to_numpy()
    n = len(load_bus)
    _assign_table(net, "sgen", {
        "name": np.array([f"shed_{b}" for b in load_bus]),
        "bus": load_bus,
        "p_mw": 0.0,
        "q_mvar": 0.0,
        "sn_mva": np.nan,
        "scaling": 1.0,
        "in_service": True,
        "type": None,
        "current_source": True,
        "controllable": True,
        "min_p_mw": 0.0,
        "max_p_mw": load_mw,
    }, n)
    _set_poly_cost(net, "sgen", np.arange(n), np.full(n, voll))
    return np.arange(n, dtype=np.int64)


def add_slack(net, mode: str = "per_synchronous_area") -> np.ndarray:
    """Angle reference per electrical island.

    A DC OPF needs one reference bus per island. These are pure references -
    `min_p_mw = max_p_mw = 0` - so they cannot inject phantom energy. That
    matters: the D3.3 script places a real, unbounded `ext_grid` per country per
    component, which silently supplies any island that would otherwise shed and
    is one of the largest single determinants of its ENS numbers. Making the
    choice explicit is the `slack_placement` uncertainty factor.

    mode:
      per_synchronous_area - one reference per connected island (default)
      per_country          - one per (island, country), i.e. every country can
                             balance itself; closer to D3.3's behaviour
    """
    labels = island_labels(net)
    if mode == "per_synchronous_area":
        keys = labels
    elif mode == "per_country":
        zone = net.bus["zone"].fillna("").to_numpy()
        keys = np.array([f"{a}|{b}" for a, b in zip(labels, zone)])
    else:
        raise ValueError(f"unknown slack mode {mode!r}")

    # Prefer a bus that actually hosts generation, so the reference sits
    # somewhere electrically sensible.
    has_gen = np.zeros(len(net.bus), dtype=bool)
    if len(net.gen):
        has_gen[net.gen["bus"].to_numpy()] = True

    order = np.lexsort((~has_gen, keys))
    _, first = np.unique(keys[order], return_index=True)
    slack_buses = order[first]

    _assign_table(net, "ext_grid", {
        "name": np.array([f"ref_{b}" for b in slack_buses]),
        "bus": slack_buses,
        "vm_pu": 1.0,
        "va_degree": 0.0,
        "slack_weight": 1.0,
        "in_service": True,
        "min_p_mw": 0.0,
        "max_p_mw": 0.0,
    }, len(slack_buses))
    return slack_buses


def island_labels(net) -> np.ndarray:
    """Connected-component label per bus, over in-service lines and trafos."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n = len(net.bus)
    li = net.line[net.line["in_service"]]
    ti = net.trafo[net.trafo["in_service"]]
    rows = np.concatenate([li["from_bus"].to_numpy(), ti["hv_bus"].to_numpy()])
    cols = np.concatenate([li["to_bus"].to_numpy(), ti["lv_bus"].to_numpy()])
    adj = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    _, labels = connected_components(adj, directed=False)
    return labels


def _set_poly_cost(net, et: str, idx: np.ndarray, cp1: np.ndarray) -> None:
    """Append linear costs (cp1 * p) for one element type."""
    block = pd.DataFrame(
        {
            "element": idx.astype(np.int64),
            "et": et,
            "cp0_eur": 0.0,
            "cp1_eur_per_mw": cp1,
            "cp2_eur_per_mw2": 0.0,
            "cq0_eur": 0.0,
            "cq1_eur_per_mvar": 0.0,
            "cq2_eur_per_mvar2": 0.0,
        }
    )
    existing = net.poly_cost
    net.poly_cost = (
        block if existing is None or not len(existing)
        else pd.concat([existing, block], ignore_index=True)
    )
    net.poly_cost.index = np.arange(len(net.poly_cost), dtype=np.int64)