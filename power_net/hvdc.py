"""The HVDC layer: DC links and the converters that couple them to the AC grid.

`build_base_network` reads bus/line/trafo/geodata and nothing else, so the DC
layer has never been in the model. The consequences are large and were measured
on 2026-09-14:

  * The 70 DC-side converter terminals appear as **degree-0 buses** - no line,
    no transformer, in service, each its own connected component. All 70 are
    endpoints in `links.csv`; none is a genuinely orphaned substation.
  * The load disaggregation spreads national demand across every bus in a
    country, so those DC terminals receive **6,766 MW (1.36% of European
    demand)** they should never hold.
  * With no supply path the OPF sheds 100% of it, in every scenario and at every
    hour. 17 of them shed in all 24 scenarios tested and carry **53% of all shed
    MW** - a large share of the 3.3 GW "structural floor".
  * Adding the links as edges collapses the topology from **76 connected
    components to 40**.

How the two files join, which is not obvious from the column names:

    links.csv       bus0, bus1  are BOTH DC terminals   (bare OSM id)
    converters.csv  bus0        is the DC terminal      (bare OSM id)
                    bus1        is the AC bus           (OSM id + "-<kV>")

so a complete path is  AC bus -> converter -> DC terminal -> link -> DC terminal
-> converter -> AC bus. Reading `links.csv` alone would only build 2-bus DC
islands that never touch the AC network; the converters are what connect it.

Matching is by EXACT bus name. An earlier diagnostic matched on the OSM id with
the voltage suffix stripped, preferring the highest voltage - a guess, and the
wrong one: the DC terminal `relation/10629005` and the AC bus
`relation/10629005-400` share a stripped id, so the guess silently chose between
them. 38 of the 70 terminals have such a twin. `converters.csv` states the AC
connection point outright, so there is nothing to infer.

In the LP both files describe the same kind of object - a controllable transfer
bounded by `p_nom` - so they are returned as one table and enter `dcopf_lp` as
injection pairs. The connectivity model has no notion of capacity and treats
them as plain edges, which OVERSTATES what a 1,400 MW link can do; that
difference between the two network models is real and is left visible rather
than hidden.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_hvdc(em_dir: Path, net) -> pd.DataFrame:
    """DC links and converters as capacity-bounded transfer elements.

    Returns one frame with `from_bus`, `to_bus` (pandapower indices), `p_nom`
    (MW) and `kind` ("link" or "converter"). Rows whose endpoints are not both
    present in the network are dropped and counted in `.attrs["dropped"]`.
    """
    ns = Path(em_dir) / "Net_structure_data"
    idx = pd.Series(np.arange(len(net.bus)),
                    index=net.bus["name"].astype(str)).to_dict()

    rows, dropped = [], {"link": 0, "converter": 0}
    lk = pd.read_csv(ns / "links.csv",
                     usecols=["bus0", "bus1", "p_nom", "under_construction"])
    lk = lk[~lk["under_construction"].astype(bool)]
    for _, r in lk.iterrows():
        f, t = idx.get(str(r.bus0)), idx.get(str(r.bus1))
        if f is None or t is None or f == t:
            dropped["link"] += 1
            continue
        rows.append((f, t, float(r.p_nom), "link"))

    cv = pd.read_csv(ns / "converters.csv", usecols=["bus0", "bus1", "p_nom"])
    for _, r in cv.iterrows():
        f, t = idx.get(str(r.bus0)), idx.get(str(r.bus1))
        if f is None or t is None or f == t:
            dropped["converter"] += 1
            continue
        rows.append((f, t, float(r.p_nom), "converter"))

    out = pd.DataFrame(rows, columns=["from_bus", "to_bus", "p_nom", "kind"])
    out.attrs["dropped"] = dropped
    return out


def dc_terminal_mask(em_dir: Path, net) -> np.ndarray:
    """Buses that are DC-side converter terminals.

    A converter terminal is a station, not a demand centre: it has no local
    load whether or not the DC layer is modelled. Assigning it a share of
    national demand is wrong in either configuration, which is why the switch
    that excludes it is independent of the one that reads the links.

    Identified as `converters.csv` bus0 plus every `links.csv` endpoint - the
    union, because 3 of the 70 terminals appear in links.csv but in no
    converter row, and those would otherwise keep their phantom load.
    """
    ns = Path(em_dir) / "Net_structure_data"
    names = pd.read_csv(ns / "converters.csv", usecols=["bus0"])["bus0"].astype(str)
    lk = pd.read_csv(ns / "links.csv", usecols=["bus0", "bus1"])
    ids = set(names) | set(lk.bus0.astype(str)) | set(lk.bus1.astype(str))
    bus_name = net.bus["name"].astype(str).to_numpy()
    return np.isin(bus_name, list(ids))
