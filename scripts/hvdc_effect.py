"""What reading the HVDC layer actually changes.

Four configurations of the same model, so the two switches can be told apart:

    base            as the study runs today
    +hvdc           links.csv + converters.csv wired in
    +noload         converter terminals excluded from the load split
    +both           the intended end state

Reports connected components, the structural floor, and one real EAENS
evaluation, because a change in topology is only interesting if it moves the
number the study reports.

Run:
    python scripts/hvdc_effect.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from power_net.ens_model import _islands, compute_ens, load_net_data
from power_net.paths import load_config

KW = dict(include_river=True, event_footprint="basin",
          failure_correlation=0.3, gen_availability=0.9, failure_seed=7,
          outage_draw=0.5, flood_mapping="lognormal",
          flood_frag_median=0.6, flood_frag_beta=0.4,
          flood_threshold=0.3, depth_scale=1.0,
          protection_abs_rp=100.0, warming="current")

CONFIGS = [("base",   False, False),
           ("+hvdc",  True,  False),
           ("+noload", False, True),
           ("+both",  True,  True)]

rows = []
for name, hvdc, noload in CONFIGS:
    cfg = load_config()
    cfg["country"] = "LUX"
    n = cfg.setdefault("network", {})
    n["opf_backend"] = "highs"
    n["dispatch_objective"] = "min_shed"
    n["include_hvdc"] = hvdc
    n["exclude_dc_terminal_load"] = noload
    t = time.time()
    data = load_net_data(cfg)
    build = time.time() - t

    lab = _islands(data, np.zeros(data.n_line, bool), np.zeros(data.n_bus, bool))
    ncomp = int(lab.max()) + 1
    sizes = np.bincount(lab)
    total_load = float(data.bus_load.sum())

    out = {}
    for model in ("connectivity", "dc_opf"):
        data._memo.clear()
        out[model] = compute_ens(data, network_model=model, **KW)

    rows.append({
        "config": name, "hvdc": hvdc, "no_dc_load": noload,
        "components": ncomp, "largest_island": int(sizes.max()),
        "singletons": int((sizes == 1).sum()),
        "total_load_MW": total_load,
        "structural_floor_MW": data.structural_floor_mw,
        "EAENS_conn": out["connectivity"]["EAENS_MWh"],
        "EAENS_dcopf": out["dc_opf"]["EAENS_MWh"],
        "build_s": build,
    })
    r = rows[-1]
    print(f"{name:8s} components {r['components']:4d} (largest {r['largest_island']}, "
          f"{r['singletons']} singletons)  load {total_load:11,.0f} MW  "
          f"floor {r['structural_floor_MW']:9,.1f} MW  "
          f"EAENS conn {r['EAENS_conn']:10,.3f} / opf {r['EAENS_dcopf']:10,.3f} MWh",
          flush=True)

df = pd.DataFrame(rows)
out_dir = Path("results/diagnostics"); out_dir.mkdir(parents=True, exist_ok=True)
df.to_csv(out_dir / "hvdc_effect.csv", index=False)
print(f"\nwrote {out_dir / 'hvdc_effect.csv'}")
print(df.to_string(index=False))
