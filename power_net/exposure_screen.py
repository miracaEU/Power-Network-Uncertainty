"""Which (country, hazard) combinations are structural zeros, and why.

`applicable_hazards()` answers whether a hazard CAN reach a country (coastal
needs a coast). It does not answer whether anything the network model can fail
is actually exposed, and several coastal combinations are certain zeros:

    ROU  coastal   Stage 1 profile exists but holds 0 rows
    EST  coastal   230 exposed power segments (poles, minor lines) - none of
                   them part of a transmission-grid component
    NLD  coastal   2 exposed segments (1 plant, 1 line), same

A Sobol run on such a combination spends its full walltime and returns
EAENS = 0 in every draw and all-zero (or NaN) indices. The zero is exact, not
a sampling result: `compute_ens` evaluates only this country's owned lev07
basins, and a component with intensity 0 at every return period has
P(fail) = 0 under every factor setting - `depth_scale` multiplies intensity,
`warming` shifts return periods, neither can make a zero positive.

The categories mirror the direct-damage study's (`plot_drivers.NO_EXPOSURE`)
so the two papers report the same thing the same way, plus one that only a
network model has:

    no_exposure       Stage 1 found no power asset in the hazard footprint.
                      Same meaning as the direct-damage "No exposure".
    no_grid_exposure  Stage 1 assets ARE exposed, but none of them belongs to
                      a grid component in this country's owned basins -
                      typically distribution assets (poles, minor lines)
                      that the transmission model does not represent.
    exposed           at least one grid component can fail; run it.
    unknown           a cache is missing, so the screen cannot decide; run it
                      (submit_network_study.sh refuses to submit without the
                      caches anyway).

Each skipped combination leaves a JSON record in results/<ISO3>/ so the result
is an explicit "EAENS = 0, reason X" rather than a missing file.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

NO_EXPOSURE = "no_exposure"
NO_GRID_EXPOSURE = "no_grid_exposure"
EXPOSED = "exposed"
UNKNOWN = "unknown"
STRUCTURAL_ZEROS = (NO_EXPOSURE, NO_GRID_EXPOSURE)

LABELS = {
    NO_EXPOSURE: "No exposure",
    NO_GRID_EXPOSURE: "Exposed assets, no exposed grid component",
    EXPOSED: "Exposed",
    UNKNOWN: "Unknown (cache missing)",
}


def _stage1_exposed_segments(cfg: dict, hazard: str) -> int | None:
    path = (Path(cfg["intermediate_dir"])
            / f"{cfg['country']}_{cfg['asset_type']}_{hazard}_profiles.parquet")
    if not path.is_file():
        return None
    prof = pd.read_parquet(path, columns=["seg", "intensity"])
    return int(prof.loc[prof["intensity"] > 0, "seg"].nunique())


def screen(cfg: dict, hazard: str) -> dict:
    """Classify one (cfg['country'], hazard). Cheap: parquet/npz reads only."""
    out = {"country": cfg["country"], "hazard": hazard,
           "stage1_exposed_segments": _stage1_exposed_segments(cfg, hazard),
           "grid_buses_exposed": None, "grid_lines_exposed": None}
    if out["stage1_exposed_segments"] == 0:
        out["status"] = NO_EXPOSURE
        return out

    cache = Path(cfg["cache_dir"])
    need = [cache / "component_basin.parquet", cache / "basin_owner.parquet",
            cache / f"hazard_paneu_{hazard}.npz"]
    if not all(p.is_file() for p in need):
        out["status"] = UNKNOWN
        return out

    owner = pd.read_parquet(need[1])
    owned = owner.loc[owner["country"] == cfg["country"], "event_basin"]
    comp = pd.read_parquet(need[0])
    comp = comp[comp["event_basin"].isin(owned)]
    z = np.load(need[2])
    for kind, key in (("bus", "grid_buses_exposed"), ("line", "grid_lines_exposed")):
        hit = (np.nan_to_num(z[f"{kind}_intensity"]) > 0).any(axis=1)
        idx = comp.loc[comp["kind"] == kind, "idx"].to_numpy(np.int64)
        out[key] = int(hit[idx].sum())
    n = out["grid_buses_exposed"] + out["grid_lines_exposed"]
    out["status"] = EXPOSED if n else NO_GRID_EXPOSURE
    return out


def record_structural_zero(cfg: dict, scenario: str, info: dict) -> Path:
    """Write the explicit "EAENS = 0, and why" record for a skipped combination."""
    from .paths import country_results_dir

    path = country_results_dir(cfg) / f"structural_zero_{cfg['country']}_{scenario}.json"
    rec = dict(info, scenario=scenario, label=LABELS[info["status"]],
               EAENS_MWh=0.0, exact=True)
    path.write_text(json.dumps(rec, indent=2))
    return path
