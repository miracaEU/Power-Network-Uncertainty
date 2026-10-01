"""Assign every grid component to a lev07 basin, pan-European.

**Why this exists.** Basin labels used to come from `HYBAS_ID` in the Stage 1
exposure table, which `hazard.py` loads for ONE country. So `basin_lev07`
described only the study country: measured on HRV, 20 of 6,737 buses carried a
label and 6,717 were -1. Three things broke as a result.

  1. Events were not Europe-wide. `compute_ens` matches a basin's members with
     `labels[BUS] == fid`; a label that only exists inside the study country
     cannot reach across a border, so removing the country mask changed nothing.
  2. The basin ownership partition had nothing to partition, because no basin
     could be shared between two runs.
  3. The attribution table was empty. Shed is recorded against
     `basin_of_bus[shed_bus]`, and with 6,549 of 6,568 shed-capable buses
     unlabelled, every affected_basin came out -1 - the one output that cannot
     be reconstructed without re-running the campaign.

**The level was never wrong.** Stage 1's `HYBAS_ID` is HydroBASINS level 08 and
`_footprint_labels` derives lev07 by Pfafstetter-prefix truncation
(`PFAF_ID // 10`). Checked against an independent point-in-polygon join of the
bus coordinates onto the lev07 polygons: **20 of 20 agree, 0 disagree.** So the
derivation is sound and exposure does NOT need recomputing - only the coverage
was missing, and coordinates plus polygons are enough to supply it.

**What this does not fix.** A component outside the study country still has no
Stage 1 hazard intensity, so it carries all-zero anchors and is immune. Labels
alone therefore make attribution correct and events *addressable* Europe-wide,
but a foreign component still cannot fail until its country's Stage 1 profiles
are loaded too. That is a separate, larger change; this one is a prerequisite
for it and is worth having on its own for the attribution table.

Run:
    python scripts/build_component_basins.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LEV07_PATH = ("S:/eks510/projects/AssetRisk_PanEU/data/"
              "basins_abs_shift_return_periods_lev07.parquet")

# A coastal substation can sit just outside every basin polygon - the polygons
# stop at the coastline, the grid does not. Snapping those to the nearest basin
# within this distance is a CARTOGRAPHIC assignment for reporting, never a claim
# that the basin's hazard applies to them; failure is still driven entirely by
# each component's own Stage 1 intensity. Matches `network.snap_max_m`.
NEAREST_MAX_M = 25_000.0


def _assign(pts, bas):
    """lev07 id per point: containment first, nearest within tolerance after."""
    import geopandas as gpd

    j = gpd.sjoin(pts, bas, how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")].sort_index()
    out = j["PFAF_ID"].to_numpy()
    out = np.where(pd.isna(out), -1, out).astype(np.int64)
    dist = np.zeros(len(out))
    method = np.where(out >= 0, "within", "unassigned").astype(object)

    miss = np.flatnonzero(out < 0)
    if len(miss):
        near = gpd.sjoin_nearest(pts.iloc[miss], bas, how="left",
                                 max_distance=NEAREST_MAX_M,
                                 distance_col="_d")
        near = near[~near.index.duplicated(keep="first")].sort_index()
        v = near["PFAF_ID"].to_numpy()
        d = near["_d"].to_numpy()
        ok = ~pd.isna(v)
        idx = miss[ok]
        out[idx] = v[ok].astype(np.int64)
        dist[idx] = d[ok]
        method[idx] = "nearest"
    return out, dist, method


def main() -> None:
    import geopandas as gpd

    from power_net.network import bus_coords_3035
    from power_net.paths import load_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache")
    args = ap.parse_args()
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)

    cfg = load_config()
    nsd = Path(cfg["energy_model_dir"]) / "Net_structure_data"
    geo = pd.read_csv(nsd / "bus_geodata.csv")
    line = pd.read_csv(nsd / "line_data.csv")
    trafo = pd.read_csv(nsd / "trafo_data.csv")
    xy = bus_coords_3035(geo)
    print(f"buses {len(xy):,}   lines {len(line):,}   trafos {len(trafo):,}")

    bas = gpd.read_parquet(LEV07_PATH)[["PFAF_ID", "geometry"]]
    print(f"lev07 polygons {len(bas):,}, CRS EPSG:{bas.crs.to_epsg()}")

    rows = []

    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(xy[:, 0], xy[:, 1]),
                           crs=3035)
    if bas.crs != pts.crs:
        bas = bas.to_crs(pts.crs)
    bid, bdist, bmeth = _assign(pts, bas)
    rows.append(pd.DataFrame({"kind": "bus", "idx": np.arange(len(bid)),
                              "event_basin": bid, "dist_m": bdist,
                              "method": bmeth}))
    print(f"\nbus   : {(bid >= 0).sum():,} assigned "
          f"({(bmeth == 'within').sum():,} within, "
          f"{(bmeth == 'nearest').sum():,} snapped), "
          f"{(bid < 0).sum():,} unassigned")

    # A line spans basins; its midpoint picks one. That is the same granularity
    # Stage 1 imposes by giving each segment a single HYBAS_ID, so this keeps
    # the two routes comparable rather than inventing a finer rule.
    mid = 0.5 * (xy[line["from_bus"].to_numpy(np.int64)]
                 + xy[line["to_bus"].to_numpy(np.int64)])
    p = gpd.GeoDataFrame(geometry=gpd.points_from_xy(mid[:, 0], mid[:, 1]),
                         crs=3035)
    lid, ldist, lmeth = _assign(p, bas)
    rows.append(pd.DataFrame({"kind": "line", "idx": np.arange(len(lid)),
                              "event_basin": lid, "dist_m": ldist,
                              "method": lmeth}))
    print(f"line  : {(lid >= 0).sum():,} assigned "
          f"({(lmeth == 'within').sum():,} within, "
          f"{(lmeth == 'nearest').sum():,} snapped), "
          f"{(lid < 0).sum():,} unassigned")

    # Transformers are NOT independent components in the hazard tables -
    # `linkage` gives each one its HV bus (`trafo_hv_bus`). Taking a midpoint
    # here would quietly contradict that, so the cache reuses the HV bus's
    # basin and stays consistent with how the model treats them.
    hv = trafo["hv_bus"].to_numpy(np.int64)
    rows.append(pd.DataFrame({"kind": "trafo", "idx": np.arange(len(hv)),
                              "event_basin": bid[hv], "dist_m": bdist[hv],
                              "method": bmeth[hv]}))
    print(f"trafo : {(bid[hv] >= 0).sum():,} assigned (inherited from HV bus), "
          f"{(bid[hv] < 0).sum():,} unassigned")

    out = pd.concat(rows, ignore_index=True)
    path = cache / "component_basin.parquet"
    out.to_parquet(path, index=False)
    print(f"\nwrote {path}: {len(out):,} rows, "
          f"{out.loc[out.event_basin >= 0, 'event_basin'].nunique():,} "
          f"distinct lev07 basins")


if __name__ == "__main__":
    main()
