"""Assign every lev07 basin to exactly one country run, and map LAUs to basins.

Two precomputations, both one-off, both written to `cache/`.

**Why ownership is needed.** A basin event now fails every component in the
basin, Europe-wide - it has to, because masking to the study country meant a
basin straddling FR/DE had only its French half fail in the FR run and only its
German half in the DE run, so neither run modelled the actual event and summing
them both double-counted the basin and never represented it whole. Once events
are Europe-wide, the same basin would be evaluated by every run whose network
touches it, so ownership is what keeps the European total a sum rather than an
overlap.

Ownership goes to the country holding the most grid components in the basin,
with area share as the tie-break. That rule is arbitrary at the margin but it is
a partition: every basin has exactly one owner, so concatenating the runs gives
each basin once and only once.

**Why LAUs.** Basins are the event grain but they are large and unfamiliar;
LAUs are the reporting grain for maps. Each LAU is assigned to the lev07 basin
covering the largest share of its area. This is a cartographic convenience and
nothing more - it smears a basin-level number across administrative units, and
the ENS was never computed per LAU. The `attribution` table is the honest
per-location record.

Run:
    python scripts/build_basin_owner.py [--countries AUT EST HRV HUN ITA NLD ROU]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LAU_PATH = "S:/eks510/projects/AssetRisk_PanEU/data/LAU_RG_01M_2024_3035.parquet"
LEV07_PATH = ("S:/eks510/projects/AssetRisk_PanEU/data/"
              "basins_abs_shift_return_periods_lev07.parquet")

# HRV replaces ALB. Albania has no OPSD demand series (`NO_OPSD_DEMAND`), so
# there is no load to place on its buses and it cannot be run at all - not a
# bug to fix here but a data gap, and one that only surfaced when this script
# tried to build all seven. Croatia is the closest substitute on every axis ALB
# was chosen for: Western Balkan, Adriatic coast, seismically active, Sava/Drava
# river flooding, below-EU-average income, and 25 buses against ALB's 22 - so
# the cost budget behind the N=128 gate is unchanged. Montenegro is a nearer
# match on income and size but has only 12 buses, too small to produce the
# capacity-constrained shedding this study exists to measure; Bulgaria
# duplicates ROU's Black Sea coast; Greece's island-heavy topology would make
# it an outlier for reasons unrelated to hazard.
SEVEN = ["AUT", "EST", "HRV", "HUN", "ITA", "NLD", "ROU"]


def build_ownership(countries: list[str], out: Path,
                    freeze: Path | None = None) -> pd.DataFrame:
    """One row per lev07 basin: its owner, and how the decision was made.

    `freeze` extends an existing partition instead of replacing it: every row
    of that table is kept verbatim, and only basins it does not contain are
    assigned, by the same majority rule, among `countries`. This is what lets
    countries be added to a campaign that is already running - each run reads
    only its own country's rows when it starts, so as long as those rows are
    unchanged, runs queued before the rebuild and runs queued after it evaluate
    one consistent partition. The price is that a border basin keeps its
    earlier owner even where a newly added neighbour holds more components in
    it; the rule becomes "earlier countries first, then majority", which is
    still a partition, so the European total stays a sum.
    """
    from power_net.ens_model import load_net_data
    from power_net.linkage import BUS, LINE
    from power_net.paths import load_config

    frozen = None
    if freeze is not None:
        # Checked before loading anything: loading 22 countries takes a while.
        frozen = pd.read_parquet(freeze)
        clash = sorted(set(countries) & set(frozen["country"]))
        if clash:
            raise SystemExit(f"{clash} already own basins in {freeze}; pass only "
                             "the countries being added")

    rows = []
    for c in countries:
        cfg = load_config()
        cfg["country"] = c
        cfg["scenario"] = "net_flood"
        try:
            data = load_net_data(cfg)
        except Exception as e:                                   # noqa: BLE001
            print(f"  {c}: SKIPPED ({type(e).__name__}: {str(e)[:60]})")
            continue
        lab = data.footprints["basin_lev07"]
        # Count only the country's OWN components per basin - that is what
        # "whose basin is this" means. Membership for failure is Europe-wide;
        # ownership is about which run is responsible for evaluating it.
        for kind, mask in ((BUS, data.country_bus), (LINE, data.country_line)):
            ids = lab[kind][mask]
            ids = ids[ids >= 0]
            if len(ids):
                v = pd.Series(ids).value_counts()
                rows.append(pd.DataFrame({"event_basin": v.index,
                                          "country": c,
                                          "n_comp": v.to_numpy()}))
        print(f"  {c}: {len(np.unique(lab[BUS][data.country_bus])):4d} bus basins")

    if not rows:
        raise SystemExit("no countries produced basins")
    tally = (pd.concat(rows, ignore_index=True)
             .groupby(["event_basin", "country"], as_index=False)["n_comp"].sum())
    if frozen is not None:
        n_all = tally["event_basin"].nunique()
        tally = tally[~tally["event_basin"].isin(frozen["event_basin"])]
        print(f"  freeze: keeping {len(frozen):,} basins of "
              f"{frozen['country'].nunique()} countries verbatim; "
              f"{n_all - tally['event_basin'].nunique():,} of the new countries' "
              "basins are already owned and stay so")
    # Highest component count wins; ties broken by country code so the result is
    # reproducible rather than dependent on row order.
    tally = tally.sort_values(["event_basin", "n_comp", "country"],
                              ascending=[True, False, True])
    owner = tally.drop_duplicates("event_basin", keep="first").copy()
    owner["contested"] = owner["event_basin"].isin(
        tally[tally.duplicated("event_basin", keep=False)]["event_basin"]
    )
    if frozen is not None:
        owner = pd.concat([frozen, owner[frozen.columns]], ignore_index=True)
        kept = owner.iloc[:len(frozen)].reset_index(drop=True)
        if not kept.equals(frozen.reset_index(drop=True)):
            raise SystemExit("frozen rows changed - refusing to write")
    # Atomic: runs already queued read this file when they start, and must see
    # either the old table or the new one, never a half-written file.
    tmp = out.with_suffix(".parquet.tmp")
    owner.to_parquet(tmp, index=False)
    os.replace(tmp, out)
    print(f"\nwrote {out}: {len(owner):,} basins, "
          f"{int(owner.contested.sum()):,} contested by >1 country")
    print(owner.groupby("country").size().to_string())
    return owner


def build_lau_basin(out: Path) -> None:
    """Each LAU -> the lev07 basin covering the largest share of its area."""
    import geopandas as gpd

    lau = gpd.read_parquet(LAU_PATH)[["GISCO_ID", "CNTR_CODE", "geometry"]]
    bas = gpd.read_parquet(LEV07_PATH)
    # PFAF_ID, NOT HYBAS_ID. `_footprint_labels` builds `basin_lev07` as the
    # lev08 PFAF_ID with its last digit dropped - Pfafstetter codes nest by
    # prefix - which yields the lev07 file's 7-digit PFAF_ID. Verified: the two
    # sets overlap on all 3,335 basins, while the lev07 HYBAS_ID (a 10-digit
    # 207xxxxxxx code) overlaps on ZERO. Joining on HYBAS_ID silently assigns
    # nothing instead of failing.
    bas = bas[["PFAF_ID", "geometry"]].rename(columns={"PFAF_ID": "event_basin"})
    print(f"LAU {len(lau):,} rows, lev07 basins {len(bas):,} rows; "
          f"CRS {lau.crs} / {bas.crs}")
    if lau.crs != bas.crs:
        bas = bas.to_crs(lau.crs)

    ov = gpd.overlay(lau, bas, how="intersection", keep_geom_type=False)
    ov["share"] = ov.geometry.area
    ov = ov.sort_values(["GISCO_ID", "share"], ascending=[True, False])
    best = ov.drop_duplicates("GISCO_ID", keep="first").copy()
    # Share of the LAU's own area, so a value near 1 means the LAU sits almost
    # entirely inside one basin and a low value flags a smeared assignment.
    lau_area = lau.set_index("GISCO_ID").geometry.area
    best["area_frac"] = (best["share"].to_numpy()
                         / lau_area.reindex(best["GISCO_ID"]).to_numpy())
    if best.empty:
        raise SystemExit("overlay produced no assignments - check the join key")
    best[["GISCO_ID", "CNTR_CODE", "event_basin", "share",
          "area_frac"]].to_parquet(out, index=False)
    print(f"wrote {out}: {len(best):,} LAUs assigned "
          f"({len(lau) - len(best):,} unassigned - outside every basin)")
    print(f"  area fraction in the chosen basin: median "
          f"{best.area_frac.median():.2f}, "
          f"{(best.area_frac < 0.5).sum():,} LAUs below 0.5 "
          f"(mostly split across basins - the map is a smear there)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--countries", nargs="+", default=SEVEN)
    ap.add_argument("--skip-lau", action="store_true")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--extend", action="store_true",
                    help="keep the existing basin_owner.parquet verbatim and add "
                         "--countries for the basins it does not yet own")
    args = ap.parse_args()
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)

    print("=== basin ownership ===")
    out = cache / "basin_owner.parquet"
    if args.extend and not out.is_file():
        raise SystemExit(f"--extend needs an existing {out}")
    build_ownership(args.countries, out, freeze=out if args.extend else None)
    if not args.skip_lau:
        print("\n=== LAU -> basin ===")
        build_lau_basin(cache / "lau_basin.parquet")


if __name__ == "__main__":
    main()
