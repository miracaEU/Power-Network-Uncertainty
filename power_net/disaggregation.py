"""Per-bus load disaggregation from NUTS3 population and GDP.

`bus_data.csv` carries no load column, so national demand has to be placed on
buses by some rule. The default rule - split a country's demand equally across
its buses - is what D3.3 does, and it is defensible only as a placeholder: it
puts load where none exists, which is the leading suspect for the 12.9% of
lines that exceed 100% loading before any hazard is applied (BENCHMARK.md).
Because ENS *is* "how much load sat at the buses that got disconnected", this
assumption sits directly underneath every number the study produces.

This module implements the alternative: weight each bus by the population
and/or GDP of the NUTS3 region it falls in, which is what PyPSA-Eur does
upstream when it builds the load it later aggregates away.

Three things are deliberate:

  * **A region's weight is shared equally among the buses inside it.** Within a
    NUTS3 region there is no further information about where load sits, so the
    equal split is pushed down one level rather than removed. The improvement
    is that it is now equal *within a region of known demand* instead of equal
    across a whole country.
  * **Regions with no bus are not dropped, they are redistributed.** Shares are
    normalised across the regions that actually contain a bus, so a region's
    demand goes proportionally to the rest of its country rather than
    vanishing. `weight_in_busless_regions` in the report says how much moved.
  * **A country either gets real weights or it gets the equal split, never a
    mixture.** If any bus-containing region in a country lacks Eurostat data,
    the whole country falls back and says so. Assigning a missing region zero
    weight would silently move its load onto its neighbours; inventing a value
    would repeat D3.3's `.fillna(500.0)`, the traceable origin of its
    245,864 MW artefact.

Coverage, measured 2026-09-09 on the pan-European network (6,737 buses): 5,824
buses land inside a NUTS3 polygon of their own country directly. Of the 73
near-misses in countries that have NUTS3 geometry, 50 sit within 3 km of their
own country's boundary - generalisation artefacts of the 1:1M polygons - and
are recovered by the nearest-polygon snap. The remaining 23 are 15-62 km
offshore and are all synthetic PyPSA-Eur wind-farm connection nodes
(NL2-220, BE1-220, DE2-220, DK7-220 and similar); they are given zero load
weight, which is what a population/GDP disaggregation means at sea.

The countries that fall back to the equal split are those with no usable
Eurostat NUTS3 statistics - GB, UA, BA, MD (no NUTS3 geometry at all), plus NO
and XK (geometry present, statistics absent). NO is a NUTS vintage mismatch:
8 of its 17 NUTS 2024 regions are splits that the Eurostat export still
reports under NUTS 2021 codes, and no correspondence table ships with the data
drop. All four current study countries - LUX, PRT, SVN, EST - are fully
covered.

Note that Luxembourg is a single NUTS3 region (LU000), so for LUX these
weights reproduce the equal split exactly. That is the correct answer, not a
failure: NUTS3 carries no sub-national information for LUX.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

# Grid `zone` -> NUTS country code. Eurostat writes Greece as EL, and the grid
# model (like OPSD and ISO 3166) writes it as GR.
NUTS_ZONE_ALIAS = {"GR": "EL"}

NUTS_SHP = "NUTS_RG_01M_2024_4326.shp/NUTS_RG_01M_2024_4326.shp"
POP_CSV = "estat_nama_10r_3popgdp$defaultview_filtered_en.csv"
GDP_CSV = "estat_nama_10r_3gdp$defaultview_filtered_en.csv"

# Eurostat unit codes to select. Population comes as thousands of persons; GDP
# is published in several units and only MIO_EUR is a regional *total* - the
# EUR_HAB default view is per inhabitant and would need multiplying by
# population to be an extensive quantity.
POP_UNIT = "THS"
GDP_UNIT = "MIO_EUR"

# Reference year for the statistics. 2021 has the widest joint NUTS3 coverage
# in this export (GDP 1,343 regions, population 1,309); 2023 is still being
# filled in and covers roughly half.
DEFAULT_STATS_YEAR = 2021

MODES = ("equal", "nuts3", "nuts3_pop", "nuts3_gdp")


@lru_cache(maxsize=2)
def _nuts3_polygons(em_dir: str):
    """NUTS3 polygons in EPSG:4326, cached per process.

    Cached for the same reason the OPSD frame is: every spawned worker
    re-imports this module and would otherwise re-read the shapefile.
    """
    import geopandas as gpd

    g = gpd.read_file(Path(em_dir) / NUTS_SHP)
    g = g[g["LEVL_CODE"] == 3][["NUTS_ID", "CNTR_CODE", "geometry"]]
    return g.reset_index(drop=True)


def _eurostat_series(em_dir: str, filename: str, unit: str) -> pd.Series:
    """Latest observation at or before the reference year, per NUTS3 region.

    Taking the most recent available observation rather than insisting on one
    fixed year buys back regions that simply report late, without inventing
    anything: every value returned is a real published figure for that region.
    """
    d = pd.read_csv(Path(em_dir) / filename)
    d = d[(d["geo"].astype(str).str.len() == 5) & (d["unit"] == unit)]
    d = d.dropna(subset=["OBS_VALUE"])
    d = d.sort_values("TIME_PERIOD")
    return d.groupby("geo")["OBS_VALUE"].last()


@lru_cache(maxsize=4)
def _region_stats(em_dir: str, year: int) -> pd.DataFrame:
    """Population (thousands) and GDP (MEUR) per NUTS3 region."""
    pop = _eurostat_series(em_dir, POP_CSV, POP_UNIT)
    gdp = _eurostat_series(em_dir, GDP_CSV, GDP_UNIT)
    return pd.DataFrame({"pop": pop, "gdp": gdp})


def bus_nuts3_ids(net, geo: pd.DataFrame, em_dir: Path,
                  snap_max_m: float = 25_000.0) -> np.ndarray:
    """The NUTS3 region each bus falls in, "" where none applies.

    A bus is only ever matched to a region of its own country: a point-in-
    polygon test near a border would otherwise hand a French bus to a Belgian
    region and move load across a national demand series. Buses that fall
    outside every polygon of their own country - almost all of them coastal,
    where the 1:1M generalised boundary cuts inside the shoreline - are snapped
    to the nearest polygon of that country within `snap_max_m`.
    """
    import geopandas as gpd

    n3 = _nuts3_polygons(str(em_dir))
    zone = net.bus["zone"].fillna("").astype(str).to_numpy()
    want = np.array([NUTS_ZONE_ALIAS.get(z, z) for z in zone])

    pts = gpd.GeoDataFrame(
        {"want": want},
        geometry=gpd.points_from_xy(geo["x"].to_numpy(), geo["y"].to_numpy()),
        crs=4326,
    )
    j = gpd.sjoin(pts, n3, how="left", predicate="within")
    # A point on a shared border can match two polygons; keep the first so the
    # result stays row-aligned with net.bus.
    j = j[~j.index.duplicated(keep="first")]

    out = j["NUTS_ID"].where(j["CNTR_CODE"] == j["want"]).to_numpy(dtype=object)
    out = np.array(["" if pd.isna(v) else str(v) for v in out], dtype=object)

    # Nearest-polygon fallback, per country and in metres.
    todo = out == ""
    have_geom = set(n3["CNTR_CODE"].unique())
    for cc in sorted(set(want[todo]) & have_geom):
        mask = todo & (want == cc)
        if not mask.any():
            continue
        left = pts.loc[mask].to_crs(3035)
        right = n3[n3["CNTR_CODE"] == cc].to_crs(3035)
        near = gpd.sjoin_nearest(left, right, how="left",
                                 max_distance=snap_max_m)
        near = near[~near.index.duplicated(keep="first")]
        got = near["NUTS_ID"]
        out[near.index.to_numpy()] = [
            "" if pd.isna(v) else str(v) for v in got.to_numpy()
        ]
    return out


def nuts3_bus_weights(net, geo: pd.DataFrame, em_dir: Path, *,
                      mode: str = "nuts3",
                      year: int = DEFAULT_STATS_YEAR,
                      gdp_weight: float = 0.5,
                      snap_max_m: float = 25_000.0
                      ) -> tuple[np.ndarray, dict]:
    """Per-bus weights for `demand.bus_load_mw`, plus a coverage report.

    Returns (weights, report). A bus whose country has no usable statistics is
    given `np.nan`, which `bus_load_mw` reads as "no information" and answers
    with the equal split for that whole country.

    `gdp_weight` blends the two shares within a country, w = (1-x)*pop + x*gdp.
    It only applies to mode "nuts3"; the pure variants exist so the blend
    itself can be tested rather than assumed. Electricity demand tracks
    population and economic activity in different proportions by sector, and
    this study has no basis for asserting which - a good reason to keep it a
    stated, movable number rather than a constant buried in the code.
    """
    if mode not in MODES:
        raise ValueError(f"unknown load_disaggregation '{mode}'; choose from {list(MODES)}")
    if mode == "equal":
        raise ValueError("nuts3_bus_weights called with mode 'equal'")
    if not 0.0 <= gdp_weight <= 1.0:
        raise ValueError(f"nuts3_gdp_weight must be in [0, 1], got {gdp_weight}")

    region = bus_nuts3_ids(net, geo, em_dir, snap_max_m)
    stats = _region_stats(str(em_dir), year)
    zone = net.bus["zone"].fillna("").astype(str).to_numpy()

    weights = np.full(len(net.bus), np.nan)
    pop_bus = np.full(len(net.bus), np.nan)
    gdp_bus = np.full(len(net.bus), np.nan)
    report: dict = {"mode": mode, "year": year, "gdp_weight": gdp_weight,
                    "countries": {}, "fallback_countries": []}

    for iso2 in sorted(set(zone) - {""}):
        mask = zone == iso2
        regs = region[mask]
        info: dict = {"n_bus": int(mask.sum())}

        offshore = int((regs == "").sum())
        named = sorted(set(regs) - {""})
        missing = [r for r in named if r not in stats.index
                   or not np.isfinite(stats.loc[r, "pop"])
                   or not np.isfinite(stats.loc[r, "gdp"])]

        info["n_regions"] = len(named)
        info["n_bus_offshore"] = offshore
        info["n_regions_missing_stats"] = len(missing)

        if not named or missing:
            # A hole in a country's *statistics* sends the whole country to the
            # equal split - see the module docstring on why not a mixture.
            info["fallback"] = ("no NUTS3 geometry" if not named else
                                "regions without Eurostat statistics")
            info["missing_regions"] = missing[:10]
            report["countries"][iso2] = info
            report["fallback_countries"].append(iso2)
            continue

        sub = stats.loc[named]
        pop_share = sub["pop"] / sub["pop"].sum()
        gdp_share = sub["gdp"] / sub["gdp"].sum()

        # Share each region's weight across the buses inside it, so summing the
        # per-bus weights over a region returns the region's weight. Buses that
        # matched no region get weight 0 rather than sending the country back
        # to the equal split: every one of them is further offshore than
        # `snap_max_m`, which in this network means a wind-farm connection node
        # (NL/BE/DE/DK 220 kV, 26-62 km out). No population or GDP lives there,
        # so zero demand is the right answer and not a missing value.
        n_in = pd.Series(regs).value_counts()

        def _spread(w):
            per_bus = (w / n_in.reindex(w.index)).to_dict()
            per_bus[""] = 0.0
            return np.array([per_bus[r] for r in regs])

        # Both pure shares are kept per bus so the blend can be re-formed later
        # without repeating the spatial join - see `blend_shares`. The join is
        # ~35 s; the blend is arithmetic, and `nuts3_gdp_weight` is a sampled
        # factor, so they must not be computed together.
        pop_bus[mask] = _spread(pop_share)
        gdp_bus[mask] = _spread(gdp_share)
        if mode == "nuts3_pop":
            weights[mask] = pop_bus[mask]
        elif mode == "nuts3_gdp":
            weights[mask] = gdp_bus[mask]
        else:
            weights[mask] = ((1.0 - gdp_weight) * pop_bus[mask]
                             + gdp_weight * gdp_bus[mask])

        # How much of the country's population sits in regions holding no bus
        # and was therefore redistributed. Large values mean the grid model has
        # no representation of a populated area, which is worth knowing.
        all_regs = [r for r in stats.index
                    if r[:2] == NUTS_ZONE_ALIAS.get(iso2, iso2)]
        tot = stats.loc[all_regs, "pop"].sum() if all_regs else np.nan
        held = sub["pop"].sum()
        info["weight_in_busless_regions"] = (
            float(1.0 - held / tot) if np.isfinite(tot) and tot > 0 else float("nan")
        )
        report["countries"][iso2] = info

    report["n_bus_weighted"] = int(np.isfinite(weights).sum())
    report["n_bus_equal_fallback"] = int((~np.isfinite(weights)).sum())
    report["n_bus_offshore"] = sum(
        c.get("n_bus_offshore", 0) for c in report["countries"].values()
        if "fallback" not in c
    )
    # The two pure per-bus share vectors, so the blend can be re-formed per
    # evaluation without redoing the spatial join.
    report["pop_share_bus"] = pop_bus
    report["gdp_share_bus"] = gdp_bus
    return weights, report


def blend_shares(pop_bus: np.ndarray, gdp_bus: np.ndarray,
                 gdp_weight: float) -> np.ndarray:
    """w = (1 - x) * population share + x * GDP share, per bus.

    The cheap half of `nuts3_bus_weights`, split out so `nuts3_gdp_weight` can
    be a sampled factor. The spatial join behind `pop_bus` / `gdp_bus` costs
    ~35 s and depends only on geometry; this is vector arithmetic and is what
    changes per experiment.

    NaN propagates deliberately: a bus in a country with no usable Eurostat
    statistics stays NaN, which `bus_load_mw` reads as "no information" and
    answers with the equal split for that country alone.
    """
    if not 0.0 <= gdp_weight <= 1.0:
        raise ValueError(f"nuts3_gdp_weight must be in [0, 1], got {gdp_weight}")
    return (1.0 - gdp_weight) * pop_bus + gdp_weight * gdp_bus
