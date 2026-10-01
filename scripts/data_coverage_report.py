"""Per-country data coverage: what is runnable, what is not, and why.

Every exclusion in this study traces to a missing input, and they are missing
for different reasons at different stages. This collects all of them in one
workbook so a country's status is a lookup rather than an investigation.

The four independent requirements, in the order they bite:

  demand      OPSD hourly demand. Without it there is no load to place on the
              country's buses and the archetype hour cannot even be chosen, so
              it cannot be a STUDY country. (AL, BA, MD, MK, XK.)
  exposure    Stage 1 output from the direct-damage study. Without it every
              component is unexposed and therefore immune, so the country can
              neither be studied NOR contribute to a Europe-wide basin event
              reaching into it. (UA, BA, ME, XK, MD.)
  localised   A lev07 basin for each bus, from `build_component_basins.py`.
              Unlocalised buses cannot be placed in any event footprint and
              cannot receive attribution.
  statistics  Eurostat NUTS3 population and GDP, for the load disaggregation
              factor. A country without them falls back to an equal split.

Note the two lists differ: Albania has exposure but no demand, Ukraine has
demand but no exposure. They fail at different stages and for different
reasons, which is why this is a matrix and not a list.

Run:
    python scripts/data_coverage_report.py [--out results/data_coverage.xlsx]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

HAZARDS = ["river", "coastal", "earthquake", "windstorm"]
STUDY = ["AUT", "EST", "HRV", "HUN", "ITA", "NLD", "ROU"]


def main() -> None:
    from power_net.demand import _demand_frame
    from power_net.network import (ISO2_TO_ISO3, ISO3_TO_ISO2,
                                   NO_OPSD_DEMAND, OPSD_DEMAND_ALIAS)
    from power_net.paths import load_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/data_coverage.xlsx")
    args = ap.parse_args()

    cfg = load_config()
    em = Path(cfg["energy_model_dir"])
    inter = Path(cfg["intermediate_dir"])
    cache = Path(cfg["cache_dir"])

    nsd = em / "Net_structure_data"
    bus = pd.read_csv(nsd / "bus_data.csv")
    line = pd.read_csv(nsd / "line_data.csv")
    zone = bus["zone"].astype(str).to_numpy()
    lz = zone[line["from_bus"].to_numpy(np.int64)]

    demand_cols = set(_demand_frame(str(em)).columns)
    seg_have = {p.name.split("_")[0]
                for p in inter.glob(f"*_{cfg['asset_type']}_segments.parquet")}

    # NUTS3 statistics, by the 2-letter prefix Eurostat uses (EL = GR, UK = GB).
    stat_pref = set()
    for f in ("estat_nama_10r_3popgdp$defaultview_filtered_en.csv",
              "estat_nama_10r_3gdp$defaultview_filtered_en.csv"):
        d = pd.read_csv(em / f, low_memory=False)
        stat_pref |= set(d["geo"].astype(str).str[:2])
    ESTAT_ALIAS = {"GR": "EL", "GB": "UK"}

    # Basin localisation, pan-European.
    loc = None
    p = cache / "component_basin.parquet"
    if p.is_file():
        t = pd.read_parquet(p)
        loc = t[t["kind"] == "bus"].sort_values("idx")["event_basin"].to_numpy()

    # Per-hazard exposure, from the pan-EU caches: `matched` is the honest
    # "this component has Stage 1 data", one array per hazard.
    matched = {}
    for hz in HAZARDS:
        f = cache / f"hazard_paneu_{hz}.npz"
        if f.is_file():
            matched[hz] = np.load(f)["bus_matched"]

    # Per-country, per-hazard profile status.
    prof = []
    for z in sorted(set(zone)):
        i3 = ISO2_TO_ISO3.get(z, "???")
        for hz in HAZARDS:
            f = inter / f"{i3}_{cfg['asset_type']}_{hz}_profiles.parquet"
            if not f.exists():
                prof.append({"iso2": z, "iso3": i3, "hazard": hz,
                             "status": "no file", "rows": 0, "return_periods": ""})
                continue
            x = pd.read_parquet(f, columns=["rp"])
            prof.append({
                "iso2": z, "iso3": i3, "hazard": hz,
                "status": "OK" if len(x) else "EMPTY (no asset exposed)",
                "rows": len(x),
                "return_periods": ", ".join(
                    str(int(v)) for v in sorted(x["rp"].unique())) if len(x) else "",
            })
    prof = pd.DataFrame(prof)

    rows = []
    for z in sorted(set(zone)):
        i3 = ISO2_TO_ISO3.get(z, "???")
        m = zone == z
        nb = int(m.sum())
        dem = z in demand_cols or OPSD_DEMAND_ALIAS.get(z) in demand_cols
        exp = i3 in seg_have
        hz_ok = sorted(prof[(prof.iso2 == z) & (prof.status == "OK")]["hazard"])
        nloc = int((loc[m] >= 0).sum()) if loc is not None else -1
        nexp = {hz: int(a[m].sum()) for hz, a in matched.items()}

        blockers = []
        if not dem:
            blockers.append("no OPSD demand series")
        if not exp:
            blockers.append("no Stage 1 exposure")
        if not hz_ok:
            blockers.append("no non-empty hazard profile")
        rows.append({
            "iso2": z,
            "iso3": i3,
            "study_country": i3 in STUDY,
            "buses": nb,
            "lines": int((lz == z).sum()),
            "has_demand": dem,
            "demand_series": OPSD_DEMAND_ALIAS.get(z, z) if dem else "",
            "has_exposure": exp,
            "has_nuts3_stats": ESTAT_ALIAS.get(z, z) in stat_pref,
            "buses_localised": nloc,
            "buses_unlocalised": nb - nloc if nloc >= 0 else -1,
            "buses_with_river_data": nexp.get("river", -1),
            "buses_with_coastal_data": nexp.get("coastal", -1),
            "buses_with_eq_data": nexp.get("earthquake", -1),
            "buses_with_wind_data": nexp.get("windstorm", -1),
            "hazards_available": ", ".join(hz_ok),
            "runnable_as_study_country": bool(dem and exp and hz_ok),
            "can_fail_in_europe_wide_event": exp,
            "blockers": "; ".join(blockers),
        })
    cov = pd.DataFrame(rows).sort_values(
        ["runnable_as_study_country", "buses"], ascending=[False, False])

    # The campaign grid: which (country, hazard) pairs are worth submitting.
    camp = []
    for i3 in STUDY:
        z = ISO3_TO_ISO2[i3]
        for hz in HAZARDS:
            r = prof[(prof.iso2 == z) & (prof.hazard == hz)].iloc[0]
            if r["status"] == "OK":
                verdict, why = "RUN", ""
            elif r["status"].startswith("EMPTY"):
                verdict, why = "SKIP", "no asset of this country is exposed"
            else:
                verdict, why = "SKIP", "Stage 1 never produced this hazard"
            camp.append({"iso3": i3, "hazard": hz,
                         "scenario": f"net_{'flood' if hz == 'river' else hz}",
                         "verdict": verdict, "reason": why,
                         "profile_rows": r["rows"]})
    camp = pd.DataFrame(camp)

    notes = pd.DataFrame({"item": [
        "demand", "exposure", "localised", "statistics", "",
        "runnable_as_study_country", "can_fail_in_europe_wide_event", "",
        "buses_with_*_data", "buses_unlocalised", "", "ROU coastal", "generated",
    ], "meaning": [
        "OPSD hourly demand series. Missing -> no load to place, archetype hour "
        "cannot be selected. Missing for AL, BA, MD, MK, XK.",
        "Stage 1 exposure from the direct-damage study. Missing -> every "
        "component unexposed and immune.",
        "lev07 basin per bus, from build_component_basins.py. Missing -> the "
        "bus is in no event footprint and receives no attribution.",
        "Eurostat NUTS3 population and GDP, for the load-placement factor. "
        "Missing -> equal split fallback.",
        "",
        "demand AND exposure AND at least one non-empty hazard profile.",
        "exposure alone. A country can contribute components to a basin event "
        "reaching into it without being studyable itself (e.g. ALB).",
        "",
        "Buses with Stage 1 data for that hazard, from the pan-EU cache. The "
        "shortfall against `buses` is mostly the 27% of buses carrying no OSM "
        "identifier, not a country-level gap.",
        "Buses outside every lev07 polygon after a 25 km nearest-snap - "
        "offshore and island buses.",
        "",
        "Profile file exists but holds zero rows: no Romanian power asset is "
        "coastally exposed, so the scenario is a guaranteed zero. Same for "
        "BGR, BEL, ISL.",
        pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
    ]})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out, engine="openpyxl") as xl:
        cov.to_excel(xl, sheet_name="country_coverage", index=False)
        prof.to_excel(xl, sheet_name="hazard_profiles", index=False)
        camp.to_excel(xl, sheet_name="campaign_grid", index=False)
        notes.to_excel(xl, sheet_name="notes", index=False)
        for name, df in (("country_coverage", cov), ("hazard_profiles", prof),
                         ("campaign_grid", camp), ("notes", notes)):
            ws = xl.sheets[name]
            ws.freeze_panes = "A2"
            for i, c in enumerate(df.columns, start=1):
                width = max(len(str(c)),
                            *(len(str(v)) for v in df[c].head(200))) + 2
                ws.column_dimensions[
                    ws.cell(row=1, column=i).column_letter].width = min(width, 60)

    print(f"wrote {out}\n")
    print("not runnable as a study country:")
    bad = cov[~cov.runnable_as_study_country]
    print(bad[["iso3", "buses", "blockers"]].to_string(index=False))
    print(f"\nrunnable: {int(cov.runnable_as_study_country.sum())} of {len(cov)} "
          f"grid countries")
    print(f"campaign: {int((camp.verdict == 'RUN').sum())} combinations to run, "
          f"{int((camp.verdict == 'SKIP').sum())} skipped")


if __name__ == "__main__":
    main()
