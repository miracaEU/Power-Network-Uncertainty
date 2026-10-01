"""Aggregate Stage 1 exposure for ALL countries onto the pan-European network.

**Why.** `hazard.py` loads `{ISO3}_power_segments.parquet` for one country, so
every per-component hazard array described only the study country: a component
abroad had zero intensity, all-zero warming anchors, and `matched=False`, which
makes it immune. Once basin events stopped being masked to the study country
that became the binding defect - a basin straddling HR/BA would fail only its
Croatian half however correct its labels were.

Nothing here recomputes GIS. Stage 1's validated output already exists per
country; this replays the SAME aggregation (`build_component_hazard`, unchanged)
once per country and merges the results onto the pan-European component index,
which the arrays were always shaped for. The output is small - intensity is
(n_comp, n_rp) - so workers read a few MB instead of ~600 MB of profiles.

**Return-period grids are not identical across countries** and that is the one
place this cannot be a plain merge:

    river       33 countries share 9 RPs; AND/CYP/MLT have empty profiles
    windstorm   all 36 share 7 RPs
    earthquake  32 share 6 RPs; DNK/EST/FIN/IRL lack RP50
    coastal     [1,100,1000] / [100,1000] / [1000] / empty - four variants

So each country's intensities are placed on the union grid by EXACT return
period where it has one, and interpolated in log-RP within its own support
otherwise. Outside a country's support the nearest endpoint is held rather than
extrapolated: a country that has no RP50 layer has no RP50 information, and
inventing a zero there would say "safe at RP50", which biases EAENS low at the
frequent end where it is weighted most.

`rp_lo` / `rp_hi` record each component's native support so the run-time
resample can tell an interpolated value from a measured one.

Run:
    python scripts/build_paneu_hazard.py [--hazards river coastal ...]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

HAZARDS = ["river", "coastal", "earthquake", "windstorm"]


def _union_rps(cfg, countries, hazard) -> tuple[np.ndarray, dict]:
    """Union return-period grid, and each country's native grid."""
    import pandas as pd

    native, vals = {}, set()
    for c in countries:
        p = (cfg["intermediate_dir"]
             / f"{c}_{cfg['asset_type']}_{hazard}_profiles.parquet")
        if not p.exists():
            continue
        rp = pd.read_parquet(p, columns=["rp"])["rp"].unique()
        if not len(rp):
            continue                      # structurally unexposed, e.g. ROU coastal
        native[c] = np.array(sorted(float(r) for r in rp))
        vals.update(native[c].tolist())
    return np.array(sorted(vals)), native


def _place(inten: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Move (n_comp, len(src)) intensities onto the `dst` return periods.

    Exact matches are copied; the rest are interpolated in log-RP and held flat
    outside `src` - `np.interp` already clamps to the end values, which is the
    "no new information" choice rather than an extrapolation.
    """
    if len(src) == len(dst) and np.allclose(src, dst):
        return inten
    out = np.empty((inten.shape[0], len(dst)))
    ls, ld = np.log(src), np.log(dst)
    for j, d in enumerate(ld):
        hit = np.flatnonzero(np.isclose(ls, d))
        if len(hit):
            out[:, j] = inten[:, hit[0]]
        else:
            out[:, j] = np.array([np.interp(d, ls, row) for row in inten])
    return out


def main() -> None:
    from power_net.hazard import build_component_hazard
    from power_net.linkage import BUS, LINE, load_linkage
    from power_net.paths import load_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--hazards", nargs="+", default=HAZARDS)
    ap.add_argument("--cache", default="cache")
    args = ap.parse_args()
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)

    cfg = load_config()
    link = load_linkage(cfg["energy_model_dir"])
    countries = sorted(p.name.split("_")[0] for p in
                       cfg["intermediate_dir"].glob(
                           f"*_{cfg['asset_type']}_segments.parquet"))
    print(f"{len(countries)} countries with Stage 1 output")
    print(f"network: {link.n_bus:,} buses, {link.n_line:,} lines\n")

    for hazard in args.hazards:
        t0 = time.perf_counter()
        rps, native = _union_rps(cfg, countries, hazard)
        if not len(rps):
            print(f"{hazard}: no country has profiles - skipped")
            continue
        print(f"=== {hazard} ===")
        print(f"  union grid {rps.tolist()} from {len(native)} countries")

        n_rp = len(rps)
        acc = {}
        for kind, n in ((BUS, link.n_bus), (LINE, link.n_line)):
            acc[kind] = {
                "intensity": np.zeros((n, n_rp)),
                "prot_rp": np.full(n, np.inf),
                "hybas": np.zeros(n, np.int64),
                "anchors": np.zeros((n, 3, 4)),
                "matched": np.zeros(n, bool),
                "rp_lo": np.full(n, np.inf),
                "rp_hi": np.full(n, -np.inf),
                # Cross-border components match in more than one country, so
                # anchors are averaged over all of them rather than taken from
                # whichever country happened to be processed first - that made
                # the result depend on iteration order.
                "_anch_n": np.zeros(n),
            }

        done = 0
        for c in native:
            sub = dict(cfg)
            sub["country"] = c
            try:
                t = build_component_hazard(sub, link, hazard)
            except FileNotFoundError:
                continue
            for kind in (BUS, LINE):
                a, s = acc[kind], t[kind]
                m = s["matched"]
                if not m.any():
                    continue
                inten = _place(s["intensity"], t["rps"], rps)
                # `max` over countries matches the within-country aggregation
                # rule: for a component built from features on both sides of a
                # border, the worst intensity any of them sees governs.
                np.maximum(a["intensity"], np.where(m[:, None], inten, 0.0),
                           out=a["intensity"])
                a["prot_rp"] = np.where(m, np.minimum(a["prot_rp"],
                                                      s["prot_rp"]), a["prot_rp"])
                # Mean over every country that matched this component, which is
                # the same rule `build_component_hazard` already applies over a
                # component's constituent features within one country.
                a["anchors"][m] += s["anchors"][m]
                a["_anch_n"][m] += 1.0
                a["rp_lo"][m] = np.minimum(a["rp_lo"][m], t["rps"][0])
                a["rp_hi"][m] = np.maximum(a["rp_hi"][m], t["rps"][-1])
                # `hybas` is a single basin id and cannot be averaged. First
                # match wins, over a sorted country list so it is at least
                # reproducible. It feeds only the `basin`/`main_basin` levels;
                # the study's `basin_lev07` comes from component_basin.parquet.
                fresh = m & ~a["matched"]
                a["hybas"][fresh] = s["hybas"][fresh]
                a["matched"] |= m
            done += 1
            if done % 6 == 0:
                print(f"    {done}/{len(native)} countries")

        out = {"rps": rps}
        for kind in (BUS, LINE):
            a = acc[kind]
            a["prot_rp"] = np.where(np.isfinite(a["prot_rp"]), a["prot_rp"], 0.0)
            n_multi = int((a["_anch_n"] > 1).sum())
            np.divide(a["anchors"], np.maximum(a["_anch_n"], 1.0)[:, None, None],
                      out=a["anchors"])
            a["rp_lo"] = np.where(np.isfinite(a["rp_lo"]), a["rp_lo"], np.nan)
            a["rp_hi"] = np.where(np.isfinite(a["rp_hi"]), a["rp_hi"], np.nan)
            del a["_anch_n"]
            for k, v in a.items():
                out[f"{kind}_{k}"] = v
            print(f"  {kind:4s}: {int(a['matched'].sum()):,} of "
                  f"{len(a['matched']):,} matched "
                  f"({100 * a['matched'].mean():.1f}%), "
                  f"{n_multi:,} matched in >1 country")
        path = cache / f"hazard_paneu_{hazard}.npz"
        np.savez_compressed(path, **out)
        print(f"  wrote {path} ({path.stat().st_size / 1e6:.1f} MB) "
              f"in {time.perf_counter() - t0:.0f}s\n")


if __name__ == "__main__":
    main()
