"""Stage 2: apply N-1 and flood defence to the stored event table.

Both are cheap - they filter events and re-integrate - but making them sampled
Sobol factors would multiply the expensive stage-1 solves for nothing, because
neither changes a single network solve. So stage 1 runs Sobol over the network
factors and persists every event; this script sweeps the cheap factors over that
table afterwards, at no marginal solve cost.

  N-1            An event in which exactly ONE component failed is survivable:
                 a transmission network is operated to withstand any single
                 contingency, and the model has no notion of that. Such events
                 contribute 0.
  flood defence  A defence is a physical structure whose design standard is
                 labelled by a PRESENT-DAY return period. Warming does not move
                 the wall, it changes how often the water reaches it. So the
                 cutoff tests `rp_nominal`, never `rp_shifted`. Testing the
                 shifted value against a fixed threshold is what previously made
                 warming appear to strengthen defences - at a 200-year standard
                 flood EAENS fell to exactly zero from 2.0C upward.

The re-integration reuses `miraca_uq.risk_model._integrate_ead`, the same
function stage 1 uses, so with both filters off this reproduces stage 1's
EAENS exactly. That identity is the correctness test for the whole split.

Run:
    python scripts/postprocess_factors.py results/LUX/events
    python scripts/postprocess_factors.py <events_dir> [--out <csv>]
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from miraca_uq.risk_model import _integrate_ead

# Present-day design standards to sweep. 0 means "no defence"; the rest bracket
# the FLOPROS range seen in the study countries (PRT 0-100, LUX ~150).
DEFENCE_RPS = [0.0, 10.0, 25.0, 50.0, 100.0, 200.0]


def _load(d: Path, table: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(d / f"{table}_*.parquet")))
    if not files:
        raise SystemExit(f"no {table}_*.parquet in {d}")
    return pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)


def eaens(events: pd.DataFrame, grid: pd.DataFrame, *,
          n1: bool, defence_rp: float) -> pd.DataFrame:
    """EAENS per experiment under one (n1, defence_rp) setting.

    Densifies each experiment's (basin, RP) matrix against the stored grid,
    because a zero cell still sets the width of the trapezoid strip beside a
    non-zero one - integrating over the sparse rows alone would understate it.
    """
    ev = events
    if n1:
        ev = ev[ev["n_failed"] != 1]
    if defence_rp > 0:
        # Present-day frame, deliberately. See the module docstring.
        ev = ev[ev["rp_nominal"] > defence_rp]

    # Every experiment must appear in the output, including those the filters
    # emptied. An experiment whose every event was a single-component failure
    # has EAENS of exactly 0 under N-1 - that is the answer, not missing data,
    # and reporting it as absent would hide the most consequential result the
    # N-1 rule can produce.
    all_keys = set(map(tuple, events[["exp_key", "hazard"]].drop_duplicates()
                       .to_numpy()))
    out = []
    for (key, hazard), part in ev.groupby(["exp_key", "hazard"], sort=False):
        warm = part["warming"].iloc[0] if "warming" in part else "current"
        g = grid[(grid["hazard"] == hazard) & (grid["warming"] == warm)]
        basins = part["event_basin"].unique()
        g = g[g["event_basin"].isin(basins)]
        if g.empty:
            continue
        rp_mat = g.pivot_table(index="event_basin", columns="rp_nominal",
                               values="rp_shifted", aggfunc="first")
        ens_mat = (part.pivot_table(index="event_basin", columns="rp_nominal",
                                    values="ens_MWh", aggfunc="sum")
                   .reindex(index=rp_mat.index, columns=rp_mat.columns)
                   .fillna(0.0))
        rp = rp_mat.to_numpy(float)
        en = ens_mat.to_numpy(float)
        order = np.argsort(rp, axis=1)
        rp = np.take_along_axis(rp, order, axis=1)
        en = np.take_along_axis(en, order, axis=1)
        out.append({"exp_key": key, "hazard": hazard, "n1": n1,
                    "defence_rp": defence_rp,
                    "EAENS_MWh": float(_integrate_ead(en, rp,
                                                      np.zeros(len(en))).sum())})
        all_keys.discard((key, hazard))
    # Whatever the filters emptied is a genuine zero.
    out += [{"exp_key": k, "hazard": h, "n1": n1, "defence_rp": defence_rp,
             "EAENS_MWh": 0.0} for k, h in sorted(all_keys)]
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("events_dir")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    d = Path(args.events_dir)

    events = _load(d, "events")
    grid = _load(d, "rp_grid")
    print(f"events {len(events):,} rows, {events.exp_key.nunique():,} experiments; "
          f"grid {len(grid):,} rows")
    if events["opf_fallback"].any():
        n = int(events["opf_fallback"].sum())
        print(f"  WARNING {n:,} rows came from the connectivity fallback after an "
              f"OPF failure - a different model. Excluded from the sweep.")
        events = events[~events["opf_fallback"]]

    if "warming" not in events.columns:
        raise SystemExit(
            "events carry no `warming` column, so an event cannot be paired "
            "with its own integration grid - re-run stage 1 on current code."
        )

    frames = [eaens(events, grid, n1=n1, defence_rp=rp)
              for n1 in (False, True) for rp in DEFENCE_RPS]
    res = pd.concat(frames, ignore_index=True)
    out = Path(args.out) if args.out else d.parent / "postprocessed_eaens.csv"
    res.to_csv(out, index=False)
    print(f"\nwrote {out}  ({len(res):,} rows)")

    piv = res.pivot_table(index="defence_rp", columns="n1", values="EAENS_MWh",
                          aggfunc="mean")
    print("\nmean EAENS (MWh) by defence standard and N-1:")
    print(piv.to_string(float_format=lambda v: f"{v:,.3f}"))


if __name__ == "__main__":
    main()
