"""Per-basin, per-return-period event records, written beside the EMA results.

`compute_ens` already builds the full picture internally: `ens[fi, ri]` is the
energy not supplied by the event in basin `fp_ids[fi]` at return period `ri`.
It is then collapsed to one scalar and thrown away. Everything this study wants
to do afterwards - map ENS to basins and LAUs, re-aggregate cross-basin
propagation, apply N-1 and flood defence post-hoc - needs that matrix, so it is
persisted here instead of recomputed.

**Why a side file and not an EMA outcome.** `DefaultCallback._store_outcomes`
allocates its array from the FIRST experiment's shape and then assigns into it,
so an outcome's shape must be identical across every experiment in a run. The
number of footprints varies by design (it is itself an outcome), so a per-basin
vector is ragged and cannot be an `ArrayOutcome`. It also refuses anything above
2-D. A parquet sidecar has none of those limits.

**Why one file per worker.** `MultiprocessingEvaluator` spawns rather than forks,
so workers share no state with the parent and `compute_ens` never learns its
experiment index. Each worker therefore writes its own file, and every row
carries `exp_key` - a hash of the fully resolved kwargs - so rows can be joined
back to `experiments.csv` afterwards. The hash covers constants too, because EMA
does not write `Constant`s to `experiments.csv`: without them, two runs that
differed only in `duration_sampling` would produce colliding keys.

Two tables:

  events       one row per (experiment, event basin, return period). Carries
               BOTH `rp_nominal` and `rp_shifted` - see the note below - plus
               `n_failed`, which is what the post-hoc N-1 rule filters on.
  attribution  one row per (experiment, event basin, return period, affected
               basin). This is what makes the A -> B propagation question
               answerable later: ENS is credited to the basin where the event
               happened, but the split across the basins where load was
               actually lost is recorded, so the aggregation can be changed
               without re-running anything.

**The two return-period columns are not redundant.** `rp_nominal` is the
present-day label of the event; `rp_shifted` is how often that event occurs
under the sampled warming level. Protection is a physical standard labelled by a
present-day return period, so the post-hoc defence cutoff tests `rp_nominal`,
while the probability integration uses `rp_shifted`. Collapsing them to one
column reintroduces the bug where warming appeared to strengthen defences.
"""

from __future__ import annotations

import atexit
import hashlib
import os
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

# Rows are buffered and flushed in batches: one parquet write per experiment
# would produce thousands of tiny files and dominate the runtime on a network
# share, which this project's is.
_FLUSH_ROWS = 50_000

_BUF: dict[str, list] = {"events": [], "attribution": [], "rp_grid": []}
# (basin, warming) pairs already recorded, so the grid is written once rather
# than on every experiment that happens to share them.
_GRID_SEEN: set = set()
_DEST: dict[str, Path | None] = {"dir": None, "stem": None}
_RUN_ID = f"{os.getpid()}_{uuid.uuid4().hex[:8]}"
_REGISTERED = False


def configure(out_dir: Path, stem: str) -> None:
    """Where this worker's event files go. Safe to call repeatedly."""
    global _REGISTERED
    _DEST["dir"] = Path(out_dir)
    _DEST["stem"] = stem
    if not _REGISTERED:
        # Spawned workers exit without unwinding the parent's code, so the
        # final partial batch is only written if flushing is tied to process
        # exit rather than to the end of a loop somewhere.
        atexit.register(flush)
        _REGISTERED = True


def exp_key(kwargs: dict) -> str:
    """Stable short hash of one experiment's fully resolved factor settings.

    Includes constants deliberately - EMA writes only uncertainties to
    `experiments.csv`, so hashing the sampled columns alone would collide across
    runs that differ by a constant.
    """
    items = sorted((str(k), repr(v)) for k, v in kwargs.items())
    blob = "|".join(f"{k}={v}" for k, v in items)
    return hashlib.blake2b(blob.encode("utf-8"), digest_size=8).hexdigest()


def record_events(key: str, hazard: str, country: str, warming: str,
                  fp_ids: np.ndarray,
                  rp_nominal: np.ndarray, rp_shifted: np.ndarray,
                  ens: np.ndarray, n_failed: np.ndarray, peak_mw: np.ndarray,
                  fallback: np.ndarray | None = None) -> None:
    """One row per (basin, return period) with non-zero energy or failures.

    Zero-everywhere cells are dropped: most (basin, RP) pairs produce no
    failures at all, and keeping them would inflate the table by orders of
    magnitude for no information. A row therefore means "something happened
    here", and its absence means nothing did.
    """
    if _DEST["dir"] is None:
        return
    keep = (ens > 0) | (n_failed > 0)
    if not keep.any():
        return
    fi, ri = np.nonzero(keep)
    _BUF["events"].append(pd.DataFrame({
        "exp_key": key,
        "hazard": hazard,
        "country_run": country,
        # `warming` is a sampled factor for river flood, and it selects which
        # rp_grid applies. EMA does not write constants to experiments.csv and
        # the grid is keyed on (basin, warming), so without it here stage 2
        # cannot pair an event with its own integration grid.
        "warming": warming,
        "event_basin": np.asarray(fp_ids)[fi],
        "rp_nominal": np.asarray(rp_nominal)[ri],
        "rp_shifted": np.asarray(rp_shifted)[fi, ri],
        "ens_MWh": ens[fi, ri],
        "n_failed": n_failed[fi, ri].astype(np.int32),
        "peak_MW": peak_mw[fi, ri],
        # True where the DC OPF failed and the connectivity model answered
        # instead. Those rows come from a different model and must not be
        # silently pooled with the rest.
        "opf_fallback": (np.zeros(len(fi), bool) if fallback is None
                         else fallback[fi, ri]),
    }))
    _maybe_flush()


def record_attribution(key: str, hazard: str, event_basin: int,
                       rp_nominal: float, affected_basin: np.ndarray,
                       ens_MWh: np.ndarray) -> None:
    """Where the energy was actually lost, for one event.

    Recorded separately from `events` because it is a different question: the
    event happened in `event_basin`, the load was lost in `affected_basin`, and
    which of the two an aggregate should be keyed on is a choice this study
    wants to defer rather than bake in.
    """
    if _DEST["dir"] is None or len(affected_basin) == 0:
        return
    _BUF["attribution"].append(pd.DataFrame({
        "exp_key": key,
        "hazard": hazard,
        "event_basin": int(event_basin),
        "rp_nominal": float(rp_nominal),
        "affected_basin": np.asarray(affected_basin),
        "ens_MWh": np.asarray(ens_MWh),
    }))
    _maybe_flush()


def record_rp_grid(hazard: str, warming: str, fp_ids: np.ndarray,
                   rp_nominal: np.ndarray, rp_shifted: np.ndarray) -> None:
    """The complete (basin, RP) integration grid, dense.

    `events` is sparse - rows where nothing failed are dropped, because most
    (basin, RP) cells produce no failures and keeping them would inflate the
    table roughly tenfold for no information. But re-integrating in stage 2
    needs the FULL grid: the trapezoid's strip between a zero cell and a
    non-zero one is `0.5 * (0 + d) * (p_i - p_i+1)`, which is not zero, so the
    widths of the empty cells still matter.

    Storing it separately rather than densifying `events` works because
    `rp_shifted` depends only on (basin, warming) - the footprint-mean anchor
    shift - and not on the rest of the factor vector. So this table is
    n_basins x n_warming x n_rp rows regardless of how many experiments run.
    """
    if _DEST["dir"] is None:
        return
    fp_ids = np.asarray(fp_ids)
    new = [i for i, f in enumerate(fp_ids) if (int(f), warming) not in _GRID_SEEN]
    if not new:
        return
    for i in new:
        _GRID_SEEN.add((int(fp_ids[i]), warming))
    idx = np.asarray(new)
    n_rp = len(rp_nominal)
    _BUF["rp_grid"].append(pd.DataFrame({
        "hazard": hazard,
        "warming": warming,
        "event_basin": np.repeat(fp_ids[idx], n_rp),
        "rp_nominal": np.tile(np.asarray(rp_nominal), len(idx)),
        "rp_shifted": np.asarray(rp_shifted)[idx].ravel(),
    }))


def _maybe_flush() -> None:
    if sum(len(d) for d in _BUF["events"]) >= _FLUSH_ROWS:
        flush()


def flush() -> None:
    """Append the buffers to this worker's parquet files.

    Two things here are shutdown-hardening, not style.

    `pd.DataFrame.to_parquet` converts columns through a `ThreadPoolExecutor`.
    At interpreter shutdown - which is exactly when the `atexit` flush runs -
    that pool is already torn down and the conversion dies with "cannot
    schedule new futures after interpreter shutdown". Building the Arrow table
    with `nthreads=1` skips the executor entirely. Measured: the atexit flush
    wrote `events` and then lost `attribution` and `rp_grid` to this.

    And each table is guarded separately. The loop used to abort on the first
    failure, so one table's error discarded the others' buffered rows; on the
    cluster that would silently cost whole workers' attribution with a zero
    exit code. A table that cannot be written now reports itself and the rest
    still go out.
    """
    d, stem = _DEST["dir"], _DEST["stem"]
    if d is None:
        return
    for table, frames in _BUF.items():
        if not frames:
            continue
        out = Path(d) / f"{table}_{stem}_{_RUN_ID}.parquet"
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq

            df = pd.concat(frames, ignore_index=True)
            if out.exists():
                df = pd.concat([pd.read_parquet(out), df], ignore_index=True)
            out.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(
                pa.Table.from_pandas(df, preserve_index=False, nthreads=1), out
            )
            frames.clear()
        except Exception as e:                                    # noqa: BLE001
            # Never raise from atexit: it would mask the real exit status and
            # take the remaining tables with it.
            print(f"eventlog: FAILED to write {out.name}: "
                  f"{type(e).__name__}: {e}", file=sys.stderr)
