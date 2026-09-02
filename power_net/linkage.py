"""Map grid components to the OSM features the direct-damage study exposed.

This is what makes Stage 1 reusable instead of re-run. The Energy Model ships
two views of the same network:

  Net_structure_data/*.csv   pandapower tables (what we solve)
  NetworkOSM/*.csv           the PyPSA-Eur `osm-prebuilt` source (what carries
                             OSM provenance in its `tags` column)

The two are row-identical in order - verified here on every load, not assumed -
so a grid component is addressed by row position in both. `tags` holds
`way/<id>` / `relation/<id>` / `node/<id>` references; stripping the prefix and
the `-<voltage>` suffix yields exactly the bare numeric `osm_id` used by
`{ISO3}_power_exposure.parquet` and hence by Stage 1's segments table.

Measured match rates against the exposure (2026-08-27):

    LUX   buses 11/11 (100%)    line ways  27/27  (100%)
    PRT   buses 83/87 (95.4%)   line ways 312/328 (95.1%)

PRT's misses are 3 cross-border ways that live in ESP's exposure file and 17
(0.4%) OSM vintage drift between the two snapshots.

Two structural limits, both reported rather than papered over:

  * 1,806 of 6,737 buses (27%) are synthetic PyPSA-Eur nodes (`LU1-220`) with
    no OSM identifier at all. No join can reach them - see
    `network.unmatched_bus_policy`.
  * The mapping is one-to-many: one grid line aggregates several OSM ways
    (LUX 16 lines -> 27 ways -> 63 exposure rows). Aggregation happens in
    hazard.py, where the choice of rule interacts with the damagescanner
    defects.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# `way/393286545-225` -> 393286545. Node/way/relation all appear; the exposure
# carries node and way ids only, so relation references simply find no match.
_OSM_REF = re.compile(r"(?:node|way|relation)/(\d+)")

BUS, LINE, TRAFO = "bus", "line", "trafo"


def read_osm_lines(em_dir: Path) -> pd.DataFrame:
    """Read NetworkOSM/lines.csv, which pandas cannot parse correctly.

    Its `geometry` field is wrapped in SINGLE quotes and contains commas, so
    every CSV reader splits it into dozens of phantom columns and silently
    shifts `tags` off its header. Because geometry is the LAST column, a
    fixed-count split on the header width is exact.

    Getting this wrong is not loud - it yields a parsed frame of the right
    length whose columns are misaligned, which is why the row-order assertion
    in `load_linkage` exists.
    """
    path = em_dir / "NetworkOSM" / "lines.csv"
    with open(path, encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split(",")
        n = len(header)
        rows = [ln.rstrip("\n").split(",", n - 1) for ln in fh]
    rows = [r for r in rows if len(r) == n]
    return pd.DataFrame(rows, columns=header)


def osm_ids(tag_value: object) -> set[str]:
    """Bare numeric OSM ids referenced by one `tags` cell."""
    if tag_value is None or (isinstance(tag_value, float) and np.isnan(tag_value)):
        return set()
    return set(_OSM_REF.findall(str(tag_value)))


@dataclass
class Linkage:
    """Per-component OSM references, row-aligned to the pandapower tables."""

    bus_osm: list[set[str]]
    line_osm: list[set[str]]
    bus_zone: np.ndarray          # (n_bus,) ISO2, "" where unknown
    trafo_hv_bus: np.ndarray      # (n_trafo,) transformers inherit their HV bus

    @property
    def n_bus(self) -> int:
        return len(self.bus_osm)

    @property
    def n_line(self) -> int:
        return len(self.line_osm)

    @property
    def bus_matched(self) -> np.ndarray:
        return np.array([len(s) > 0 for s in self.bus_osm])

    @property
    def line_matched(self) -> np.ndarray:
        return np.array([len(s) > 0 for s in self.line_osm])

    def pairs(self, kind: str) -> pd.DataFrame:
        """Long-form (component index, osm_id) for joining to Stage 1 segments."""
        src = self.bus_osm if kind == BUS else self.line_osm
        idx, oid = [], []
        for i, ids in enumerate(src):
            for o in ids:
                idx.append(i)
                oid.append(o)
        return pd.DataFrame({"comp_idx": np.array(idx, np.int64),
                             "osm_id": pd.array(oid, dtype="string")})


def load_linkage(em_dir: Path) -> Linkage:
    """Build the component -> OSM id mapping, asserting the row alignment."""
    ns = em_dir / "Net_structure_data"
    bus = pd.read_csv(ns / "bus_data.csv")
    line = pd.read_csv(ns / "line_data.csv")
    trafo = pd.read_csv(ns / "trafo_data.csv")
    osm_bus = pd.read_csv(em_dir / "NetworkOSM" / "buses.csv",
                          usecols=["bus_id", "tags"], dtype=str)
    osm_line = read_osm_lines(em_dir)

    # The whole scheme rests on positional identity. Assert it every load: a
    # silent mis-parse of lines.csv would otherwise attach each line's hazard
    # to the wrong component and change which lines fail.
    if not np.array_equal(bus["name"].astype(str).to_numpy(),
                          osm_bus["bus_id"].astype(str).to_numpy()):
        raise ValueError("bus_data.csv and NetworkOSM/buses.csv are not row-aligned")
    if not np.array_equal(line["name"].astype(str).to_numpy(),
                          osm_line["line_id"].astype(str).to_numpy()):
        raise ValueError(
            "line_data.csv and NetworkOSM/lines.csv are not row-aligned - "
            "most likely lines.csv was parsed with a plain CSV reader "
            "(see read_osm_lines)"
        )

    return Linkage(
        bus_osm=[osm_ids(t) for t in osm_bus["tags"]],
        line_osm=[osm_ids(t) for t in osm_line["tags"]],
        bus_zone=bus["zone"].fillna("").astype(str).to_numpy(),
        trafo_hv_bus=trafo["hv_bus"].to_numpy(np.int64),
    )


def country_mask(link: Linkage, iso2: str, line_from: np.ndarray,
                 line_to: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Buses in `iso2`, and lines with at least one end there.

    Cross-border lines are included for both countries on purpose: a line that
    fails on the Portuguese side still removes the Spanish interconnection, and
    a study run for PRT has to see it.
    """
    bmask = link.bus_zone == iso2
    lmask = bmask[line_from] | bmask[line_to]
    return bmask, lmask
