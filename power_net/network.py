"""Build the pan-European pandapower network from the MIRACA Energy Model drop.

The Energy Model folder is an input-data snapshot (PyPSA-Eur `osm-prebuilt`
base network converted to pandapower CSVs, OPSD hourly demand,
powerplantmatching capacities, Eurostat NUTS3). It contains no code; this
module is the only place that reads it.

Deliberate departures from the existing MIRACA D3.3 `spof_electrical_analysis.py`
(see the plan's "do not inherit" list) - each one is a correctness fix, not a
preference:

  * No `.fillna(500.0)` MW-per-bus default for countries missing from OPSD.
    Countries without demand data are recorded and left at zero load, never
    silently invented. That default is the likely source of D3.3's 245,864 MW
    single-line artefact.
  * Generators are snapped to buses with a KDTree in EPSG:3035 (metres), not in
    raw lon/lat degrees, where a "nearest" match is distorted by cos(latitude).
  * No global `0.7/(max_loading)` rescale of loads and generation. If the
    baseline is infeasible we say so rather than quietly moving the operating
    point and then accounting against the un-rescaled values.

Buses, lines and transformers are assigned as whole DataFrames rather than via
`pp.create_*` loops, which costs ~200 s versus well under a second for 6.7k
buses. The CSVs were written against pandapower 2.x, so `conform_to_schema`
translates them to whatever version is installed (3.x renamed
tap_phase_shifter and the load const_z/const_i columns, and replaced the
bus_geodata frame with a `geo` column).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ENERGY_MODEL_DIR = Path(
    r"C:\Users\yma794\OneDrive - Vrije Universiteit Amsterdam\Documents\MIRACA"
    r"\D 3.3 Network-level risk assessment for CI in Europe\Energy Model"
)

# ISO2 (grid `zone` / OPSD prefix) -> ISO3 (MIRACA exposure file naming).
ISO2_TO_ISO3 = {
    "AL": "ALB", "AT": "AUT", "BA": "BIH", "BE": "BEL", "BG": "BGR",
    "CH": "CHE", "CZ": "CZE", "DE": "DEU", "DK": "DNK", "EE": "EST",
    "ES": "ESP", "FI": "FIN", "FR": "FRA", "GB": "GBR", "GR": "GRC",
    "HR": "HRV", "HU": "HUN", "IE": "IRL", "IT": "ITA", "LT": "LTU",
    "LU": "LUX", "LV": "LVA", "MD": "MDA", "ME": "MNE", "MK": "MKD",
    "NL": "NLD", "NO": "NOR", "PL": "POL", "PT": "PRT", "RO": "ROU",
    "RS": "SRB", "SE": "SWE", "SI": "SVN", "SK": "SVK", "UA": "UKR",
    "XK": "XKX",
}
ISO3_TO_ISO2 = {v: k for k, v in ISO2_TO_ISO3.items()}

# OPSD publishes some countries only under a sub-zone code. Great Britain has no
# plain "GB" demand series - only GB_GBN (Great Britain), GB_NIR (Northern
# Ireland) and GB_UKM (the whole UK). The grid model's GB zone is the Great
# Britain synchronous area, with Northern Ireland belonging to the all-island
# SEM alongside IE, so GB_GBN is the matching series.
OPSD_DEMAND_ALIAS = {"GB": "GB_GBN"}

# Countries in the grid model with no OPSD demand series at all. Recorded here
# so they are a known, reported gap rather than something a default silently
# papers over. ALB and MKD fall inside the 31-country study set and will need
# demand from another source (e.g. ENTSO-E annual totals shaped by a regional
# profile) before their results mean anything.
NO_OPSD_DEMAND = ("AL", "BA", "MD", "MK", "XK")

# Value of lost load, EUR/MWh. Priced far above any generator's marginal cost so
# the OPF sheds only as a genuine last resort. The absolute number does not
# affect how much is shed (only the ordering matters); it is factored back out
# when reporting ENS in MWh.
VOLL_EUR_MWH = 10_000.0

# Marginal cost by fuel, EUR/MWh - sets merit order in the DC OPF. Values follow
# the D3.3 script's dict so dispatch stays comparable, but here they are
# actually used by the optimiser rather than stored and ignored.
MARGINAL_COST = {
    "Nuclear": 5.0, "Hard Coal": 40.0, "Lignite": 35.0, "Natural Gas": 60.0,
    "Hydro": 10.0, "Wind": 0.0, "Solar": 0.0, "Solid Biomass": 30.0,
    "Oil": 80.0, "Geothermal": 15.0, "Waste": 25.0, "Bioenergy": 30.0,
    "Other": 40.0,
}
DEFAULT_MARGINAL_COST = 40.0


@dataclass
class NetworkBundle:
    """A built network plus the lookups Stage 2 needs to fail components fast."""

    net: object                       # pandapower net
    bus_country: np.ndarray           # (n_bus,) ISO3, "" where unknown
    bus_xy_3035: np.ndarray           # (n_bus, 2) metres
    line_name: np.ndarray             # (n_line,) osm_id, row-aligned to TENT lines
    bus_name: np.ndarray              # (n_bus,) osm_id, row-aligned to TENT buses
    missing_demand: list[str] = field(default_factory=list)
    notes: dict = field(default_factory=dict)


# The Energy Model CSVs were written against the pandapower 2.x schema, which
# 3.x renamed in places. Rather than hard-code either generation, tables are
# conformed to whatever pandapower is installed. Mapping to None drops a column.
SCHEMA_RENAMES = {
    # All 875 transformers have tap_phase_shifter=False and no tap position at
    # all, so there is nothing to translate into 3.x's tap_changer_type.
    "trafo": {"tap_phase_shifter": None},
    "load": {
        "const_z_percent": "const_z_p_percent",
        "const_i_percent": "const_i_p_percent",
    },
}


def _schema_default(dtype):
    """A neutral 'not set' value matching a pandapower column's dtype."""
    if pd.api.types.is_bool_dtype(dtype):
        return False
    if isinstance(dtype, pd.core.dtypes.dtypes.BaseMaskedDtype):
        return pd.NA
    if pd.api.types.is_numeric_dtype(dtype):
        return np.nan
    return None


def conform_to_schema(table: str, df: pd.DataFrame) -> pd.DataFrame:
    """Fit a frame to the installed pandapower's schema for one element table.

    Applies SCHEMA_RENAMES, fills any column the installed version expects but
    the source lacks with a dtype-appropriate default, and keeps extra columns
    (that is how pandapower carries optional OPF fields such as min_p_mw). This
    is what lets whole-DataFrame assignment stay fast without silently omitting
    a column the solver later needs.
    """
    import pandapower as pp

    template = getattr(pp.create_empty_network(), table)
    out = df.copy()
    for old, new in SCHEMA_RENAMES.get(table, {}).items():
        if old in out.columns:
            if new is None:
                out = out.drop(columns=[old])
            else:
                out = out.rename(columns={old: new})
    for col in template.columns:
        if col not in out.columns:
            out[col] = _schema_default(template[col].dtype)
    extras = [c for c in out.columns if c not in template.columns]
    out = out[list(template.columns) + extras]
    out.index = np.arange(len(out), dtype=np.int64)
    return out


def _read(em_dir: Path, *parts: str) -> pd.DataFrame:
    return pd.read_csv(em_dir.joinpath(*parts))


def build_base_network(em_dir: Path = ENERGY_MODEL_DIR):
    """Buses, lines and transformers only - no injections yet.

    Returns (net, bus_geo) where bus_geo has x/y in EPSG:4326.
    """
    import pandapower as pp

    bus = _read(em_dir, "Net_structure_data", "bus_data.csv")
    line = _read(em_dir, "Net_structure_data", "line_data.csv")
    trafo = _read(em_dir, "Net_structure_data", "trafo_data.csv")
    geo = _read(em_dir, "Net_structure_data", "bus_geodata.csv")

    net = pp.create_empty_network()
    # Whole-DataFrame assignment: pp.create_* in a loop would cost ~200 s for
    # 6.7k buses. conform_to_schema keeps that safe across pandapower versions.
    net.bus = conform_to_schema("bus", bus)
    net.line = conform_to_schema("line", line)
    net.trafo = conform_to_schema("trafo", trafo)

    # Coordinates are returned alongside rather than written into net: 3.x
    # replaced the bus_geodata frame with a GeoJSON `geo` column, and nothing
    # here needs pandapower's own plotting.
    return net, geo


def bus_coords_3035(geo: pd.DataFrame) -> np.ndarray:
    """Bus coordinates reprojected to EPSG:3035 metres.

    Everything spatial in this project happens in 3035, matching the
    direct-damage pipeline. Distance work in 4326 degrees or in 3857 (as the
    D3.3 UC5 code does) distorts by ~1/cos(latitude) - about 35% at 46 deg N
    and 65% at 55 deg N.
    """
    from pyproj import Transformer

    tf = Transformer.from_crs(4326, 3035, always_xy=True)
    x, y = tf.transform(geo["x"].to_numpy(), geo["y"].to_numpy())
    return np.column_stack([x, y])


def topology_report(net) -> dict:
    """Characterise connectivity before any ENS number is believed.

    D3.3 reports 206 components from lines alone; transformers merge most of
    them. Many remaining 'islands' are artefacts of incomplete topology rather
    than real electrical islands, and an artefact island produces spurious
    unserved energy, so this has to be quantified up front.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n = len(net.bus)
    edges = [
        (net.line.from_bus.to_numpy(), net.line.to_bus.to_numpy()),
        (net.trafo.hv_bus.to_numpy(), net.trafo.lv_bus.to_numpy()),
    ]
    out = {}
    for label, use in (("lines_only", edges[:1]), ("lines_and_trafos", edges)):
        rows = np.concatenate([e[0] for e in use])
        cols = np.concatenate([e[1] for e in use])
        adj = coo_matrix(
            (np.ones(len(rows)), (rows, cols)), shape=(n, n)
        ).tocsr()
        ncomp, labels = connected_components(adj, directed=False)
        sizes = np.bincount(labels)
        out[label] = {
            "n_components": int(ncomp),
            "largest": int(sizes.max()),
            "top5": sorted(sizes.tolist(), reverse=True)[:5],
            "singletons": int((sizes == 1).sum()),
        }
    return out