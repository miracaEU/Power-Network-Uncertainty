"""Stage 1 hazard intensities, aggregated from OSM features onto grid components.

Reads the direct-damage study's validated Stage 1 output and re-expresses it
per grid component, so no GIS is recomputed here. For LUX and PRT that output
already exists and carries a `_validated.ok` marker.

What Stage 1 gives us, per country and hazard:

    {ISO3}_power_segments.parquet    osm_id, object_type, prot_rp, HYBAS_ID,
                                     new_rp{10,100,500}_w{15,20,30,40}
    {ISO3}_power_{hz}_profiles.parquet   seg, rp, intensity, quantity

`seg` indexes rows of the segments table. We join `osm_id` to the component
linkage, then collapse many features onto one component.

Aggregation rules and why the default is `max`
----------------------------------------------
A grid line is many OSM ways; a substation bus may be several exposure rows.
`max` takes the worst intensity any constituent feature sees. Besides being the
right physical rule for a series element (the weakest point governs), it is
**immune by construction to both damagescanner defects**, which corrupt
`quantity` / cell coverage and never `intensity`:

  (a) `_get_cell_area_m2` omits the cos(lat) shrink of a geographic cell,
      inflating polygon area by ~1/cos(lat) - polygon quantity only.
  (b) `_remove_duplicates` concatenates across processing tiles, double-counting
      cells for features spanning a tile border - up to 2.09x per feature.

`qty_weighted` is offered for a sensitivity contrast and is NOT immune to (b):
duplicated cells get double weight. It is not the default for that reason.

Protection is aggregated with `min`: a component is only as protected as its
least-protected constituent feature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .linkage import BUS, LINE, Linkage

ANCHOR_COLS = [f"new_rp{rp}_w{w}" for rp in (10, 100, 500)
               for w in (15, 20, 30, 40)]

# Direct-damage object types that correspond to each grid component kind. The
# transmission model has no towers or poles as separate elements - they are
# part of the line - so they are not consumed here; a line's hazard comes from
# the `line`/`cable`/`minor_line` features it is built from.
BUS_OBJECT_TYPES = {"substation", "transformer", "portal", "terminal",
                    "plant", "generator", "switch"}
LINE_OBJECT_TYPES = {"line", "cable", "minor_line"}


def load_segments(cfg: dict) -> pd.DataFrame:
    path = cfg["intermediate_dir"] / f"{cfg['country']}_{cfg['asset_type']}_segments.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Stage 1 for ({cfg['country']}, {cfg['asset_type']}) "
            "must have been run by the direct-damage study first."
        )
    seg = pd.read_parquet(path)
    seg["osm_id"] = seg["osm_id"].astype(str)
    return seg


def available_hazards(cfg: dict) -> list[str]:
    """Hazards with a Stage 1 profile on disk for this country."""
    stem = f"{cfg['country']}_{cfg['asset_type']}"
    return [hz for hz in ("river", "coastal", "earthquake", "windstorm")
            if (cfg["intermediate_dir"] / f"{stem}_{hz}_profiles.parquet").exists()]


def _component_map(link: Linkage, seg: pd.DataFrame, kind: str) -> pd.DataFrame:
    """(seg row, component index) pairs restricted to sensible object types."""
    types = BUS_OBJECT_TYPES if kind == BUS else LINE_OBJECT_TYPES
    keep = seg.index[seg["object_type"].isin(types)]
    sub = pd.DataFrame({"seg": keep.to_numpy(np.int64),
                        "osm_id": seg.loc[keep, "osm_id"].astype("string").to_numpy()})
    pairs = link.pairs(kind)
    return sub.merge(pairs, on="osm_id", how="inner")[["seg", "comp_idx"]]


def build_component_hazard(cfg: dict, link: Linkage, hazard: str,
                           seg: pd.DataFrame | None = None) -> dict:
    """Collapse Stage 1 profiles onto grid components for one hazard.

    Returns a dict with, per component kind:
        rps        (n_rp,)                ascending return periods
        intensity  (n_comp, n_rp)         aggregated hazard intensity, 0 = unexposed
        prot_rp    (n_comp,)              min protection RP over constituents
        hybas      (n_comp,)              basin id (0 = none), for event footprints
        anchors    (n_comp, 3, 4)         warming RP anchors, mean over constituents
        matched    (n_comp,) bool         had at least one exposure feature
    """
    if seg is None:
        seg = load_segments(cfg)
    prof = pd.read_parquet(
        cfg["intermediate_dir"]
        / f"{cfg['country']}_{cfg['asset_type']}_{hazard}_profiles.parquet"
    )
    rps = np.array(sorted(prof["rp"].unique()), dtype=np.float64)
    rp_index = {int(r): i for i, r in enumerate(rps)}
    prof = prof.assign(rp_idx=prof["rp"].astype(int).map(rp_index))

    agg = cfg["network"].get("intensity_aggregation", "max")
    out = {"rps": rps, "aggregation": agg}

    for kind, n_comp in ((BUS, link.n_bus), (LINE, link.n_line)):
        cmap = _component_map(link, seg, kind)
        joined = prof.merge(cmap, on="seg", how="inner")
        intensity = np.zeros((n_comp, len(rps)))
        if len(joined):
            flat = joined["comp_idx"].to_numpy() * len(rps) + joined["rp_idx"].to_numpy()
            vals = joined["intensity"].to_numpy(np.float64)
            if agg == "max":
                acc = np.zeros(n_comp * len(rps))
                np.maximum.at(acc, flat, vals)
            elif agg == "qty_weighted":
                q = joined["quantity"].to_numpy(np.float64)
                num = np.bincount(flat, weights=vals * q, minlength=n_comp * len(rps))
                den = np.bincount(flat, weights=q, minlength=n_comp * len(rps))
                acc = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
            else:
                raise ValueError(f"unknown intensity_aggregation '{agg}'")
            intensity = acc.reshape(n_comp, len(rps))

        # Per-component scalars, taken over the SAME constituent features.
        prot = np.zeros(n_comp)
        hybas = np.zeros(n_comp, dtype=np.int64)
        anchors = np.zeros((n_comp, 3, 4))
        matched = np.zeros(n_comp, dtype=bool)
        # Coastal protection is a different standard from river protection:
        # COASTPROS-EU joined to NUTS2, not FLOPROS. Stage 1 writes it as
        # `coast_prot_rp` for coastal countries only. It was never loaded, so
        # coastal scenarios ran against the river standard - for PRT that is
        # 55.2 yr instead of 141.8 yr, i.e. ~2.6x too permissive.
        prot_col = ("coast_prot_rp"
                    if hazard == "coastal" and "coast_prot_rp" in seg.columns
                    else "prot_rp")
        if len(cmap):
            g = cmap.merge(
                seg[[prot_col, "HYBAS_ID"] + ANCHOR_COLS].reset_index(names="seg"),
                on="seg", how="left",
            )
            byc = g.groupby("comp_idx")
            idx = byc.size().index.to_numpy()
            matched[idx] = True
            prot[idx] = byc[prot_col].min().to_numpy()
            # Modal basin: a component spanning basins is assigned the one most
            # of its features sit in, so footprint membership stays single-valued.
            hybas[idx] = byc["HYBAS_ID"].agg(
                lambda s: s.mode().iloc[0] if len(s.mode()) else 0
            ).to_numpy()
            # Clip and NaN-fill the warming anchors exactly as the direct-damage
            # original does (miraca_uq.risk_model, ANCHOR_CLIPS): a shifted
            # return period outside the source grid's support is extrapolation,
            # not signal, and NaN silently propagates through _shift_rps into
            # the integration grid. PRT carries 7,305 NaN per anchor column and
            # maxima well outside the clips (new_rp500_w40 reaches 3,634 against
            # a 1,000 ceiling); LUX is clean, which is why the benchmark never
            # showed this.
            a = byc[ANCHOR_COLS].mean().to_numpy().reshape(-1, 3, 4)
            nominal = np.array([10.0, 100.0, 500.0])[None, :, None]
            a = np.where(np.isfinite(a), a, np.broadcast_to(nominal, a.shape))
            lo = np.array([1.0, 1.0, 1.0])[None, :, None]
            hi = np.array([99.0, 499.0, 1000.0])[None, :, None]
            a = np.clip(a, lo, hi)
            # Force the (RP10, RP100, RP500) anchors to be non-decreasing. They
            # are a groupby-mean over a component's features, so averaging and
            # clipping can leave them out of order, which makes `_shift_rps`
            # non-monotone - and a non-monotone shift both unsorts the
            # integration grid and breaks the equivalence between testing
            # protection in the present-day frame and in the shifted frame
            # (see the plan, section 6.1). Monotone anchors make that
            # equivalence hold by construction rather than by luck.
            anchors[idx] = np.maximum.accumulate(a, axis=1)
        out[kind] = {"intensity": intensity, "prot_rp": prot, "hybas": hybas,
                     "anchors": anchors, "matched": matched}
    return out
