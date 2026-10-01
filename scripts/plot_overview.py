"""Overview figures for the network study, in the style of the direct-damage
study's `overview_figures/` (Uncertainty-Modelling/src/plot_drivers.py,
plot_heatmap.py, ead_ranges.py, plot_maps.py), so the two can sit side by side.

Reads every `results/<ISO3>/experiments_*_sobol_n<N>_*.tar.gz` plus the
`structural_zero_*.json` records and writes, under overview_figures/:

    drivers_ALL_hazards_overview.png   one pie per hazard: top Sobol driver
    heatmaps/Spread_and_zeros.png      country x hazard: p95/p5 and zero share
    eaens_ranges/all_hazards_by_country.png   p5-p95 per country and hazard
    maps/drivers_by_hazard.png         choropleth of the top driver
    maps/spread_by_hazard.png          choropleth of p95/p5
    EAENS_Ranges.csv, Sobol_Indices.csv       the numbers behind them

Sobol indices use SALib's Saltelli-2010 (S1) and Jansen (ST) estimators,
reimplemented here so the script needs no SALib. The sample layout is checked
rather than assumed: within each block of 2k+2 rows, row 1+j must differ from
row 0 only in factor j.

    python scripts/plot_overview.py [--n 512] [--geo PATH]
"""
from __future__ import annotations

import argparse
import io
import json
import re
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
OUT = REPO / "overview_figures"
DEFAULT_GEO = REPO.parent / "Uncertainty-Modelling" / "geo" / "CNTR_RG_20M_2024_4326.geojson"

HAZARDS = ["flood", "coastal", "earthquake", "windstorm"]
HAZARD_LABEL = {"flood": "River flood", "coastal": "Coastal flood",
                "earthquake": "Earthquake", "windstorm": "Windstorm"}
HAZARD_COLOR = {"flood": "#1baf7a", "coastal": "#2a78d6",
                "earthquake": "#d6453d", "windstorm": "#e8a33a"}

# --- palette: shared with the direct-damage figures where the meaning is shared
SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BORDER, CONTEXT_FILL = "#b9b8b0", "#f1f0ea"

CATEGORY = {
    "curve_bus": "Fragility curve", "curve_line": "Fragility curve",
    "pga_scale": "Hazard intensity", "gust_scale": "Hazard intensity",
    "depth_scale": "Hazard intensity",
    "warming": "Climate warming",
    "nuts3_gdp_weight": "Demand placement",
    "failure_correlation": "System assumptions", "gen_availability": "System assumptions",
    "failure_seed": "Failure sampling (aleatory)",
}
NO_EXPOSURE = "No exposure"
NO_GRID = "Exposed, no grid component"
NEVER_ENS = "Exposed, no ENS in any draw"
NEGLIGIBLE = "Negligible (< 1 MWh/yr)"
COLORS = {
    "Fragility curve": "#eb6834",      # = "Vulnerability curve" in the damage study
    "Hazard intensity": "#1baf7a",     # same meaning, same hue
    "Climate warming": "#c2185b",      # same meaning, same hue
    "Demand placement": "#d4a017",
    "System assumptions": "#7a5195",
    "Failure sampling (aleatory)": "#5f6b7a",
    NO_EXPOSURE: "#bfbdb1", NO_GRID: "#a09c8e", NEVER_ENS: "#b0ada0", NEGLIGIBLE: "#d9d8d1",
}
HATCH = {NEVER_ENS: "///"}
NEUTRAL = (NO_EXPOSURE, NO_GRID, NEVER_ENS, NEGLIGIBLE)

# Spread classes: the damage study's edges, so a colour means the same p95/p5 in both
SPREAD_RAMP = ("#8fb4e0", "#6f99cc", "#5480b6", "#3b649c", "#264a80", "#132a55")
SPREAD_EDGES = (5.0, 10.0, 25.0, 60.0, 200.0)
SPREAD_LABELS = ("under 5x", "5 - 10x", "10 - 25x", "25 - 60x", "60 - 200x", "over 200x")
P5_ZERO_FILL, P5_ZERO_HATCH_COLOR = "#132a55", "#9dc2ee"
ZERO_EDGES = (0.05, 0.20, 0.40, 0.60, 0.90)
ZERO_LABELS = ("under 5%", "5 - 20%", "20 - 40%", "40 - 60%", "60 - 90%", "90 - under 100%")
NEVER_ZERO_FILL = "#f7f6f1"

FLOOR = 1.0     # MWh/yr; the range plot's left strip, like the damage study's 1,000 EUR/yr


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def _sobol(X: pd.DataFrame, y: np.ndarray, rng, n_boot=500):
    cols = [c for c in X.columns if c not in ("scenario", "policy", "model")]
    k = len(cols); B = 2 * k + 2
    if len(y) % B:
        raise ValueError(f"{len(y)} rows is not a multiple of 2k+2={B}")
    N = len(y) // B
    V = X[cols].astype(str).to_numpy().reshape(N, B, k)
    for j in range(k):
        other = [i for i in range(k) if i != j]
        if not (V[:, 1 + j, other] == V[:, 0, other]).all():
            raise ValueError(f"row {1 + j} of each block differs from A outside {cols[j]}"
                             " - not the Saltelli layout this script assumes")
    Y = y.reshape(N, B)
    fA, fB, fAB = Y[:, 0], Y[:, -1], Y[:, 1:k + 1]

    def est(ii):
        a, b, ab = fA[ii], fB[ii], fAB[ii]
        v = np.var(np.concatenate([a, b]))
        if v == 0:
            return np.full(k, np.nan), np.full(k, np.nan)
        return ((b[:, None] * (ab - a[:, None])).mean(0) / v,
                0.5 * ((a[:, None] - ab) ** 2).mean(0) / v)

    s1, st = est(np.arange(N))
    boot = [est(rng.integers(0, N, N)) for _ in range(n_boot)]
    s1c = 1.96 * np.nanstd([b[0] for b in boot], axis=0)
    stc = 1.96 * np.nanstd([b[1] for b in boot], axis=0)
    return pd.DataFrame({"factor": cols, "S1": s1, "S1_conf": s1c, "ST": st, "ST_conf": stc})


def load(n: int):
    rng = np.random.default_rng(0)
    ranges, indices = [], []
    pat = re.compile(rf"experiments_([A-Z]{{3}})_powergrid_net_([a-z]+)_.*_sobol_n{n}_\d{{8}}_\d{{6}}\.tar\.gz$")
    for f in sorted(RESULTS.glob(f"*/experiments_*_sobol_n{n}_*.tar.gz")):
        m = pat.search(f.name)
        if not m:
            continue                                   # tagged / variant runs
        c, hz = m.groups()
        with tarfile.open(f) as t:
            X = pd.read_csv(t.extractfile("experiments.csv"))
            y = np.loadtxt(io.BytesIO(t.extractfile("EAENS_MWh.cls").read()))
            fb = np.loadtxt(io.BytesIO(t.extractfile("n_opf_fallback.cls").read()))
        q = np.percentile(y, [1, 5, 25, 50, 75, 95, 99])
        ranges.append(dict(country=c, hazard=hz, status="run", n_draws=len(y),
                           mean=y.mean(), p1=q[0], p5=q[1], p25=q[2], median=q[3],
                           p75=q[4], p95=q[5], p99=q[6], max=y.max(),
                           zero_fraction=(y == 0).mean(),
                           evals_with_opf_fallback=(fb > 0).mean(), source=f.name))
        s = _sobol(X, y, rng)
        s.insert(0, "hazard", hz); s.insert(0, "country", c)
        indices.append(s)
        print(f"  {c} {hz:10s} mean {y.mean():12,.1f} MWh/yr")
    for f in sorted(RESULTS.glob("*/structural_zero_*.json")):
        r = json.loads(f.read_text())
        ranges.append(dict(country=r["country"], hazard=r["hazard"], status=r["status"],
                           mean=0.0, zero_fraction=1.0, source=f.name))
    R = pd.DataFrame(ranges)
    R["hazard"] = R["hazard"].replace({"river": "flood"})
    S = pd.concat(indices, ignore_index=True)
    return R, S


def drivers(R: pd.DataFrame, S: pd.DataFrame) -> pd.DataFrame:
    """One row per (country, hazard): the category shown on pies and maps."""
    out = []
    for r in R.itertuples():
        top = second = None
        if r.status == "no_exposure":
            cat = NO_EXPOSURE
        elif r.status == "no_grid_exposure":
            cat = NO_GRID
        elif r.mean == 0:
            cat = NEVER_ENS
        elif r.mean < FLOOR:
            cat = NEGLIGIBLE
        else:
            g = S[(S.country == r.country) & (S.hazard == r.hazard)].sort_values("ST", ascending=False)
            top, second = g.iloc[0], g.iloc[1]
            cat = CATEGORY[top.factor]
        out.append(dict(country=r.country, hazard=r.hazard, category=cat,
                        top_factor=None if top is None else top.factor,
                        top_ST=None if top is None else top.ST,
                        top_ST_conf=None if top is None else top.ST_conf,
                        second_factor=None if second is None else second.factor,
                        second_ST=None if second is None else second.ST,
                        ranking_resolved=None if top is None else
                        bool(top.ST - top.ST_conf > second.ST + second.ST_conf)))
    return pd.DataFrame(out)


def spread_class(r) -> int | None:
    if r.status != "run" or r.mean == 0:
        return None
    if r.p5 == 0:
        return -1
    return int(np.searchsorted(SPREAD_EDGES, r.p95 / r.p5, side="right"))


def _style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.facecolor": SURFACE, "text.color": INK, "axes.labelcolor": INK_2,
        "xtick.color": INK_2, "ytick.color": INK_2, "axes.edgecolor": BORDER,
    })


def _legend_patch(cat):
    return Patch(facecolor=COLORS[cat], edgecolor=INK_2 if cat in HATCH else SURFACE,
                 hatch=HATCH.get(cat), label=cat)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_pies(D: pd.DataFrame, n_countries: int, path: Path):
    order = [k for k in COLORS]
    fig, axes = plt.subplots(1, 4, figsize=(20, 7.2))
    used = set()
    for ax, hz in zip(axes, HAZARDS):
        d = D[D.hazard == hz]
        counts = d.category.value_counts().reindex(order).dropna()
        counts = counts[counts > 0]
        used |= set(counts.index)
        wedges, _ = ax.pie(counts.values, colors=[COLORS[c] for c in counts.index],
                           startangle=90, counterclock=False,
                           wedgeprops=dict(edgecolor=SURFACE, linewidth=1.5))
        for w, c in zip(wedges, counts.index):
            if c in HATCH:
                w.set_hatch(HATCH[c]); w.set_edgecolor(INK_2)
        for w, (c, v) in zip(wedges, counts.items()):
            th = np.deg2rad((w.theta1 + w.theta2) / 2)
            light = c not in NEUTRAL
            ax.text(0.68 * np.cos(th), 0.68 * np.sin(th), int(v), ha="center", va="center",
                    fontsize=13, fontweight="bold", color="#ffffff" if light else INK)
        ax.set_title(HAZARD_LABEL[hz], fontsize=15, fontweight="bold", pad=10)
        na = n_countries - len(d)
        ax.text(0, -1.32, f"{len(d)} applicable of {n_countries} countries"
                + (f"\n{na} landlocked: not applicable" if na else ""),
                ha="center", va="top", fontsize=10.5, color=INK_2)
    fig.suptitle("Most influential parameter for expected energy not supplied - by hazard",
                 fontsize=19, fontweight="bold", y=0.97)
    fig.text(0.5, 0.875, "Share of countries whose highest Sobol total-effect index (ST) on EAENS "
             "falls in each parameter group.\nGrey slices are not drivers: there is nothing to "
             "attribute (no exposure, no grid component exposed, no ENS in any draw, or < 1 MWh/yr).",
             ha="center", fontsize=12, color=INK_2)
    fig.legend(handles=[_legend_patch(c) for c in order if c in used], loc="lower center",
               ncol=5, frameon=False, fontsize=11, bbox_to_anchor=(0.5, 0.01))
    fig.subplots_adjust(top=0.76, bottom=0.17, wspace=0.15)
    fig.savefig(path, dpi=150); plt.close(fig)


def _country_order(R):
    return list(R.groupby("country")["median"].sum().sort_values(ascending=False).index)


def fig_heatmaps(R: pd.DataFrame, countries: list, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.32 * len(countries) + 3.2))
    for ax, kind in zip(axes, ("spread", "zeros")):
        for i, c in enumerate(countries):
            for j, hz in enumerate(HAZARDS):
                r = R[(R.country == c) & (R.hazard == hz)]
                if r.empty:
                    continue                           # not applicable: blank
                r = r.iloc[0]
                hatch, ec = None, SURFACE
                if r.status != "run" or r["mean"] == 0:
                    fc = COLORS[NO_EXPOSURE] if r.status == "no_exposure" else (
                        COLORS[NO_GRID] if r.status == "no_grid_exposure" else COLORS[NEVER_ENS])
                    if r.status == "run":
                        hatch, ec = HATCH[NEVER_ENS], INK_2
                elif kind == "spread":
                    cls = spread_class(r)
                    if cls == -1:
                        fc, hatch, ec = P5_ZERO_FILL, "////", P5_ZERO_HATCH_COLOR
                    else:
                        fc = SPREAD_RAMP[cls]
                else:
                    z = r.zero_fraction
                    fc = NEVER_ZERO_FILL if z == 0 else SPREAD_RAMP[int(np.searchsorted(ZERO_EDGES, z, side="right"))]
                ax.add_patch(Rectangle((j, i), 1, 1, facecolor=fc, edgecolor=ec, hatch=hatch, linewidth=0.8))
        ax.set_xlim(0, 4); ax.set_ylim(len(countries), 0)
        ax.set_yticks(np.arange(len(countries)) + 0.5, countries, fontsize=9.5)
        ax.set_xticks(np.arange(4) + 0.5, [HAZARD_LABEL[h] for h in HAZARDS], rotation=30, ha="right")
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title("How wide is the plausible range?\np95 / p5 of EAENS" if kind == "spread"
                     else "How often is there no ENS at all?\nshare of draws with EAENS = 0",
                     fontsize=12.5, fontweight="bold")
    neutral = [Patch(facecolor=COLORS[NO_EXPOSURE], label=NO_EXPOSURE),
               Patch(facecolor=COLORS[NO_GRID], label=NO_GRID),
               _legend_patch(NEVER_ENS)]
    h1 = [Patch(facecolor=c, label=l) for c, l in zip(SPREAD_RAMP, SPREAD_LABELS)]
    h1.append(Patch(facecolor=P5_ZERO_FILL, edgecolor=P5_ZERO_HATCH_COLOR, hatch="////",
                    label="p5 = 0 (band reaches zero)"))
    h2 = [Patch(facecolor=NEVER_ZERO_FILL, edgecolor=BORDER, linewidth=0.4, label="0% - ENS in every draw")]
    h2 += [Patch(facecolor=c, label=l) for c, l in zip(SPREAD_RAMP, ZERO_LABELS)]
    axes[0].legend(handles=h1 + neutral, loc="upper center", bbox_to_anchor=(0.5, -0.09),
                   frameon=False, fontsize=8.5, ncol=2)
    axes[1].legend(handles=h2 + [_legend_patch(NEVER_ENS)], loc="upper center", bbox_to_anchor=(0.5, -0.09),
                   frameon=False, fontsize=8.5, ncol=2)
    fig.text(0.5, 0.005, "countries ordered by total median EAENS  |  blank = hazard not applicable "
             "(landlocked)  |  spread classes identical to the direct-damage heatmap",
             ha="center", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(path, dpi=150); plt.close(fig)


def fig_ranges(R: pd.DataFrame, countries: list, path: Path):
    rows = []
    for c in countries:
        for hz in HAZARDS:
            r = R[(R.country == c) & (R.hazard == hz)]
            if not r.empty:
                rows.append(r.iloc[0])
    n = len(rows)
    fig, ax = plt.subplots(figsize=(11, 0.17 * n + 2.5))
    lo, hi = FLOOR / 3, R["p99"].max() * 2
    ax.set_xscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(n, -1)
    ax.axvspan(lo, FLOOR, color="#efeee8", zorder=0)
    ylab, prev, band = [], None, False
    for i, r in enumerate(rows):
        if r.country != prev:
            band = not band; prev = r.country
        if band:
            ax.axhspan(i - 0.5, i + 0.5, color="#eef3fb", zorder=0, lw=0)
        ylab.append(f"{r.country}  {HAZARD_LABEL[r.hazard]}")
        col = HAZARD_COLOR[r.hazard]
        xm = FLOOR * 0.62
        if r.status != "run" or r["mean"] == 0:
            ax.plot(xm, i, marker="x", color=INK_2, ms=5)
            continue
        if r.p95 < FLOOR:
            ax.plot(xm, i, marker="<", color=INK_2, ms=5)
            continue
        if r.p5 == 0:
            ax.plot(xm, i, marker="o", mfc="none", color=INK_2, ms=5)
        ax.hlines(i, max(r.p5, FLOOR), r.p95, color=col, lw=1.2, zorder=2)
        ax.hlines(i, max(r.p25, FLOOR), max(r.p75, FLOOR), color=col, lw=4, zorder=3)
        if r["median"] >= FLOOR:
            ax.plot(r["median"], i, marker="|", color=INK, ms=8, mew=2, zorder=4)
        ax.plot(r["mean"], i, marker="D", mfc="white", mec=INK, ms=4.5, zorder=5)
    ax.set_yticks(range(n), ylab, fontsize=7.5)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, lw=0.6, which="major")
    ax.set_xlabel("EAENS (MWh / yr)")
    ax.xaxis.set_ticks_position("both"); ax.tick_params(axis="x", labeltop=True)
    for s in ("left", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("Expected annual energy not supplied - all hazards, by country\n"
                 "Sobol design, N = 512 base samples per combination  |  grey strip = below "
                 f"{FLOOR:g} MWh/yr, marks there say what is off the scale",
                 fontsize=11, loc="left", color=INK)
    h = [Line2D([], [], color=INK_2, lw=1.2, label="p5 - p95"),
         Line2D([], [], color=INK_2, lw=4, label="p25 - p75"),
         Line2D([], [], color=INK, marker="|", ls="", ms=8, mew=2, label="median"),
         Line2D([], [], marker="D", mfc="white", mec=INK, ls="", label="mean"),
         Line2D([], [], marker="o", mfc="none", color=INK_2, ls="", label="p5 = 0"),
         Line2D([], [], marker="<", color=INK_2, ls="", label="whole band below floor"),
         Line2D([], [], marker="x", color=INK_2, ls="", label="no ENS in any draw / structural zero")]
    h += [Line2D([], [], color=HAZARD_COLOR[k], lw=4, label=HAZARD_LABEL[k]) for k in HAZARDS]
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, fontsize=8.5)
    fig.savefig(path, dpi=150); plt.close(fig)


def _geometry(geo_path: Path):
    import geopandas as gpd
    from shapely.geometry import box
    g = gpd.read_file(geo_path)
    g = g[g["ISO3_CODE"].notna()].copy()
    g = gpd.clip(g, box(-26.0, 33.0, 45.0, 72.0))
    g = g[~g.geometry.is_empty & g.geometry.notna()].to_crs(3035)
    g = g.dissolve(by="ISO3_CODE")
    g["is_micro"] = (g.geometry.area / 1e6) < 5_000.0
    g["point"] = g.geometry.representative_point()
    return g


def _map_panel(ax, geo, fills, extent, hatched=(), title="", never=()):
    study = [c for c in fills if c in geo.index]
    geo.loc[geo.index.difference(study)].plot(ax=ax, facecolor=CONTEXT_FILL, edgecolor=BORDER, lw=0.3)
    for c in study:
        g = geo.loc[[c]]
        h = c in hatched
        nv = c in never
        g.plot(ax=ax, facecolor=P5_ZERO_FILL if h else fills[c],
               hatch="////" if h else (HATCH[NEVER_ENS] if nv else None),
               edgecolor=P5_ZERO_HATCH_COLOR if h else (INK_2 if nv else SURFACE), lw=0.5)
        p = geo.loc[c, "point"]
        if geo.loc[c, "is_micro"]:
            ax.scatter([p.x], [p.y], s=60, marker="s", color=fills[c], edgecolor=INK, lw=0.8, zorder=5)
        ax.text(p.x, p.y, c, fontsize=6.5, ha="center", va="center", color=INK, zorder=6,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.6))
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ax.set_axis_off(); ax.set_title(title, fontsize=13)


def fig_maps(R, D, geo_path: Path, out_dir: Path):
    geo = _geometry(geo_path)
    present = [c for c in R.country.unique() if c in geo.index]
    minx, miny, maxx, maxy = geo.loc[present].total_bounds
    ext = (minx - 0.03 * (maxx - minx), maxx + 0.03 * (maxx - minx),
           miny - 0.03 * (maxy - miny), maxy + 0.03 * (maxy - miny))

    fig, axes = plt.subplots(2, 2, figsize=(11, 13))
    used = set()
    for ax, hz in zip(axes.flat, HAZARDS):
        d = D[D.hazard == hz]
        fills = {r.country: COLORS[r.category] for r in d.itertuples()}
        used |= set(d.category)
        _map_panel(ax, geo, fills, ext, title=f"{HAZARD_LABEL[hz]}  (n={len(d)})",
                   never=set(d.loc[d.category == NEVER_ENS, "country"]))
    fig.suptitle("What drives the uncertainty in expected energy not supplied?",
                 fontsize=18, y=0.985)
    fig.legend(handles=[_legend_patch(c) for c in COLORS if c in used], loc="lower center",
               ncol=3, frameon=False, fontsize=10.5, bbox_to_anchor=(0.5, 0.03))
    fig.text(0.5, 0.012, "colour = the parameter group with the highest Sobol total effect on EAENS  |  "
             "grey = nothing to attribute  |  not coloured = landlocked or not in the study",
             ha="center", fontsize=10, color=MUTED)
    fig.subplots_adjust(top=0.94, bottom=0.14, left=0.01, right=0.99, hspace=0.08, wspace=0.02)
    fig.savefig(out_dir / "drivers_by_hazard.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 13))
    for ax, hz in zip(axes.flat, HAZARDS):
        fills, hatched = {}, []
        for r in R[R.hazard == hz].itertuples():
            cls = spread_class(r)
            if cls is None:
                fills[r.country] = COLORS[NO_EXPOSURE] if r.status == "no_exposure" else (
                    COLORS[NO_GRID] if r.status == "no_grid_exposure" else COLORS[NEVER_ENS])
            elif cls == -1:
                fills[r.country] = P5_ZERO_FILL; hatched.append(r.country)
            else:
                fills[r.country] = SPREAD_RAMP[cls]
        never = [r.country for r in R[R.hazard == hz].itertuples() if r.status == "run" and r.mean == 0]
        _map_panel(ax, geo, fills, ext, hatched, title=HAZARD_LABEL[hz], never=never)
    h = [Patch(facecolor=c, label=l) for c, l in zip(SPREAD_RAMP, SPREAD_LABELS)]
    h.append(Patch(facecolor=P5_ZERO_FILL, edgecolor=P5_ZERO_HATCH_COLOR, hatch="////", label="p5 = 0"))
    h += [_legend_patch(k) for k in (NO_EXPOSURE, NO_GRID, NEVER_ENS)]
    fig.suptitle("How wide is the plausible range of EAENS?  (p95 / p5)", fontsize=18, y=0.985)
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, fontsize=10.5)
    fig.subplots_adjust(top=0.94, bottom=0.11, left=0.01, right=0.99, hspace=0.08, wspace=0.02)
    fig.savefig(out_dir / "spread_by_hazard.png", dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--geo", type=Path, default=DEFAULT_GEO)
    a = ap.parse_args()
    for sub in ("heatmaps", "eaens_ranges", "maps"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    _style()
    R, S = load(a.n)
    D = drivers(R, S)
    countries = _country_order(R)
    R.to_csv(OUT / "EAENS_Ranges.csv", index=False)
    S.merge(D[["country", "hazard"]], on=["country", "hazard"]).to_csv(OUT / "Sobol_Indices.csv", index=False)
    D.to_csv(OUT / "Top_Drivers.csv", index=False)
    fig_pies(D, len(countries), OUT / "drivers_ALL_hazards_overview.png")
    fig_heatmaps(R, countries, OUT / "heatmaps" / "Spread_and_zeros.png")
    fig_ranges(R, countries, OUT / "eaens_ranges" / "all_hazards_by_country.png")
    if a.geo.is_file():
        fig_maps(R, D, a.geo, OUT / "maps")
    else:
        print(f"no country geometry at {a.geo} - maps skipped")
    print(f"\n{len(R)} combinations, {R.country.nunique()} countries -> {OUT}")


if __name__ == "__main__":
    main()
