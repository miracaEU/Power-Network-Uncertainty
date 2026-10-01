"""2x4 maps of base-case network state: 4 disaggregations x HVDC on/off.

Each figure is one hour of the OPSD series solved eight ways:

                equal      pop+GDP      pop only     GDP only
    AC only       .            .            .            .
    AC + HVDC     .            .            .            .

so the two independent modelling choices - where demand sits, and whether the
DC layer exists - can be read off separately, and the three NUTS3 weightings
compared against each other and against the equal-split placeholder.

Two things the colours mean, and they are NOT the same thing:

  * **AC lines have no thermal limit in this solve** (`max_loading_percent` is
    unset, see ens_model.load_net_data). They are free to exceed 100%, and the
    point of the figure is that ~13% of them do before any hazard is applied.
    Each panel reports the worst line, which is the number that says whether
    this is a mild modelling artefact or a broken base case.
  * **HVDC links ARE capacity-bounded.** A pandapower dcline is a controllable
    injection with transfer bounded [0, max_p_mw], so a DC link can never read
    over 100% - it saturates instead. For DC, "at cap" means "binding, and
    would carry more if it could".

Because a dcline is unidirectional, every real link is added as two opposing
dclines each capped at `p_nom`; a small per-MW cost stops both running at once.
A DC link does not merge synchronous areas - it lets an otherwise isolated AC
island import - which is why it moves the shed a great deal and the AC loadings
very little.

Coloured frames mark, per hour, the scenario with the fewest overloaded AC
lines (green) and the one with the least load shed (purple). A panel can win
both, in which case it carries both frames.

Run:
    python scripts/map_network_state.py [--outdir results/diagnostics]
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from power_net.dcopf_lp import MIN_SHED_GEN_COST, build_lp_net, solve_dc_opf
from power_net.hvdc import dc_terminal_mask, load_hvdc
from power_net.demand import _demand_frame, bus_load_mw, national_demand_at
from power_net.disaggregation import nuts3_bus_weights
from power_net.injections import (add_shedding_units, add_slack,
                                  assign_generation, assign_loads)
from power_net.network import (ISO2_TO_ISO3, VOLL_EUR_MWH, build_base_network,
                               bus_coords_3035)

# Hours spanning the daily and seasonal cycle. The first is the hour
# BENCHMARK.md measured, kept so its 12.9% stays directly comparable. The
# minimum-load hour of the whole series is found at run time and appended.
HOURS = [
    ("2018-01-24T17:00:00Z", "winter weekday evening peak"),
    ("2018-01-25T03:00:00Z", "winter night trough"),
    ("2018-04-11T08:00:00Z", "spring morning ramp"),
    ("2018-07-18T14:00:00Z", "summer weekday afternoon"),
    ("2018-07-19T03:00:00Z", "summer night trough"),
    ("2018-10-17T18:00:00Z", "autumn evening peak"),
]

# Columns of the figure: how national demand is placed on buses.
DISAGG = [
    ("equal",     "equal split (current default)"),
    ("nuts3",     "NUTS3 population + GDP (50/50)"),
    ("nuts3_pop", "NUTS3 population only"),
    ("nuts3_gdp", "NUTS3 GDP only"),
]

# Top edge is finite, not inf: BoundaryNorm propagates inf into the colorbar's
# axis limits and matplotlib rejects them.
BANDS = [0, 25, 50, 75, 100, 150, 200, 400, 10_000]
BAND_COLORS = ["#2c7fb8", "#41b6c4", "#7fcdbb", "#c7e9b4",
               "#fed976", "#fd8d3c", "#e31a1c", "#67000d"]

# With thermal limits enforced nothing can exceed 100% (or 100 x rating_factor),
# so the unconstrained scale above spends seven of its eight bands on a range
# that is empty and crushes everything real into the first two. These bands step
# every 10% instead, with a dedicated top band for lines AT their limit - the
# ones that are actually binding, which is the thing worth seeing.
BANDS_LIMITED = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99, 110]
BAND_COLORS_LIMITED = ["#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#4292c6",
                       "#41b6c4", "#78c679", "#c7e9b4", "#fed976", "#fd8d3c",
                       "#e31a1c"]


def loading_scale(limits: bool, rating_factor: float = 1.0):
    """(bands, colours, tick labels) for the line-loading colour scale."""
    if not limits:
        return (BANDS, BAND_COLORS,
                ["0", "25", "50", "75", "100", "150", "200", "400", ">400"])
    bands = list(BANDS_LIMITED)
    if rating_factor > 1.0:
        bands[-1] = max(110.0, 100.0 * rating_factor)
    labels = [f"{b:g}" for b in bands[:-1]] + [f"{bands[-1]:g}"]
    labels[-2] = "99"          # the band below the cap
    return bands, BAND_COLORS_LIMITED, labels
SATURATED = 99.0          # a dcline this close to its cap is binding
WIN_LINES = "#1a9850"     # frame: fewest overloaded AC lines
WIN_SHED = "#6a3d9a"      # frame: least load shed


def lowest_load_hour(em: Path, zones: set[str]) -> tuple[str, float]:
    """The hour of the whole series with the least total demand.

    Restricted to hours where EVERY country in the grid model reports, so the
    answer is a genuine demand minimum and not simply an hour with the most
    missing data - which is what an unguarded argmin returns.
    """
    df = _demand_frame(str(em))
    cols = [c for c in df.columns if c in zones]
    full = df[cols].dropna()
    tot = full.sum(axis=1)
    ts = tot.idxmin()
    print(f"  {len(full):,} of {len(df):,} hours have all {len(cols)} countries; "
          f"min {tot.min():,.0f} MW at {ts}  (max {tot.max():,.0f} MW)")
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ"), float(tot.min())


def hvdc_link_table(net, em: Path) -> pd.DataFrame:
    """DC links AND the converters that couple them to the AC grid.

    Delegates to `power_net.hvdc.load_hvdc`, which joins both files on the EXACT
    bus name. This function used to strip the voltage suffix off each link
    endpoint and take the highest-voltage bus sharing the remaining OSM id -
    a guess, and the wrong one. 38 of the 70 DC terminals have an AC twin
    differing only by that suffix, so the heuristic silently chose between a
    converter's DC terminal and its own AC bus; on the 2 GW GB-France link the
    AC bus is `way/1271288345-400` while the terminal is `way/1271288346`, so no
    id-based rule could have found it. `converters.csv` states the AC bus
    outright. The old behaviour is what put injections on corridors that could
    not evacuate them and produced the 1,670% loading.

    The returned frame carries a `kind` column ("link" / "converter"): every row
    is a capacity-bounded transfer for the LP, but only the links are
    interconnectors worth drawing - a converter joins two co-located buses and
    would plot as a zero-length segment.
    """
    return load_hvdc(em, net)


def add_dclines(net, links: pd.DataFrame) -> pd.DataFrame:
    """Add the links as opposing dcline pairs (pandapower backend only)."""
    import pandapower as pp

    fwd, bwd = [], []
    for _, r in links.iterrows():
        idx = []
        for a, b in ((int(r.from_bus), int(r.to_bus)), (int(r.to_bus), int(r.from_bus))):
            k = pp.create_dcline(
                net, from_bus=a, to_bus=b, p_mw=0.0,
                loss_percent=0.0, loss_mw=0.0, vm_from_pu=1.0, vm_to_pu=1.0,
                max_p_mw=float(r.p_nom), min_q_from_mvar=-1.0, max_q_from_mvar=1.0,
                min_q_to_mvar=-1.0, max_q_to_mvar=1.0)
            pp.create_poly_cost(net, k, "dcline", cp1_eur_per_mw=0.1)
            idx.append(k)
        fwd.append(idx[0]); bwd.append(idx[1])
    out = links.copy()
    out["fwd"], out["bwd"] = fwd, bwd
    return out


def _pack(loading, dc, links, shed, total, shed_bus=None, shed_mw=None,
          shed_frac=None):
    return {"lp": loading, "dc": dc, "links": links, "shed": shed,
            "over": int(np.nansum(loading > 100)),
            "n_ac": int(np.isfinite(loading).sum()),
            "max_load": float(np.nanmax(loading)), "total": total,
            "shed_pct": 100.0 * shed / total if total > 0 else np.nan,
            "shed_bus": shed_bus, "shed_mw": shed_mw, "shed_frac": shed_frac}


def run_variant(net0, xy, plants, load, em, with_dc, backend, objective, limits,
                rating_factor=1.0):
    """Build one variant and solve it. Returns a result dict."""
    net = copy.deepcopy(net0)
    assign_loads(net, load)
    assign_generation(net, plants, xy, availability=1.0)
    add_shedding_units(net, VOLL_EUR_MWH)
    add_slack(net, mode="per_synchronous_area")
    links = hvdc_link_table(net, em) if with_dc else None
    total = float(load.sum())

    if backend == "highs":
        lpn = build_lp_net(net, voll=VOLL_EUR_MWH)
        if links is not None and len(links):
            lpn.dc_from = links.from_bus.to_numpy(np.int64)
            lpn.dc_to = links.to_bus.to_numpy(np.int64)
            lpn.dc_cap = links.p_nom.to_numpy(float)
        r = solve_dc_opf(lpn, np.zeros(lpn.n_line, bool), np.zeros(lpn.n_bus, bool),
                         1.0, 1.0, thermal_limits=limits, objective=objective,
                         rating_factor=rating_factor)
        if not r["success"]:
            print(f"      LP FAILED: {r['status'][:80]}")
            return {"lp": None, "dc": None, "links": links}
        loading = r["loading"][:lpn.n_line]
        dc = None
        if links is not None and len(links):
            dc = 100.0 * np.abs(r["dc_flow"]) / links.p_nom.to_numpy()
        return _pack(loading, dc, links, r["shed"], total,
                     r["shed_bus"], r["shed_unit"], r["shed_frac"])

    import pandapower as pp
    if links is not None and len(links):
        links = add_dclines(net, links)
    if limits:
        net.line["max_loading_percent"] = 100.0 * rating_factor
        net.trafo["max_loading_percent"] = 100.0 * rating_factor
    if objective == "min_shed":
        g = net.poly_cost.et == "gen"
        net.poly_cost.loc[g, "cp1_eur_per_mw"] = MIN_SHED_GEN_COST
    try:
        pp.rundcopp(net)
    except Exception as e:                                   # noqa: BLE001
        print(f"      OPF FAILED: {type(e).__name__}: {str(e)[:80]}")
        return {"lp": None, "dc": None, "links": links}
    loading = net.res_line.loading_percent.replace([np.inf, -np.inf], np.nan).to_numpy()
    shed = float(net.res_sgen.p_mw.sum())
    dc = None
    if links is not None and len(links):
        pf = net.res_dcline["p_from_mw"].to_numpy()
        flow = np.maximum(pf[links.fwd.to_numpy()], pf[links.bwd.to_numpy()])
        dc = 100.0 * np.abs(flow) / links.p_nom.to_numpy()
    return _pack(loading, dc, links, shed, total)


def _frame(ax, xy):
    ax.set_aspect("equal")
    ax.set_xlim(xy[:, 0].min() - 5e4, xy[:, 0].max() + 5e4)
    ax.set_ylim(xy[:, 1].min() - 5e4, xy[:, 1].max() + 5e4)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("#bbbbbb"); s.set_linewidth(1.0)


def figure_2x4(xy, seg, results, stamp, label, out: Path, note: str = "",
               limits: bool = False, rating_factor: float = 1.0):
    bands, colors, tick_labels = loading_scale(limits, rating_factor)
    cmap, norm = ListedColormap(colors), BoundaryNorm(bands, len(colors))
    fig, axes = plt.subplots(2, 4, figsize=(26, 14.2))

    ok = {k: v for k, v in results.items() if v["lp"] is not None}
    best_lines = min(ok, key=lambda k: ok[k]["over"]) if ok else None
    best_shed = min(ok, key=lambda k: ok[k]["shed"]) if ok else None

    for r, dc_on in enumerate((False, True)):
        for c, (mode, mode_label) in enumerate(DISAGG):
            ax = axes[r, c]
            res = results[(mode, dc_on)]
            _frame(ax, xy)
            if res["lp"] is None:
                ax.text(.5, .5, "OPF did not converge", ha="center",
                        transform=ax.transAxes, fontsize=12)
                continue
            v = res["lp"]
            order = np.argsort(np.nan_to_num(v))            # hot lines on top
            lc = LineCollection(seg[order], array=v[order], cmap=cmap, norm=norm,
                                linewidths=.28 + 1.3 * (np.nan_to_num(v)[order] > 100),
                                zorder=2)
            ax.add_collection(lc)
            extra = ""
            if dc_on and res["dc"] is not None:
                lk = res["links"]
                # Draw interconnectors only. A converter joins a DC terminal to
                # its own AC bus - the same place - so it plots as a dot.
                is_link = (lk["kind"] == "link").to_numpy() if "kind" in lk \
                    else np.ones(len(lk), bool)
                dseg = np.stack([xy[lk.from_bus.to_numpy()[is_link]],
                                 xy[lk.to_bus.to_numpy()[is_link]]], axis=1)
                ax.add_collection(LineCollection(dseg, colors="black",
                                                 linewidths=5.0, zorder=3))
                ax.add_collection(LineCollection(dseg, array=res["dc"][is_link],
                                                 cmap=cmap, norm=norm,
                                                 linewidths=3.0, zorder=4))
                nl_ = int(is_link.sum()); nc_ = int((~is_link).sum())
                extra = (f"\nHVDC links {int((res['dc'][is_link] >= SATURATED).sum())}"
                         f"/{nl_} at cap  ·  median {np.median(res['dc'][is_link]):.0f}%"
                         + (f"  ·  converters {int((res['dc'][~is_link] >= SATURATED).sum())}"
                            f"/{nc_} at cap" if nc_ else ""))
            ax.set_title(
                f"{mode_label}\n"
                f"{res['over']:,}/{res['n_ac']:,} AC over 100% "
                f"({100*res['over']/res['n_ac']:.1f}%)  ·  worst {res['max_load']:,.0f}%\n"
                f"shed {res['shed']:,.0f} MW = {res['shed_pct']:.2f}% of demand{extra}",
                fontsize=10.2, linespacing=1.35)

            key = (mode, dc_on)
            if key == best_lines:
                for s in ax.spines.values():
                    s.set_edgecolor(WIN_LINES); s.set_linewidth(4.0)
            if key == best_shed:
                ax.add_patch(Rectangle((.012, .012), .976, .976,
                                       transform=ax.transAxes, fill=False,
                                       edgecolor=WIN_SHED, linewidth=3.4,
                                       zorder=10, clip_on=False))

    for r, rl in enumerate(("AC only\n(as modelled today)", "AC + HVDC links\n(+32.5 GW)")):
        axes[r, 0].set_ylabel(rl, fontsize=13, labelpad=14)

    cb = fig.colorbar(lc, ax=axes, fraction=.014, pad=.012, spacing="uniform")
    cb.set_ticks(bands)
    cb.set_ticklabels(tick_labels)
    cb.set_label(
        ("utilisation (% of nameplate rating). Thermal limits ENFORCED, so the top "
         "band is the set of lines actually AT their limit." if limits else
         "utilisation (% of rating).  AC lines are UNLIMITED here and may exceed "
         "100%;  HVDC is capacity-bounded and saturates at 100%."), fontsize=10)

    fig.legend(handles=[
        Line2D([], [], color=WIN_LINES, lw=4, label="fewest overloaded AC lines"),
        Line2D([], [], color=WIN_SHED, lw=3.4, label="least load shed"),
        Line2D([], [], color="black", lw=5, label="HVDC link (coloured core = its own utilisation)"),
    ], loc="lower center", ncol=3, frameon=False, fontsize=11.5,
        bbox_to_anchor=(.5, .045))
    tot = next(iter(ok.values()))["total"] if ok else float("nan")
    fig.suptitle(f"Base-case network state - {label}  ({stamp[:16]}Z)   ·   "
                 f"total demand {tot:,.0f} MW\n"
                 "no hazard applied; columns = load disaggregation, rows = HVDC layer"
                 + (f"\n{note}" if note else ""),
                 fontsize=16, y=.965)
    fig.savefig(out, dpi=125, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    wrote {out.name}")


def figure_shed_2x4(xy, seg, results, stamp, label, out: Path, note: str = ""):
    """Where the unserved load sits, rather than how hard the lines work.

    Marker AREA is the MW shed at that bus; COLOUR is the fraction of that bus's
    own demand that went unserved. The two say different things and both matter:
    a large pale marker is a big node curtailed a little to relieve a corridor,
    a small dark one is a small node cut off completely. Only colour would hide
    the first; only size would hide the second.
    """
    fig, axes = plt.subplots(2, 4, figsize=(26, 14.2))
    ok = {k: v for k, v in results.items() if v["lp"] is not None}
    best_shed = min(ok, key=lambda k: ok[k]["shed"]) if ok else None
    # One size scale across all eight panels, or they cannot be compared.
    vmax = max((v["shed_mw"].max() for v in ok.values()
                if v["shed_mw"] is not None and len(v["shed_mw"])), default=1.0)

    for r, dc_on in enumerate((False, True)):
        for c, (mode, mode_label) in enumerate(DISAGG):
            ax = axes[r, c]
            res = results[(mode, dc_on)]
            ax.add_collection(LineCollection(seg, colors="#e2e2e2",
                                             linewidths=.25, zorder=1))
            _frame(ax, xy)
            if res["lp"] is None or res["shed_mw"] is None:
                ax.text(.5, .5, "no solution", ha="center", transform=ax.transAxes)
                continue
            m = res["shed_mw"] > 1e-6
            b = res["shed_bus"][m]
            mw = res["shed_mw"][m]
            fr = 100.0 * res["shed_frac"][m]
            size = 6 + 300 * np.sqrt(np.clip(mw / vmax, 0, 1))
            sc = ax.scatter(xy[b, 0], xy[b, 1], s=size, c=fr, cmap="inferno_r",
                            vmin=0, vmax=100, alpha=.85, linewidths=.3,
                            edgecolors="#333333", zorder=3)
            full = int((fr > 99.5).sum())
            ax.set_title(
                f"{mode_label}   |   {'AC + HVDC' if dc_on else 'AC only'}\n"
                f"shed {res['shed']:,.0f} MW = {res['shed_pct']:.2f}% of demand "
                f"at {int(m.sum()):,} buses\n"
                f"largest {mw.max():,.0f} MW at one bus  ·  {full:,} buses cut off "
                f"entirely", fontsize=10.2, linespacing=1.35)
            if (mode, dc_on) == best_shed:
                for s in ax.spines.values():
                    s.set_edgecolor(WIN_SHED); s.set_linewidth(4.0)

    for r, rl in enumerate(("AC only\n(as modelled today)", "AC + HVDC links\n(+32.5 GW)")):
        axes[r, 0].set_ylabel(rl, fontsize=13, labelpad=14)
    cb = fig.colorbar(sc, ax=axes, fraction=.014, pad=.012)
    cb.set_label("share of that bus's own demand left unserved (%).  "
                 "Marker area is MW shed, on one scale across all panels.",
                 fontsize=10)
    fig.legend(handles=[Line2D([], [], color=WIN_SHED, lw=4, label="least total shed")],
               loc="lower center", ncol=1, frameon=False, fontsize=11.5,
               bbox_to_anchor=(.5, .055))
    tot = next(iter(ok.values()))["total"] if ok else float("nan")
    fig.suptitle(f"Where the unserved load is - {label}  ({stamp[:16]}Z)   ·   "
                 f"total demand {tot:,.0f} MW\n"
                 "no hazard applied; columns = load disaggregation, rows = HVDC layer"
                 + (f"\n{note}" if note else ""), fontsize=16, y=.965)
    fig.savefig(out, dpi=125, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    wrote {out.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--backend", default="highs", choices=("highs", "pandapower"))
    ap.add_argument("--objective", default="min_shed", choices=("min_shed", "economic"))
    ap.add_argument("--limits", action="store_true", help="enforce thermal limits")
    ap.add_argument("--exclude-dc-load", action="store_true",
                    help="do not give HVDC converter terminals a share of "
                         "national demand (they are stations, not demand centres)")
    ap.add_argument("--shed-maps", action="store_true",
                    help="map where the unserved load sits, not line loading")
    ap.add_argument("--rating-factor", type=float, default=1.0,
                    help="scale the ENFORCED limit; loading is still reported "
                         "against nameplate, so 1.1 can read up to 110%%")
    args = ap.parse_args()
    default_dir = ("results/diagnostics/economic dispatch"
                   if args.objective == "economic" and args.backend == "pandapower"
                   else "results/diagnostics/min-shed HiGHS")
    out = Path(args.outdir or default_dir); out.mkdir(parents=True, exist_ok=True)
    print(f"backend={args.backend}  objective={args.objective}  "
          f"thermal_limits={args.limits}  ->  {out}")

    from power_net.paths import load_config
    em = load_config()["energy_model_dir"]

    print("building network...")
    net0, geo = build_base_network(em)
    xy = bus_coords_3035(geo)
    seg = np.stack([xy[net0.line.from_bus.to_numpy()],
                    xy[net0.line.to_bus.to_numpy()]], axis=1)
    plants = pd.read_csv(em / "powerplants.csv")

    print("computing NUTS3 weights (3 variants)...")
    weights = {"equal": None}
    for mode, _ in DISAGG[1:]:
        weights[mode], rep = nuts3_bus_weights(net0, geo, em, mode=mode)
        print(f"  {mode}: {rep['n_bus_weighted']} weighted, "
              f"{rep['n_bus_equal_fallback']} equal fallback")

    dc_excl = None
    if args.exclude_dc_load:
        dc_excl = dc_terminal_mask(em, net0)
        print(f"excluding {int(dc_excl.sum())} HVDC converter terminals from the "
              f"load split")

    print("finding the minimum-load hour of the series...")
    zones = set(net0.bus["zone"].dropna().astype(str)) & set(ISO2_TO_ISO3)
    min_stamp, min_mw = lowest_load_hour(em, zones)
    hours = HOURS + [(min_stamp, f"MINIMUM-LOAD HOUR of the series ({min_mw:,.0f} MW)")]

    rows = []
    for stamp, label in hours:
        print(f"\n{stamp}  ({label})")
        try:
            dem = national_demand_at(em, stamp)
        except KeyError:
            print("  not in the OPSD series - skipped"); continue
        loads = {m: bus_load_mw(net0, dem, mode=m, weights=weights[m],
                                exclude=dc_excl)[0]
                 for m, _ in DISAGG}
        results = {}
        for mode, _ in DISAGG:
            for dc_on in (False, True):
                t = time.time()
                res = run_variant(net0, xy, plants, loads[mode], em, dc_on,
                                  args.backend, args.objective, args.limits,
                                  args.rating_factor)
                results[(mode, dc_on)] = res
                if res["lp"] is None:
                    continue
                nsat = int((res["dc"] >= SATURATED).sum()) if res["dc"] is not None else None
                print(f"    {mode:10s} {'AC+HVDC' if dc_on else 'AC only':8s} "
                      f"{time.time()-t:5.1f}s  over100={res['over']:5d}  "
                      f"worst={res['max_load']:8,.0f}%  "
                      f"shed={res['shed']:8,.0f} MW ({res['shed_pct']:.2f}%)" +
                      (f"  dc_sat={nsat}/{len(res['links'])}" if nsat is not None else ""))
                rows.append({"stamp": stamp, "label": label, "disagg": mode,
                             "hvdc": dc_on, "total_load_MW": res["total"],
                             "lines_over_100": res["over"],
                             "max_loading_pct": res["max_load"],
                             "shed_MW": res["shed"], "shed_pct_of_demand": res["shed_pct"],
                             "dc_saturated": nsat, "backend": args.backend,
                             "objective": args.objective, "thermal_limits": args.limits,
                             "rating_factor": args.rating_factor})
        tag = "min_load" if label.startswith("MINIMUM") else stamp[:13].replace('-', '').replace('T', '_')
        note = (f"solver: {args.backend}  ·  objective: {args.objective}  ·  "
                f"thermal limits: {'ON' if args.limits else 'off'}"
                + ("  ·  DC terminal load excluded" if args.exclude_dc_load else "")
                + (f"  ·  rating factor {args.rating_factor:g}x "
                   f"(loading shown vs NAMEPLATE, so up to "
                   f"{100*args.rating_factor:.0f}% is within the enforced limit)"
                   if args.limits and args.rating_factor != 1.0 else ""))
        if args.shed_maps:
            figure_shed_2x4(xy, seg, results, stamp, label,
                            out / f"shed_{tag}.png", note)
        else:
            figure_2x4(xy, seg, results, stamp, label, out / f"state_{tag}.png",
                       note, limits=args.limits, rating_factor=args.rating_factor)

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(out / "state_summary.csv", index=False)
        print(f"\nwrote {out/'state_summary.csv'}\n")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
