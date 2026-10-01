"""DC OPF as an explicit sparse LP, solved with HiGHS.

Replaces `pandapower.rundcopp` in the inner loop. That path routes through
PYPOWER's vendored `opf()` to `dcopf_solver`, which at `alg=200` is PIPS - a
pure-Python/NumPy interior-point method. BENCHMARK.md measured its cost as a
fixed ~0.6 s per call that does not shrink with problem size, and it fails to
converge on the constrained continental case even though a feasible point
provably exists. This module is BENCHMARK.md's own escape hatch.

The formulation is the textbook B-theta DC OPF, NOT a PTDF one. PTDF would be
dense at 9,869 branches x 6,737 buses (~530 MB) and needs a matrix inverse;
B-theta stays sparse and lets an outage be applied by zeroing one branch
susceptance instead of rebuilding anything.

    variables   theta (per bus)          free, except one reference per island
                p_g   (per generator)    [0, max_p_mw * availability]
                s     (per load bus)     [0, load]            <- load shedding

    minimise    c_g' p_g + c_s' s

    subject to  B(theta) - A_g p_g - A_s s = -load            (nodal balance)
                |Bf theta| <= rating                          (thermal, optional)

Susceptances come from pandapower's own `_ppc["internal"]["Bf"]`, built once on
the intact network, so the impedance and tap-ratio conversion is pandapower's
rather than a reimplementation of it. Branch k contributes Bf[k, from] = +b_k
and Bf[k, to] = -b_k, so b_k is read straight off that matrix.

Three defects of the pandapower path disappear by construction here:

  * **Stranded load is counted.** A de-energised bus keeps its shed variable and
    loses its branches, so nodal balance forces s = load and the energy appears
    in the total. `_unserved_dc_opf` instead let the load and its shedding
    injector leave the optimisation together, which is why that branch returned
    exactly 0.000 MWh.
  * **References are never stale.** Islands are recomputed from the *active*
    branches on every solve, so a failure that splits an island cannot leave a
    piece unreferenced and silently dropped.
  * **Thermal limits are usable.** A feasible point always exists (shed
    everything, generate nothing, zero flow), and HiGHS finds it or reports
    infeasibility honestly rather than failing to converge.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog
from scipy.sparse.csgraph import connected_components

# Objective modes.
#   min_shed  generation is (almost) free, only unserved energy is priced. The
#             right objective for an ENS study: it asks "can this load be
#             served", not "what is the cheapest way to serve it". Measured on
#             the 2018-01-24T17:00Z base case, it leaves the shed identical to
#             the megawatt while cutting lines over 100% from 1,163 to 204 and
#             cross-border transport from 255 GW to 93 GW - the economic
#             objective's long-distance flows are pure artefact for this metric.
#   economic  fuel-cost merit order plus VoLL, i.e. what the model did before.
OBJECTIVES = ("min_shed", "economic")

# Tiny per-MW price on generation under `min_shed`. Not an economic signal - it
# breaks the degeneracy of a wholly free dispatch and, among the many optima
# that serve the same load, prefers the one that generates (and therefore
# transports) least.
MIN_SHED_GEN_COST = 1e-3


@dataclass
class LPNet:
    """Everything the LP needs, built once per network."""

    n_bus: int
    br_from: np.ndarray        # (nbr,) bus index
    br_to: np.ndarray          # (nbr,)
    b_mw: np.ndarray           # (nbr,) susceptance, MW per radian
    rating_mw: np.ndarray      # (nbr,) thermal rating, MW
    n_line: int                # branches [0:n_line) are lines, rest transformers
    incidence: sp.csr_matrix   # (nbr, n_bus) +1 at from, -1 at to
    gen_bus: np.ndarray        # (ng,)
    gen_max: np.ndarray        # (ng,) MW
    gen_cost: np.ndarray       # (ng,) EUR/MW, economic objective
    shed_bus: np.ndarray       # (ns,) buses carrying load
    shed_max: np.ndarray       # (ns,) MW at load_scale 1
    voll: float
    # Optional HVDC layer. A DC link is not a branch with an impedance - it is
    # a controllable injection pair, withdrawing d at one end and delivering it
    # at the other - so it never enters B or Bf. Modelled lossless and
    # bidirectional through a non-negative pair (d+, d-) so a small cost can
    # keep the optimiser from circulating power for free.
    dc_from: np.ndarray = None    # (nd,) bus index
    dc_to: np.ndarray = None      # (nd,)
    dc_cap: np.ndarray = None     # (nd,) MW, p_nom


# Per-MW price on DC transfer. Same purpose as MIN_SHED_GEN_COST: break ties
# rather than express an economics.
DC_COST = 1e-4


def build_lp_net(net, voll: float = 10_000.0) -> LPNet:
    """Extract the LP data from a built pandapower net.

    The net must have loads, generation and shedding units already assigned;
    `add_slack` is NOT required - references are chosen per solve.
    """
    import pandapower as pp
    from pandapower.pypower.idx_brch import F_BUS, T_BUS

    # One DC power flow to populate _ppc. Its result is discarded; only the
    # matrices are wanted. An angle reference is needed for it to run, so add a
    # throwaway one if the caller has not.
    tmp_ref = False
    if not len(net.ext_grid):
        pp.create_ext_grid(net, net.bus.index[0], vm_pu=1.0, va_degree=0.0)
        tmp_ref = True
    try:
        pp.rundcpp(net)
    finally:
        if tmp_ref:
            net.ext_grid.drop(net.ext_grid.index[-1], inplace=True)

    from pandapower.pypower.idx_brch import BR_X, TAP

    ppc = net._ppc
    base = float(ppc["baseMVA"])
    bl = net._pd2ppc_lookups["bus"]           # pandapower bus -> ppc bus
    brs = ppc["branch"]
    f_ppc = np.real(brs[:, F_BUS]).astype(np.int64)
    t_ppc = np.real(brs[:, T_BUS]).astype(np.int64)

    # PYPOWER's makeBdc: b = 1 / (x * tap). BR_X and TAP are already the result
    # of pandapower's own impedance and tap conversion from net.line/net.trafo,
    # so this is its arithmetic rather than a reimplementation of it. Taking b
    # from the `Bf` matrix instead would be wrong here: `_ppc["internal"]` is
    # built on the *internal* (ppci) bus set, which drops out-of-service buses,
    # while `ppc["branch"]` indexes the full ppc set.
    x = np.real(brs[:, BR_X]).astype(float)
    tap = np.real(brs[:, TAP]).astype(float)
    tap = np.where(tap == 0.0, 1.0, tap)
    with np.errstate(divide="ignore", invalid="ignore"):
        b_pu = 1.0 / (x * tap)
    b_pu = np.where(np.isfinite(b_pu), b_pu, 0.0)

    rng = net._pd2ppc_lookups["branch"]
    l0, l1 = rng["line"]
    t0, t1 = rng.get("trafo", (l1, l1))
    keep = np.r_[np.arange(l0, l1), np.arange(t0, t1)]
    n_line = int(l1 - l0)

    # ppc bus indices -> pandapower bus indices
    inv = np.full(int(bl.max()) + 1, -1, np.int64)
    inv[bl] = np.arange(len(bl))
    br_from = inv[f_ppc[keep]]
    br_to = inv[t_ppc[keep]]
    b_mw = np.abs(b_pu[keep]) * base

    # Thermal ratings, matching how pandapower computes loading_percent.
    vn = net.bus["vn_kv"].to_numpy()
    par = net.line["parallel"].to_numpy(float) if "parallel" in net.line else 1.0
    df_ = net.line["df"].to_numpy(float) if "df" in net.line else 1.0
    line_mva = (np.sqrt(3) * vn[net.line["from_bus"].to_numpy()]
                * net.line["max_i_ka"].to_numpy() * par * df_)
    trafo_mva = net.trafo["sn_mva"].to_numpy(float)
    rating = np.r_[line_mva, trafo_mva]

    n_bus = len(net.bus)
    nbr = len(br_from)
    rows = np.r_[np.arange(nbr), np.arange(nbr)]
    cols = np.r_[br_from, br_to]
    vals = np.r_[np.ones(nbr), -np.ones(nbr)]
    inc = sp.csr_matrix((vals, (rows, cols)), shape=(nbr, n_bus))

    gen_bus = net.gen["bus"].to_numpy(np.int64)
    gen_max = net.gen["max_p_mw"].to_numpy(float)
    cost = np.zeros(len(net.gen))
    pc = net.poly_cost
    if len(pc):
        g = pc[pc["et"] == "gen"]
        cost[g["element"].to_numpy(np.int64)] = g["cp1_eur_per_mw"].to_numpy(float)

    shed_bus = net.sgen["bus"].to_numpy(np.int64)
    shed_max = net.sgen["max_p_mw"].to_numpy(float)

    return LPNet(n_bus=n_bus, br_from=br_from, br_to=br_to, b_mw=b_mw,
                 rating_mw=rating, n_line=n_line, incidence=inc,
                 gen_bus=gen_bus, gen_max=gen_max, gen_cost=cost,
                 shed_bus=shed_bus, shed_max=shed_max, voll=voll)


def _references(lp: LPNet, active: np.ndarray, bus_out: np.ndarray) -> np.ndarray:
    """One reference bus per island of the ACTIVE network.

    Recomputed every solve. A de-energised bus is its own island and gets its
    own reference, which is what keeps its load in the problem instead of
    being dropped as an unreferenced area.
    """
    f, t = lp.br_from[active], lp.br_to[active]
    live = ~bus_out[f] & ~bus_out[t]
    adj = sp.coo_matrix((np.ones(live.sum()), (f[live], t[live])),
                        shape=(lp.n_bus, lp.n_bus)).tocsr()
    n, lab = connected_components(adj, directed=False)
    order = np.argsort(lab, kind="stable")
    _, first = np.unique(lab[order], return_index=True)
    return order[first]


def solve_dc_opf(lp: LPNet, line_out: np.ndarray, bus_out: np.ndarray,
                 gen_availability: float = 1.0, load_scale: float = 1.0,
                 thermal_limits: bool = False, objective: str = "min_shed",
                 rating_factor: float = 1.0) -> dict:
    """One DC OPF. Returns shed MW, branch flows and the solver status.

    `rating_factor` scales the thermal limit that is ENFORCED, but the returned
    `loading` is always a percentage of the NAMEPLATE rating. So at factor 1.1 a
    line may legitimately read 110%, and the map shows which corridors are using
    the extra headroom rather than every panel flattening to 100% again.

    The factor is the honest way to treat a rating: it is a modelling assumption,
    not a measurement. Ratings vary seasonally, and a value below 1.0
    approximates the security margin an operator holds for N-1 without modelling
    contingencies explicitly.
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective '{objective}'; choose from {list(OBJECTIVES)}")

    nbr = len(lp.br_from)
    br_out = np.zeros(nbr, bool)
    br_out[:lp.n_line] = line_out
    # A branch touching a de-energised bus carries nothing.
    br_out |= bus_out[lp.br_from] | bus_out[lp.br_to]
    active = ~br_out

    b_act = np.where(active, lp.b_mw, 0.0)
    Bf = sp.diags(b_act) @ lp.incidence            # (nbr, n_bus), MW per radian
    B = lp.incidence.T @ Bf                        # (n_bus, n_bus)

    ng, ns = len(lp.gen_bus), len(lp.shed_bus)
    A_g = sp.csr_matrix((np.ones(ng), (lp.gen_bus, np.arange(ng))),
                        shape=(lp.n_bus, ng))
    A_s = sp.csr_matrix((np.ones(ns), (lp.shed_bus, np.arange(ns))),
                        shape=(lp.n_bus, ns))

    load = lp.shed_max * load_scale                # MW at each shed bus
    bus_load = np.zeros(lp.n_bus)
    np.add.at(bus_load, lp.shed_bus, load)

    nd = 0 if lp.dc_from is None else len(lp.dc_from)
    blocks = [B, -A_g, -A_s]
    if nd:
        # +d delivers at `to` and withdraws at `from`; the (d+, d-) pair gives
        # a bidirectional link, so the two columns carry opposite signs.
        rows = np.r_[lp.dc_from, lp.dc_to]
        cols = np.r_[np.arange(nd), np.arange(nd)]
        vals = np.r_[-np.ones(nd), np.ones(nd)]
        A_d = sp.csr_matrix((vals, (rows, cols)), shape=(lp.n_bus, nd))
        blocks += [-A_d, A_d]
    A_eq = sp.hstack(blocks, format="csr")
    b_eq = -bus_load

    if objective == "min_shed":
        c = np.r_[np.zeros(lp.n_bus),
                  np.full(ng, MIN_SHED_GEN_COST),
                  np.ones(ns)]
    else:
        c = np.r_[np.zeros(lp.n_bus), lp.gen_cost, np.full(ns, lp.voll)]
    if nd:
        c = np.r_[c, np.full(2 * nd, DC_COST)]

    gmax = lp.gen_max * gen_availability
    gmax = np.where(bus_out[lp.gen_bus], 0.0, gmax)
    lb = np.r_[np.full(lp.n_bus, -np.inf), np.zeros(ng), np.zeros(ns)]
    ub = np.r_[np.full(lp.n_bus, np.inf), gmax, load]
    if nd:
        # A link with either end de-energised carries nothing.
        cap = np.where(bus_out[lp.dc_from] | bus_out[lp.dc_to], 0.0, lp.dc_cap)
        lb = np.r_[lb, np.zeros(2 * nd)]
        ub = np.r_[ub, cap, cap]
    refs = _references(lp, active, bus_out)
    lb[refs] = 0.0
    ub[refs] = 0.0

    A_ub, b_ub = None, None
    if thermal_limits:
        Z = sp.csr_matrix((nbr, ng + ns + 2 * nd))
        A_ub = sp.vstack([sp.hstack([Bf, Z]), sp.hstack([-Bf, Z])], format="csr")
        cap = lp.rating_mw * rating_factor
        b_ub = np.r_[cap, cap]

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=np.c_[lb, ub], method="highs")
    if not res.success:
        return {"success": False, "status": res.message, "shed": np.nan,
                "flow": None, "loading": None}

    theta = res.x[:lp.n_bus]
    shed_unit = res.x[lp.n_bus + ng: lp.n_bus + ng + ns]
    shed = float(shed_unit.sum())
    flow = Bf @ theta
    loading = np.where(lp.rating_mw > 0, 100.0 * np.abs(flow) / lp.rating_mw, np.nan)
    loading[br_out] = np.nan
    dc_flow = None
    if nd:
        off = lp.n_bus + ng + ns
        dc_flow = res.x[off: off + nd] - res.x[off + nd: off + 2 * nd]
    # Per-bus shed, aligned to lp.shed_bus, plus the fraction of each bus's own
    # demand that went unserved. The fraction is what separates a bus that was
    # cut off entirely (1.0 - islanded, or de-energised) from one merely
    # curtailed to relieve a corridor.
    shed_frac = np.divide(shed_unit, load, out=np.zeros_like(shed_unit),
                          where=load > 0)
    return {"success": True, "status": res.message, "shed": shed,
            "flow": flow, "loading": loading,
            "gen": res.x[lp.n_bus:lp.n_bus + ng], "dc_flow": dc_flow,
            "shed_unit": shed_unit, "shed_frac": shed_frac,
            "shed_bus": lp.shed_bus, "bus_demand": load}
