# Step-1 benchmark: is pan-European DC OPF Sobol feasible?

Measured 2026-08-26 on the MIRACA Energy Model network (6,737 buses, 8,994
lines, 875 transformers), pandapower 3.5.4 / numpy 2.4.6 / scipy 1.18.1, single
core, `scripts/benchmark_opf.py` and `scripts/diagnose_opf.py`.

The plan deferred every feasibility decision to this number because it could
not be measured during planning (pandapower was not installed).

## Results

| Operation | Time | Note |
|---|---|---|
| Build base network from CSVs | **0.3 s** | whole-DataFrame assignment |
| Connectivity screen (`island_labels`) | **1.1 ms** | scipy sparse, whole continent |
| Plain DC power flow (`rundcpp`) | **1.15 s** | |
| **Full-continent DC OPF, no thermal limits** | **6.7 s ± 0.09** | converges; sheds 3,292 MW |
| Full-continent DC OPF, 100% thermal limits | **fails at 17 s** | `OPFNotConverged` - see below |
| Localised OPF, 26-bus k-hop subnetwork | 1.41 s | |
| Localised OPF, 55-bus | 0.68 s | |
| Localised OPF, 91-bus | 0.62 s | |
| Localised OPF, 147-bus | 0.76 s | |

## What this settles

**Full-continent DC OPF per (footprint, return period) is definitively out.**
Sobol at N=1024 with k≈12 needs ~26,600 model evaluations. At 6.7 s per solve,
even 100 surviving solves per evaluation is 26,600 × 100 × 6.7 s ≈ **206
core-days**. The plan's localised-OPF lever is therefore mandatory, not
optional.

**But localisation alone does not solve it, and the reason is not what the plan
assumed.** Solve time is almost flat in subnetwork size - 26 buses costs about
the same as 147. That is a **fixed ~0.6 s of pandapower per-call overhead**,
not LP work. Shrinking the problem buys roughly one order of magnitude (6.7 s →
0.6 s) and then stops.

At 0.6 s the arithmetic is: 26,600 evaluations × *S* solves each × 0.6 s.
- *S* = 10 → ~44 core-hours (comfortable)
- *S* = 100 → ~444 core-hours (fine on the cluster)
- *S* = 1000 → ~4,400 core-hours (borderline)

So the governing unknown is now **S, how many footprint × RP pairs actually
produce failures that survive the connectivity screen**. That needs Stage 1 for
the `powergrid` asset and is the next measurement, not a guess.

The screen itself is 1.1 ms - about 6,000× cheaper than a solve - so aggressive
screening plus memoisation on the failed-component set is clearly worth it.

**If S turns out to be large**, the escape hatch is to bypass pandapower's
per-call overhead entirely: a DC OPF with load shedding is just an LP
(minimise `c_g'p_g + VoLL'·s` subject to DC power balance, PTDF line limits and
box bounds), which `scipy.optimize.linprog` with HiGHS solves in milliseconds
once the PTDF is built. pandapower would then serve as the reference
implementation to validate against rather than the inner loop.

## The escape hatch, taken (2026-09-10)

`power_net/dcopf_lp.py` implements exactly that, and is now the default
(`opf_backend: highs`). One departure from the sketch above: it is a **B-theta**
formulation, not PTDF. PTDF here would be dense at 9,869 branches x 6,737 buses
(~530 MB) and needs a matrix inverse, whereas B-theta stays sparse and lets an
outage be applied by zeroing a single branch susceptance. Susceptances are read
from pandapower's own `BR_X`/`TAP`, so the impedance and tap conversion is still
pandapower's arithmetic rather than a reimplementation.

Measured on the same reference hour, equal-split load:

| | time | shed | over 100% | worst line |
|---|---|---|---|---|
| pandapower, economic, no limits | 24.7 s | 3,291.9 MW | 1,163 | 1,668% |
| HiGHS, economic, no limits | **0.9 s** | **3,291.9 MW** | 1,201 | 1,668% |
| HiGHS, min-shed, no limits | 0.8 s | 3,291.9 MW | **333** | **298%** |
| HiGHS, min-shed, **limits ON** | 6.3 s | 3,507.7 MW | 0 | **100%** |
| HiGHS, economic, **limits ON** | 3.9 s | 3,507.7 MW | 0 | **100%** |

**The thermal-limit non-convergence above was a solver failure, not an
infeasibility.** A feasible point always exists - shed everything, generate
nothing, zero flow on every branch - and HiGHS finds it in seconds where PIPS
could not converge at all. Enforcing the ratings costs 216 MW of unavoidable
shedding that the unconstrained solve hid.

Validation, since "faster" is worthless if it is a different model:

- **Shed matches to +0.000 MW** on the like-for-like case (economic, no limits).
- **Line flows match to 3.8e-5 MW** against a 622 MW mean, correlation
  1.0000000000, with the dispatch held fixed at pandapower's own optimum. The
  larger discrepancies seen when both solvers choose their own dispatch are
  degeneracy - fuel-type costs make many dispatches tie - not a model difference.
- **Stranded load is counted.** De-energising bus 5758 (262.5 MW) moves the shed
  by exactly +262.5 MW; the pandapower path moved it by +0.0.
- **End-to-end on LUX**, `network_model=dc_opf`: pandapower returns
  **0.0000 MWh** (the known regression), HiGHS returns **17.7617 MWh**, matching
  the connectivity model, in 3.4 s against 91.4 s.

## The thermal-limit non-convergence

With `max_loading_percent = 100` on every line the OPF fails. The cause is
visible in the no-limits solve: **12.9% of lines (1,163 of 8,994) are already
over 100%, with a maximum of 1,668%**, before any hazard is applied. The base
case is massively infeasible to begin with.

The original reading of this was that it is "very likely an artefact of the
**placeholder** load disaggregation" - each country's demand split equally
across its buses, putting load where none exists - and that the proper NUTS3
disaggregation had to land before the figure meant anything.

**Tested 2026-09-09, and that reading is wrong.** With NUTS3
population/GDP disaggregation (`load_disaggregation: nuts3`, see
`power_net/disaggregation.py`) the same no-limits DC OPF at the same reference
hour gives:

| | over 100% | max loading | p90 / p99 | structural shed |
|---|---|---|---|---|
| `equal` | 1,163 (12.9%) | 1,668% | 118% / 307% | 3,292 MW |
| `nuts3` | 1,282 (14.3%) | 1,722% | 126% / 315% | 3,295 MW |

Moving load onto population and GDP makes the base case *slightly worse*, not
better. So the infeasibility is not the load disaggregation, and no loading
figure should be blamed on it any more.

**The leading candidate is now the missing HVDC layer.**
`Net_structure_data/links.csv` holds 38 DC links totalling **32,530 MW**, and
`converters.csv` a further 67 converters; every one of the 76 link endpoints
resolves to a real bus in the network, none is under construction, and
`build_base_network` reads none of them - it loads only `bus_data`,
`line_data`, `trafo_data` and `bus_geodata`. Adding those links as edges
collapses the topology from **76 connected components to 40** (largest
component 5,486 -> 6,104 buses), which means 36 of the islands behind the
3.3 GW structural floor are not electrical islands at all. Power that should
move by DC is currently forced onto the AC network, which is exactly the shape
of defect that produces overloads no redispatch can relieve.

This is a hypothesis with a strong prior, not yet a demonstration: nobody has
re-run the limited OPF with the links in. That is the next measurement.

## Incidental findings worth keeping

- **42 of 76 islands are short of generation even with no hazard**, a
  structural unserved floor of **3.3 GW**. The OPF's shed of 3,292 MW matches
  this almost exactly, which is a good independent check that the
  load-shedding formulation is behaving correctly. It also confirms the plan's
  concern that some "islands" are topology artefacts: that floor would
  otherwise be reported as hazard-driven ENS.

  **Resolved 2026-09-14: the floor is entirely an artefact, and the "islands"
  are not islands.** 70 of the 76 components are single buses, and all 70 are
  HVDC converter terminals whose only connections live in `links.csv` and
  `converters.csv` - files `build_base_network` never read. The load
  disaggregation then handed those DC nodes 6,766 MW of national demand, which
  had no supply path and was shed in full. Setting either
  `exclude_dc_terminal_load: true` (remove the phantom load) or
  `include_hvdc: true` (give it a path) drops the structural floor from
  3,522 MW to **exactly 0.0 MW**, independently. With `include_hvdc` the
  topology goes from 76 components to **1** - Europe is a single connected
  graph once the DC layer exists, which is what the real network is.

  EAENS was not biased by this, because `_event_energy` subtracts a no-hazard
  baseline computed the same way, so the artefact cancels. What it did corrupt
  is the `structural_floor_MW` diagnostic and every statement made about it -
  including the one above.
- **Topology**: 206 components from lines alone (largest 1,022); **76
  components including transformers** (largest 5,486), 70 singletons. Matches
  D3.3's reported figures.
- **Demand at the 2018-01-24 17:00Z winter evening peak: 498.4 GW** across
  6,652 load buses - a plausible European peak.
- **5 grid countries have no OPSD demand series at all**: AL, BA, MD, MK, XK.
  Of these, **ALB and MKD are inside the 31-country study set** and need demand
  from another source. They are left at zero and reported, never defaulted -
  D3.3's `.fillna(500.0)` MW-per-bus default for exactly this case is the
  likely origin of its 245,864 MW artefact.
- **GB has no plain `GB` demand series**; OPSD publishes GB_GBN / GB_NIR /
  GB_UKM. GB_GBN is the match for the grid model's GB zone (Northern Ireland
  belongs to the all-island SEM with IE). Without this alias GB silently
  dropped out, costing ~49 GW.

## Environment note

pandapower 2.14.11 (which UC5 pins) is **incompatible with pandas 3.x**: it
writes into `.values` in place and pandas 3 copy-on-write makes those arrays
read-only (`ValueError: assignment destination is read-only` from
`results_bus.py`). Pinning it also drags numpy back to 1.26 and scipy to 1.13,
diverging from the direct-damage environment.

pandapower **3.5.4** is used instead: it keeps numpy 2.4.6 and scipy 1.18.1,
matching paper 1's stack, so Stage 1 outputs cannot differ for environment
reasons. Only pandas differs (2.3.3 here vs 3.0.5 there), which affects APIs
rather than float arithmetic. The cost is that 3.x renamed several columns
(`tap_phase_shifter` → `tap_changer_type`, load `const_z_percent` →
`const_z_p_percent`) and dropped `net.bus_geodata`; `power_net.network.conform_to_schema`
translates the 2.x-era CSVs to whatever version is installed rather than
hard-coding either.

Reproducing the UC5 Slovenia cross-check (verification step 8) will need its own
pandapower 2.14.11 environment.
