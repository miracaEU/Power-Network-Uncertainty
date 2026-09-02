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

## The thermal-limit non-convergence

With `max_loading_percent = 100` on every line the OPF fails. The cause is
visible in the no-limits solve: **12.9% of lines (1,163 of 8,994) are already
over 100%, with a maximum of 1,668%**, before any hazard is applied. The base
case is massively infeasible to begin with.

This is expected and is very likely an artefact of the **placeholder** load
disaggregation used for the benchmark (each country's demand split equally
across its buses), which puts load where none exists and creates unphysical
flows. It is *not* yet evidence of a problem with the network data. The proper
NUTS3 population/GDP disaggregation must land before this is re-tested or any
loading figure is interpreted.

## Incidental findings worth keeping

- **42 of 76 islands are short of generation even with no hazard**, a
  structural unserved floor of **3.3 GW**. The OPF's shed of 3,292 MW matches
  this almost exactly, which is a good independent check that the
  load-shedding formulation is behaving correctly. It also confirms the plan's
  concern that some "islands" are topology artefacts: that floor would
  otherwise be reported as hazard-driven ENS.
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
