# Power-Network-Uncertainty (MIRACA)

Which modelling choices and uncertainties matter most when estimating the
**network-level** impact of natural hazards on the European power system?

Companion to [Uncertainty-Modelling](../Uncertainty-Modelling), which asks the
same question about **direct damage** to critical infrastructure. Same hazards,
same return periods, same protection data, same exposure - so the two answers
are directly comparable. The outcome here is **expected annual energy not
supplied (EAENS, MWh/yr)** rather than expected annual damage.

Three things make this structurally different from the direct-damage case:

1. **Direct damage is additive; network impact is not.** EAD is a sum of
   per-feature expectations, so hazard spatial correlation does not bias it -
   which is why treating "the RP100 map" as one simultaneous event is harmless
   there. For energy not supplied it is not harmless, because islanding is
   super-additive. Hazard events are therefore decomposed into spatially
   coherent footprints (HydroBASINS for flood, NUTS2 for coastal,
   country-coherent for windstorm), and the footprint choice is itself a
   first-class uncertainty factor bracketed by its two extremes.
2. **Damage ratios must become failure probabilities.** A substation with
   damage ratio 0.3 is not 30% working. Wind and earthquake have real fragility
   curves in the MIRACA Table D2 workbook (`W_Frag_V10m`, `E_Frag_PGA`); flood
   does not, so the damage-ratio-to-failure mapping becomes an explicitly
   tested assumption rather than a hidden one.
3. **Time enters the answer** - when the event happens sets the demand level.
   Timing is aleatory and is treated as a conditioning dimension, not mixed
   into the Sobol factors with the epistemic ones.

## Status

Runnable end-to-end. Steps 1-9 of the plan are done: the model builds, solves,
integrates to EAENS, and runs under EMA Workbench with a resumable orchestrator
and a SLURM submitter.

**Stage 1 is not re-run.** The direct-damage study's validated `power` output is
reused directly, joined to grid components through the OSM identifiers both
datasets carry (`NetworkOSM/*.csv` `tags` -> `{ISO3}_power_exposure.parquet`
`osm_id`). Measured join rates: LUX 11/11 buses and 27/27 line ways; PRT 83/87
(95.4%) and 312/328 (95.1%).

See **[BENCHMARK.md](BENCHMARK.md)** for the DC OPF timings that shaped the
design.

**Three one-off precomputes must exist before any run** (they live in `cache/`
and are country-independent, so build them once):

```powershell
& $py scripts/build_component_basins.py   # every component -> lev07 basin
& $py scripts/build_paneu_hazard.py       # Stage 1 exposure, all 36 countries
& $py scripts/build_basin_owner.py        # the basin partition + LAU overlay
```

Build them in that order - ownership reads the basin labels. Without the first
two, basin labels and hazard intensities cover only the study country, which
silently reduces a Europe-wide basin event to its study-country half and leaves
the attribution table empty. Total cost ~15 min, ~4 MB.

**Study countries: AUT, EST, HRV, HUN, ITA, NLD, ROU.** Croatia replaces
Albania, which was chosen for the study set before it was known that Albania has
no OPSD demand series at all (see the limitation below) and so cannot be run.
Croatia matches what ALB was picked for - Western Balkan, Adriatic coast,
seismic, river-flood exposed, below-EU-average income - at a near-identical grid
size (25 buses against 22). Of the 28 nominal (country, hazard) combinations,
**23 are run**:

| combination | category | why | cost |
|---|---|---|---|
| AUT, HUN coastal | not applicable | landlocked; Stage 1 never produced the hazard | none |
| ROU coastal | **No exposure** | Stage 1 profile exists but holds 0 rows - no Romanian power asset in the coastal footprint | none, recorded as EAENS = 0 |
| EST coastal | **Exposed assets, no exposed grid component** | 230 power segments exposed, but they are poles, minor lines and the like; none belongs to a grid component in EST's owned basins | none, recorded as EAENS = 0 |
| NLD coastal | **Exposed assets, no exposed grid component** | 2 segments exposed (1 plant, 1 line), neither part of a grid component in NLD's owned basins | none, recorded as EAENS = 0 |

"No exposure" is the direct-damage study's category (`plot_drivers.NO_EXPOSURE`)
with the same meaning. "Exposed assets, no exposed grid component" exists only
at the network level: the direct-damage study sees those assets and damages
them, but the transmission model does not represent them, so they cannot
disconnect load. Both zeros are **exact**, not sampling results: `compute_ens`
evaluates only the country's owned lev07 basins, and a component with zero
intensity at every return period has P(fail) = 0 under every factor setting
(`depth_scale` multiplies intensity, `warming` shifts return periods).

`power_net/exposure_screen.py` decides this from the same caches the model
reads, and `run_network_study.py` writes one
`results/<ISO3>/structural_zero_<ISO3>_<scenario>.json` per skipped
combination (category, exposed segment and component counts, `EAENS_MWh: 0`),
so aggregation reports them as a result category rather than a missing file.
The screen is re-evaluated on every submit, so a Stage 1 or cache change that
creates exposure puts the combination back into the campaign automatically.

### Known limitations, stated not hidden

- **27% of buses (1,806 of 6,737) have no OSM identifier** - synthetic
  PyPSA-Eur nodes. They cannot be joined to any hazard and are left unexposed
  (`unmatched_bus_policy: immune`), which understates risk. LUX 2 of 14, PRT
  44 of 145.
- **Per-bus load is disaggregated, not measured.** `bus_data.csv` carries no
  load column, so some rule is unavoidable. `load_disaggregation: equal`
  (the D3.3 behaviour) remains the default for comparability;
  `nuts3` / `nuts3_pop` / `nuts3_gdp` weight each bus by the population and
  GDP of its NUTS3 region (`power_net/disaggregation.py`). GB, UA, BA, MD, NO
  and XK have no usable Eurostat NUTS3 statistics and fall back to the equal
  split individually, which the run's `disagg_report` records. LUX is a single
  NUTS3 region, so for LUX the two are identical by construction.
- **The HVDC layer is modelled, as of 2026-09-16** (`include_hvdc: true`).
  `power_net/hvdc.py` reads `Net_structure_data/links.csv` (38 DC links,
  32,530 MW) and `converters.csv` (67 AC/DC converters) and joins both on the
  exact bus name. Links connect DC terminal to DC terminal; the converters are
  what couple the layer to the AC grid, so reading links alone would build 2-bus
  DC islands that never touch it. In the LP both are capacity-bounded
  controllable transfers; in the connectivity model both are plain edges, which
  overstates what a finite link can move - that asymmetry between the two
  network models is real and left visible. With the layer on, the topology is a
  **single connected component** rather than 76.
- **Thermal limits are off by default, but no longer blocked.** 12.9% of lines
  exceed 100% loading before any hazard under the old economic dispatch. Two
  hypotheses for that have been tested and falsified: it is not the equal-split
  load disaggregation (`nuts3` makes it marginally worse, 14.3%) and it is not
  the missing HVDC (12.9% -> 13.0%). The cause is the **objective**: an
  unconstrained economic dispatch with fuel-type-only marginal costs ships cheap
  generation across the continent, pushing 42 GW across a France-Spain border
  whose real NTC is ~2.8 GW. Switching to `dispatch_objective: min_shed` cuts
  lines over 100% from 1,163 to 333 and the worst line from 1,668% to 298%,
  while leaving the shed identical to the megawatt. With
  `thermal_limits: true` under the `highs` backend the base case now solves in
  6.3 s with no line over 100%, at the cost of 216 MW of unavoidable shedding.
  Left off by default only so results stay comparable with earlier runs.
- **The DC OPF is solved as an explicit LP with HiGHS**
  (`power_net/dcopf_lp.py`, `opf_backend: highs`), not by
  `pandapower.rundcopp`. Same model - flows agree to 3.8e-5 MW with the dispatch
  held fixed - but ~25x faster, able to converge with thermal limits on, and it
  counts load stranded at a de-energised bus. That last point fixes the
  regression that made `network_model=dc_opf` return exactly 0.000 MWh. The
  pandapower path is retained as a cross-check via `opf_backend: pandapower`.
- **The 3.5 GW structural floor was an artefact, and is now fixed.**
  70 of the 76 "islands" were single buses, and every one an HVDC converter
  terminal left unconnected because the DC layer was never read. The load split
  then gave those DC nodes 6,766 MW of demand with no supply path, shed in full
  - 53% of all shed MW, and the whole of the floor. As of 2026-09-16 both
  switches are on and the floor is **0.0 MW**. They fix different things and are
  both needed: `include_hvdc` corrects the topology but would otherwise leave
  the grid *serving* demand that does not exist, and
  `exclude_dc_terminal_load` removes that demand - including on the 3 terminals
  that have no converter row and stay isolated regardless.

  EAENS was never biased by this, because `_event_energy` subtracts a no-hazard
  baseline computed the same way, so the artefact cancelled. What it corrupted
  was the `structural_floor_MW` diagnostic and every statement made about
  generation-short islands. **Results from before 2026-09-16 are not comparable
  with later ones.**
- **What remains unserved is delivery-limited, not supply-limited.** With both
  switches on, the winter-peak base case sheds 228 MW (equal split) at 5 buses,
  every one of them inside an island holding a **675 GW surplus**, with an
  incident branch pinned at 100%. Adding the 32.5 GW HVDC layer changes that
  number by 0.0 MW, because a converter can only inject at its own bus and the
  binding constraint is an AC corridor downstream of it. HVDC relieves an
  energy-balance shortage; it cannot relieve local congestion.
- **Transmission only** (220-750 kV). Most customer outages originate in
  distribution, so EAENS here is a lower bound on total unserved energy.
- **The five demand-less countries are spurious exporters.** AL, BA, MD, MK and
  XK carry zero load but keep 11.53 GW of generation, of which **6.7 GW
  dispatches in the base case** - 1.24% of European demand, injected with
  nothing to serve. It makes **Bosnia the fourth largest exporter in the
  model**, ahead of Lithuania, Sweden and Greece, which is an artefact of
  missing demand data and not a property of the grid.

  **EAENS is insensitive to it**: removing that generation moves one-draw EAENS
  by +0.004 MWh (HRV flood), -0.959 (HUN flood), -0.037 (ROU flood) and +0.002
  (HRV windstorm) - all under 0.03%, and of both signs, so re-dispatch noise
  rather than bias. ENS is a difference of two solves that both contain the
  artefact, and the residual shed is delivery-limited, so distant supply cannot
  relieve it. `network.zero_load_country_policy: neutral` switches their
  generation off to re-check that; the default keeps it.

  What it *does* contaminate is any **absolute** statement about flows or line
  loading in the Balkan corridor. Treat base-case loading figures there as
  indicative, not measured.
- **Five grid countries have no Stage 1 exposure at all**: UA, BA, ME, XK, MD
  (361 buses, 5.4%). Their components are unexposed and therefore immune, so a
  basin event reaching into them fails only the covered side. This touches
  **7 of the 314 owned basins** (HRV 2, HUN 2, ROU 3). Closing it means running
  Stage 1 for those countries in the direct-damage study; it is not fixable
  here.
- **ROU coastal is structurally empty.** `ROU_power_coastal_profiles.parquet`
  exists but has zero rows, so no Romanian power asset is coastally exposed and
  the scenario is a guaranteed zero. Same for BGR, BEL and ISL. Coastal is
  therefore meaningful for four study countries, not five.
- **ALB and MKD have no demand data** and are excluded rather than defaulted.
  OPSD publishes no series for AL, BA, MD, MK or XK, so their buses carry zero
  load and the archetype hour cannot even be selected. This is why ALB is not a
  study country: running it would need Albanian demand from another source
  (ENTSO-E annual totals shaped by a regional profile), not a default.

## Reuse

Stage 1 (the expensive GIS: exposure x hazard rasters, FLOPROS protection,
HydroBASINS join, warming return-period anchors), the EAD integration and the
curve-group machinery are **reused, not copied**, from the direct-damage repo
via an editable install:

```powershell
uv venv $env:USERPROFILE\.venvs\miraca_net --python 3.12
$py = "$env:USERPROFILE\.venvs\miraca_net\Scripts\python.exe"
uv pip install --python $py -r ..\Uncertainty-Modelling\requirements.txt
uv pip install --python $py -r requirements-network.txt
uv pip install --python $py -e ..\Uncertainty-Modelling   # -> import miraca_uq
```

## Layout

```
power_net/network.py       pandapower network from the Energy Model CSVs;
                           version-robust schema translation; topology report
power_net/injections.py    loads, merit-order generation, slack placement, and
                           the load-shedding units that make ENS a decision
                           variable rather than a connectivity artefact
power_net/linkage.py       grid component <-> OSM id mapping; asserts the
                           row alignment both views depend on
power_net/hazard.py        Stage 1 intensities aggregated onto components
power_net/fragility.py     fragility sheets -> P(fail); the three flood
                           mappings; correlated common-random-number draws
power_net/demand.py        OPSD series, timing archetypes, disaggregation
power_net/ens_model.py     Stage 2: failure -> islanding/OPF -> ENS -> EAENS
power_net/ema_model.py     scenarios and factor declarations
power_net/run_experiments.py   one (country, scenario, archetype) run
run_network_study.py       resumable orchestrator across combinations
submit_network_study.sh    SLURM submitter
scripts/benchmark_opf.py   the step-1 feasibility benchmark
scripts/diagnose_opf.py    solver ladder used to isolate non-convergence
```

## Running

Locally:

```powershell
$py = "$env:USERPROFILE\.venvs\miraca_net\Scripts\python.exe"
& $py -m power_net.run_experiments --country LUX --scenario net_flood_absprot --n 200
& $py run_network_study.py --countries LUX PRT --sampler lhs --n 200 --dry-run
```

On the cluster:

```bash
./submit_network_study.sh setup    # once
./submit_network_study.sh dry      # print the plan
./submit_network_study.sh test     # LUX + PRT, LHS
./submit_network_study.sh sobol    # the real thing
```

`n_solves` is reported as an outcome on purpose: it is the quantity that
decides whether pan-European Sobol is affordable, and the first runs exist to
measure it.
