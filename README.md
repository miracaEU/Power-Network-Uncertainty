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

### Known limitations, stated not hidden

- **27% of buses (1,806 of 6,737) have no OSM identifier** - synthetic
  PyPSA-Eur nodes. They cannot be joined to any hazard and are left unexposed
  (`unmatched_bus_policy: immune`), which understates risk. LUX 2 of 14, PRT
  44 of 145.
- **Per-bus load is an equal split** of national demand. `bus_data.csv` carries
  no load column, so some disaggregation is unavoidable; the equal split
  matches the existing D3.3 behaviour and is a named switch
  (`load_disaggregation`), not a hard-coded choice.
- **Thermal limits are off.** With the equal split, 12.9% of lines exceed 100%
  loading before any hazard, so a limited OPF does not converge even in the
  base case. This blinds the DC OPF to capacity-constrained shedding - the one
  failure mode a connectivity model also cannot see.
- **A 3.5 GW structural floor.** 42 of 76 islands are generation-short with no
  hazard applied. Every outcome is reported net of that baseline, and the floor
  itself is an outcome so it can be checked.
- **Transmission only** (220-750 kV). Most customer outages originate in
  distribution, so EAENS here is a lower bound on total unserved energy.
- **ALB and MKD have no demand data** and are excluded rather than defaulted.

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
