"""Orchestrate the network-level study across countries, scenarios and archetypes.

    python run_network_study.py --countries LUX PRT --sampler sobol --n 64
    python run_network_study.py --countries LUX --scenarios net_flood_absprot \\
        --archetypes winter_peak summer_night --n 200

Every (country, scenario, archetype) combination is a separate EMA run and a
separate output file, so the study is resumable: a combination whose result
file already exists is skipped unless --force is given. That is what makes a
cluster campaign restartable after a walltime kill without redoing finished
work - the same skip-token pattern the direct-damage orchestrator uses.

Scenarios that do not apply to a country are skipped automatically (coastal for
landlocked countries; any hazard Stage 1 did not preprocess). So are STRUCTURAL
ZEROS - the hazard applies, but no grid component in the country's owned basins
is exposed (power_net/exposure_screen.py). Those are not dropped silently: each
leaves results/<ISO3>/structural_zero_<ISO3>_<scenario>.json stating EAENS = 0
and the reason, so they stay a reported result category without costing a run.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

from power_net.demand import ARCHETYPES, DURATION_SAMPLING
from power_net.ema_model import (DEFAULT_SCENARIOS, SCENARIO_HAZARD, SCENARIOS,
                                 applicable_scenarios)
from power_net.exposure_screen import (LABELS, STRUCTURAL_ZEROS,
                                       record_structural_zero, screen)
from power_net.paths import (
    country_results_dir,
    load_config,
    set_archetype_override,
    set_country_override,
    set_scenario_override,
)

REPO = Path(__file__).resolve().parent


def result_exists(country: str, scenario: str, archetype: str, sampler: str,
                  n: int, duration: str = "fixed", hourly: bool = False) -> bool:
    set_country_override(country)
    set_scenario_override(scenario)
    set_archetype_override(archetype)
    cfg = load_config()
    d = country_results_dir(cfg, create=False)
    if not d.is_dir():
        return False
    variant = f"_{duration}" + ("_hourly" if hourly else "")
    # The `n` must be delimited, not a prefix: `n{n}*` let `--n 4` match an
    # existing `_n400_*.tar.gz`, so a 4-run smoke test read as a finished
    # 400-run campaign. run_experiments.py writes `..._n{n}{tag}_{stamp}.tar.gz`
    # with tag either empty or `_<tag>`, so anchor on the `_` before the stamp
    # and reject anything that continues the digits.
    pattern = (f"experiments_{country}_powergrid_{scenario}_{archetype}{variant}_"
               f"{sampler}_n{n}_*.tar.gz")
    hits = [p for p in d.glob(pattern)]
    # A tagged run is a smoke test or a variant, not the real campaign - `--tag`
    # was absent from the pattern entirely, so any tagged run marked the
    # combination done. Accept only untagged results: `_n<N>_<stamp>.tar.gz`,
    # where the stamp is exactly YYYYmmdd_HHMMSS.
    stamp = re.compile(rf"_n{n}_\d{{8}}_\d{{6}}\.tar\.gz$")
    return any(stamp.search(p.name) for p in hits)


def run_one(country: str, scenario: str, archetype: str, sampler: str, n: int,
            workers: int, duration: str = "fixed", hourly: bool = False) -> int:
    cmd = [sys.executable, "-m", "power_net.run_experiments",
           "--country", country, "--scenario", scenario,
           "--archetype", archetype, "--sampler", sampler,
           "--n", str(n), "--workers", str(workers),
           "--duration-sampling", duration]
    if hourly:
        cmd.append("--hourly-resolve")
    tag = f" / {duration}" + ("/hourly" if hourly else "")
    print(f"\n=== {country} / {scenario} / {archetype}{tag} "
          f"({sampler}, n={n}, workers={workers})", flush=True)
    t = time.perf_counter()
    rc = subprocess.call(cmd, cwd=REPO)
    print(f"=== exit {rc} in {time.perf_counter() - t:.1f}s", flush=True)
    return rc


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    # No `choices=` on the nargs="+" arguments, and blanks are dropped in
    # _clean below. A shell that leaves a stray whitespace token in the
    # command line - which a backslash continuation inside a heredoc can do,
    # differently on different bash versions - would otherwise have that token
    # swallowed as an extra value and kill the job at argument parsing with
    # "invalid choice: ' '". Validation still happens, just with a message
    # that names the offending value.
    p.add_argument("--countries", nargs="+", required=True, help="ISO3 codes")
    p.add_argument("--scenarios", nargs="+", default=None,
                   help=f"default: {DEFAULT_SCENARIOS}")
    p.add_argument("--archetypes", nargs="+", default=["winter_peak"])
    p.add_argument("--sampler", choices=["lhs", "sobol"], default="lhs")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--force", action="store_true",
                   help="re-run combinations that already have a result file")
    p.add_argument("--duration-sampling", choices=DURATION_SAMPLING, default="fixed")
    p.add_argument("--hourly-resolve", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--plan-out", type=Path, default=None,
                   help="write the runnable 'COUNTRY SCENARIO' pairs to this file "
                        "(what submit_network_study.sh submits; ignores done-ness, "
                        "which depends on the variant)")
    args = p.parse_args()

    def _clean(values, valid, what):
        cleaned = [v.strip() for v in values if v and v.strip()]
        bad = [v for v in cleaned if v not in valid]
        if bad:
            p.error(f"invalid {what}: {bad} (choose from {sorted(valid)})")
        return cleaned

    args.countries = [c.strip().upper() for c in args.countries if c and c.strip()]
    args.archetypes = _clean(args.archetypes, set(ARCHETYPES), "archetype")
    if args.scenarios is not None:
        args.scenarios = _clean(args.scenarios, set(SCENARIOS), "scenario")

    scenarios = args.scenarios or DEFAULT_SCENARIOS
    planned, skipped, runnable = [], [], []
    for country in args.countries:
        ok = set(applicable_scenarios(country))
        for scenario in scenarios:
            if scenario not in ok:
                skipped.append((country, scenario, "hazard not applicable"))
                continue
            set_country_override(country)
            set_scenario_override(scenario)
            cfg = load_config()
            info = screen(cfg, SCENARIO_HAZARD[scenario])
            if info["status"] in STRUCTURAL_ZEROS:
                record_structural_zero(cfg, scenario, info)
                skipped.append((country, scenario,
                                f"{LABELS[info['status']]} -> EAENS = 0 (recorded)"))
                continue
            runnable.append((country, scenario))
            for archetype in args.archetypes:
                if not args.force and result_exists(
                        country, scenario, archetype, args.sampler, args.n,
                        args.duration_sampling, args.hourly_resolve):
                    skipped.append((country, scenario, f"{archetype}: done"))
                    continue
                planned.append((country, scenario, archetype))

    print(f"{len(planned)} combination(s) to run, {len(skipped)} skipped")
    for c, s, why in skipped:
        print(f"  skip {c:4s} {s:24s} {why}")
    for c, s, a in planned:
        print(f"  run  {c:4s} {s:24s} {a}")
    if args.plan_out is not None:
        args.plan_out.write_text("".join(f"{c} {s}\n" for c, s in runnable),
                                 newline="\n")
    if args.dry_run:
        return

    failures = []
    for country, scenario, archetype in planned:
        if run_one(country, scenario, archetype, args.sampler, args.n,
                   args.workers, args.duration_sampling,
                   args.hourly_resolve) != 0:
            failures.append((country, scenario, archetype))

    print(f"\nfinished: {len(planned) - len(failures)} ok, {len(failures)} failed")
    for f in failures:
        print("  FAILED", f)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
