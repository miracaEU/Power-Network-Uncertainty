"""Run EMA Workbench experiments for one (country, scenario) combination.

    python -m power_net.run_experiments --country LUX --scenario net_flood_absprot --n 200
    python -m power_net.run_experiments --country PRT --scenario net_earthquake \\
        --sampler sobol --n 64 --workers 8

With --sampler sobol, --n is the SALib base sample size N (use a power of 2);
the number of model runs is N * (2k + 2), k being the scenario's factor count
(15 for flood, 9 for earthquake/windstorm - see power_net/ema_model.py).

Results are saved as a tar.gz under results/<ISO3>/, named so no (country,
scenario, sampler, archetype) combination can overwrite another.
"""

from __future__ import annotations

import argparse
from datetime import datetime

from ema_workbench import (
    MultiprocessingEvaluator,
    Samplers,
    SequentialEvaluator,
    ema_logging,
    save_results,
)

from .demand import ARCHETYPES, DURATION_SAMPLING
from .ema_model import SCENARIOS, build_model
from .paths import (
    country_results_dir,
    load_config,
    result_stem,
    set_archetype_override,
    set_country_override,
    set_duration_override,
    set_hourly_override,
    set_scenario_override,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=200,
                        help="experiments (LHS) or base sample size N (Sobol)")
    parser.add_argument("--sampler", choices=["lhs", "sobol"], default="lhs")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--country", default=None, help="ISO3 override")
    parser.add_argument("--scenario", choices=SCENARIOS, default=None)
    parser.add_argument("--archetype", choices=list(ARCHETYPES), default=None,
                        help="timing archetype (conditioning dimension, not a factor)")
    parser.add_argument("--duration-sampling", choices=DURATION_SAMPLING, default=None,
                        help="outage duration: loguniform/uniform over [6 h, 14 d], "
                             "or fixed at network.outage_window_hours")
    parser.add_argument("--hourly-resolve", action="store_true",
                        help="re-solve unserved power at every hour of the outage "
                             "instead of scaling one solve by the demand profile")
    parser.add_argument("--tag", default="", help="extra tag for the output filename")
    args = parser.parse_args()

    set_country_override(args.country)
    set_scenario_override(args.scenario)
    set_archetype_override(args.archetype)
    set_duration_override(args.duration_sampling)
    set_hourly_override(args.hourly_resolve)
    ema_logging.log_to_stderr(ema_logging.INFO)

    cfg = load_config()
    archetype = cfg["network"].get("timing_archetype", "winter_peak")
    dur = cfg["network"].get("duration_sampling", "fixed")
    hourly = bool(cfg["network"].get("hourly_resolve", False))
    print(f"Country: {cfg['country']}  Scenario: {cfg['scenario']}  "
          f"Archetype: {archetype}  Duration: {dur}  Hourly: {hourly}")

    model = build_model(cfg)
    kwargs = {"scenarios": args.n}
    if args.sampler == "sobol":
        if args.n & (args.n - 1) != 0:
            print(f"WARNING: Sobol base sample n={args.n} is not a power of 2; "
                  "SALib convergence is better with one.")
        k = len(model.uncertainties)
        print(f"Sobol/Saltelli: N={args.n}, k={k} -> {args.n * (2 * k + 2)} model runs")
        kwargs["uncertainty_sampling"] = Samplers.SOBOL

    if args.workers > 1:
        with MultiprocessingEvaluator(model, n_processes=args.workers) as ev:
            results = ev.perform_experiments(**kwargs)
    else:
        with SequentialEvaluator(model) as ev:
            results = ev.perform_experiments(**kwargs)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    variant = f"_{dur}" + ("_hourly" if hourly else "")
    out = (country_results_dir(cfg)
           / f"experiments_{result_stem(cfg)}_{archetype}{variant}_{args.sampler}"
             f"_n{args.n}{tag}_{stamp}.tar.gz")
    save_results(results, out)
    print(f"\nSaved results to {out}")


if __name__ == "__main__":
    main()
