#!/usr/bin/env bash
# Submit the network-level power study to SLURM.
#
#   ./submit_network_study.sh setup                 # build the venv, once
#   ./submit_network_study.sh test                  # HRV + EST smoke test (LHS)
#   ./submit_network_study.sh gate                  # benchmark gate: EST + ITA, small N
#   ./submit_network_study.sh duration              # log-uniform vs uniform duration
#   ./submit_network_study.sh hourly                # profile-scaled vs hourly re-solve
#   ./submit_network_study.sh sobol                 # the campaign: 7 countries, N=512
#   ./submit_network_study.sh owner                 # extend basin ownership to the 22 others
#   ./submit_network_study.sh europe                # the 22 others, N=512, after `owner`
#   ./submit_network_study.sh dry                   # print the plan, submit nothing
#
# Run `gate` BEFORE `sobol`. Cost per evaluation is unmeasured on the current
# model - lev07 basins are ~3x larger than the lev08 ones the old timings came
# from, the country mask is gone so events fail components Europe-wide, and
# every draw now runs a DC OPF where half used to take the 1 ms connectivity
# screen. The gate exists to turn `sobol`'s TIME from a guess into a
# measurement, and to check that N=128 actually converges the Sobol indices.
#
# One SLURM job per (country, scenario, archetype) combination, so a walltime
# kill costs one combination rather than the campaign, and re-running the same
# command resumes: run_network_study.py skips combinations whose result file
# already exists.
#
# Unlike the direct-damage study there is NO Stage 1 stage here. Hazard
# intensities are read from the direct-damage study's validated
# data/intermediate output, so this script only ever runs Stage 2.
#
# Environment overrides (all optional):
#   REPO       clone to run from            (default: this script's directory)
#   USER_DIR   your own space               (default: /scistor/ivm/$USER)
#   VENV       python environment           (default: $USER_DIR/.venvs/miraca_net)
#   NET_INTERMEDIATE_DIR   Stage 1 output   (default: the cluster config's path)
#   PARTITION, ACCOUNT     if your cluster requires them
#   EXCLUDE    nodes to avoid               (default: node009, ~2.5x slower)
#   MAIL_USER  e-mail address: each job mails on FAIL, and one final job that
#              waits for all of them mails once the whole submission is done
#   COUNTRIES, SCENARIOS, ARCHETYPES, N, WORKERS, TIME
#
# WALLTIME: every tier asks for 120 h, the defq maximum that is proven to
# schedule. Deliberately not sized to an estimate - no cluster run exists yet
# on the current model (HiGHS OPF, HVDC, lev07 basins, pan-EU exposure), node
# speeds differ ~2.5x, and the direct-damage study lost near-finished work
# (99.3%, 99.6%) to walltimes sized from estimates. Idle walltime costs
# nothing; a job exits when it is done.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO:-$SCRIPT_DIR}
USER_DIR=${USER_DIR:-/scistor/ivm/${USER:-$(id -un)}}
VENV=${VENV:-${USER_DIR}/.venvs/miraca_net}
PYTHON=${PYTHON:-${VENV}/bin/python}
LOG_DIR=${LOG_DIR:-${USER_DIR}/MIRACA_NET/logs}

# The direct-damage clone supplying Stage 1 output and the miraca_uq package.
DIRECT_REPO=${DIRECT_REPO:-/scistor/ivm/yma794/Uncertainty-Modelling}

export MIRACA_NET_CONFIG=${MIRACA_NET_CONFIG:-${REPO}/config.network.cluster.yml}

MODULES=${MODULES:-"gdal/3.6.2"}
UV_PYTHON=${UV_PYTHON:-3.12}

PARTITION=${PARTITION:-}
if [[ -n "$PARTITION" ]]; then
    SB_PARTITION="#SBATCH --partition=${PARTITION}"
else
    SB_PARTITION="# (no --partition set; using the cluster default)"
fi
ACCOUNT=${ACCOUNT:-}
if [[ -n "$ACCOUNT" ]]; then
    SB_ACCOUNT="#SBATCH --account=${ACCOUNT}"
else
    SB_ACCOUNT="# (no --account set; export ACCOUNT=... if your cluster needs one)"
fi

EXCLUDE=${EXCLUDE-node009}
if [[ -n "$EXCLUDE" ]]; then
    SB_EXCLUDE="#SBATCH --exclude=${EXCLUDE}"
else
    SB_EXCLUDE="# (no nodes excluded)"
fi

MAIL_USER=${MAIL_USER:-}
if [[ -n "$MAIL_USER" ]]; then
    SB_MAIL="#SBATCH --mail-user=${MAIL_USER}
#SBATCH --mail-type=FAIL"
else
    SB_MAIL="# (no MAIL_USER set; no e-mail)"
fi

WALL=120:00:00

mode=${1:-test}

# The 22 runnable countries outside the study set, largest predicted cost
# first (see the `europe` tier for how the order was derived).
EUROPE_COUNTRIES="ESP FRA DEU PRT CHE BGR GBR NOR SWE SRB POL FIN GRC BEL SVK LVA IRL DNK SVN CZE LTU LUX"

# --- tiers --------------------------------------------------------------------
# Stage 2 only. Each worker holds its own copy of the pan-European network
# (~6.7k buses, 9k lines) plus the hazard tables, so memory is per-CPU. Measured
# footprint of one NetData is well under 2 GB; 4 GB/cpu leaves headroom for the
# pandapower OPF's internal matrices.
case "$mode" in
    test)
        # Smallest thing that exercises the whole path on a study country.
        COUNTRIES=${COUNTRIES:-"HRV EST"}
        SCENARIOS=${SCENARIOS:-"net_flood net_coastal net_earthquake net_windstorm"}
        ARCHETYPES=${ARCHETYPES:-"winter_peak"}
        VARIANTS=${VARIANTS:-"fixed:0"}
        SAMPLER=lhs;   N=${N:-32};   WORKERS=${WORKERS:-8}
        TIME=${TIME:-$WALL};      MEM=${MEM:-4G}
        ;;
    gate)
        # The benchmark gate the design calls for: one cheap country and one
        # expensive one, same design, small N. EST has 19 buses / 12 owned
        # basins, ITA 592 / 101 - the two ends of the cost range, so the pair
        # brackets what the campaign will cost. Read three things off it:
        # solves per evaluation, wall-clock per combination (-> sobol's TIME),
        # and whether the first-order/total indices have separated at this N.
        # If their confidence intervals still straddle the factor ranking,
        # raise N rather than running the campaign and discovering it later.
        COUNTRIES=${COUNTRIES:-"EST ITA"}
        SCENARIOS=${SCENARIOS:-"net_flood net_coastal net_earthquake net_windstorm"}
        ARCHETYPES=${ARCHETYPES:-"winter_peak"}
        VARIANTS=${VARIANTS:-"fixed:0"}
        SAMPLER=sobol; N=${N:-16};   WORKERS=${WORKERS:-16}
        TIME=${TIME:-$WALL};      MEM=${MEM:-4G}
        ;;
    duration)
        # Does the duration prior matter? Same LHS design, two mappings of the
        # same unit draw over [6 h, 14 d], so the comparison is PAIRED - any
        # difference is the prior, not sampling noise.
        # LUX/SVN are not study countries; these two tiers are methodology
        # side-studies whose country choice was made for the reason documented
        # here, so they keep it. `net_flood_absprot` became `net_flood` when
        # protection went post-hoc.
        COUNTRIES=${COUNTRIES:-"LUX SVN EST"}
        SCENARIOS=${SCENARIOS:-"net_windstorm net_flood"}
        ARCHETYPES=${ARCHETYPES:-"winter_peak"}
        VARIANTS=${VARIANTS:-"loguniform:0 uniform:0"}
        SAMPLER=lhs;   N=${N:-400};  WORKERS=${WORKERS:-8}
        TIME=${TIME:-$WALL};      MEM=${MEM:-4G}
        ;;
    hourly)
        # Is one DC OPF at t0 plus profile scaling a good enough stand-in for
        # re-solving every hour? Same design, same draws, one switch flipped.
        #
        # Flood is the primary scenario, not windstorm: measured 3-sec gusts are
        # 51 m/s in LUX but only 35-42 in SVN/EST, and the wind fragility curves
        # are flat below ~40 m/s, so SVN/EST windstorm returns near-zero and
        # carries no signal about the approximation. Windstorm is kept because
        # LUX does exercise it.
        COUNTRIES=${COUNTRIES:-"LUX SVN EST"}
        SCENARIOS=${SCENARIOS:-"net_flood_absprot net_windstorm"}
        ARCHETYPES=${ARCHETYPES:-"winter_peak"}
        VARIANTS=${VARIANTS:-"loguniform:0 loguniform:1"}
        SAMPLER=lhs;   N=${N:-300};  WORKERS=${WORKERS:-8}
        TIME=${TIME:-$WALL};      MEM=${MEM:-4G}
        ;;
    sobol)
        # The campaign. All seven study countries, 28 nominal combinations,
        # 23 submitted: coastal is not applicable for landlocked AUT and HUN,
        # and ROU, EST and NLD coastal are structural zeros (no exposed grid
        # component in their owned basins) - recorded as EAENS = 0 by the
        # dry run below, not run. ONE archetype: every extra archetype
        # multiplies the campaign for a conditioning dimension, not a factor.
        #
        # N=512, not 128: the 2026-09-23 LHS N=32 pilot of all 23 combinations
        # showed windstorm and earthquake EAENS are zero-inflated and heavy-
        # tailed (top 3 of 32 draws carry 52-86% of the windstorm total), which
        # variance-based indices converge on slowly. Projected from that pilot
        # (8 workers, start-up included, so an overestimate): ~2.9k core-h in
        # total, every job under ~27 h on 16 workers except ITA earthquake at
        # ~48 h. WALL is already the 120 h cap, so that one job gets more
        # workers instead of more time: at 24 it projects to ~32 h, which
        # survives even a 2.5x slower node. 24 x 4G = 96 GB fits one 64-core,
        # ~123 GB node; 32 x 4G would not.
        COUNTRIES=${COUNTRIES:-"AUT EST HRV HUN ITA NLD ROU"}
        SCENARIOS=${SCENARIOS:-"net_flood net_coastal net_earthquake net_windstorm"}
        ARCHETYPES=${ARCHETYPES:-"winter_peak"}
        VARIANTS=${VARIANTS:-"fixed:0"}
        SAMPLER=sobol; N=${N:-512}; WORKERS=${WORKERS:-16}
        BIG_COMBOS=${BIG_COMBOS:-"ITA:net_earthquake"}; BIG_WORKERS=${BIG_WORKERS:-24}
        TIME=${TIME:-$WALL};      MEM=${MEM:-4G}
        ;;
    owner)
        # Extend cache/basin_owner.parquet from the 7 study countries to the
        # 22 others, keeping the 7's rows verbatim (build_basin_owner.py
        # --extend), so `sobol` jobs still queued read an unchanged partition.
        # Must finish before `europe` is submitted: that tier's exposure screen
        # reads the new owners at submit time.
        COUNTRIES=${COUNTRIES:-"$EUROPE_COUNTRIES"}
        ;;
    europe)
        # The 22 remaining runnable countries (results/data_coverage.xlsx:
        # demand AND Stage 1 exposure; ALB, MKD, BIH, MNE, XKX, MDA, UKR
        # cannot run). Same design as `sobol` so all 29 are one campaign.
        #
        # ORDER IS DELIBERATE: largest predicted cost first, whole countries
        # at a time. SLURM starts equal-priority jobs in submission order and
        # the loop below is country-major, so countries complete roughly in
        # this order and whatever has finished at any point is a clean prefix
        # (the largest systems) rather than a patchwork of hazards. Largest-
        # first also means no multi-day job is left running alone at the end.
        # Costs are predicted, not measured: per-hazard log-log fits of the
        # N=32 pilot's cost against the number of exposed components in each
        # country's owned basins (residual x1.1-1.6), ~6.8k core-h in total;
        # ESP/FRA/DEU extrapolate beyond ITA, the largest calibration point.
        #
        # 24 workers for every combination predicted above 200 core-h; the
        # largest (ESP earthquake, ~350) then projects to ~15 h, ~60 h even at
        # 1.6x the prediction on a 2.5x slower node - inside the 120 h cap.
        COUNTRIES=${COUNTRIES:-"$EUROPE_COUNTRIES"}
        SCENARIOS=${SCENARIOS:-"net_earthquake net_flood net_windstorm net_coastal"}
        ARCHETYPES=${ARCHETYPES:-"winter_peak"}
        VARIANTS=${VARIANTS:-"fixed:0"}
        SAMPLER=sobol; N=${N:-512}; WORKERS=${WORKERS:-16}
        BIG_COMBOS=${BIG_COMBOS:-"ESP:net_earthquake ESP:net_flood ESP:net_windstorm FRA:net_earthquake FRA:net_flood FRA:net_windstorm DEU:net_earthquake DEU:net_flood PRT:net_earthquake BGR:net_earthquake GBR:net_flood"}
        BIG_WORKERS=${BIG_WORKERS:-24}
        TIME=${TIME:-$WALL};      MEM=${MEM:-4G}
        ;;
    dry)
        COUNTRIES=${COUNTRIES:-"AUT EST HRV HUN ITA NLD ROU"}
        SCENARIOS=${SCENARIOS:-"net_flood net_coastal net_earthquake net_windstorm"}
        ARCHETYPES=${ARCHETYPES:-"winter_peak"}
        VARIANTS=${VARIANTS:-"fixed:0"}
        SAMPLER=sobol; N=${N:-512}; WORKERS=1
        TIME=00:10:00; MEM=4G
        ;;
    setup) ;;
    *) echo "usage: $0 {setup|test|gate|duration|hourly|sobol|owner|europe|dry}" >&2; exit 2 ;;
esac

# --- setup --------------------------------------------------------------------
if [[ "$mode" == "setup" ]]; then
    module load ${MODULES}
    command -v uv >/dev/null || { echo "uv not on PATH" >&2; exit 1; }
    rm -rf "$VENV"
    uv venv "$VENV" --python "${UV_PYTHON}" --python-preference only-managed
    uv pip install --python "${PYTHON}" -r "${DIRECT_REPO}/requirements.txt"
    uv pip install --python "${PYTHON}" -r "${REPO}/requirements-network.txt"
    # The direct-damage repo installed as a package: this is what supplies
    # miraca_uq.risk_model._integrate_ead and the curve machinery, so the two
    # studies share one implementation rather than a copy.
    uv pip install --python "${PYTHON}" -e "${DIRECT_REPO}"
    "${PYTHON}" -c "import pandapower, ema_workbench, miraca_uq.risk_model; print('env OK', pandapower.__version__)"
    echo "venv ready at ${VENV}"
    echo
    echo "Now build the three caches (once, ~15 min) before submitting:"
    echo "  ${PYTHON} scripts/build_component_basins.py"
    echo "  ${PYTHON} scripts/build_paneu_hazard.py"
    echo "  ${PYTHON} scripts/build_basin_owner.py"
    exit 0
fi

# --- required caches ----------------------------------------------------------
# Every one of these is country-independent and built once. A missing cache is
# not a crash - the model falls back to study-country-only labels and exposure,
# which quietly turns a Europe-wide basin event back into its study-country
# half and writes an empty attribution table. That is a wrong result with a
# zero exit code, so it is checked here rather than discovered afterwards.
if [[ "$mode" != "dry" ]]; then
    missing=0
    for f in component_basin.parquet basin_owner.parquet \
             hazard_paneu_river.npz hazard_paneu_earthquake.npz \
             hazard_paneu_windstorm.npz hazard_paneu_coastal.npz; do
        if [[ ! -f "${REPO}/cache/${f}" ]]; then
            echo "MISSING cache/${f}" >&2
            missing=1
        fi
    done
    if [[ "$missing" == "1" ]]; then
        echo "Run '$0 setup' and then the three build_*.py scripts it lists." >&2
        exit 1
    fi
fi

mkdir -p "$LOG_DIR"

if [[ ! -x "$PYTHON" ]]; then
    echo "No python at ${PYTHON}. Run '$0 setup' first." >&2
    exit 1
fi

# --- basin ownership for added countries --------------------------------------
if [[ "$mode" == "owner" ]]; then
    if [[ -n "$MAIL_USER" ]]; then
        SB_OWNER_MAIL="#SBATCH --mail-user=${MAIL_USER}
#SBATCH --mail-type=END,FAIL"
    else
        SB_OWNER_MAIL="# (no MAIL_USER set; no e-mail)"
    fi
    jid=$(sbatch --parsable <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=net_basin_owner
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
${SB_PARTITION}
${SB_ACCOUNT}
${SB_EXCLUDE}
${SB_OWNER_MAIL}
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=${LOG_DIR}/owner_%j
#SBATCH --error=${LOG_DIR}/owner_err_%j

set -euo pipefail
module load ${MODULES}
export MIRACA_NET_CONFIG=${MIRACA_NET_CONFIG}
cd ${REPO}
cp cache/basin_owner.parquet "cache/basin_owner.before_extend_\$(date +%Y%m%d_%H%M%S).parquet"
"${PYTHON}" scripts/build_basin_owner.py --extend --skip-lau --countries ${COUNTRIES}
EOF
    )
    echo "submitted ${jid}  basin ownership for: ${COUNTRIES}"
    echo "When it has finished (log: ${LOG_DIR}/owner_${jid}), submit: $0 europe"
    exit 0
fi

# `europe` screens its countries against basin_owner.parquet at submit time;
# a country missing from it would silently fall back to "every basin it
# touches" and double-count border basins. Refuse rather than do that.
if [[ "$mode" == "europe" ]]; then
    missing=$("${PYTHON}" -c "import sys, pandas as pd; o=set(pd.read_parquet('${REPO}/cache/basin_owner.parquet').country); print(' '.join(c for c in sys.argv[1:] if c not in o))" ${COUNTRIES})
    if [[ -n "$missing" ]]; then
        echo "basin_owner.parquet has no basins for: ${missing}" >&2
        echo "Run '$0 owner' first and wait for it to finish." >&2
        exit 1
    fi
fi

# --- plan ---------------------------------------------------------------------
echo "repo        ${REPO}"
echo "config      ${MIRACA_NET_CONFIG}"
echo "stage 1 in  ${DIRECT_REPO}/data/intermediate"
echo "countries   ${COUNTRIES}"
echo "scenarios   ${SCENARIOS}"
echo "archetypes  ${ARCHETYPES}"
echo "variants    ${VARIANTS}   (duration_sampling:hourly_resolve)"
echo "sampler     ${SAMPLER} n=${N} workers=${WORKERS}"
echo

cd "$REPO"
# The dry run is also the screen: it drops non-applicable hazards and
# structural zeros (writing each zero's results/<ISO3>/structural_zero_*.json)
# and hands back the pairs worth a job. Only those are submitted below.
PLAN=$(mktemp)
trap 'rm -f "$PLAN"' EXIT
"${PYTHON}" run_network_study.py --countries ${COUNTRIES} --scenarios ${SCENARIOS} \
    --archetypes ${ARCHETYPES} --sampler "${SAMPLER}" --n "${N}" --dry-run \
    --plan-out "$PLAN"

if [[ "$mode" == "dry" ]]; then
    echo; echo "(dry mode: nothing submitted)"
    exit 0
fi

# --- submit -------------------------------------------------------------------
n_sub=0
jids=""
for country in ${COUNTRIES}; do
  for scenario in ${SCENARIOS}; do
    grep -qx "${country} ${scenario}" "$PLAN" || continue
    for archetype in ${ARCHETYPES}; do
      for variant in ${VARIANTS}; do
      dur="${variant%%:*}"; hr="${variant##*:}"
      hrflag=""; hrtag=""
      if [[ "$hr" == "1" ]]; then hrflag="--hourly-resolve"; hrtag="_hourly"; fi
      label="${country}_${scenario}_${archetype}_${dur}${hrtag}"
      w=${WORKERS}
      if [[ " ${BIG_COMBOS:-} " == *" ${country}:${scenario} "* ]]; then w=${BIG_WORKERS}; fi
      jid=$(sbatch --parsable <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=net_${label}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${w}
${SB_PARTITION}
${SB_ACCOUNT}
${SB_EXCLUDE}
${SB_MAIL}
#SBATCH --mem-per-cpu=${MEM}
#SBATCH --time=${TIME}
#SBATCH --output=${LOG_DIR}/out_${label}_%j
#SBATCH --error=${LOG_DIR}/err_${label}_%j

set -euo pipefail
module load ${MODULES}
export MIRACA_NET_CONFIG=${MIRACA_NET_CONFIG}
cd ${REPO}
# sbatch exports the submitting shell's environment. When this script itself
# runs inside a SLURM job (e.g. a dependency-chained submitter), that includes
# the parent's SLURM_MEM_PER_NODE, which collides with this job's own
# SLURM_MEM_PER_CPU and makes srun abort with "mutually exclusive" before the
# model starts - that failed all 78 europe jobs on 2026-09-26. This job always
# sets --mem-per-cpu, so the other two can only be inherited: drop them.
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_GPU
# Deliberately ONE line, with every expansion quoted. A backslash line
# continuation inside an unquoted heredoc is not portable across bash
# versions - some collapse it, some leave a stray whitespace token - and
# because --countries/--scenarios/--archetypes all use nargs="+", such a
# token is silently swallowed as an extra value and fails as
# "invalid choice: ' '". Keep this on one line.
srun "${PYTHON}" run_network_study.py --countries "${country}" --scenarios "${scenario}" --archetypes "${archetype}" --sampler "${SAMPLER}" --n "${N}" --workers "${w}" --duration-sampling "${dur}" ${hrflag}
EOF
      )
      echo "submitted ${jid}  ${label}  (${w} workers)"
      n_sub=$((n_sub + 1))
      jids="${jids:+${jids}:}${jid}"
      done
    done
  done
done

# One e-mail for the whole submission rather than one per job: a 1-CPU job
# that starts only once every job above has ended (afterany: succeeded, failed,
# timed out or cancelled alike) and mails on its own END. Its output lists
# each job's final state, so the log says which ones need attention.
if [[ -n "$MAIL_USER" && "$n_sub" -gt 0 ]]; then
    njid=$(sbatch --parsable --dependency="afterany:${jids}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=net_${mode}_done
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
${SB_PARTITION}
${SB_ACCOUNT}
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --mail-user=${MAIL_USER}
#SBATCH --mail-type=END,FAIL
#SBATCH --output=${LOG_DIR}/done_${mode}_%j

sacct -X -j "${jids//:/,}" --format=JobID,JobName%60,State,Elapsed
EOF
    )
    echo "submitted ${njid}  completion notice -> ${MAIL_USER} (after all ${n_sub} jobs)"
fi

echo
echo "${n_sub} job(s) submitted. Watch with:  squeue -u \$USER"
echo "Logs: ${LOG_DIR}"
