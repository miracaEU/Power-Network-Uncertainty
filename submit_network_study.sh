#!/usr/bin/env bash
# Submit the network-level power study to SLURM.
#
#   ./submit_network_study.sh setup                 # build the venv, once
#   ./submit_network_study.sh test                  # LUX + PRT smoke test (LHS)
#   ./submit_network_study.sh duration              # log-uniform vs uniform duration
#   ./submit_network_study.sh hourly                # profile-scaled vs hourly re-solve
#   ./submit_network_study.sh sobol                 # Sobol on the default set
#   ./submit_network_study.sh dry                   # print the plan, submit nothing
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
#   COUNTRIES, SCENARIOS, ARCHETYPES, N, WORKERS

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

mode=${1:-test}

# --- tiers --------------------------------------------------------------------
# Stage 2 only. Each worker holds its own copy of the pan-European network
# (~6.7k buses, 9k lines) plus the hazard tables, so memory is per-CPU. Measured
# footprint of one NetData is well under 2 GB; 4 GB/cpu leaves headroom for the
# pandapower OPF's internal matrices.
case "$mode" in
    test)
        COUNTRIES=${COUNTRIES:-"LUX PRT"}
        SCENARIOS=${SCENARIOS:-"net_flood_absprot net_earthquake net_windstorm"}
        ARCHETYPES=${ARCHETYPES:-"winter_peak"}
        VARIANTS=${VARIANTS:-"fixed:0"}
        SAMPLER=lhs;   N=${N:-200};  WORKERS=${WORKERS:-8}
        TIME=${TIME:-04:00:00};      MEM=${MEM:-4G}
        ;;
    duration)
        # Does the duration prior matter? Same LHS design, two mappings of the
        # same unit draw over [6 h, 14 d], so the comparison is PAIRED - any
        # difference is the prior, not sampling noise.
        COUNTRIES=${COUNTRIES:-"LUX SVN EST"}
        SCENARIOS=${SCENARIOS:-"net_windstorm net_flood_absprot"}
        ARCHETYPES=${ARCHETYPES:-"winter_peak"}
        VARIANTS=${VARIANTS:-"loguniform:0 uniform:0"}
        SAMPLER=lhs;   N=${N:-400};  WORKERS=${WORKERS:-8}
        TIME=${TIME:-06:00:00};      MEM=${MEM:-4G}
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
        TIME=${TIME:-12:00:00};      MEM=${MEM:-4G}
        ;;
    sobol)
        COUNTRIES=${COUNTRIES:-"LUX PRT"}
        SCENARIOS=${SCENARIOS:-"net_flood_absprot net_flood_noprot net_earthquake net_windstorm"}
        ARCHETYPES=${ARCHETYPES:-"winter_peak summer_night"}
        VARIANTS=${VARIANTS:-"loguniform:0"}
        SAMPLER=sobol; N=${N:-128}; WORKERS=${WORKERS:-16}
        TIME=${TIME:-24:00:00};      MEM=${MEM:-4G}
        ;;
    dry)
        COUNTRIES=${COUNTRIES:-"LUX PRT"}
        SCENARIOS=${SCENARIOS:-"net_flood_absprot net_earthquake net_windstorm"}
        ARCHETYPES=${ARCHETYPES:-"winter_peak"}
        VARIANTS=${VARIANTS:-"fixed:0"}
        SAMPLER=lhs;   N=${N:-200};  WORKERS=1
        TIME=00:10:00; MEM=4G
        ;;
    setup) ;;
    *) echo "usage: $0 {setup|test|sobol|dry}" >&2; exit 2 ;;
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
    exit 0
fi

mkdir -p "$LOG_DIR"

if [[ ! -x "$PYTHON" ]]; then
    echo "No python at ${PYTHON}. Run '$0 setup' first." >&2
    exit 1
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
"${PYTHON}" run_network_study.py --countries ${COUNTRIES} --scenarios ${SCENARIOS} \
    --archetypes ${ARCHETYPES} --sampler "${SAMPLER}" --n "${N}" --dry-run

if [[ "$mode" == "dry" ]]; then
    echo; echo "(dry mode: nothing submitted)"
    exit 0
fi

# --- submit -------------------------------------------------------------------
n_sub=0
for country in ${COUNTRIES}; do
  for scenario in ${SCENARIOS}; do
    for archetype in ${ARCHETYPES}; do
      for variant in ${VARIANTS}; do
      dur="${variant%%:*}"; hr="${variant##*:}"
      hrflag=""; hrtag=""
      if [[ "$hr" == "1" ]]; then hrflag="--hourly-resolve"; hrtag="_hourly"; fi
      label="${country}_${scenario}_${archetype}_${dur}${hrtag}"
      jid=$(sbatch --parsable <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=net_${label}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${WORKERS}
${SB_PARTITION}
${SB_ACCOUNT}
#SBATCH --mem-per-cpu=${MEM}
#SBATCH --time=${TIME}
#SBATCH --output=${LOG_DIR}/out_${label}_%j
#SBATCH --error=${LOG_DIR}/err_${label}_%j

set -euo pipefail
module load ${MODULES}
export MIRACA_NET_CONFIG=${MIRACA_NET_CONFIG}
cd ${REPO}
# Deliberately ONE line, with every expansion quoted. A backslash line
# continuation inside an unquoted heredoc is not portable across bash
# versions - some collapse it, some leave a stray whitespace token - and
# because --countries/--scenarios/--archetypes all use nargs="+", such a
# token is silently swallowed as an extra value and fails as
# "invalid choice: ' '". Keep this on one line.
srun "${PYTHON}" run_network_study.py --countries "${country}" --scenarios "${scenario}" --archetypes "${archetype}" --sampler "${SAMPLER}" --n "${N}" --workers "${WORKERS}" --duration-sampling "${dur}" ${hrflag}
EOF
      )
      echo "submitted ${jid}  ${label}"
      n_sub=$((n_sub + 1))
      done
    done
  done
done

echo
echo "${n_sub} job(s) submitted. Watch with:  squeue -u \$USER"
echo "Logs: ${LOG_DIR}"
