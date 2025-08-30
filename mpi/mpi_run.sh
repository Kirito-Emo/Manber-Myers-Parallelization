#!/usr/bin/env bash
set -euo pipefail

# Created by Emanuele (https://github.com/Kirito-Emo)
# Measure an MPI binary with different ranks and collect:
#  1) CPU per core (mpstat)
#  2) CPU/RSS per process (ps)
#  3) stdout (MPI metrics, printed by rank 0 and rows per-rank)
#
# Output in: mpi_chunks/np_<ranks>/{cpu.log, ps.log, out.log}

# -----------------------
# CONFIGURATION
# -----------------------
BIN="../cmake-build-debug/hpc_mpi"       # Path to MPI binary
MB=100                                   # Size in MB of input file
INPUT_FILE="../random_strings/string_${MB}MB.bin"
OUTPUT_ROOT="mpi_chunks"
RANKS=("2" "4" "8")
MPIRUN_OPTS=()                           # es. (--bind-to core --map-by socket)
PS_INTERVAL=1                            # sampling ps/mpstat interval in seconds
# -----------------------

# Ensure the actual folder is perf-stats
if [[ $PWD != *"perf-stats"* ]]; then
  cd perf-stats || exit 1
fi

# Ensure binary exists
if [[ ! -x "$BIN" ]]; then
  echo "Executable not found or not executable at: $BIN"
  exit 2
fi

# Ensure input file exists
if [[ ! -f "$INPUT_FILE" ]]; then
  echo "Input file $INPUT_FILE not found. Please generate it first."
  exit 3
fi

mkdir -p "$OUTPUT_ROOT"

# Check for mpstat
if ! command -v mpstat >/dev/null 2>&1; then
  echo "WARNING: 'mpstat' not found. Install sysstat to get CPU per-core." >&2
fi

# Cleanup on exit
mpstat_pid=""
psloop_pid=""
cleanup() {
  [[ -n "${mpstat_pid}" && -e "/proc/${mpstat_pid}" ]] && kill "${mpstat_pid}" 2>/dev/null || true
  [[ -n "${psloop_pid}" && -e "/proc/${psloop_pid}" ]] && kill "${psloop_pid}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

monitor_and_run() {
  local np="$1"
  local outdir="${OUTPUT_ROOT}/np_${np}/"
  mkdir -p "${outdir}"

  local cpu_log="${outdir}/cpu.log"
  local ps_log="${outdir}/ps.log"
  local out_log="${outdir}/out.log"

  echo "=== Launching: mpirun -np ${np} ${BIN} ${MB} ==="
  echo "  logs → ${outdir}/{cpu.log,ps.log,out.log}"

  # Start mpstat if available
  if command -v mpstat >/dev/null 2>&1; then
    mpstat -P ALL "${PS_INTERVAL}" > "${cpu_log}" &
    mpstat_pid=$!
  else
    mpstat_pid=""
  fi

  # Loop ps in background
  (
    while sleep "${PS_INTERVAL}"; do
      ps -eLo pid,psr,pcpu,pmem,rss,etimes,comm,args \
        | awk -v bn="$(basename "$BIN")" '
            NR==1 { print; next }                # header
            $6 ~ bn || $7 ~ bn || $8 ~ bn { print }'
    done
  ) > "${ps_log}" &
  psloop_pid=$!

  # MPI execution (stdout → out.log to maintain per-rank output)
  if ! mpirun -np "${np}" "${MPIRUN_OPTS[@]}" "$BIN" "$MB" | tee "${out_log}"; then
    echo "mpirun fallito per np=${np}" >&2
  fi

  # Stop monitoring
  cleanup

  echo "---- Done run for np=${np}. Logs in ${outdir}/ ----"
  echo
}

# Loop on ranks
for ranks in "${RANKS[@]}"; do
  monitor_and_run "${ranks}"
done