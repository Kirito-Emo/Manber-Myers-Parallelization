#!/usr/bin/env bash
set -euo pipefail

# Created by Emanuele (https://github.com/Kirito-Emo)
# This script runs an MPI program at varying ranks and:
#   1) collects per-core CPU stats via mpstat
#   2) collects per-process CPU/RSS stats via ps
# It will produce four log files for each run: cpu_<np>.log and ps_<np>.log

# CONFIGURATION
BIN="../cmake-build-debug-mpi/hpc_mpi"
MB=10

# Check if the executable exists
if [ ! -f "../cmake-build-debug-mpi/hpc_mpi" ]; then
    echo "Executable not found. Please build the project first."
    exit 1
fi

monitor_and_run(){
  local np=$1
  local cpu_log="cpu_${np}.log"
  local ps_log="ps_${np}.log"

  echo "=== Launching mpirun -np $np ==="
  echo "  logging 'mpstat -P ALL 1' → $cpu_log"
  echo "  logging 'ps -eLo pid,psr,pcpu,pmem,cmd | grep hpc_mpi' → $ps_log"

  # Start mpstat in the background
  mpstat -P ALL 1 > "$cpu_log" &
  mpstat_pid=$!

  # Start a ps‐loop in background
  (
    while sleep 1; do
      ps -eLo pid,psr,pcpu,pmem,cmd \
        | grep "[h]pc_mpi"               \
        >> "$ps_log"
    done
  ) &
  ps_pid=$!

  # Run MPI job (this will block until MPI jobs finish)
  mpirun -np "$np" "$BIN" "$MB"

  # Stop the monitors
  kill $mpstat_pid $ps_pid

  echo "---- Done run for np=$np, logs in $cpu_log and $ps_log ----"
  echo
}

# Run for 2, 4 and 8 ranks
for ranks in 2 4 8; do
  monitor_and_run "$ranks"
done