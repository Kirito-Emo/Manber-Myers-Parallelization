#!/usr/bin/env bash
set -euo pipefail

# Created by Emanuele (https://github.com/Kirito-Emo)

# CONFIGURATION
BIN=../cmake-build-debug-mpi-omp/hpc_mpi_omp
SIZES=(1 50 100 200 500)
OUTPUT_DIR="mpi_omp_measurements"
RUNS=10

# Pairs of (MPI ranks, OMP threads)
CONFIGS=(
  "2 8"
  "4 4"
  "8 2"
)

# Check if the current dir is perf-stats otherwise cd into it
if [[ $PWD != *"perf-stats"* ]]; then
  cd perf-stats || exit 1
fi

# Check if binary exists
if [ ! -f "$BIN" ]; then
  echo "Error: Binary not found at $BIN. Please build the project first."
  exit 2
fi

# Create output directory if it doesn't exist
if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

run_measurement() {
  local NP="$1"
  local NT="$2"
  local STATS_FILE="mpiomp_stats_${NP}r_${NT}t.csv"
  local SUMMARY_FILE="mpi_omp_summary_${NP}r_${NT}t.csv"

  echo "Running measurement: $NP MPI ranks × $NT OMP threads"

  echo "mode,size_mb,run,rank,time_s" > "$STATS_FILE"

  for mb in "${SIZES[@]}"; do
    INPUT_FILE="../random_strings/string_${mb}MB.bin"
    if [ ! -f "$INPUT_FILE" ]; then
      echo "Skipping size ${mb}MB → file not found: $INPUT_FILE"
      continue
    fi

    for run in $(seq 1 "$RUNS"); do
      OMP_NUM_THREADS="$NT" mpirun -np "$NP" "$BIN" "$mb" 2>/dev/null \
        | grep 'rank=' \
        | awk -v mode="mpi+omp" -v mb="$mb" -v run="$run" '{
            for(i=1;i<=NF;i++) {
              if ($i ~ /^rank=/) rank=substr($i,6);
              if ($i ~ /^time_build=/) time=substr($i,12);
            }
            print mode "," mb "," run "," rank "," time
          }' \
        >> "$STATS_FILE"
    done
  done

  # Generate summary
  echo "mode,size_mb,rank,avg_time_s" > "$SUMMARY_FILE"
  awk -F, 'NR>1 { key=$1","$2","$3; sum[key]+=$5; cnt[key]++ }
           END { for (k in sum) printf "%s,%.6f\n", k, sum[k]/cnt[k] }' "$STATS_FILE" >> "$SUMMARY_FILE"

  # Sort and move
  sort -t, -k2n -k3n -o "$SUMMARY_FILE" "$SUMMARY_FILE"
  mv "$STATS_FILE" "$OUTPUT_DIR/"
  mv "$SUMMARY_FILE" "$OUTPUT_DIR/"
}

# Run for all configs
for cfg in "${CONFIGS[@]}"; do
  read -r ranks threads <<< "$cfg"
  run_measurement "$ranks" "$threads"
done

echo "MPI+OpenMP benchmarks completed → $OUTPUT_DIR/"