#!/usr/bin/env bash
set -euo pipefail

# Created by Emanuele (https://github.com/Kirito-Emo)

# CONFIGURATION
BIN=../cmake-build-debug-mpi/hpc_mpi
SIZES=(1 50 100 200 500)
OUTPUT_DIR="mpi_measurements"

# Create output directory if it doesn't exist
if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

# RUNNING
run_measurement() {
  local NP="$1"
  local STATS_FILE="mpi_stats_${NP}.csv"
  local SUMMARY_FILE="mpi_summary_${NP}.csv"

  echo "Running measurement with $NP ranks..."

  echo "mode,size_mb,run,rank,time_s" > "$STATS_FILE"

  for mb in "${SIZES[@]}"; do
    INPUT_FILE="../random_strings/string_${mb}MB.bin"
    if [ ! -f "$INPUT_FILE" ]; then
      echo "Skipping size ${mb}MB → file not found: $INPUT_FILE"
      continue
    fi

    for run in $(seq 1 10); do
      mpirun -np "$NP" "$BIN" "$mb" 2>/dev/null \
        | grep 'rank=' \
        | awk -v mode="mpi" -v mb="$mb" -v run="$run" '{
            for(i=1;i<=NF;i++) {
              if ($i ~ /^rank=/) rank=substr($i,6);
              if ($i ~ /^time_build=/) time=substr($i,12);
            }
            print mode "," mb "," run "," rank "," time
          }' \
        >> "$STATS_FILE"
    done
  done

  # Generate summary (avg time per mode and size)
  echo "mode,size_mb,rank,avg_time_s" > "$SUMMARY_FILE"
  awk -F, 'NR>1 { key=$1","$2","$4; sum[key]+=$5; cnt[key]++ }
           END { for (k in sum) printf "%s,%.6f\n", k, sum[k]/cnt[k] }' "$STATS_FILE" >> "$SUMMARY_FILE"

  # Sort the summary by size and then rank
  sort -t, -k2n -k3n -o "$SUMMARY_FILE" "$SUMMARY_FILE"

  # Move to output directory
  mv "$STATS_FILE" "$OUTPUT_DIR/"
  mv "$SUMMARY_FILE" "$OUTPUT_DIR/"
}

# Run measurements for different ranks (cores)
for ranks in 2 4 8; do
  run_measurement "$ranks"
done