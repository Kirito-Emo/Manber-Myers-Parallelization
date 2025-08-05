#!/usr/bin/env bash
set -euo pipefail

# Created by Emanuele (https://github.com/Kirito-Emo)

# CONFIGURATION
BIN=../cmake-build-debug-omp/hpc_omp    # path to OpenMP binary
RUNS=10                                 # how many repeats per size
SIZES=(1 50 100 200 500)                # sizes in MB
THREADS=(4 8)                           # OpenMP thread counts
OUTPUT_DIR=omp_measurements             # output directory root

# Check if current directory is perf-stats
if [[ $PWD != *"perf-stats"* ]]; then
  cd perf-stats || exit 1
fi

# Check if binary exists
if [[ ! -f "$BIN" ]]; then
  echo "Error: Binary not found at $BIN. Please build the project first."
  exit 2
fi

# Create output directory if it doesn't exist
if [[ ! -d "$OUTPUT_DIR" ]]; then
  mkdir -p "$OUTPUT_DIR"
fi

# Loop over thread counts
for th in "${THREADS[@]}"; do
  OUTDIR="$OUTPUT_DIR/threads_${th}"
  mkdir -p "$OUTDIR"

  # CSV headers
  STATS_CSV="$OUTDIR/omp_stats_${th}.csv"
  SUMMARY_CSV="$OUTDIR/omp_summary_${th}.csv"
  echo "threads,size,run,build_time" > "$STATS_CSV"

  echo "Running OpenMP tests with $th threads..."

  for mb in "${SIZES[@]}"; do
    for run in $(seq 1 "$RUNS"); do
      export OMP_NUM_THREADS="$th"
      t=$("$BIN" "$mb" \
        | grep -m1 -o 'time_build=[0-9]\+\(\.[0-9]\+\)\?' \
        | cut -d= -f2)
      echo "$th,${mb}MB,${run},${t}s" >> "$STATS_CSV"
    done
  done

  # Generate summary
  echo "threads,size,avg_build_time" > "$SUMMARY_CSV"
  awk -F, '
    NR>1 {
      val = substr($4, 1, length($4)-1)
      key = $1 "," $2
      sum[key] += val
      cnt[key]++
    }
    END {
      for (k in sum) {
        printf "%s,%.6fs\n", k, sum[k]/cnt[k]
      }
    }
  ' "$STATS_CSV" | sort -t, -k1n -k2,2V >> "$SUMMARY_CSV"

  echo "- Completed: OpenMP $th threads"
done

echo "All OpenMP runs completed. Results saved in: $OUTPUT_DIR/"