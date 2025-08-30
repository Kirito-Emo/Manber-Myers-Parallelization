#!/usr/bin/env bash
set -euo pipefail

# Created by Emanuele (https://github.com/Kirito-Emo)

# CONFIGURATION
BIN=../cmake-build-release/hpc      # path to binary
RUNS=10                             # how many repeats per size
SIZES=(1 50 100 200 500)            # sizes in MB
OUTPUT_DIR=seq_measurements         # output directory

# Check if the current dir is perf-stats otherwise cd into it
if [[ $PWD != *"perf-stats"* ]]; then
  cd perf-stats || exit 1
fi

# Check if binary exists
if [[ ! -f "$BIN" ]]; then
  echo "Error: Binary not found at $BIN. Please build the project first."
  exit 2
fi

# Create output directory if it doesn't exist
if [[ ! -d $PWD/$OUTPUT_DIR ]]; then
  mkdir -p "$OUTPUT_DIR"
fi

# Stats CSV
STATS_CSV="seq_stats.csv"
echo "size,run,time_compute_pure_s" > "$STATS_CSV"

for mb in "${SIZES[@]}"; do
  for run in $(seq 1 "$RUNS"); do
    out=$("$BIN" "$mb")
    t=$(echo "$out" | grep -m1 -o 'time_compute_pure=[0-9]\+\(\.[0-9]\+\)\?' | cut -d= -f2)
    if [[ -z "${t}" ]]; then
      echo "Error: could not parse time_compute_pure for size ${mb}MB (run ${run})."
      echo "Program output was:"
      echo "$out"
      exit 3
    fi
    echo "${mb}MB,${run},${t}" >> "$STATS_CSV"
  done
done

# Summary CSV
SUMMARY_CSV="seq_summary.csv"
echo "size,avg_time_compute_pure_s" > "$SUMMARY_CSV"
awk -F, '
  NR>1 {
    sum[$1] += $3
    cnt[$1]++
  }
  END {
    for (sz in sum) {
      printf "%s,%.6f\n", sz, sum[sz]/cnt[sz]
    }
  }
' "$STATS_CSV" | sort -t, -k1,1V >> "$SUMMARY_CSV"

# Report
echo "Sequential performance statistics:"
echo "- Detailed runs:   $STATS_CSV"
echo "- Averages:        $SUMMARY_CSV"

# Move output files to the output directory
mv "$STATS_CSV" "$OUTPUT_DIR"/
mv "$SUMMARY_CSV" "$OUTPUT_DIR"/
echo "Output files moved to $OUTPUT_DIR/"