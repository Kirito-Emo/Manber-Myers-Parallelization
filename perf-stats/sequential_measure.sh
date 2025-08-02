#!/usr/bin/env bash
set -euo pipefail

# CONFIGURATION
BIN=../cmake-build-debug/hpc        # path to binary
RUNS=10                             # how many repeats per size
SIZES=(1 5 10 50 100 500)           # sizes in MB

# Stats CSV
echo "size,run,build_time" > seq_stats.csv

for mb in "${SIZES[@]}"; do
  for run in $(seq 1 $RUNS); do
    # Invoke program and grab time_build
    t=$("$BIN" "$mb" \
          | grep -m1 -o 'time_build=[0-9]\+\(\.[0-9]\+\)\?' \
          | cut -d= -f2)
    # Append MB to size and “s” to build time
    echo "${mb}MB,${run},${t}s" >> seq_stats.csv
  done
done

# Summary CSV
echo "size,avg_build_time" > seq_summary.csv

awk -F, '
  NR>1 {
    val = substr($3, 1, length($3)-1)
    sum[$1] += val
    cnt[$1]++
  }
  END {
    for (sz in sum) {
      printf "%s,%.6fs\n", sz, sum[sz]/cnt[sz]
    }
  }
' seq_stats.csv \
  | sort -t, -k1,1V \
  >> seq_summary.csv

# Report
echo "Sequential performance statistics:"
echo "- Detailed runs in   seq_stats.csv"
echo "- Average times in   seq_summary.csv"