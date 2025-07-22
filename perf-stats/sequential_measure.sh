#!/usr/bin/env bash
set -euo pipefail

# This script measures the build time of the hpc program for each size of the input files (1, 5, 10, 50, 100, and 500 MB)
# Then, it computes the average build time for each file size across multiple runs (default is 10 runs)

# CONFIGURATION
HPC_BINARY=../cmake-build-debug/hpc
FILES=( ../random_strings/string_1MB.txt \
        ../random_strings/string_5MB.txt \
        ../random_strings/string_10MB.txt \
        ../random_strings/string_50MB.txt \
        ../random_strings/string_100MB.txt \
        ../random_strings/string_500MB.txt )
RUNS=10

# Header for the per‐run CSV file
echo "file,run,build_time_s" > seq_stats.csv

# Run hpc and append timings
for f in "${FILES[@]}"; do
  for ((i=1; i<=RUNS; i++)); do
    t=$(${HPC_BINARY} "$f" \
        | grep time_build \
        | awk '{print $2}')
    echo "$(basename "$f"),$i,$t" >> seq_stats.csv
  done
done

# Header for the averages CSV file
echo "file,avg_build_time_s" > seq_summary.csv

# Compute average per file
for f in "${FILES[@]}"; do
  base=$(basename "$f")
  avg=$(awk -F, -v file="$base" \
        '$1==file{sum+=$3;count++} \
         END { printf "%.6f", sum/count }' \
        seq_stats.csv)
  echo "$base,$avg" >> seq_summary.csv
done

echo "Done. Raw data in seq_stats.csv, averages in seq_summary.csv"