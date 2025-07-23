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
    t=$("${HPC_BINARY}" "$f" \
        | grep -m1 -o 'time_build=[0-9]\+\(\.[0-9]\+\)\?' \
        | cut -d= -f2)
    echo "$(basename "$f"),$i,$t" >> seq_stats.csv
  done
done

# Compute averages grouping by file, directly from seq_stats.csv
awk -F, '
  NR==1 { next }                # Skip header
  { sum[$1] += $3; cnt[$1]++ }  # Sums and counts
  END {
    # Print header
    print "file,avg_build_time_s"
    # For each file emit file,average
    for (f in sum) {
      printf "%s,%.6f\n", f, sum[f]/cnt[f]
    }
  }
' seq_stats.csv > seq_summary.csv

# Sort the summary file by file name and keep the header
{
  head -n1 seq_summary.csv
  grep '^string_1MB.txt,'   seq_summary.csv
  grep '^string_5MB.txt,'   seq_summary.csv
  grep '^string_10MB.txt,'  seq_summary.csv
  grep '^string_50MB.txt,'  seq_summary.csv
  grep '^string_100MB.txt,' seq_summary.csv
  grep '^string_500MB.txt,' seq_summary.csv
} > tmp && mv tmp seq_summary.csv

echo "Done. Raw data in seq_stats.csv, averages in seq_summary.csv"