#!/usr/bin/env bash
set -euo pipefail

# Created by Emanuele (https://github.com/Kirito-Emo)

# CONFIGURATION
BIN=../cmake-build-debug-cuda/hpc_cuda
SIZES=(1 50 100 200)
RUNS=10
OUTPUT_DIR="cuda_measurements"

# Check if the binary exists
if [[ ! -f "$BIN" ]]; then
  echo "Error: Binary $BIN not found. Please build the project first."
  exit 1
fi

# Check if the output directory exists and create it if not
if [[ ! -d "$OUTPUT_DIR" ]]; then
  mkdir -p "$OUTPUT_DIR"
fi

# Check if the random strings directory exists
if [[ ! -d "../random_strings" ]]; then
  echo "Error: Directory ../random_strings not found. Please generate random strings first."
  exit 2
fi

# Check if the random strings files exist
for mb in "${SIZES[@]}"; do
  INPUT="../random_strings/string_${mb}MB.bin"
  if [[ ! -f "$INPUT" ]]; then
    echo "Error: Input file $INPUT not found. Please generate random strings first."
    exit 3
  fi
done

# CSV output
CSV_RAW="$OUTPUT_DIR/cuda_stats.csv"
CSV_SUM="$OUTPUT_DIR/cuda_summary.csv"
echo "mode,size_mb,run,time_s" > "$CSV_RAW"

# Run benchmark
for mb in "${SIZES[@]}"; do
  INPUT="../random_strings/string_${mb}MB.bin"

  if [[ ! -f "$INPUT" ]]; then
    echo "Warning: skipping ${mb}MB → file not found: $INPUT"
    continue
  fi

  for run in $(seq 1 $RUNS); do
    echo "Running: $mb MB (run $run)"
    time_build=$("$BIN" "$mb" | grep -oP 'time_build=\K[0-9.]+')
    printf "cuda,%s,%s,%.6f\n" "$mb" "$run" "$time_build" >> "$CSV_RAW"
  done
done

# Compute summary with average time
echo "mode,size_mb,avg_time_s" > "$CSV_SUM"
awk -F, 'NR>1 { sum[$2]+=$4; count[$2]++ }
         END { for (s in sum) printf "cuda,%s,%.6f\n", s, sum[s]/count[s] }' \
    "$CSV_RAW" | sort -t, -k2n >> "$CSV_SUM"

echo "Finished CUDA benchmarks. Output saved to:"
echo " - $CSV_RAW"
echo " - $CSV_SUM"