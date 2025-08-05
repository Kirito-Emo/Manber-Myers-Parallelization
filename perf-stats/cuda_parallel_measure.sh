#!/usr/bin/env bash
set -euo pipefail

# Created by Emanuele (https://github.com/Kirito-Emo)

# CONFIGURATION
BIN=../cmake-build-debug-cuda-parallel/hpc_cuda_parallel
SIZES=(1 50 100 200)
RUNS=10
OUTPUT_DIR="cuda_parallel_measurements"

RAW="$OUTPUT_DIR/cuda_parallel_stats.csv"
SUMMARY="$OUTPUT_DIR/cuda_parallel_summary.csv"

# Check binary
if [[ ! -f "$BIN" ]]; then
  echo "Error: Binary not found → $BIN"
  exit 1
fi

# Check input files
mkdir -p "$OUTPUT_DIR"
if [[ ! -d "../random_strings" ]]; then
  echo "Missing ../random_strings directory"
  exit 2
fi

for mb in "${SIZES[@]}"; do
  if [[ ! -f "../random_strings/string_${mb}MB.bin" ]]; then
    echo "Missing input: string_${mb}MB.bin"
    exit 3
  fi
done

# CSV Header
echo "mode,size_mb,run,time_s" > "$RAW"

# Benchmark Loop
for mb in "${SIZES[@]}"; do
  echo "[CUDA PARALLEL] Running size: ${mb}MB..."
  for run in $(seq 1 $RUNS); do
    t=$("$BIN" "$mb" | grep -oP 'time_build=\K[0-9.]+')
    printf "cuda_parallel,%s,%s,%.6f\n" "$mb" "$run" "$t" >> "$RAW"
  done
done

# Summary
echo "mode,size_mb,avg_time_s" > "$SUMMARY"
awk -F, 'NR>1 { sum[$2]+=$4; count[$2]++ }
         END { for (s in sum) printf "cuda_parallel,%s,%.6f\n", s, sum[s]/count[s] }' \
    "$RAW" | sort -t, -k2n >> "$SUMMARY"

echo "Finished Parallel CUDA benchmark"
echo " - Raw:     $RAW"
echo " - Summary: $SUMMARY"