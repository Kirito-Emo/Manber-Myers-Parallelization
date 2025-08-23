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
RAW="$OUTDIR_DIR/cuda_stats.csv"
SUM="$OUTDIR_DIR/cuda_summary.csv"
echo "mode,size_mb,run,sa_ms,lcp_ms,time_ms" > "$RAW"

# Run benchmark
for mb in "${SIZES[@]}"; do
  echo "[CUDA] size=${mb}MB"
  for run in $(seq 1 $RUNS); do
    # Run and capture full output
    out="$("$BIN" "$mb")" || { echo "Run failed on ${mb}MB"; exit 4; }

    # Extract times (they are printed by your program)
    sa_ms=$(echo "$out"  | grep -oP '\[CUDA\] SA built in \K[0-9.]+(?= ms)' || echo 0)
    lcp_ms=$(echo "$out" | grep -oP '\[Kasai-(CPU|GPU)\] LCP built in \K[0-9.]+(?= ms)' || echo 0)

    # Fallbacks / checks
    [[ "$sa_ms" != "" ]] || sa_ms=0
    [[ "$lcp_ms" != "" ]] || lcp_ms=0

    time_ms=$(awk -v a="$sa_ms" -v b="$lcp_ms" 'BEGIN{printf("%.2f", a+b)}')
    printf "cuda,%s,%s,%.2f,%.2f,%.2f\n" "$mb" "$run" "$sa_ms" "$lcp_ms" "$time_ms" >> "$RAW"
  done
done

# Summary by size (avg)
echo "mode,size_mb,avg_sa_ms,avg_lcp_ms,avg_total_ms" > "$SUM"
awk -F, 'NR>1 {sa[$2]+=$4; lcp[$2]+=$5; tot[$2]+=$6; cnt[$2]++}
         END {for (s in sa) printf "cuda,%s,%.2f,%.2f,%.2f\n", s, sa[s]/cnt[s], lcp[s]/cnt[s], tot[s]/cnt[s] }' \
    "$RAW" | sort -t, -k2n >> "$SUM"

echo "Finished CUDA benchmarks. Output saved to:"
echo " - Raw : $RAW"
echo " - Mean: $SUM"