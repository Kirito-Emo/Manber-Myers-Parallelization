#!/usr/bin/env bash
set -euo pipefail

# Created by Emanuele (https://github.com/Kirito-Emo)

# CONFIGURATION
BIN=../cmake-build-debug-cuda-parallel/hpc_cuda_ms
SIZES=(1 50 100 200)
STREAMS=(2 4 8)
RUNS=10
OUTPUT_DIR="cuda_ms_measurements"

if [[ ! -d "$OUTPUT_DIR" ]]; then
  mkdir -p "$OUTPUT_DIR"
fi

[[ -f "$BIN" ]] || { echo "Missing binary: $BIN"; exit 1; }
[[ -d ../random_strings ]] || { echo "Missing ../random_strings dir"; exit 2; }

for mb in "${SIZES[@]}"; do
  [[ -f "../random_strings/string_${mb}MB.bin" ]] || { echo "Missing ../random_strings/string_${mb}MB.bin"; exit 3; }
done

RAW="$OUTDIR/cuda_ms_stats.csv"
SUM="$OUTDIR/cuda_ms_summary.csv"
echo "mode,size_mb,streams,run,sa_ms,lcp_ms,total_ms" > "$RAW"

for mb in "${SIZES[@]}"; do
  for s in "${STREAMS[@]}"; do
    echo "[CUDA-MS] size=${mb}MB streams=${s}"
    for run in $(seq 1 $RUNS); do
      out="$("$BIN" "$mb" --streams "$s")" || { echo "Run failed on ${mb}MB s=$s"; exit 4; }
      sa_ms=$(echo "$out"  | grep -oP '\[CUDA-MS\] SA built in \K[0-9.]+(?= ms)' || echo 0)
      lcp_ms=$(echo "$out" | grep -oP '\[Kasai-(CPU|GPU)\] LCP built in \K[0-9.]+(?= ms)' || echo 0)
      total_ms=$(awk -v a="$sa_ms" -v b="$lcp_ms" 'BEGIN{printf("%.2f", a+b)}')
      printf "cuda_ms,%s,%s,%s,%.2f,%.2f,%.2f\n" "$mb" "$s" "$run" "$sa_ms" "$lcp_ms" "$total_ms" >> "$RAW"
    done
  done
done

echo "mode,size_mb,streams,avg_sa_ms,avg_lcp_ms,avg_total_ms" > "$SUM"
awk -F, 'NR>1 {key=$2","$3; sa[key]+=$5; lcp[key]+=$6; tot[key]+=$7; cnt[key]++}
         END {for (k in sa) {split(k,a,","); printf "cuda_ms,%s,%s,%.2f,%.2f,%.2f\n", a[1], a[2], sa[k]/cnt[k], lcp[k]/cnt[k], tot[k]/cnt[k]} }' \
    "$RAW" | sort -t, -k2,2n -k3,3n >> "$SUM"

echo "Finished Parallel CUDA benchmark"
echo " - Raw:     $RAW"
echo " - Summary: $SUMMARY"