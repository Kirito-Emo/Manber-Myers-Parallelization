#!/usr/bin/env bash
set -euo pipefail

# Created by Emanuele (https://github.com/Kirito-Emo)

# CONFIGURATION
BIN=../cmake-build-release/hpc_cuda
SIZES=(1 50 100 200 500)
RUNS=10
OUTPUT_DIR="cuda_measurements"

# Check if the binary exists
if [[ ! -f "$BIN" ]]; then
  echo "Error: Binary $BIN not found. Please build the project first."
  exit 1
fi

# Check if the output directory exists and create it if not
mkdir -p "$OUTPUT_DIR"

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

RAW="${OUTPUT_DIR}/cuda_stats.csv"
SUM="${OUTPUT_DIR}/cuda_summary.csv"
echo "mode,size_mb,run,time_io_s,time_alloc_host_dev_s,time_h2d_s,time_kernel_gpu_s,time_lcp_cpu_s,time_d2h_s,time_compute_pure_s,time_total_compute_s,time_transfers_comm_s,throughput_MBps,speedup,efficiency_pct,memory_overhead_ratio_pct" > "$RAW"

for mb in "${SIZES[@]}"; do
  echo "[CUDA] size=${mb}MB"
  for run in $(seq 1 "$RUNS"); do
    # Run and capture full output
    out="$("$BIN" "$mb" 2>/dev/null)" || { echo "Run failed on ${mb}MB"; exit 4; }

    # Extract times
    tio=$(echo "$out"     | grep -m1 '^time_io='               | cut -d= -f2 | awk '{print $1}')
    thalloc=$(echo "$out" | grep -m1 '^time_alloc_host_dev='   | cut -d= -f2 | awk '{print $1}')
    th2d=$(echo "$out"    | grep -m1 '^time_h2d='              | cut -d= -f2 | awk '{print $1}')
    tkern=$(echo "$out"   | grep -m1 '^time_kernel_gpu='       | cut -d= -f2 | awk '{print $1}')
    tlcp=$(echo "$out"    | grep -m1 '^time_lcp_cpu='          | cut -d= -f2 | awk '{print $1}')
    td2h=$(echo "$out"    | grep -m1 '^time_d2h='              | cut -d= -f2 | awk '{print $1}')
    tcomp=$(echo "$out"   | grep -m1 '^time_compute_pure='     | cut -d= -f2 | awk '{print $1}')
    ttot=$(echo "$out"    | grep -m1 '^time_total_compute='    | cut -d= -f2 | awk '{print $1}')
    tcomm=$(echo "$out"   | grep -m1 '^time_transfers_comm='   | cut -d= -f2 | awk '{print $1}')
    thpt=$(echo "$out"    | grep -m1 '^throughput='            | cut -d= -f2 | awk '{print $1}')
    spd=$(echo "$out"     | grep -m1 '^speedup='               | cut -d= -f2 | awk '{print $1}')
    eff=$(echo "$out"     | grep -m1 '^efficiency='            | sed -E 's/efficiency=//; s/ %//')
    mo=$(echo "$out"      | grep -m1 '^memory_overhead_ratio=' | sed -E 's/memory_overhead_ratio=//; s/ %//')

    # Fallback n/a/empty -> NaN
    [[ -z "$tio"      ]] && tio="nan"
    [[ -z "$thalloc"  ]] && thalloc="nan"
    [[ -z "$th2d"     ]] && th2d="nan"
    [[ -z "$tkern"    ]] && tkern="nan"
    [[ -z "$tlcp"     ]] && tlcp="nan"
    [[ -z "$td2h"     ]] && td2h="nan"
    [[ -z "$tcomp"    ]] && tcomp="nan"
    [[ -z "$ttot"     ]] && ttot="nan"
    [[ -z "$tcomm"    ]] && tcomm="nan"
    [[ -z "$thpt"     ]] && thpt="nan"
    [[ "$spd" =~ ^n/a ]] && spd="nan"
    [[ -z "$spd"      ]] && spd="nan"
    [[ -z "$eff"      ]] && eff="nan"
    [[ -z "$mo"       ]] && mo="nan"

    echo "cuda,${mb},${run},${tio},${thalloc},${th2d},${tkern},${tlcp},${td2h},${tcomp},${ttot},${tcomm},${thpt},${spd},${eff},${mo}" >> "$RAW"
  done
done

# CSV summary per size (averages; speedup/eff can be NaN if baseline doesn't exist)
echo "mode,size_mb,avg_time_io_s,avg_time_alloc_host_dev_s,avg_time_h2d_s,avg_time_kernel_gpu_s,avg_time_lcp_cpu_s,avg_time_d2h_s,avg_time_compute_pure_s,avg_time_total_compute_s,avg_time_transfers_comm_s,avg_throughput_MBps,avg_speedup,avg_efficiency_pct,avg_memory_overhead_ratio_pct" > "$SUM"

awk -F, '
  NR>1 {
    key=$2
    io[key]+=$4; alloc[key]+=$5; h2d[key]+=$6; kern[key]+=$7; lcp[key]+=$8
    d2h[key]+=$9; comp[key]+=$10; tot[key]+=$11; comm[key]+=$12; thr[key]+=$13
    spd[key]+=$14; eff[key]+=$15; mem[key]+=$16
    cnt[key]++
  }
  END {
    for (k in cnt) {
      printf "cuda,%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
        k, io[k]/cnt[k], alloc[k]/cnt[k], h2d[k]/cnt[k], kern[k]/cnt[k], lcp[k]/cnt[k],
        d2h[k]/cnt[k], comp[k]/cnt[k], tot[k]/cnt[k], comm[k]/cnt[k], thr[k]/cnt[k],
        spd[k]/cnt[k], eff[k]/cnt[k], mem[k]/cnt[k]
    }
  }
' "$RAW" | sort -t, -k2,2n >> "$SUM"

echo "Finished CUDA benchmarks. Output saved to:"
echo " - Raw : $RAW"
echo " - Mean: $SUM"