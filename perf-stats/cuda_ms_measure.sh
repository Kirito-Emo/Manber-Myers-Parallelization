#!/usr/bin/env bash
set -euo pipefail

# Created by Emanuele (https://github.com/Kirito-Emo)

# CONFIGURATION
BIN=../cmake-build-release/hpc_cuda_ms
SIZES=(1 50 100 200 500)
STREAMS=(2 4 8)
RUNS=10
OUTPUT_DIR="cuda_ms_measurements"

[[ -f "$BIN" ]] || { echo "Missing binary: $BIN"; exit 1; }
[[ -d ../random_strings ]] || { echo "Missing ../random_strings dir"; exit 2; }
mkdir -p "$OUTPUT_DIR"

for mb in "${SIZES[@]}"; do
  [[ -f "../random_strings/string_${mb}MB.bin" ]] || { echo "Missing ../random_strings/string_${mb}MB.bin"; exit 3; }
done

RAW="${OUTPUT_DIR}/cuda_ms_stats.csv"
SUM="${OUTPUT_DIR}/cuda_ms_summary.csv"
echo "mode,size_mb,streams,run,time_io_s,time_alloc_host_dev_s,time_h2d_s,time_kernel_gpu_s,time_lcp_cpu_s,time_d2h_s,time_compute_pure_s,time_total_compute_s,time_transfers_comm_s,throughput_MBps,speedup,efficiency_pct,memory_overhead_ratio_pct" > "$RAW"

for mb in "${SIZES[@]}"; do
  for s in "${STREAMS[@]}"; do
    echo "[CUDA-MS] size=${mb}MB streams=${s}"
    for run in $(seq 1 "$RUNS"); do
      # Run and capture full output
      out="$("$BIN" "$mb" --streams "$s" 2>/dev/null)" || { echo "Run failed on ${mb}MB streams=$s"; exit 4; }

      # Extract times
      tio=$(echo "$out"   | grep -m1 '^time_io='               | cut -d= -f2 | awk '{print $1}')
      tall=$(echo "$out"  | grep -m1 '^time_alloc_host_dev='   | cut -d= -f2 | awk '{print $1}')
      th2d=$(echo "$out"  | grep -m1 '^time_h2d='              | cut -d= -f2 | awk '{print $1}')
      tkern=$(echo "$out" | grep -m1 '^time_kernel_gpu='       | cut -d= -f2 | awk '{print $1}')
      tlcp=$(echo "$out"  | grep -m1 '^time_lcp_cpu='          | cut -d= -f2 | awk '{print $1}')
      td2h=$(echo "$out"  | grep -m1 '^time_d2h='              | cut -d= -f2 | awk '{print $1}')
      tcmp=$(echo "$out"  | grep -m1 '^time_compute_pure='     | cut -d= -f2 | awk '{print $1}')
      ttot=$(echo "$out"  | grep -m1 '^time_total_compute='    | cut -d= -f2 | awk '{print $1}')
      tcom=$(echo "$out"  | grep -m1 '^time_transfers_comm='   | cut -d= -f2 | awk '{print $1}')
      thpt=$(echo "$out"  | grep -m1 '^throughput='            | cut -d= -f2 | awk '{print $1}')
      spd=$(echo "$out"   | grep -m1 '^speedup='               | cut -d= -f2 | awk '{print $1}')
      eff=$(echo "$out"   | grep -m1 '^efficiency='            | sed -E 's/efficiency=//; s/ %//')
      mo=$(echo "$out"    | grep -m1 '^memory_overhead_ratio=' | sed -E 's/memory_overhead_ratio=//; s/ %//')

      # Fallback → NaN
      [[ -z "$tio"      ]] && tio="nan"
      [[ -z "$tall"     ]] && tall="nan"
      [[ -z "$th2d"     ]] && th2d="nan"
      [[ -z "$tkern"    ]] && tkern="nan"
      [[ -z "$tlcp"     ]] && tlcp="nan"
      [[ -z "$td2h"     ]] && td2h="nan"
      [[ -z "$tcmp"     ]] && tcmp="nan"
      [[ -z "$ttot"     ]] && ttot="nan"
      [[ -z "$tcom"     ]] && tcom="nan"
      [[ -z "$thpt"     ]] && thpt="nan"
      [[ "$spd" =~ ^n/a ]] && spd="nan"
      [[ -z "$spd"      ]] && spd="nan"
      [[ -z "$eff"      ]] && eff="nan"
      [[ -z "$mo"       ]] && mo="nan"

      echo "cuda_ms,${mb},${s},${run},${tio},${tall},${th2d},${tkern},${tlcp},${td2h},${tcmp},${ttot},${tcom},${thpt},${spd},${eff},${mo}" >> "$RAW"
    done
  done
done

# CSV summary per size (avgs per size×streams)
echo "mode,size_mb,streams,avg_time_io_s,avg_time_alloc_host_dev_s,avg_time_h2d_s,avg_time_kernel_gpu_s,avg_time_lcp_cpu_s,avg_time_d2h_s,avg_time_compute_pure_s,avg_time_total_compute_s,avg_time_transfers_comm_s,avg_throughput_MBps,avg_speedup,avg_efficiency_pct,avg_memory_overhead_ratio_pct" > "$SUM"

awk -F, '
  NR>1 {
    key=$2","$3
    io[key]+=$5; alloc[key]+=$6; h2d[key]+=$7; kern[key]+=$8; lcp[key]+=$9
    d2h[key]+=$10; comp[key]+=$11; tot[key]+=$12; comm[key]+=$13; thr[key]+=$14
    spd[key]+=$15; eff[key]+=$16; mem[key]+=$17
    cnt[key]++
  }
  END {
    for (k in cnt) {
      split(k,a,",")
      printf "cuda_ms,%s,%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
        a[1], a[2],
        io[k]/cnt[k], alloc[k]/cnt[k], h2d[k]/cnt[k], kern[k]/cnt[k], lcp[k]/cnt[k],
        d2h[k]/cnt[k], comp[k]/cnt[k], tot[k]/cnt[k], comm[k]/cnt[k], thr[k]/cnt[k],
        spd[k]/cnt[k], eff[k]/cnt[k], mem[k]/cnt[k]
    }
  }
' "$RAW" | sort -t, -k2,2n -k3,3n >> "$SUM"

echo "Finished CUDA Multi-Stream benchmark"
echo " - Raw:     $RAW"
echo " - Summary: $SUM"