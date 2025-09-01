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
mkdir -p "$OUTPUT_DIR"

# Stats CSV
STATS_CSV="$OUTPUT_DIR/seq_stats.csv"
echo "size,run,time_io_s,time_alloc_s,time_sa_s,time_lcp_s,time_lrs_scan_s,time_compute_pure_s,time_total_compute_s,time_transfers_comm_s,throughput_MBps,speedup,efficiency_pct,memory_overhead_ratio_pct" > "$STATS_CSV"

for mb in "${SIZES[@]}"; do
  for run in $(seq 1 "$RUNS"); do
    out=$("$BIN" "$mb")

    # Parsing
    time_io=$(echo "$out" | grep -m1 "time_io=" | cut -d= -f2 | cut -d' ' -f1)
    time_alloc=$(echo "$out" | grep -m1 "time_alloc=" | cut -d= -f2 | cut -d' ' -f1)
    time_sa=$(echo "$out" | grep -m1 "time_sa=" | cut -d= -f2 | cut -d' ' -f1)
    time_lcp=$(echo "$out" | grep -m1 "time_lcp=" | cut -d= -f2 | cut -d' ' -f1)
    time_lrs_scan=$(echo "$out" | grep -m1 "time_lrs_scan=" | cut -d= -f2 | cut -d' ' -f1)
    time_compute_pure=$(echo "$out" | grep -m1 "time_compute_pure=" | cut -d= -f2 | cut -d' ' -f1)
    time_total_compute=$(echo "$out" | grep -m1 "time_total_compute=" | cut -d= -f2 | cut -d' ' -f1)
    time_transfers_comm=$(echo "$out" | grep -m1 "time_transfers_comm=" | cut -d= -f2 | cut -d' ' -f1)
    throughput=$(echo "$out" | grep -m1 "throughput=" | cut -d= -f2 | cut -d' ' -f1)
    speedup=$(echo "$out" | grep -m1 "speedup=" | cut -d= -f2)
    efficiency=$(echo "$out" | grep -m1 "efficiency=" | cut -d= -f2 | cut -d' ' -f1)
    mem_overhead=$(echo "$out" | grep -m1 "memory_overhead_ratio=" | cut -d= -f2 | cut -d' ' -f1)

    echo "${mb}MB,${run},${time_io},${time_alloc},${time_sa},${time_lcp},${time_lrs_scan},${time_compute_pure},${time_total_compute},${time_transfers_comm},${throughput},${speedup},${efficiency},${mem_overhead}" >> "$STATS_CSV"
  done
done

# Summary CSV
SUMMARY_CSV="$OUTPUT_DIR/seq_summary.csv"
echo "size,avg_time_io_s,avg_time_alloc_s,avg_time_sa_s,avg_time_lcp_s,avg_time_lrs_scan_s,avg_time_compute_pure_s,avg_time_total_compute_s,avg_time_transfers_comm_s,avg_throughput_MBps,avg_speedup,avg_efficiency_pct,avg_memory_overhead_ratio_pct" > "$SUMMARY_CSV"

awk -F, '
  NR>1 {
    sz=$1
    io[sz]+=$3; alloc[sz]+=$4; sa[sz]+=$5; lcp[sz]+=$6; lrs[sz]+=$7
    comp[sz]+=$8; tot[sz]+=$9; comm[sz]+=$10; thr[sz]+=$11
    spd[sz]+=$12; eff[sz]+=$13; mem[sz]+=$14
    cnt[sz]++
  }
  END {
    for (sz in cnt) {
      printf "%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
        sz, io[sz]/cnt[sz], alloc[sz]/cnt[sz], sa[sz]/cnt[sz], lcp[sz]/cnt[sz],
        lrs[sz]/cnt[sz], comp[sz]/cnt[sz], tot[sz]/cnt[sz], comm[sz]/cnt[sz],
        thr[sz]/cnt[sz], spd[sz]/cnt[sz], eff[sz]/cnt[sz], mem[sz]/cnt[sz]
    }
  }
' "$STATS_CSV" | sort -t, -k1,1V >> "$SUMMARY_CSV"

# Report
echo "Sequential performance statistics:"
echo "- Detailed runs:   $STATS_CSV"
echo "- Averages:        $SUMMARY_CSV"