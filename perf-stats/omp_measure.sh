#!/usr/bin/env bash
set -euo pipefail

# Created by Emanuele (https://github.com/Kirito-Emo)

# CONFIGURATION
BIN=../cmake-build-release/hpc_omp      # path to OpenMP binary
RUNS=10                                 # how many repeats per size
SIZES=(1 50 100 200 500)                # sizes in MB
THREADS=(2 4 8)                         # OpenMP thread counts
OUTPUT_DIR=omp_measurements             # output directory root

# Check if current directory is perf-stats
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

# Loop over thread counts
for th in "${THREADS[@]}"; do
  OUTDIR="$OUTPUT_DIR/threads_${th}"
  mkdir -p "$OUTDIR"

  # CSV headers
  STATS_CSV="$OUTDIR/omp_stats_${th}.csv"
  SUMMARY_CSV="$OUTDIR/omp_summary_${th}.csv"
  echo "threads,size,run,time_io_s,time_alloc_s,time_sa_s,time_lcp_s,time_lrs_scan_s,time_compute_pure_s,time_total_compute_s,time_transfers_comm_s,throughput_MBps,speedup,efficiency_pct,memory_overhead_ratio_pct" > "$STATS_CSV"

  echo "Running OpenMP tests with $th threads..."

  for mb in "${SIZES[@]}"; do
    for run in $(seq 1 "$RUNS"); do
      export OMP_NUM_THREADS="$th"
      export OMP_DYNAMIC=false
      export OMP_PROC_BIND=spread
      export OMP_PLACES=cores
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
      speedup=$(echo "$out" | grep -m1 "speedup=" | cut -d= -f2 | cut -d' ' -f1)
      efficiency=$(echo "$out" | grep -m1 "efficiency=" | sed 's/efficiency=//; s/ %//' | awk '{print $1}')
      mem_overhead=$(echo "$out" | grep -m1 "memory_overhead_ratio=" | cut -d= -f2 | cut -d' ' -f1)

      echo "$th,${mb}MB,${run},${time_io},${time_alloc},${time_sa},${time_lcp},${time_lrs_scan},${time_compute_pure},${time_total_compute},${time_transfers_comm},${throughput},${speedup},${efficiency},${mem_overhead}" >> "$STATS_CSV"
    done
  done

  # Generate summary
  echo "threads,size,avg_time_io_s,avg_time_alloc_s,avg_time_sa_s,avg_time_lcp_s,avg_time_lrs_scan_s,avg_time_compute_pure_s,avg_time_total_compute_s,avg_time_transfers_comm_s,avg_throughput_MBps,avg_speedup,avg_efficiency_pct,avg_memory_overhead_ratio_pct" > "$SUMMARY_CSV"

  awk -F, '
    NR>1 {
      key = $1 "," $2
      io[key]+=$4; alloc[key]+=$5; sa[key]+=$6; lcp[key]+=$7; lrs[key]+=$8
      comp[key]+=$9; tot[key]+=$10; comm[key]+=$11; thr[key]+=$12
      spd[key]+=$13; eff[key]+=$14; mem[key]+=$15
      cnt[key]++
    }
    END {
      for (k in cnt) {
        printf "%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
          k, io[k]/cnt[k], alloc[k]/cnt[k], sa[k]/cnt[k], lcp[k]/cnt[k], lrs[k]/cnt[k],
          comp[k]/cnt[k], tot[k]/cnt[k], comm[k]/cnt[k], thr[k]/cnt[k],
          spd[k]/cnt[k], eff[k]/cnt[k], mem[k]/cnt[k]
      }
    }
  ' "$STATS_CSV" | sort -t, -k1n -k2,2V >> "$SUMMARY_CSV"

  echo "- Completed: OpenMP $th threads"
done

echo "All OpenMP runs completed. Results saved in: $OUTPUT_DIR/"