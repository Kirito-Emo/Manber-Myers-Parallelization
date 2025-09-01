#!/usr/bin/env bash
set -euo pipefail

# Created by Emanuele (https://github.com/Kirito-Emo)

# CONFIGURATION
BIN=../cmake-build-release/hpc_mpi
SIZES=(1 50 100 200 500)
RUNS=10
RANKS=(2 4 8)
OUTPUT="mpi_measurements"

# Check if the current dir is perf-stats otherwise cd into it
if [[ $PWD != *"perf-stats"* ]]; then
  cd perf-stats || exit 1
fi

if [[ ! -f "$BIN" ]]; then
  echo "Error: Binary not found at $BIN. Please build the project first."
  exit 2
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT"

# Helper to parse a key=value token from a line set
kv() {
  awk -v k="$1" '{
    for(i=1;i<=NF;i++){
      if ($i ~ ("^"k"=")) { v=$i; sub("^"k"=","",v); print v }
    }
  }'
}

run_measurement() {
  local NP="$1"
  local OUTDIR="${OUTPUT}/${NP}r/"
  mkdir -p "$OUTDIR"

  local STATS_FILE="$OUTDIR/mpi_stats_${NP}.csv"
  local AGG_FILE="$OUTDIR/mpi_agg_${NP}.csv"
  local SUMMARY_FILE="$OUTDIR/mpi_summary_${NP}.csv"

  echo "Running measurement with $NP ranks..."

  # per-rank stats
  echo "mode,size_mb,run,rank,time_sa_local_s,time_comm_s,time_io_s,time_alloc_s" > "$STATS_FILE"
  # aggregate (rank 0)
  echo "mode,size_mb,run,ranks,time_io_max_s,time_alloc_max_s,time_sa_max_s,time_merge_s,time_lcp_s,time_lrs_scan_s,time_compute_pure_s,time_total_compute_s,time_transfers_comm_s,throughput_MBps,speedup,efficiency_pct,memory_overhead_ratio_pct" > "$AGG_FILE"

  for mb in "${SIZES[@]}"; do
    INPUT_FILE="../random_strings/string_${mb}MB.bin"
    if [[ ! -f "$INPUT_FILE" ]]; then
      echo "Skipping size ${mb}MB → file not found"
      continue
    fi

    for run in $(seq 1 "$RUNS"); do
      # Get full output
      out="$(mpirun -np "$NP" "$BIN" "$mb" 2>/dev/null)"

      # ----- per-rank lines -----
      echo "$out" | grep '^rank=' | while read -r line; do
        r=$(echo "$line" | kv "rank")
        tsa=$(echo "$line" | kv "time_sa_local")
        tcm=$(echo "$line" | kv "time_comm")
        tio=$(echo "$line" | kv "time_io")
        tal=$(echo "$line" | kv "time_alloc")
        echo "mpi,${mb},${run},${r},${tsa},${tcm},${tio},${tal}" >> "$STATS_FILE"
      done || true

      # ----- aggregate (rank 0) -----
      tio=$(echo "$out"   | grep -m1 "time_io_max="           | cut -d= -f2 | awk '{print $1}')
      tall=$(echo "$out"  | grep -m1 "time_alloc_max="        | cut -d= -f2 | awk '{print $1}')
      tsa=$(echo "$out"   | grep -m1 "time_sa_max="           | cut -d= -f2 | awk '{print $1}')
      tmerge=$(echo "$out"| grep -m1 "time_merge="            | cut -d= -f2 | awk '{print $1}')
      tlcp=$(echo "$out"  | grep -m1 "time_lcp="              | cut -d= -f2 | awk '{print $1}')
      tlrs=$(echo "$out"  | grep -m1 "time_lrs_scan="         | cut -d= -f2 | awk '{print $1}')
      tcomp=$(echo "$out" | grep -m1 "time_compute_pure="     | cut -d= -f2 | awk '{print $1}')
      ttot=$(echo "$out"  | grep -m1 "time_total_compute="    | cut -d= -f2 | awk '{print $1}')
      tcomm=$(echo "$out" | grep -m1 "time_transfers_comm="   | cut -d= -f2 | awk '{print $1}')
      thpt=$(echo "$out"  | grep -m1 "throughput="            | cut -d= -f2 | awk '{print $1}')
      spd=$(echo "$out"   | grep -m1 "speedup="               | cut -d= -f2 | awk '{print $1}')
      eff=$(echo "$out"   | grep -m1 "efficiency="            | sed -E 's/efficiency=//; s/ %//')
      mo=$(echo "$out"    | grep -m1 "memory_overhead_ratio=" | sed -E 's/memory_overhead_ratio=//; s/ %//')

      [[ -z "$tio"      ]] && tio="nan"
      [[ -z "$tall"     ]] && tall="nan"
      [[ -z "$tsa"      ]] && tsa="nan"
      [[ -z "$tmerge"   ]] && tmerge="nan"
      [[ -z "$tlcp"     ]] && tlcp="nan"
      [[ -z "$tlrs"     ]] && tlrs="nan"
      [[ -z "$tcomp"    ]] && tcomp="nan"
      [[ -z "$ttot"     ]] && ttot="nan"
      [[ -z "$tcomm"    ]] && tcomm="nan"
      [[ -z "$thpt"     ]] && thpt="nan"
      [[ "$spd" =~ ^n/a ]] && spd="nan"
      [[ -z "$spd"      ]] && spd="nan"
      [[ -z "$eff"      ]] && eff="nan"
      [[ -z "$mo"       ]] && mo="nan"

      echo "mpi,${mb},${run},${NP},${tio},${tall},${tsa},${tmerge},${tlcp},${tlrs},${tcomp},${ttot},${tcomm},${thpt},${spd},${eff},${mo}" >> "$AGG_FILE"
    done
  done

  # Summary (average over runs)
  echo "mode,size_mb,ranks,avg_time_io_max_s,avg_time_alloc_max_s,avg_time_sa_max_s,avg_time_merge_s,avg_time_lcp_s,avg_time_lrs_scan_s,avg_time_compute_pure_s,avg_time_total_compute_s,avg_time_transfers_comm_s,avg_throughput_MBps,avg_speedup,avg_efficiency_pct,avg_memory_overhead_ratio_pct" > "$SUMMARY_FILE"

  awk -F, '
    NR>1 {
      key = $1 "," $2 "," $4
      tio[key]+=$5; tall[key]+=$6; tsa[key]+=$7; merge[key]+=$8; lcp[key]+=$9
      lrs[key]+=$10; comp[key]+=$11; tot[key]+=$12; comm[key]+=$13; thr[key]+=$14
      spd[key]+=$15; eff[key]+=$16; mem[key]+=$17
      cnt[key]++
    }
    END {
      for (k in cnt) {
        printf "%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
          k, tio[k]/cnt[k], tall[k]/cnt[k], tsa[k]/cnt[k], merge[k]/cnt[k], lcp[k]/cnt[k],
          lrs[k]/cnt[k], comp[k]/cnt[k], tot[k]/cnt[k], comm[k]/cnt[k], thr[k]/cnt[k],
          spd[k]/cnt[k], eff[k]/cnt[k], mem[k]/cnt[k]
      }
    }
  ' "$AGG_FILE" | sort -t, -k2,2n -k3,3n >> "$SUMMARY_FILE"

  # Sort per-rank stats
  sort -t, -k2,2n -k3,3n -k4,4n -o "$STATS_FILE" "$STATS_FILE"
  # Sort aggregate stats
  sort -t, -k2,2n -k3,3n -o "$AGG_FILE" "$AGG_FILE"
}

# Run measurements for different ranks (cores)
for ranks in "${RANKS[@]}"; do
  run_measurement "$ranks"
done

echo "MPI measurements completed. Results in $OUTPUT/"