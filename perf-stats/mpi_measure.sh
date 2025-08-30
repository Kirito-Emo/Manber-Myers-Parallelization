#!/usr/bin/env bash
set -euo pipefail

# Created by Emanuele (https://github.com/Kirito-Emo)

# CONFIGURATION
BIN=../cmake-build-release/hpc_mpi
SIZES=(1 50 100 200 500)
RUNS=10
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
if [ ! -d "$OUTPUT" ]; then
  mkdir -p "$OUTPUT"
fi

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
  local OUTPUT_DIR="${OUTPUT}/${NP}r/"
  local STATS_FILE="mpi_stats_${NP}.csv"
  local AGG_FILE="mpi_agg_${NP}.csv"

  echo "Running measurement with $NP ranks..."

  # per-rank stats
  echo "mode,size_mb,run,rank,time_sa_local_s,time_comm_s,time_io_s,time_alloc_s" > "$STATS_FILE"
  # aggregate (rank 0)
  echo "mode,size_mb,run,ranks,time_merge_s,time_lcp_s,time_lrs_scan_s,time_compute_pure_s,time_total_compute_s,time_transfers_comm_s,throughput_MBps,speedup,efficiency_pct,memory_overhead_ratio_pct" > "$AGG_FILE"

  for mb in "${SIZES[@]}"; do
    INPUT_FILE="../random_strings/string_${mb}MB.bin"
    if [[ ! -f "$INPUT_FILE" ]]; then
      echo "Skipping size ${mb}MB → file not found: $INPUT_FILE"
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
      done

      # ----- aggregate (rank 0) -----
      ranks=$(echo "$out"  | grep -m1 '^size='                 | awk '{for(i=1;i<=NF;i++){if($i ~ /^ranks=/){print substr($i,7)}}}')
      tmerge=$(echo "$out" | grep -m1 '^time_merge='           | cut -d= -f2 | awk '{print $1}')
      tlcp=$(echo "$out"   | grep -m1 '^time_lcp='             | cut -d= -f2 | awk '{print $1}')
      tlrs=$(echo "$out"   | grep -m1 '^time_lrs_scan='        | cut -d= -f2 | awk '{print $1}')
      tcomp=$(echo "$out"  | grep -m1 '^time_compute_pure='    | cut -d= -f2 | awk '{print $1}')
      ttot=$(echo "$out"   | grep -m1 '^time_total_compute='   | cut -d= -f2 | awk '{print $1}')
      tcomm=$(echo "$out"  | grep -m1 '^time_transfers_comm='  | cut -d= -f2 | awk '{print $1}')
      thpt=$(echo "$out"   | grep -m1 '^throughput='           | cut -d= -f2 | awk '{print $1}')
      spd=$(echo "$out"    | grep -m1 '^speedup='              | cut -d= -f2 | awk '{print $1}')
      eff=$(echo "$out"    | grep -m1 '^efficiency='           | sed -E 's/efficiency=//; s/ %//')
      mo=$(echo "$out"     | grep -m1 '^memory_overhead_ratio='| sed -E 's/memory_overhead_ratio=//; s/ %//')

      # fallback n/a
      [[ -z "$ranks" ]] && ranks="$NP"
      [[ -z "$tmerge" ]] && tmerge="nan"
      [[ -z "$tlcp"   ]] && tlcp="nan"
      [[ -z "$tlrs"   ]] && tlrs="nan"
      [[ -z "$tcomp"  ]] && tcomp="nan"
      [[ -z "$ttot"   ]] && ttot="nan"
      [[ -z "$tcomm"  ]] && tcomm="nan"
      [[ -z "$thpt"   ]] && thpt="nan"
      [[ "$spd" =~ ^n/a ]] && spd="nan"
      [[ -z "$spd" ]] && spd="nan"
      [[ -z "$eff" ]] && eff="nan"
      [[ -z "$mo"  ]] && mo="nan"

      echo "mpi,${mb},${run},${ranks},${tmerge},${tlcp},${tlrs},${tcomp},${ttot},${tcomm},${thpt},${spd},${eff},${mo}" >> "$AGG_FILE"
    done
  done

  # Sorting (by size, then run, then rank)
  sort -t, -k2,2n -k3,3n -k4,4n -o "$STATS_FILE" "$STATS_FILE"
  sort -t, -k2,2n -k3,3n -o "$AGG_FILE" "$AGG_FILE"

  # Move to output directory
  mv "$STATS_FILE" "$OUTPUT_DIR/"
  mv "$AGG_FILE" "$OUTPUT_DIR/"
}

# Run measurements for different ranks (cores)
for ranks in 2 4 8; do
  run_measurement "$ranks"
done