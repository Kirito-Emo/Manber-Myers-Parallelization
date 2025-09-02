#!/usr/bin/env bash
set -euo pipefail

# Created by Emanuele (https://github.com/Kirito-Emo)

# CONFIGURATION
BIN=../cmake-build-release/hpc_mpi_omp
SIZES=(1 50 100 200 500)
OUTPUT_DIR="mpi_omp_measurements"
RUNS=10

# Pairs of (MPI ranks, OMP threads)
CONFIGS=(
  "2 4"
  "2 8"
  "4 2"
  "4 4"
  "8 2"
)

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

# Helper to parse a key=value token from a line set
kv() { # usage: echo "$line" | kv key
  awk -v k="$1" '{
    for(i=1;i<=NF;i++){
      if ($i ~ ("^"k"=")) { v=$i; sub("^"k"=","",v); print v }
    }
  }'
}

run_measurement() {
  local NP="$1"
  local NT="$2"
  local SUBDIR="${OUTPUT_DIR}/${NP}r_${NT}t"
  mkdir -p "$SUBDIR"

  local STATS_FILE="${SUBDIR}/mpi_omp_stats_${NP}r_${NT}t.csv"
  local AGG_FILE="${SUBDIR}/mpi_omp_agg_${NP}r_${NT}t.csv"
  local SUMMARY_FILE="${SUBDIR}/mpi_omp_summary_${NP}r_${NT}t.csv"

  echo "Running measurement: ${NP} MPI ranks × ${NT} OMP threads"

  # CSV per-rank
  echo "mode,size_mb,run,rank,threads,time_sa_local_s,time_comm_s,time_io_s,time_alloc_s" > "$STATS_FILE"
  # CSV aggregate (rank 0)
  echo "mode,size_mb,run,ranks,threads_per_rank,time_io_max_s,time_alloc_max_s,time_sa_max_s,time_merge_s,time_lcp_s,time_lrs_scan_s,time_compute_pure_s,time_total_compute_s,time_transfers_comm_s,throughput_MBps,speedup,efficiency_pct,memory_overhead_ratio_pct" > "$AGG_FILE"

  for mb in "${SIZES[@]}"; do
    INPUT_FILE="../random_strings/string_${mb}MB.bin"
    [[ ! -f "$INPUT_FILE" ]] && { echo "Skipping size ${mb}MB → file not found"; continue; }

    for run in $(seq 1 "$RUNS"); do
      export OMP_NUM_THREADS="$NT"
      export OMP_DYNAMIC=false
      export OMP_PROC_BIND=spread
      export OMP_PLACES=cores
      # Get full output
      out="$(mpirun -np "$NP" "$BIN" "$mb" 2>/dev/null)"

      # ----- per-rank lines -----
      while IFS= read -r line; do
        r=$(echo "$line"   | kv "rank")
        t=$(echo "$line"   | kv "threads")
        tsa=$(echo "$line" | kv "time_sa_local")
        tcm=$(echo "$line" | kv "time_comm")
        tio=$(echo "$line" | kv "time_io")
        tal=$(echo "$line" | kv "time_alloc")
        [[ -z "$t" ]] && t="$NT"
        echo "mpi+omp,${mb},${run},${r},${t},${tsa},${tcm},${tio},${tal}" >> "$STATS_FILE"
      done <<< "$(printf '%s\n' "$out" | grep '^rank=')"

      # ----- aggregate (rank 0) -----
      ranks=$(echo "$out" | grep -m1 'size=' | awk '{for(i=1;i<=NF;i++){if($i ~ /^ranks=/){print substr($i,7)}}}')
      [[ -z "$ranks" ]] && ranks="$NP"
      threads_per_rank=$(printf '%s\n' "$out" | grep '^rank=' | head -n1 | kv "threads")
      [[ -z "$threads_per_rank" ]] && threads_per_rank="$NT"

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

      # fallback n/a -> NaN
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

      echo "mpi+omp,${mb},${run},${ranks},${threads_per_rank},${tio},${tall},${tsa},${tmerge},${tlcp},${tlrs},${tcomp},${ttot},${tcomm},${thpt},${spd},${eff},${mo}" >> "$AGG_FILE"
    done
  done

  # Summary CSV
  echo "mode,size_mb,ranks,threads_per_rank,avg_time_io_max_s,avg_time_alloc_max_s,avg_time_sa_max_s,avg_time_merge_s,avg_time_lcp_s,avg_time_lrs_scan_s,avg_time_compute_pure_s,avg_time_total_compute_s,avg_time_transfers_comm_s,avg_throughput_MBps,avg_speedup,avg_efficiency_pct,avg_memory_overhead_ratio_pct" > "$SUMMARY_FILE"

  awk -F, '
    NR>1 {
      key = $1 "," $2 "," $4 "," $5
      tio[key]+=$6; tall[key]+=$7; tsa[key]+=$8; merge[key]+=$9; lcp[key]+=$10
      lrs[key]+=$11; comp[key]+=$12; tot[key]+=$13; comm[key]+=$14; thr[key]+=$15
      spd[key]+=$16; eff[key]+=$17; mem[key]+=$18
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
  ' "$AGG_FILE" | sort -t, -k2,2n -k3,3n -k4,4n >> "$SUMMARY_FILE"

  # Sorting
  sort -t, -k2,2n -k3,3n -k4,4n -k5,5n -o "$STATS_FILE" "$STATS_FILE"
  sort -t, -k2,2n -k3,3n -k4,4n -o "$AGG_FILE" "$AGG_FILE"
}

# Run measurements for different (ranks, threads) configurations
for cfg in "${CONFIGS[@]}"; do
  read -r ranks threads <<< "$cfg"
  run_measurement "$ranks" "$threads"
done

echo "MPI+OpenMP benchmarks completed → $OUTPUT_DIR/"