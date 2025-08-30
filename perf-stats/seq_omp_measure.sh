#!/usr/bin/env bash
set -euo pipefail

# Created by Emanuele (https://github.com/Kirito-Emo)

# CONFIGURATION
BIN=../cmake-build-release/hpc_omp      # path to OpenMP binary
RUNS=10                                 # how many repeats per size
SIZES=(1 50 100 200 500)                # sizes in MB
THREADS=(4 8)                           # OpenMP thread counts
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
if [[ ! -d "$OUTPUT_DIR" ]]; then
  mkdir -p "$OUTPUT_DIR"
fi

# Loop over thread counts
for th in "${THREADS[@]}"; do
  OUTDIR="$OUTPUT_DIR/threads_${th}"
  mkdir -p "$OUTDIR"

  # CSV headers
  STATS_CSV="$OUTDIR/omp_stats_${th}.csv"
  SUMMARY_CSV="$OUTDIR/omp_summary_${th}.csv"
  echo "threads,size,run,time_compute_pure_s,speedup,efficiency_pct" > "$STATS_CSV"

  echo "Running OpenMP tests with $th threads..."

  for mb in "${SIZES[@]}"; do
      for run in $(seq 1 "$RUNS"); do
        export OMP_NUM_THREADS="$th"
        export OMP_DYNAMIC=false
        export OMP_PROC_BIND=spread
        export OMP_PLACES=cores
        out=$("$BIN" "$mb")

        t=$(echo "$out" | grep -m1 -o 'time_compute_pure=[0-9]\+\(\.[0-9]\+\)\?' | cut -d= -f2 || true)
        s=$(echo "$out" | grep -m1 -o 'speedup=[0-9]\+\(\.[0-9]\+\)\?'           | cut -d= -f2 || true)
        e=$(echo "$out" | grep -m1 -o 'efficiency=[0-9]\+\(\.[0-9]\+\)\? %'      | sed 's/efficiency=//; s/ %//' || true)

        # Fallbacks in case parsing fails
        [[ -z "$t" ]] && t="nan"
        [[ -z "$s" ]] && s="nan"
        [[ -z "$e" ]] && e="nan"

        echo "$th,${mb}MB,${run},${t},${s},${e}" >> "$STATS_CSV"
      done
    done

  # Generate summary
  echo "threads,size,avg_time_compute_pure_s,avg_speedup,avg_efficiency_pct" > "$SUMMARY_CSV"
    awk -F, '
      NR>1 && $4 != "nan" {
        key = $1 "," $2
        sum_t[key] += $4; cnt_t[key]++
      }
      NR>1 && $5 != "nan" {
        sum_s[key] += $5; cnt_s[key]++
      }
      NR>1 && $6 != "nan" {
        sum_e[key] += $6; cnt_e[key]++
      }
      END {
        for (k in sum_t) {
          at = (cnt_t[k] ? sum_t[k]/cnt_t[k] : 0)
          as = (cnt_s[k] ? sum_s[k]/cnt_s[k] : 0)
          ae = (cnt_e[k] ? sum_e[k]/cnt_e[k] : 0)
          printf "%s,%.6f,%.6f,%.2f\n", k, at, as, ae
        }
      }
    ' "$STATS_CSV" | sort -t, -k1n -k2,2V >> "$SUMMARY_CSV"

  echo "- Completed: OpenMP $th threads"
done

echo "All OpenMP runs completed. Results saved in: $OUTPUT_DIR/"