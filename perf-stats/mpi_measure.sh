#!/usr/bin/env bash
set -euo pipefail

# Created by Emanuele (https://github.com/Kirito-Emo)

# CONFIGURATION
BIN=../cmake-build-debug-mpi/hpc_mpi
SIZES=(1 5 10 50 100 500)
NP=4 # Number of processes (change as needed)

# RUNNING
echo "mode,size_mb,run,rank,time_s" > mpi_stats_4.csv

for mb in "${SIZES[@]}"; do
  for run in $(seq 1 10); do
    mpirun -np "$NP" "$BIN" "$mb" 2>/dev/null \
      | grep 'rank=' \
      | awk -v mode="mpi" -v mb="$mb" -v run="$run" '{
          for(i=1;i<=NF;i++) {
            if ($i ~ /^rank=/) rank=substr($i,6);
            if ($i ~ /^time_build=/) time=substr($i,12);
          }
          print mode "," mb "," run "," rank "," time
        }' \
      >> mpi_stats_4.csv
  done
done

# Generate summary (avg time per mode and size)
echo "mode,size_mb,rank,avg_time_s" > mpi_summary_4.csv
awk -F, 'NR>1 { key=$1","$2","$4; sum[key]+=$5; cnt[key]++ } END { for (k in sum) printf "%s,%.6f\n", k, sum[k]/cnt[k] }' mpi_stats_4.csv >> mpi_summary_4.csv