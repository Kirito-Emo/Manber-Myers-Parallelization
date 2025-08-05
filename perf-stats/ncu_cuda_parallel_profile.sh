#!/usr/bin/env bash
set -euo pipefail

# Created by Emanuele (https://github.com/Kirito-Emo)

# CONFIGURATION
BIN=../cmake-build-debug-cuda-parallel/hpc_cuda_parallel    # Binary to profile
MB=100                                                      # Input size in MB
INPUT="../random_strings/string_${MB}MB.bin"                # Input binary string
OUTPUT_DIR="cuda_parallel_profiles"
PROFILE_NAME="cuda_parallel_${MB}MB_profile"

# Sanity checks
[[ -f "$BIN" ]]   || { echo "Binary not found: $BIN"; exit 1; }
[[ -f "$INPUT" ]] || { echo "Input file missing: $INPUT"; exit 1; }
if [[ ! -d "$OUTPUT_DIR" ]]; then
    mkdir -p "$OUTPUT_DIR"
fi

# Run Nsight Compute
echo "Profiling ${MB}MB (parallel SA) with Nsight Compute..."
/usr/local/cuda/bin/ncu \
    --set full \
    --target-processes all \
    --launch-skip 0 \
    --launch-count 1 \
    --csv \
    --export "$OUTPUT_DIR/${PROFILE_NAME}.ncu-rep" \
    "$BIN" "$MB" | tee "$OUTPUT_DIR/${PROFILE_NAME}.txt"

# Output
echo -e "\nProfiling completed."
echo " - Text  : $OUTPUT_DIR/${PROFILE_NAME}.txt"
echo " - Report: $OUTPUT_DIR/${PROFILE_NAME}.ncu-rep"