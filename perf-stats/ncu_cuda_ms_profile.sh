#!/usr/bin/env bash
set -euo pipefail

# Created by Emanuele (https://github.com/Kirito-Emo)

# CONFIGURATION
BIN=../cmake-build-release/hpc_cuda_ms          # Path to compiled CUDA binary
MB=${1:-100}                                    # Input file size in MB
STREAMS=${2:-4}                                 # Number of streams (default: 4)
INPUT="../random_strings/string_${MB}MB.bin"    # Input binary string
OUTPUT_DIR="cuda_ms_profiles"
PROFILE_NAME="cuda_ms_${MB}MB_${STREAMS}s"

# Sanity checks
[[ -f "$BIN" ]] || { echo "Binary not found: $BIN"; exit 1; }
[[ -f "$INPUT" ]] || { echo "Input file missing: $INPUT"; exit 2; }
if [[ ! -d "$OUTPUT_DIR" ]]; then
    mkdir -p "$OUTPUT_DIR"
fi

# Run Nsight Compute
echo "Profiling ${MB}MB (MultiStream) with Nsight Compute (streams=${STREAMS})"
/usr/local/cuda/bin/ncu \
    --set full \
    --target-processes all \
    --launch-skip 0 \
    --launch-count 1 \
    --csv \
    --export "${OUTPUT_DIR}/${PROFILE_NAME}.ncu-rep" \
    "$BIN" "$MB" --streams "$STREAMS" | tee "${OUTPUT_DIR}/${PROFILE_NAME}.txt"

# Output
echo -e "\nProfiling completed."
echo " - Text  : ${OUTPUT_DIR}/${PROFILE_NAME}.txt"
echo " - Report: ${OUTPUT_DIR}/${PROFILE_NAME}.ncu-rep"