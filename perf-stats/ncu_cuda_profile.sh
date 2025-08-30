#!/usr/bin/env bash
set -euo pipefail

# Created by Emanuele (https://github.com/Kirito-Emo)

# CONFIGURATION
BIN=../cmake-build-release/hpc_cuda             # Path to compiled CUDA binary
MB=100                                          # Input file size in MB
INPUT="../random_strings/string_${MB}MB.bin"    # Input binary string
OUTPUT_DIR="cuda_profiles"
PROFILE_NAME="cuda_${MB}MB_profile"

# Check if executable exists
if [[ ! -f "$BIN" ]]; then
  echo "Error: CUDA binary not found in $BIN. Please compile the project first."
  exit 1
fi

# Check if binary input file exists
if [[ ! -f "$INPUT" ]]; then
  echo "Error: input file not found in $INPUT. Please generate the input file first."
  exit 2
fi

# Create output directory if it doesn't exist
if [[ ! -d "$OUTPUT_DIR" ]]; then
  mkdir -p "$OUTPUT_DIR"
fi

# Run NVIDIA Nsight Compute (ncu) with full profiling
echo "Profiling ${MB}MB using NVIDIA Nsight Compute..."
/usr/local/cuda/bin/ncu \
    --set full \
    --target-processes all \
    --launch-skip 0 \
    --launch-count 1 \
    --csv \
    --export "${OUTPUT_DIR}/${PROFILE_NAME}.ncu-rep" \
    "$BIN" "$MB" | tee "${OUTPUT_DIR}/${PROFILE_NAME}.txt"

# Output
echo "Profiling completed. Output files:"
echo " - ${OUTPUT_DIR}/${PROFILE_NAME}.txt     (text report)"
echo " - ${OUTPUT_DIR}/${PROFILE_NAME}.ncu-rep (Nsight Compute GUI report)"