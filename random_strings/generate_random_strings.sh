#!/usr/bin/env bash

# Created by Emanuele (https://github.com/Kirito-Emo)
# This script generates binary files filled with pseudorandom strings of bytes using /dev/urandom so that
# STRING + AUXILIARY STRUCTURES ≈ TARGET SIZE in memory.

# Set the directory where the files will be saved
DIR="random_strings"
TARGET_SIZES=(1 50 100 200 500)   # Target sizes in MB (total memory budget = string + SA + LCP + aux)
OVERHEAD=21                       # Overhead factor (memory_total ≈ overhead * string_size)

mkdir -p "$DIR"

for TGT in "${TARGET_SIZES[@]}"; do
  # Compute string size in bytes
  STR_SIZE_BYTES=$(( (TGT * 1024 * 1024) / OVERHEAD ))
  filename="$DIR/string_${TGT}MB.bin"

  echo "Generating $filename with $STR_SIZE_BYTES bytes (target ${TGT}MB total)..."

  # Generate exactly STR_SIZE_BYTES random bytes
  dd if=/dev/urandom of="$filename" bs=1 count="$STR_SIZE_BYTES" status=progress
done

echo "All files generated in '$DIR'."