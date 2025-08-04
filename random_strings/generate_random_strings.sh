#!/usr/bin/env bash

# Created by Emanuele (https://github.com/Kirito-Emo)
# This script generates binary files filled with pseudorandom strings of bytes of specified sizes using /dev/urandom

# Set the directory where the files will be saved
DIR="random_strings"

# 1 MB
dd if=/dev/urandom of=$DIR/string_1MB.bin bs=1M count=1

# 50 MB
dd if=/dev/urandom of=$DIR/string_50MB.bin bs=1M count=50

# 100 MB
dd if=/dev/urandom of=$DIR/string_100MB.bin bs=1M count=100

# 200 MB
dd if=/dev/urandom of=$DIR/string_200MB.bin bs=1M count=200

# 500 MB
dd if=/dev/urandom of=$DIR/string_500MB.bin bs=1M count=500