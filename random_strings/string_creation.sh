#!/usr/bin/env bash

# This script generates text files filled with random strings of specified sizes using /dev/urandom

# 1 MB
dd if=/dev/urandom of=string_1MB.txt bs=1M count=1

# 5 MB
dd if=/dev/urandom of=string_5MB.txt bs=1M count=5

# 10 MB
dd if=/dev/urandom of=string_10MB.txt bs=1M count=10

# 50 MB
dd if=/dev/urandom of=string_50MB.txt bs=1M count=50

# 100 MB
dd if=/dev/urandom of=string_100MB.txt bs=1M count=100

# 500 MB
dd if=/dev/urandom of=string_500MB.txt bs=1M count=500