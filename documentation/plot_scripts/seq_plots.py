#!/usr/bin/env python3
import re

import matplotlib.pyplot as plt
import pandas as pd

# Load summary CSV
summary_file = "../../perf-stats/seq_measurements/seq_summary.csv"
df = pd.read_csv(summary_file)

# Parse size (e.g. "100MB" -> 100)
df["size_MB"] = df["size"].str.replace("MB", "").astype(int)

# Plot 1: Execution time
plt.figure(figsize=(7, 5))
plt.plot(df["size_MB"], df["avg_time_compute_pure_s"], marker="o", color="tab:blue", label="Average time")
plt.xlabel("Problem size [MB] (total budget)")
plt.ylabel("Average time [s]")
plt.title("Execution time - Sequential")
plt.grid(True)
plt.legend()
plt.savefig("../img/seq_plots/seq_time.jpg", dpi=300)
plt.close()

# Plot 2: Throughput
plt.figure(figsize=(7, 5))
plt.plot(df["size_MB"], df["avg_throughput_MBps"], marker="s", color="tab:green", label="Throughput (CSV)")
plt.xlabel("Problem size [MB] (total budget)")
plt.ylabel("Throughput [MB/s]")
plt.title("Throughput - Sequential")
plt.grid(True)
plt.legend()
plt.savefig("../img/seq_plots/seq_throughput.jpg", dpi=300)
plt.close()

# Plot 3: Speedup (always 1)
plt.figure(figsize=(7, 5))
plt.plot(df["size_MB"], df["avg_speedup"], marker="^", color="tab:orange", label="Speedup")
plt.xlabel("Problem size [MB] (total budget)")
plt.ylabel("Speedup")
plt.title("Speedup - Sequential")
plt.ylim(0, 2)
plt.grid(True)
plt.legend()
plt.savefig("../img/seq_plots/seq_speedup.jpg", dpi=300)
plt.close()

# Plot 4: Memory profile from Massif
massif_file = "../../perf-stats/seq_measurements/seq_500MB_mem_profile.txt"
times, mems = [], []

with open(massif_file) as f:
    for line in f:
        m = re.match(r"\s*\d+\s+([\d,]+)\s+([\d,]+)", line)
        if m:
            t = int(m.group(1).replace(",", ""))
            mem = int(m.group(2).replace(",", ""))
            times.append(t)
            mems.append(mem)

# Normalize time (ns → s, relative)
times_rel = [(t - times[0]) / 1e9 for t in times]
# Convert memory B → GB
mems_gb = [m / (1024 ** 3) for m in mems]

plt.figure(figsize=(7, 5))
plt.plot(times_rel, mems_gb, marker=".", linestyle="-", color="tab:red", label="Allocated memory")
plt.xlabel("Time [s] (normalized)")
plt.ylabel("Memory [MB]")
plt.title("Memory profile (seq, input 500MB)")
plt.grid(True)
plt.legend()
plt.savefig("../img/seq_plots/seq_memory_profile.jpg", dpi=300)
plt.close()

print("Plots saved in ../img/seq_plots/: seq_time.jpg, seq_throughput.jpg, seq_speedup.jpg, seq_memory_profile.jpg")
