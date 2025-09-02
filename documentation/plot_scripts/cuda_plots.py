#!/usr/bin/env python3
import re

import matplotlib.pyplot as plt
import pandas as pd

# Load CUDA summary
df = pd.read_csv("../../perf-stats/cuda_measurements/cuda_summary.csv")


# Extract memory overhead from profile
def parse_cuda_profile(filename):
    """
    Parse the CUDA profiling text file to extract memory overhead ratio (%).
    """
    overhead = None
    with open(filename, "r") as f:
        for line in f:
            if "memory_overhead_ratio" in line:
                match = re.search(r"memory_overhead_ratio=([\d\.]+)", line)
                if match:
                    overhead = float(match.group(1))
    return overhead


memory_overhead_100 = parse_cuda_profile("../../perf-stats/cuda_profiles/cuda_500MB_profile.txt")
df["peak_memory_overhead_pct"] = df.apply(
    lambda row: memory_overhead_100 if row["size_mb"] == 100 else None, axis=1
)

# Plot 1: Execution time vs Problem size (total budget)
plt.figure(figsize=(8, 6))
plt.plot(df["size_mb"], df["avg_time_total_compute_s"], marker="o", color="b")
plt.xlabel("Problem size (total budget) [MB]")
plt.ylabel("Execution time [s]")
plt.title("CUDA - Execution time vs Problem size (total budget)")
plt.grid(True)
plt.savefig("../img/cuda_plots/cuda_times.jpg", dpi=300)

# Plot 2: Throughput
plt.figure(figsize=(8, 6))
plt.plot(df["size_mb"], df["avg_throughput_MBps"], marker="o", color="g")
plt.xlabel("Problem size (total budget) [MB]")
plt.ylabel("Throughput [MB/s]")
plt.title("CUDA - Throughput vs Problem size (total budget)")
plt.grid(True)
plt.savefig("../img/cuda_plots/cuda_throughput.jpg", dpi=300)

# Plot 3: Speedup
plt.figure(figsize=(8, 6))
plt.plot(df["size_mb"], df["avg_speedup"], marker="o", color="r")
plt.xlabel("Problem size (total budget) [MB]")
plt.ylabel("Speedup")
plt.title("CUDA - Speedup vs Problem size (total budget)")
plt.grid(True)
plt.savefig("../img/cuda_plots/cuda_speedup.jpg", dpi=300)

# Plot 4: Efficiency
plt.figure(figsize=(8, 6))
plt.plot(df["size_mb"], df["avg_efficiency_pct"], marker="o", color="m")
plt.xlabel("Problem size (total budget) [MB]")
plt.ylabel("Efficiency [%]")
plt.title("CUDA - Efficiency vs Problem size (total budget)")
plt.grid(True)
plt.savefig("../img/cuda_plots/cuda_efficiency.jpg", dpi=300)

# Plot 5: Memory overhead ratio
plt.figure(figsize=(8, 6))
plt.plot(df["size_mb"], df["avg_memory_overhead_ratio_pct"], marker="o", color="c")
plt.xlabel("Problem size (total budget) [MB]")
plt.ylabel("Memory overhead [%]")
plt.title("CUDA - Memory overhead vs Problem size (total budget)")
plt.grid(True)
plt.savefig("../img/cuda_plots/cuda_memory_overhead.jpg", dpi=300)

# Plot 6: Stacked breakdown for 500MB
profile_file = "../../perf-stats/cuda_profiles/cuda_500MB_profile.txt"
times = {}
with open(profile_file, "r") as f:
    for line in f:
        if "time_alloc_host_dev" in line:
            times["alloc"] = float(line.split("=")[1].split()[0])
        elif "time_h2d" in line:
            times["h2d"] = float(line.split("=")[1].split()[0])
        elif "time_kernel_gpu" in line:
            times["kernel"] = float(line.split("=")[1].split()[0])
        elif "time_lcp_cpu" in line:
            times["lcp"] = float(line.split("=")[1].split()[0])
        elif "time_d2h" in line:
            times["d2h"] = float(line.split("=")[1].split()[0])

plt.figure(figsize=(8, 6))
labels = list(times.keys())
values = list(times.values())
plt.bar(labels, values, color=["#ffcc00", "#66ccff", "#ff6666", "#99cc66", "#cc99ff"])
plt.ylabel("Time [s]")
plt.title("CUDA 500MB - Breakdown of execution time")
plt.grid(axis="y")
plt.savefig("../img/cuda_plots/cuda_breakdown_100MB.jpg", dpi=300)

print(
    "All CUDA plots saved: cuda_times.jpg, cuda_throughput.jpg, cuda_speedup.jpg, cuda_efficiency.jpg, cuda_memory_overhead.jpg, cuda_breakdown_100MB.jpg")
