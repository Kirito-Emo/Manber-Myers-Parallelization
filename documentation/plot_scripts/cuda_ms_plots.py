#!/usr/bin/env python3
import matplotlib.pyplot as plt
import pandas as pd

# Load CUDA multi-stream summary
df = pd.read_csv("../../perf-stats/cuda_ms_measurements/cuda_ms_summary.csv")

# Ensure sorting
df = df.sort_values(by=["streams", "size_mb"])


# Utility: parse profiling files for breakdown
def parse_cuda_ms_profile(filename):
    """
    Parse a CUDA multi-stream profile text file and return
    a dict with the main timing components and memory overhead ratio.
    """
    times = {}
    with open(filename, "r") as f:
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
            elif "memory_overhead_ratio" in line:
                times["mem_overhead_pct"] = float(line.split("=")[1].split()[0])
    return times


profile_2s = parse_cuda_ms_profile("../../perf-stats/cuda_ms_profiles/cuda_ms_500MB_2s.txt")
profile_4s = parse_cuda_ms_profile("../../perf-stats/cuda_ms_profiles/cuda_ms_500MB_4s.txt")
profile_8s = parse_cuda_ms_profile("../../perf-stats/cuda_ms_profiles/cuda_ms_500MB_8s.txt")

# Plot 1: Execution time vs Problem size (total budget)
plt.figure(figsize=(8, 6))
for s in sorted(df["streams"].unique()):
    subset = df[df["streams"] == s]
    plt.plot(subset["size_mb"], subset["avg_time_total_compute_s"], marker="o", label=f"{s} streams")
plt.xlabel("Problem size (total budget) [MB]")
plt.ylabel("Execution time [s]")
plt.title("CUDA Multi-Stream - Execution time")
plt.legend()
plt.grid(True)
plt.savefig("../img/cuda_ms_plots/cuda_ms_times.jpg", dpi=300)

# Plot 2: Throughput
plt.figure(figsize=(8, 6))
for s in sorted(df["streams"].unique()):
    subset = df[df["streams"] == s]
    plt.plot(subset["size_mb"], subset["avg_throughput_MBps"], marker="o", label=f"{s} streams")
plt.xlabel("Problem size (total budget) [MB]")
plt.ylabel("Throughput [MB/s]")
plt.title("CUDA Multi-Stream - Throughput")
plt.legend()
plt.grid(True)
plt.savefig("../img/cuda_ms_plots/cuda_ms_throughput.jpg", dpi=300)

# Plot 3: Speedup
plt.figure(figsize=(8, 6))
for s in sorted(df["streams"].unique()):
    subset = df[df["streams"] == s]
    plt.plot(subset["size_mb"], subset["avg_speedup"], marker="o", label=f"{s} streams")
plt.xlabel("Problem size (total budget) [MB]")
plt.ylabel("Speedup")
plt.title("CUDA Multi-Stream - Speedup")
plt.legend()
plt.grid(True)
plt.savefig("../img/cuda_ms_plots/cuda_ms_speedup.jpg", dpi=300)

# Plot 4: Efficiency
plt.figure(figsize=(8, 6))
for s in sorted(df["streams"].unique()):
    subset = df[df["streams"] == s]
    plt.plot(subset["size_mb"], subset["avg_efficiency_pct"], marker="o", label=f"{s} streams")
plt.xlabel("Problem size (total budget) [MB]")
plt.ylabel("Efficiency [%]")
plt.title("CUDA Multi-Stream - Efficiency")
plt.legend()
plt.grid(True)
plt.savefig("../img/cuda_ms_plots/cuda_ms_efficiency.jpg", dpi=300)

# Plot 5: Memory overhead ratio
plt.figure(figsize=(8, 6))
for s in sorted(df["streams"].unique()):
    subset = df[df["streams"] == s]
    plt.plot(subset["size_mb"], subset["avg_memory_overhead_ratio_pct"], marker="o", label=f"{s} streams")
plt.xlabel("Problem size (total budget) [MB]")
plt.ylabel("Memory overhead [%]")
plt.title("CUDA Multi-Stream - Memory overhead")
plt.legend()
plt.grid(True)
plt.savefig("../img/cuda_ms_plots/cuda_ms_memory_overhead.jpg", dpi=300)

# Plot 6: Breakdown for 500MB (2 streams vs 4 streams vs 8 streams)
labels = ["alloc", "h2d", "kernel", "lcp", "d2h"]
values_2s = [profile_2s[k] for k in labels]
values_4s = [profile_4s[k] for k in labels]
values_8s = [profile_8s[k] for k in labels]

x = range(len(labels))
width = 0.25

plt.figure(figsize=(8, 6))
plt.bar([i - width / 2 for i in x], values_2s, width, label="2 streams")
plt.bar([i + width / 2 for i in x], values_4s, width, label="4 streams")
plt.bar([i + 1.5 * width for i in x], values_8s, width, label="8 streams")
plt.xticks(x, labels)
plt.ylabel("Time [s]")
plt.title("CUDA Multi-Stream 500MB - Breakdown (2s vs 4s vs 8s)")
plt.legend()
plt.grid(axis="y")
plt.savefig("../img/cuda_ms_plots/cuda_ms_breakdown_500MB.jpg", dpi=300)

print(
    "All CUDA multi-stream plots saved: cuda_ms_times.jpg, cuda_ms_throughput.jpg, cuda_ms_speedup.jpg, cuda_ms_efficiency.jpg, cuda_ms_memory_overhead.jpg, cuda_ms_breakdown_500MB.jpg"
)
