#!/usr/bin/env python3
import re

import matplotlib.pyplot as plt
import pandas as pd


# Helper function to parse massif output for peak memory
def parse_massif_peak(filename):
    peak_gb = None
    with open(filename, "r") as f:
        for line in f:
            if re.search(r"^\s*[0-9]+\.[0-9]+\^", line):
                match = re.search(r"([0-9]+\.[0-9]+)\^", line)
                if match:
                    peak_gb = float(match.group(1))
    return peak_gb


# Load summary CSVs and add ranks info
files = {
    2: "../../perf-stats/mpi_measurements/2r/mpi_summary_2.csv",
    4: "../../perf-stats/mpi_measurements/4r/mpi_summary_4.csv",
    8: "../../perf-stats/mpi_measurements/8r/mpi_summary_8.csv",
}

dfs = []
for ranks, path in files.items():
    df = pd.read_csv(path)
    df["ranks"] = ranks
    dfs.append(df)

summary = pd.concat(dfs, ignore_index=True)

# Add peak memory usage for rank 0 from massif outputs
memory_peaks = {
    2: parse_massif_peak("../../perf-stats/mpi_measurements/2r/mpi_rank0_2r_500MB_mem_profile.txt"),
    4: parse_massif_peak("../../perf-stats/mpi_measurements/4r/mpi_rank0_4r_500MB_mem_profile.txt"),
    8: parse_massif_peak("../../perf-stats/mpi_measurements/8r/mpi_rank0_8r_500MB_mem_profile.txt"),
}
summary["peak_memory_GB"] = summary["ranks"].map(memory_peaks)

# Plot 1: Execution time
plt.figure(figsize=(8, 6))
for r in summary["ranks"].unique():
    subset = summary[summary["ranks"] == r]
    plt.plot(subset["size_mb"], subset["avg_time_total_compute_s"], marker="o", label=f"{r} ranks")
plt.xlabel("Problem size [MB] (total budget)")
plt.ylabel("Execution time [s]")
plt.title("MPI - Execution time vs Problem size (total budget)")
plt.legend()
plt.grid(True)
plt.savefig("../img/mpi_plots/mpi_times.jpg", dpi=600)

# Plot 2: Throughput
plt.figure(figsize=(8, 6))
for r in summary["ranks"].unique():
    subset = summary[summary["ranks"] == r]
    plt.plot(subset["size_mb"], subset["avg_throughput_MBps"], marker="o", label=f"{r} ranks")
plt.xlabel("Problem size [MB] (total budget)")
plt.ylabel("Throughput [MB/s]")
plt.title("MPI - Throughput vs Problem size (total budget)")
plt.legend()
plt.grid(True)
plt.savefig("../img/mpi_plots/mpi_throughput.jpg", dpi=600)

# Plot 3: Speedup
plt.figure(figsize=(8, 6))
for r in summary["ranks"].unique():
    subset = summary[summary["ranks"] == r]
    plt.plot(subset["size_mb"], subset["avg_speedup"], marker="o", label=f"{r} ranks")
plt.xlabel("Problem size [MB] (total budget)")
plt.ylabel("Speedup")
plt.title("MPI - Speedup vs Problem size (total budget)")
plt.legend()
plt.grid(True)
plt.savefig("../img/mpi_plots/mpi_speedup.jpg", dpi=600)

# Plot 4: Efficiency
plt.figure(figsize=(8, 6))
for r in summary["ranks"].unique():
    subset = summary[summary["ranks"] == r]
    plt.plot(subset["size_mb"], subset["avg_efficiency_pct"], marker="o", label=f"{r} ranks")
plt.xlabel("Problem size [MB] (total budget)")
plt.ylabel("Efficiency [%]")
plt.title("MPI - Efficiency vs Problem size (total budget)")
plt.legend()
plt.grid(True)
plt.savefig("../img/mpi_plots/mpi_efficiency.jpg", dpi=600)

# Plot 5: Memory usage
plt.figure(figsize=(8, 6))
mem_subset = summary[summary["size_mb"] == 500]
plt.bar(mem_subset["ranks"].astype(str), mem_subset["peak_memory_GB"])
plt.xlabel("MPI ranks")
plt.ylabel("Peak memory [MB]")
plt.title("MPI - Peak memory usage (500MB case, rank 0)")
plt.grid(axis="y")
plt.savefig("../img/mpi_plots/mpi_memory.jpg", dpi=600)

print("All MPI plots saved: mpi_times.jpg, mpi_throughput.jpg, mpi_speedup.jpg, mpi_efficiency.jpg, mpi_memory.jpg")
