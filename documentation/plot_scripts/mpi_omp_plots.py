#!/usr/bin/env python3
import re

import matplotlib.pyplot as plt
import pandas as pd


def parse_massif_peak(filename):
    peak_mb = None
    with open(filename, "r") as f:
        for line in f:
            if re.search(r"^\s*[0-9]+\.[0-9]+\^", line):
                m = re.search(r"([0-9]+\.[0-9]+)\^", line)
                if m:
                    peak_mb = float(m.group(1))
    return peak_mb


# Load and combine data from CSV files for different configurations (ranks, threads)
files = {
    (2, 4): "../../perf-stats/mpi_omp_measurements/2r_4t/mpi_omp_summary_2r_4t.csv",
    (2, 8): "../../perf-stats/mpi_omp_measurements/2r_8t/mpi_omp_summary_2r_8t.csv",
    (4, 2): "../../perf-stats/mpi_omp_measurements/4r_2t/mpi_omp_summary_4r_2t.csv",
    (4, 4): "../../perf-stats/mpi_omp_measurements/4r_4t/mpi_omp_summary_4r_4t.csv",
    (8, 2): "../../perf-stats/mpi_omp_measurements/8r_2t/mpi_omp_summary_8r_2t.csv",
}

dfs = []
for (ranks, threads), path in files.items():
    df = pd.read_csv(path)
    df["ranks"] = ranks
    df["threads_per_rank"] = threads
    dfs.append(df)

summary = pd.concat(dfs, ignore_index=True)

# Load peak memory usage from Massif output files for the 500MB case (rank 0)
memory_peaks = {
    (2, 4): parse_massif_peak("../../perf-stats/mpi_omp_measurements/2r_4t/mpi_omp_rank0_2r_4t_500MB_mem_profile.txt"),
    (2, 8): parse_massif_peak("../../perf-stats/mpi_omp_measurements/2r_8t/mpi_omp_rank0_2r_8t_500MB_mem_profile.txt"),
    (4, 2): parse_massif_peak("../../perf-stats/mpi_omp_measurements/4r_2t/mpi_omp_rank0_4r_2t_500MB_mem_profile.txt"),
    (4, 4): parse_massif_peak("../../perf-stats/mpi_omp_measurements/4r_4t/mpi_omp_rank0_4r_4t_500MB_mem_profile.txt"),
    (8, 2): parse_massif_peak("../../perf-stats/mpi_omp_measurements/8r_2t/mpi_omp_rank0_8r_2t_500MB_mem_profile.txt"),
}

summary["peak_memory_MB"] = summary.apply(
    lambda row: memory_peaks.get((row["ranks"], row["threads_per_rank"]), None),
    axis=1
)

# Plot 1: Execution time
plt.figure(figsize=(8, 6))
for (r, t) in summary[["ranks", "threads_per_rank"]].drop_duplicates().itertuples(index=False):
    subset = summary[(summary["ranks"] == r) & (summary["threads_per_rank"] == t)]
    plt.plot(subset["size_mb"], subset["avg_time_total_compute_s"], marker="o", label=f"{r}r {t}t/r")
plt.xlabel("Problem size [MB] (total budget)")
plt.ylabel("Execution time [s]")
plt.title("MPI+OpenMP - Execution time vs Problem size (total budget)")
plt.legend();
plt.grid(True)
plt.savefig("../img/mpi_omp_plots/mpi_omp_times.jpg", dpi=600)

# Plot 2: Throughput
plt.figure(figsize=(8, 6))
for (r, t) in summary[["ranks", "threads_per_rank"]].drop_duplicates().itertuples(index=False):
    subset = summary[(summary["ranks"] == r) & (summary["threads_per_rank"] == t)]
    plt.plot(subset["size_mb"], subset["avg_throughput_MBps"], marker="o", label=f"{r}r {t}t/r")
plt.xlabel("Problem size [MB] (total budget)")
plt.ylabel("Throughput [MB/s]")
plt.title("MPI+OpenMP - Throughput vs Problem size (total budget)")
plt.legend();
plt.grid(True)
plt.savefig("../img/mpi_omp_plots/mpi_omp_throughput.jpg", dpi=600)

# Plot 3: Speedup
plt.figure(figsize=(8, 6))
for (r, t) in summary[["ranks", "threads_per_rank"]].drop_duplicates().itertuples(index=False):
    subset = summary[(summary["ranks"] == r) & (summary["threads_per_rank"] == t)]
    plt.plot(subset["size_mb"], subset["avg_speedup"], marker="o", label=f"{r}r {t}t/r")
plt.xlabel("Problem size [MB] (total budget)")
plt.ylabel("Speedup")
plt.title("MPI+OpenMP - Speedup vs Problem size (total budget)")
plt.legend();
plt.grid(True)
plt.savefig("../img/mpi_omp_plots/mpi_omp_speedup.jpg", dpi=600)

# Plot 4: Efficiency
plt.figure(figsize=(8, 6))
for (r, t) in summary[["ranks", "threads_per_rank"]].drop_duplicates().itertuples(index=False):
    subset = summary[(summary["ranks"] == r) & (summary["threads_per_rank"] == t)]
    plt.plot(subset["size_mb"], subset["avg_efficiency_pct"], marker="o", label=f"{r}r {t}t/r")
plt.xlabel("Problem size [MB] (total budget)")
plt.ylabel("Efficiency [%]")
plt.title("MPI+OpenMP - Efficiency vs Problem size (total budget)")
plt.legend();
plt.grid(True)
plt.savefig("../img/mpi_omp_plots/mpi_omp_efficiency.jpg", dpi=600)

# Plot 5: Peak memory usage (500MB case, rank 0)
plt.figure(figsize=(8, 6))
mem_subset = summary[summary["size_mb"] == 500]
labels = [f"{r}r {t}t/r" for r, t in zip(mem_subset["ranks"], mem_subset["threads_per_rank"])]
plt.bar(labels, mem_subset["peak_memory_MB"])
plt.xlabel("Configuration")
plt.ylabel("Peak memory [MB]")
plt.title("MPI+OpenMP - Peak memory usage (500MB case, rank 0)")
plt.grid(axis="y")
plt.savefig("../img/mpi_omp_plots/mpi_omp_memory.jpg", dpi=600)

print("All MPI+OpenMP plots saved.")
