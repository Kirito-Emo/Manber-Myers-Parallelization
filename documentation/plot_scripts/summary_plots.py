#!/usr/bin/env python3
import matplotlib.pyplot as plt
import pandas as pd

# === Load best configs ===
seq = pd.read_csv("../../perf-stats/seq_measurements/seq_summary.csv")
omp = pd.read_csv("../../perf-stats/omp_measurements/threads_8/omp_summary_8.csv")
mpi = pd.read_csv("../../perf-stats/mpi_measurements/8r/mpi_summary_8.csv")
mpi_omp = pd.read_csv("../../perf-stats/mpi_omp_measurements/8r_2t/mpi_omp_summary_8r_2t.csv")
cuda = pd.read_csv("../../perf-stats/cuda_measurements/cuda_summary.csv")
cuda_ms = pd.read_csv("../../perf-stats/cuda_ms_measurements/cuda_ms_summary.csv")

# === Normalize column names ===
seq["label"] = "Seq"
omp["label"] = "OpenMP (8t)"
mpi["label"] = "MPI (8r)"
mpi_omp["label"] = "MPI+OMP (8r×2t)"
cuda["label"] = "CUDA"
cuda_ms = cuda_ms[cuda_ms["streams"] == 8].copy()
cuda_ms["label"] = "CUDA-MS (8s)"

# Convert "size" like "500MB" → int MB
if "size" in seq.columns:
    seq["size_mb"] = seq["size"].str.replace("MB", "").astype(int)
if "size" in omp.columns:
    omp["size_mb"] = omp["size"].str.replace("MB", "").astype(int)

df_list = []

df_list.append(seq.rename(columns={
    "avg_time_total_compute_s": "time",
    "avg_throughput_MBps": "throughput",
    "avg_speedup": "speedup",
    "avg_efficiency_pct": "efficiency",
    "avg_memory_overhead_ratio_pct": "memory_overhead"
}))

df_list.append(omp.rename(columns={
    "avg_time_total_compute_s": "time",
    "avg_throughput_MBps": "throughput",
    "avg_speedup": "speedup",
    "avg_efficiency_pct": "efficiency",
    "avg_memory_overhead_ratio_pct": "memory_overhead"
}))

df_list.append(mpi.rename(columns={
    "avg_time_total_compute_s": "time",
    "avg_throughput_MBps": "throughput",
    "avg_speedup": "speedup",
    "avg_efficiency_pct": "efficiency",
    "avg_memory_overhead_ratio_pct": "memory_overhead"
}))

df_list.append(mpi_omp.rename(columns={
    "avg_time_total_compute_s": "time",
    "avg_throughput_MBps": "throughput",
    "avg_speedup": "speedup",
    "avg_efficiency_pct": "efficiency",
    "avg_memory_overhead_ratio_pct": "memory_overhead"
}))

df_list.append(cuda.rename(columns={
    "avg_time_total_compute_s": "time",
    "avg_throughput_MBps": "throughput",
    "avg_speedup": "speedup",
    "avg_efficiency_pct": "efficiency",
    "avg_memory_overhead_ratio_pct": "memory_overhead"
}))

df_list.append(cuda_ms.rename(columns={
    "avg_time_total_compute_s": "time",
    "avg_throughput_MBps": "throughput",
    "avg_speedup": "speedup",
    "avg_efficiency_pct": "efficiency",
    "avg_memory_overhead_ratio_pct": "memory_overhead"
}))

df_all = pd.concat(df_list, ignore_index=True)

# Ensure sorting
df_all = df_all.sort_values(["label", "size_mb"])

# === Helper: plot with consistent x ticks ===
xticks = [1, 50, 100, 200, 500]
xticklabels = ["1MB", "50MB", "100MB", "200MB", "500MB"]


def plot_metric(metric, ylabel, title, filename):
    plt.figure(figsize=(10, 8))
    for label, subset in df_all.groupby("label"):
        subset = subset.sort_values("size_mb")
        plt.plot(subset["size_mb"], subset[metric], marker="o", label=label)
    plt.xticks(xticks, xticklabels)
    plt.xlabel("Problem size (total budget)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.7)
    plt.savefig(f"../img/overall_plots/{filename}", dpi=300)


# === Generate plots ===
plot_metric("time", "Execution time [s]", "Execution time - Best configs", "overall_times.svg")
plot_metric("throughput", "Throughput [MB/s]", "Throughput - Best configs", "overall_throughput.svg")
plot_metric("speedup", "Speedup", "Speedup - Best configs", "overall_speedup.svg")
plot_metric("efficiency", "Efficiency [%]", "Efficiency - Best configs", "overall_efficiency.svg")
plot_metric("memory_overhead", "Memory overhead [%]", "Memory Overhead - Best configs", "overall_memory_overhead.svg")

print("All plots saved.")
