#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# === Load best config ===
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

# Filter for size 500MB
df_500 = df_all[df_all["size_mb"] == 500].copy()

# Metrics to plot
metrics = ["time", "throughput", "speedup", "efficiency", "memory_overhead"]

# === Normalize metrics ===
df_norm = df_500.copy()
for m in metrics:
    vals = df_norm[m].astype(float).values
    if m in ["time", "memory_overhead"]:
        vals = 1 / vals
    vals = (vals - vals.min()) / (vals.max() - vals.min() + 1e-9)
    df_norm[m] = vals

# === Radar chart ===
labels = metrics
num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

plt.figure(figsize=(8, 6))
ax = plt.subplot(111)
r_ticks = [0.2, 0.4, 0.6, 0.8, 1.0]
for r in r_ticks:
    xs = [r * np.cos(a) for a in angles]
    ys = [r * np.sin(a) for a in angles]
    ax.plot(xs, ys, color="gray", linewidth=0.7)
    ax.fill(xs, ys, color="lightgray", alpha=0.05)

for a, label in zip(angles[:-1], labels):
    ax.plot([0, np.cos(a)], [0, np.sin(a)], color="gray", linewidth=0.7)
    ax.text(1.1 * np.cos(a), 1.1 * np.sin(a), label, ha="center", va="center")

for _, row in df_norm.iterrows():
    values = row[metrics].tolist()
    values += values[:1]
    xs = [v * np.cos(a) for v, a in zip(values, angles)]
    ys = [v * np.sin(a) for v, a in zip(values, angles)]
    ax.plot(xs, ys, label=row["label"])
    ax.fill(xs, ys, alpha=0.1)

ax.set_aspect("equal")
ax.axis("off")
plt.title("Best configs at 500MB (normalized)", y=1.05, fontsize=14, fontweight="bold", loc="left", color="black")
plt.legend(loc="upper right", bbox_to_anchor=(1.35, 1))
plt.savefig("../img/overall_plots/overall_radar_500MB.jpg", dpi=300)
plt.close()

print("Radar chart saved: overall_radar_500MB.jpg")
