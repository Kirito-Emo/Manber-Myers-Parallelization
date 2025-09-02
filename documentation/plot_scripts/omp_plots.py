import matplotlib.pyplot as plt
import pandas as pd

# Load CSV data for OpenMP (2, 4 and 8 threads)
df2 = pd.read_csv("../../perf-stats/omp_measurements/threads_2/omp_summary_2.csv")
df4 = pd.read_csv("../../perf-stats/omp_measurements/threads_4/omp_summary_4.csv")
df8 = pd.read_csv("../../perf-stats/omp_measurements/threads_8/omp_summary_8.csv")


# Helper: convert "50MB" → 50
def parse_size(x):
    return int(x.replace("MB", ""))


df2["size_MB"] = df2["size"].apply(parse_size)
df4["size_MB"] = df4["size"].apply(parse_size)
df8["size_MB"] = df8["size"].apply(parse_size)

# Plot 1: Execution time vs input size
plt.figure(figsize=(8, 5))
plt.plot(df2["size_MB"], df2["avg_time_compute_pure_s"], marker="^", label="2 threads")
plt.plot(df4["size_MB"], df4["avg_time_compute_pure_s"], marker="o", label="4 threads")
plt.plot(df8["size_MB"], df8["avg_time_compute_pure_s"], marker="s", label="8 threads")
plt.xlabel("Problem size [MB] (total budget)")
plt.ylabel("Average compute time [s]")
plt.title("OpenMP: Execution time vs input size")
plt.legend()
plt.grid(True)
plt.savefig("../img/omp_plots/omp_times.jpg", dpi=300)
plt.close()

# Plot 2: Throughput (MB/s)
plt.figure(figsize=(8, 5))
plt.plot(df2["size_MB"], df2["size_MB"] / df2["avg_time_compute_pure_s"], marker="^", label="2 threads")
plt.plot(df4["size_MB"], df4["size_MB"] / df4["avg_time_compute_pure_s"], marker="o", label="4 threads")
plt.plot(df8["size_MB"], df8["size_MB"] / df8["avg_time_compute_pure_s"], marker="s", label="8 threads")
plt.xlabel("Problem size [MB] (total budget)")
plt.ylabel("Throughput [MB/s]")
plt.title("OpenMP: Throughput")
plt.legend()
plt.grid(True)
plt.savefig("../img/omp_plots/omp_throughput.jpg", dpi=300)
plt.close()

# Plot 3: Speedup
plt.figure(figsize=(8, 5))
plt.plot(df2["size_MB"], df2["avg_speedup"], marker="^", label="2 threads")
plt.plot(df4["size_MB"], df4["avg_speedup"], marker="o", label="4 threads")
plt.plot(df8["size_MB"], df8["avg_speedup"], marker="s", label="8 threads")
plt.xlabel("Problem size [MB] (total budget)")
plt.ylabel("Speedup")
plt.title("OpenMP: Speedup vs sequential")
plt.legend()
plt.grid(True)
plt.savefig("../img/omp_plots/omp_speedup.jpg", dpi=300)
plt.close()

# Plot 4: Efficiency
plt.figure(figsize=(8, 5))
plt.plot(df2["size_MB"], df2["avg_efficiency_pct"], marker="^", label="2 threads")
plt.plot(df4["size_MB"], df4["avg_efficiency_pct"], marker="o", label="4 threads")
plt.plot(df8["size_MB"], df8["avg_efficiency_pct"], marker="s", label="8 threads")
plt.xlabel("Problem size [MB] (total budget)")
plt.ylabel("Efficiency [%]")
plt.title("OpenMP: Parallel efficiency")
plt.legend()
plt.grid(True)
plt.savefig("../img/omp_plots/omp_efficiency.jpg", dpi=300)
plt.close()


# Function to extract peak memory from Massif txt profile
def parse_massif_peak(filename):
    """
    Parse a Massif output text file and extract the peak memory usage in GB.
    It scans the 'total(B)' column, removing commas before conversion.
    """
    peak_bytes = 0
    with open(filename, "r") as f:
        for line in f:
            parts = line.strip().split()
            # We need at least 3 columns: n, time(i), total(B)
            if len(parts) >= 3 and parts[0].isdigit():
                # Remove commas from numbers like "2,228,337,216"
                total_str = parts[2].replace(",", "")
                try:
                    total_bytes = int(total_str)
                    if total_bytes > peak_bytes:
                        peak_bytes = total_bytes
                except ValueError:
                    continue
    return peak_bytes / (1024 ** 3)  # Convert to GB


mem2 = parse_massif_peak("../../perf-stats/omp_measurements/threads_2/seq_omp_2t_500MB_mem_profile.txt")
mem4 = parse_massif_peak("../../perf-stats/omp_measurements/threads_4/seq_omp_4t_500MB_mem_profile.txt")
mem8 = parse_massif_peak("../../perf-stats/omp_measurements/threads_8/seq_omp_8t_500MB_mem_profile.txt")

# Plot 5: Peak memory usage (bar chart)
plt.figure(figsize=(6, 4))
plt.bar(["2 threads", "4 threads", "8 threads"], [mem2, mem4, mem8], color=["skyblue", "salmon", "lightgreen"])
plt.ylabel("Peak memory [MB]")
plt.title("OpenMP memory profile (total 500MB)")
plt.savefig("../img/omp_plots/omp_memory.jpg", dpi=300)
plt.close()

print("Plots generated successfully.")
