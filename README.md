# Manber-Myers-Parallelization

This project implements and benchmarks sequential and parallel (OMP, MPI and CUDA) versions of the **Manber-Myers
algorithm** for suffix array construction. It is designed for **high-performance computing (HPC)** environments and
includes tools for generating test data, collecting performance statistics and profiling memory/compute usage.

---

## Features

- **Sequential Suffix Array Construction:** Efficient implementation of the Manber-Myers algorithm
- **Parallel Suffix Array Construction:**
    - **OpenMP** (shared-memory parallelism)
    - **MPI** (distributed-memory parallelism)
    - **Hybrid MPI+OpenMP** (distributed + shared memory)
    - **CUDA** (single-stream GPU parallelization)
    - **CUDA MS** (concurrent GPU kernels with multi-streams)
- **Performance Measurement:** Scripts to benchmark sequential, OpenMP, MPI and hybrid versions, producing detailed CSV
  logs
- **Profiling Tools:**
    - **Valgrind Massif** for memory profiling
      > Only rank 0 profiled in MPI/MPI+OMP versions
    - **Nsight Compute (ncu)** for GPU kernel profiling
- **Random String Generation:** Utilities to create large test strings of various sizes (1, 5, 10, 50, 100, 200, 500 MB)

---

## Directory Structure

- `src/` – Source code for suffix array construction algorithms
- `include/` – Header files for the project
- `random_strings/` – Scripts to generate test input files
- `perf-stats/` – Scripts to run benchmarks and generate CSV stats
    - `seq_measurements` – Performance stats for sequential version and profiling outputs
    - `omp_measurements/` – Performance stats for OpenMP version and profiling outputs
    - `mpi_measurements/` – Performance stats for MPI version and profiling outputs
    - `mpi_omp_measurements/` – Performance stats for hybrid MPI+OpenMP version and profiling outputs
    - `cuda_measurements/` – Performance stats for CUDA version
    - `cuda_ms_measurements/` – Performance stats for CUDA multi-stream version
    - `cuda_profiles/` – Nsight Compute reports (CUDA single-stream)
    - `cuda_ms_profiles/` – Nsight Compute reports (CUDA multi-stream)
- `cmake-build-release/` – Release build artifacts
- `cmake-build-debug/` – Debug build artifacts
- `cuda_docker` - Dockerfile to use CUDA in a containerized environment

---

## Building

This project uses CMake.

To build:

```sh
git clone git@github.com:Kirito-Emo/Manber-Myers-Parallelization.git # Using SSH to clone the repository
cd Manber-Myers-Parallelization
cmake -S . -B cmake-build-release -DCMAKE_BUILD_TYPE=Release
cmake --build cmake-build-release -j
```

---

## Usage

### Generating Random Strings

Use the script in the `random_strings/` directory to generate random strings for testing:

```sh
cd random_strings
./string_creation.sh
```

### Sequential Version

To run the sequential implementation, execute the `hpc` binary with the required input file:

```sh
# Change <input_file> to the path of your input file
cd cmake-build-release
./hpc <input_file>
```

### OpenMP Version

To run the OpenMP version, set the `OMP_NUM_THREADS` environment variable to specify the number of threads:

```sh
# Change <num_threads> to the number of threads you want to use
# Change <input_file> to the path of your input file
OMP_NUM_THREADS=<num_threads> OMP_DYNAMIC=false OMP_PROC_BIND=spread OMP_PLACES=cores
./hpc_omp <input_file> 
```

### MPI Version

To run the MPI version, use `mpirun` or `mpiexec` with the desired number of processes.
> **_NOTE:_** Ensure that you have MPI installed and configured on your system.

```sh
# Change <num_processes> to the number of processes you want to use
# Change <input_file> to the path of your input file
mpirun -np <num_processes> --bind-to none ./hpc_mpi <input_file>
```

### MPI + OpenMP Version

To run the hybrid MPI + OpenMP version, set the `OMP_NUM_THREADS` environment variable and use `mpirun` or `mpiexec`:

```sh
# Change <num_processes> to the number of MPI processes you want to use
# Change <num_threads> to the number of OpenMP threads you want to use per process
# Change <input_file> to the path of your input file
mpirun -np <num_processes> --bind-to none \
  bash -c 'OMP_NUM_THREADS=<num_threads> OMP_DYNAMIC=false OMP_PROC_BIND=spread OMP_PLACES=cores exec ./hpc_mpi_omp <input_file>'
```

### CUDA Version (Single-Stream)

To run the CUDA version, execute the `hpc_cuda` binary with the required input file:

```sh
# Change <input_file> to the path of your input file
./hpc_cuda <input_file>
```

### CUDA Version (Multi-Stream)

To run the CUDA multi-stream version, execute the `hpc_cuda_ms` binary with the required input file:

```sh
# Change <input_file> to the path of your input file
# Change <num_streams> to the number of CUDA streams you want to use (typically 2, 4, or 8)
./hpc_cuda_ms <input_file> --streams <num_streams>
```

---

## Profiling

### Sequential Version

To profile the sequential version with Valgrind Massif:

```sh
valgrind --tool=massif --massif-out-file=seq_massif.out ./hpc <input_file>
ms_print seq_massif.out > seq_massif.txt
```

### OpenMP Version

To profile the OpenMP version with Valgrind Massif:

```sh
OMP_NUM_THREADS=<num_threads> OMP_DYNAMIC=false OMP_PROC_BIND=spread OMP_PLACES=cores \
valgrind --tool=massif --stacks=yes --massif-out-file=omp_massif.out ./hpc_omp <input_file>
ms_print omp_massif.out > omp_massif.txt
```

### MPI Version

To profile the MPI version with Valgrind Massif (only rank 0 is profiled):

```sh
mpirun -np <num_ranks> --bind-to none \
  bash -c '[[ "$OMPI_COMM_WORLD_RANK" -eq 0 ]] && \
    exec valgrind --tool=massif --stacks=yes --massif-out-file=./mpi_massif.out ./hpc_mpi <input_file> || \
    exec ./hpc_mpi <input_file>'
ms_print mpi_massif.out > mpi_rank0.txt
```

### MPI + OpenMP Version

To profile the MPI + OpenMP version with Valgrind Massif (only rank 0 is profiled):

```sh
mpirun -np <num_ranks> --bind-to none -x FI_PROVIDER=tcp -x FI_OFI_RXM_USE_SRX=0 \
  bash -c '[[ "$OMPI_COMM_WORLD_RANK" -eq 0 ]] && \
    OMP_NUM_THREADS=<num_threads> OMP_DYNAMIC=false OMP_PROC_BIND=spread OMP_PLACES=cores \
    exec valgrind --tool=massif --stacks=yes --massif-out-file=mpi_omp_massif.out ./hpc_mpi_omp 100 || \
    OMP_NUM_THREADS=<num_threads> OMP_DYNAMIC=false OMP_PROC_BIND=spread OMP_PLACES=cores \
    exec ./hpc_mpi_omp 100'
ms_print mpi_omp_massif.out > mpi_omp_rank0.txt
```

### CUDA (both Single-Stream and Multi-Stream Versions)

To profile the CUDA version with Nsight Compute:

```sh
ncu --set full \
    --target-processes all \
    --launch-skip 0 \
    --launch-count 1 \
    --csv \
    --export "${OUTPUT_DIR}/${PROFILE_NAME}.ncu-rep" \
    "$BIN" "$MB" | tee "${OUTPUT_DIR}/${PROFILE_NAME}.txt"
```

## Performance Statistics

Script and CSV files in the `perf-stats/` directory can be used to analyze performance.

> If you want to run all the scripts at once, you can use the `measure_all.sh` script:
>
>```sh
># This script runs all measurement scripts and the profiling sequentially
>./measure_all.sh
>```

### Sequential

The `sequential_measure.sh` script can be run to generate performance statistics saved in CSV format:

```sh
cd perf-stats/
./sequential_measure.sh
```

### OpenMP

The `omp_measure.sh` script can be run to generate performance statistics for the OpenMP version:

```sh
./omp_measure.sh
```

### MPI

The `mpi_measure.sh` script can be run to generate performance statistics for the MPI version:

```sh
./mpi_measure.sh
```

### MPI + OpenMP

The `mpi_omp_measure.sh` script can be run to generate performance statistics for the hybrid MPI + OpenMP version:

```sh
./mpi_omp_measure.sh
```

### CUDA (Single-Stream)

The `cuda_measure.sh` script can be run to generate performance statistics for the CUDA version:

```sh
./cuda_measure.sh
```

### CUDA (Multi-Stream)

The `cuda_ms_measure.sh` script can be run to generate performance statistics for the CUDA multi-stream version:

```sh
./cuda_ms_measure.sh
```

---

## CUDA Docker

A Dockerfile is provided in the `cuda_docker/` directory to create a containerized environment with CUDA support.
To build and run the Docker container:

```sh
# Change <name> as you prefer
cd cuda_docker
docker build -t <name> .
docker run --rm -it --gpus all -v <path_to_repo>:/workspace -w /workspace <name>
```

Then inside the container, you can build and run the project as described in the previous sections.
> - To see if there is CUDA availability and compatibility, you can run `nvidia-smi` inside the container.
> - Make sure your host machine has the NVIDIA drivers installed.
> - For more details, refer to
    the [NVIDIA Container Toolkit documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

---

## License

This project is licensed under the MIT License. See the [LICENSE file](./LICENSE) for details.

---

## Author

This project was developed by [Emanuele](https://github.com/Kirito-Emo). </br>
For questions or contributions, please open an issue or pull request on the GitHub repository.