# Manber-Myers-Parallelization
This project implements and benchmarks sequential and parallel (MPI and CUDA) versions of the Manber-Myers algorithm for suffix array construction. It is designed for high-performance computing (HPC) environments and includes tools for generating test data and collecting performance statistics.

## Features
- **Sequential Suffix Array Construction:** Efficient implementation of the Manber-Myers algorithm
- **Parallel Suffix Array Construction:** MPI-based and CUDA-based parallelization for distributed memory systems
- **Performance Measurement:** Scripts and tools to benchmark and analyze performance
- **Random String Generation:** Utilities to create large test strings of various sizes

## Directory Structure
- `src/`
  Source code for suffix array construction algorithms (sequential, MPI-based and CUDA-based)
- `include/`
  Header files for the project
- `random_strings/`
  Script to generate random strings files and the resulting string files of various sizes (1MB, 5MB, 10MB, 50MB, 100MB, 500MB)
- `perf-stats/`
  Script to generate performance statistics (based on build time) and the resulting CSV files
- `cmake-build-debug/`
  Build artifacts (auto-generated)

## Building
This project uses CMake.

To build:
```sh
git clone git@github.com:Kirito-Emo/Manber-Myers-Parallelization.git # Using SSH to clone the repository
cd Manber-Myers-Parallelization
make
```

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
cd cmake-build-debug
./hpc [input_file] # Change [input_file] to the path of your input file
```

### Parallel MPI Version
To run the parallel MPI version, use `mpirun` or `mpiexec` with the desired number of processes.
Ensure that you have MPI installed and configured on your system.
```sh
# Change <num_processes> to the number of processes you want to use
# Change [input_file] to the path of your input file
mpirun -np <num_processes> ./hpc [input_file]
```

## Performance Statistics
Script and CSV files in the `perf-stats/` directory can be used to analyze performance.
The `sequential_measure.sh` script can be run to generate performance statistics saved in CSV format:
```sh
cd perf-stats/
./sequential_measure.sh
```

## License
This project is licensed under the MIT License. See the [LICENSE file](./LICENSE) for details.

## Author
This project was developed by [Emanuele](https://github.com/Kirito-Emo). </br>
For questions or contributions, please open an issue or pull request on the GitHub repository.