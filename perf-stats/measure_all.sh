#!/usr/bin/env bash
set -euo pipefail

# Created by Emanuele (https://github.com/Kirito-Emo)

##### SEQUENTIAL #####
echo ">>> Running sequential_measure.sh"
bash sequential_measure.sh

echo "<<< Profiling the memory usage of the sequential version >>>"
valgrind --tool=massif --massif-out-file=./seq_measurements/massif_seq_100MB.out ../cmake-build-debug/hpc 100\
ms_print ./seq_measurements/massif_seq_100MB.out > ./seq_measurements/seq_100MB_mem_profile.txt

##### OPENMP #####
echo ">>> Running seq_omp_measure.sh"
bash seq_omp_measure.sh

echo "<<< Profiling the memory usage of the OpenMP version >>>"
# Change the number of threads as needed (also in the name of the files for better organization)
OMP_NUM_THREADS=8 OMP_DYNAMIC=false OMP_PROC_BIND=spread OMP_PLACES=cores \
valgrind --tool=massif --massif-out-file=./omp_measurements/massif_omp_8.out --stacks=yes ../cmake-build-release/hpc_omp 100\
ms_print ./omp_measurements/massif_omp_8.out > ./omp_measurements/seq_omp_8t_100MB_mem_profile.txt

##### MPI #####
echo ">>> Running mpi_measure.sh"
bash mpi_measure.sh

echo "<<< Profiling the memory usage of the MPI version >>>"
# Change the number of processes as needed (also in the name of the files for better organization)
mpirun -np 8 --bind-to none \
bash -c '[[ "$OMPI_COMM_WORLD_RANK" -eq 0 ]] && exec valgrind --tool=massif --stacks=yes --massif-out-file=./mpi_measurements/massif_mpi_8r_100MB.out ../cmake-build-release/hpc_mpi 100 || exec ../cmake-build-release/hpc_mpi 100'
ms_print ./mpi_measurements/massif_mpi_8r_100MB.out > ./mpi_measurements/mpi_rank0_8r_100MB_mem_profile.txt

##### MPI + OPENMP #####
echo ">>> Running mpi_omp_measure.sh"
bash mpi_omp_measure.sh

echo "<<< Profiling the memory usage of the MPI + OpenMP version >>>"
# Change the number of processes and threads as needed (also in the name of the files for better organization)
mpirun -np 8 --bind-to none -x FI_PROVIDER=tcp -x FI_OFI_RXM_USE_SRX=0 \
bash -c '[[ "$OMPI_COMM_WORLD_RANK" -eq 0 ]] && \
  OMP_NUM_THREADS=2 OMP_DYNAMIC=false OMP_PROC_BIND=spread OMP_PLACES=cores \
  exec valgrind --tool=massif --stacks=yes --massif-out-file=./mpi_omp_measurements/8r_2t/massif_mpi_omp_8r_2t_100MB.out ../cmake-build-release/hpc_mpi_omp 100 || \
  OMP_NUM_THREADS=2 OMP_DYNAMIC=false OMP_PROC_BIND=spread OMP_PLACES=cores exec ../cmake-build-release/hpc_mpi_omp 100' \
ms_print ./mpi_omp_measurements/8r_2t/massif_mpi_omp_8r_2t_100MB.out > ./mpi_omp_measurements/8r_2t/mpi_omp_rank0_8r_2t_100MB_mem_profile.txt

##### CUDA #####
echo ">>> Running cuda_measure.sh"
bash cuda_measure.sh

echo "<<< Profiling the memory usage of the CUDA Single-Stream version >>>"
bash ncu_cuda_profile.sh

##### CUDA MS #####
echo ">>> Running cuda_ms_measure.sh"
bash cuda_ms_measure.sh

echo "<<< Profiling the memory usage of the CUDA Multi-Stream version >>>"
bash ncu_cuda_ms_profile.sh

echo ">>> All scripts completed!"
