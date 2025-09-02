#!/usr/bin/env bash
set -euo pipefail

# Created by Emanuele (https://github.com/Kirito-Emo)

##### SEQUENTIAL #####
echo ">>> Running sequential_measure.sh"
bash seq_measure.sh

echo "<<< Profiling the memory usage of the sequential version >>>"
valgrind --tool=massif --massif-out-file=./seq_measurements/massif_seq_500MB.out ../cmake-build-release/hpc 500
ms_print ./seq_measurements/massif_seq_500MB.out > ./seq_measurements/seq_500MB_mem_profile.txt

##### OPENMP #####
echo ">>> Running seq_omp_measure.sh"
bash omp_measure.sh

echo "<<< Profiling the memory usage of the OpenMP version >>>"
# 8 threads
OMP_NUM_THREADS=8 OMP_DYNAMIC=false OMP_PROC_BIND=spread OMP_PLACES=cores \
valgrind --tool=massif --massif-out-file=./omp_measurements/threads_8/massif_omp_8t_500MB.out --stacks=yes ../cmake-build-release/hpc_omp 500
ms_print ./omp_measurements/threads_8/massif_omp_8t_500MB.out > ./omp_measurements/threads_8/seq_omp_8t_500MB_mem_profile.txt

# 4 threads
OMP_NUM_THREADS=4 OMP_DYNAMIC=false OMP_PROC_BIND=spread OMP_PLACES=cores \
valgrind --tool=massif --massif-out-file=./omp_measurements/threads_4/massif_omp_4t_500MB.out --stacks=yes ../cmake-build-release/hpc_omp 500
ms_print ./omp_measurements/threads_4/massif_omp_4t_500MB.out > ./omp_measurements/threads_4/seq_omp_4t_500MB_mem_profile.txt

# 2 threads
OMP_NUM_THREADS=2 OMP_DYNAMIC=false OMP_PROC_BIND=spread OMP_PLACES=cores \
valgrind --tool=massif --massif-out-file=./omp_measurements/threads_2/massif_omp_2t_500MB.out --stacks=yes ../cmake-build-release/hpc_omp 500
ms_print ./omp_measurements/threads_2/massif_omp_2t_500MB.out > ./omp_measurements/threads_2/seq_omp_2t_500MB_mem_profile.txt

##### MPI #####
echo ">>> Running mpi_measure.sh"
bash mpi_measure.sh

echo "<<< Profiling the memory usage of the MPI version >>>"
# 8 ranks
mpirun -np 8 --bind-to none \
bash -c '[[ "$OMPI_COMM_WORLD_RANK" -eq 0 ]] && exec valgrind --tool=massif --stacks=yes --massif-out-file=./mpi_measurements/8r/massif_mpi_8r_500MB.out ../cmake-build-release/hpc_mpi 500 || exec ../cmake-build-release/hpc_mpi 500'
ms_print ./mpi_measurements/8r/massif_mpi_8r_500MB.out > ./mpi_measurements/8r/mpi_rank0_8r_500MB_mem_profile.txt

# 4 ranks
mpirun -np 4 --bind-to none \
bash -c '[[ "$OMPI_COMM_WORLD_RANK" -eq 0 ]] && exec valgrind --tool=massif --stacks=yes --massif-out-file=./mpi_measurements/4r/massif_mpi_4r_500MB.out ../cmake-build-release/hpc_mpi 500 || exec ../cmake-build-release/hpc_mpi 500'
ms_print ./mpi_measurements/4r/massif_mpi_4r_500MB.out > ./mpi_measurements/4r/mpi_rank0_4r_500MB_mem_profile.txt

# 2 ranks
mpirun -np 2 --bind-to none \
bash -c '[[ "$OMPI_COMM_WORLD_RANK" -eq 0 ]] && exec valgrind --tool=massif --stacks=yes --massif-out-file=./mpi_measurements/2r/massif_mpi_2r_500MB.out ../cmake-build-release/hpc_mpi 500 || exec ../cmake-build-release/hpc_mpi 500'
ms_print ./mpi_measurements/2r/massif_mpi_2r_500MB.out > ./mpi_measurements/2r/mpi_rank0_2r_500MB_mem_profile.txt

##### MPI + OPENMP #####
echo ">>> Running mpi_omp_measure.sh"
bash mpi_omp_measure.sh

echo "<<< Profiling the memory usage of the MPI + OpenMP version >>>"
# 8 ranks, 2 threads
mpirun -np 8 --bind-to none -x FI_PROVIDER=tcp -x FI_OFI_RXM_USE_SRX=0 \
bash -c '[[ "$OMPI_COMM_WORLD_RANK" -eq 0 ]] && \
  OMP_NUM_THREADS=2 OMP_DYNAMIC=false OMP_PROC_BIND=spread OMP_PLACES=cores \
  exec valgrind --tool=massif --stacks=yes --massif-out-file=./mpi_omp_measurements/8r_2t/massif_mpi_omp_8r_2t_500MB.out ../cmake-build-release/hpc_mpi_omp 500 || \
  OMP_NUM_THREADS=2 OMP_DYNAMIC=false OMP_PROC_BIND=spread OMP_PLACES=cores exec ../cmake-build-release/hpc_mpi_omp 500'
ms_print ./mpi_omp_measurements/8r_2t/massif_mpi_omp_8r_2t_500MB.out > ./mpi_omp_measurements/8r_2t/mpi_omp_rank0_8r_2t_500MB_mem_profile.txt

# 4 ranks, 4 threads
mpirun -np 4 --bind-to none -x FI_PROVIDER=tcp -x FI_OFI_RXM_USE_SRX=0 \
bash -c '[[ "$OMPI_COMM_WORLD_RANK" -eq 0 ]] && \
  OMP_NUM_THREADS=4 OMP_DYNAMIC=false OMP_PROC_BIND=spread OMP_PLACES=cores \
  exec valgrind --tool=massif --stacks=yes --massif-out-file=./mpi_omp_measurements/4r_4t/massif_mpi_omp_4r_4t_500MB.out ../cmake-build-release/hpc_mpi_omp 500 || \
  OMP_NUM_THREADS=4 OMP_DYNAMIC=false OMP_PROC_BIND=spread OMP_PLACES=cores exec ../cmake-build-release/hpc_mpi_omp 500'
ms_print ./mpi_omp_measurements/4r_4t/massif_mpi_omp_4r_4t_500MB.out > ./mpi_omp_measurements/4r_4t/mpi_omp_rank0_4r_4t_500MB_mem_profile.txt

# 4 ranks, 2 threads
mpirun -np 4 --bind-to none -x FI_PROVIDER=tcp -x FI_OFI_RXM_USE_SRX=0 \
bash -c '[[ "$OMPI_COMM_WORLD_RANK" -eq 0 ]] && \
  OMP_NUM_THREADS=2 OMP_DYNAMIC=false OMP_PROC_BIND=spread OMP_PLACES=cores \
  exec valgrind --tool=massif --stacks=yes --massif-out-file=./mpi_omp_measurements/4r_2t/massif_mpi_omp_4r_2t_500MB.out ../cmake-build-release/hpc_mpi_omp 500 || \
  OMP_NUM_THREADS=2 OMP_DYNAMIC=false OMP_PROC_BIND=spread OMP_PLACES=cores exec ../cmake-build-release/hpc_mpi_omp 500'
ms_print ./mpi_omp_measurements/4r_2t/massif_mpi_omp_4r_2t_500MB.out > ./mpi_omp_measurements/4r_2t/mpi_omp_rank0_4r_2t_500MB_mem_profile.txt

# 2 ranks, 8 threads
mpirun -np 2 --bind-to none -x FI_PROVIDER=tcp -x FI_OFI_RXM_USE_SRX=0 \
bash -c '[[ "$OMPI_COMM_WORLD_RANK" -eq 0 ]] && \
  OMP_NUM_THREADS=8 OMP_DYNAMIC=false OMP_PROC_BIND=spread OMP_PLACES=cores \
  exec valgrind --tool=massif --stacks=yes --massif-out-file=./mpi_omp_measurements/2r_8t/massif_mpi_omp_2r_8t_500MB.out ../cmake-build-release/hpc_mpi_omp 500 || \
  OMP_NUM_THREADS=8 OMP_DYNAMIC=false OMP_PROC_BIND=spread OMP_PLACES=cores exec ../cmake-build-release/hpc_mpi_omp 500'
ms_print ./mpi_omp_measurements/2r_8t/massif_mpi_omp_2r_8t_500MB.out > ./mpi_omp_measurements/2r_8t/mpi_omp_rank0_2r_8t_500MB_mem_profile.txt

# 2 ranks, 4 threads
mpirun -np 2 --bind-to none -x FI_PROVIDER=tcp -x FI_OFI_RXM_USE_SRX=0 \
bash -c '[[ "$OMPI_COMM_WORLD_RANK" -eq 0 ]] && \
  OMP_NUM_THREADS=4 OMP_DYNAMIC=false OMP_PROC_BIND=spread OMP_PLACES=cores \
  exec valgrind --tool=massif --stacks=yes --massif-out-file=./mpi_omp_measurements/2r_4t/massif_mpi_omp_2r_4t_500MB.out ../cmake-build-release/hpc_mpi_omp 500 || \
  OMP_NUM_THREADS=4 OMP_DYNAMIC=false OMP_PROC_BIND=spread OMP_PLACES=cores exec ../cmake-build-release/hpc_mpi_omp 500'
ms_print ./mpi_omp_measurements/2r_4t/massif_mpi_omp_2r_4t_500MB.out > ./mpi_omp_measurements/2r_4t/mpi_omp_rank0_2r_4t_500MB_mem_profile.txt

##### CUDA #####
echo ">>> Running cuda_measure.sh"
bash cuda_measure.sh

echo "<<< Profiling the memory usage of the CUDA Single-Stream version >>>"
bash ncu_cuda_profile.sh

##### CUDA MS #####
echo ">>> Running cuda_ms_measure.sh"
bash cuda_ms_measure.sh

echo "<<< Profiling the memory usage of the CUDA Multi-Stream version >>>"
bash ncu_cuda_ms_profile.sh # Default = 500MB and 4 streams
bash ncu_cuda_ms_profile.sh 500 2 # 500MB and 2 streams
bash ncu_cuda_ms_profile.sh 500 8 # 500MB and 8 streams

echo ">>> All scripts completed!"
