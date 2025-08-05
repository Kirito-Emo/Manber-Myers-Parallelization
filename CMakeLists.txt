cmake_minimum_required(VERSION 3.15)

# ===== CUDA architecture (required for CMake >= 3.18) =====
set(CMAKE_CUDA_ARCHITECTURES 86)  # RTX 3070 Compute Capability 8.6

project(Manber-Myers-Parallelization LANGUAGES CXX CUDA)

# ===== C++ flags =====
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -Wall")

# ===== CUDA flags =====
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O2 --expt-relaxed-constexpr --use_fast_math")

# ===== Include header directory =====
include_directories(${PROJECT_SOURCE_DIR}/include)

# ===== Sequential build =====
set(SA_ARCS "${PROJECT_SOURCE_DIR}/src/suffix_array.cpp")
add_executable(hpc ${PROJECT_SOURCE_DIR}/src/seq_main.cpp ${SA_ARCS}) # C++ sequential source file location
target_compile_definitions(hpc PRIVATE MODE_SEQ)

# ===== Sequential build with OpenMP =====
find_package(OpenMP REQUIRED)
set(SA_ARCS_OMP "${PROJECT_SOURCE_DIR}/src/suffix_array_openmp.cpp")
add_executable(hpc_omp ${PROJECT_SOURCE_DIR}/src/seq_main.cpp ${SA_ARCS_OMP}) # C++ sequential source file location with OpenMP
target_link_libraries(hpc_omp OpenMP::OpenMP_CXX)
target_compile_definitions(hpc_omp PRIVATE MODE_SEQ_OMP)

# ===== Parallel MPI build =====
set(MPI_HOME "/usr/lib64/openmpi")
set(MPI_CXX_COMPILER "${MPI_HOME}/bin/mpicxx")
find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})
add_executable(hpc_mpi
        ${PROJECT_SOURCE_DIR}/src/mpi_main.cpp
        ${PROJECT_SOURCE_DIR}/src/suffix_array_mpi.cpp
        ${SA_ARCS}
) # C++ parallel source file location with MPI
target_link_libraries(hpc_mpi MPI::MPI_CXX)
target_compile_definitions(hpc_mpi PRIVATE MODE_MPI)

# ===== Parallel MPI + OpenMP hybrid build =====
add_executable(hpc_mpi_omp
        ${PROJECT_SOURCE_DIR}/src/mpi_main.cpp
        ${PROJECT_SOURCE_DIR}/src/suffix_array_mpi_omp.cpp
        ${SA_ARCS}
) # C++ parallel source file location with MPI and OpenMP
target_link_libraries(hpc_mpi_omp MPI::MPI_CXX OpenMP::OpenMP_CXX)
target_compile_definitions(hpc_mpi_omp PRIVATE MODE_MPI_OMP)

# ===== CUDA GPU build =====
enable_language(CUDA)
add_executable(hpc_cuda
        ${PROJECT_SOURCE_DIR}/src/cuda_main.cu
        ${PROJECT_SOURCE_DIR}/src/suffix_array_cuda.cu
        ${PROJECT_SOURCE_DIR}/src/cuda_kernels.cu
        ${PROJECT_SOURCE_DIR}/src/cuda_utils.cu
        ${PROJECT_SOURCE_DIR}/src/cuda_malloc_utils.cu
)
target_include_directories(hpc_cuda PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_compile_definitions(hpc_cuda PRIVATE MODE_CUDA)
set_target_properties(hpc_cuda PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED YES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES 86
)