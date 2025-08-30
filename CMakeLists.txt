cmake_minimum_required(VERSION 3.18)

# ===== CUDA architecture (required for CMake >= 3.18) =====
set(CMAKE_CUDA_COMPILER /usr/local/cuda-13/bin/nvcc)
set(CMAKE_CUDA_ARCHITECTURES 86)  # RTX 3070 Compute Capability 8.6

project(Manber-Myers-Parallelization LANGUAGES CXX CUDA)

# ===== C++ flags =====
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ===== CUDA flags =====
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ===== Include header directory =====
set(PROJ_INC ${PROJECT_SOURCE_DIR}/include)

# ===== Helper =====
set(HOST_WARN -Wall)
set(HOST_REL -O3 -DNDEBUG)
set(HOST_DBG -O0 -g)

set(CUDA_COMMON --expt-relaxed-constexpr)
set(CUDA_FAST --use_fast_math)
set(CUDA_DBG -G -lineinfo)

# ===== Sequential build =====
add_executable(hpc
        ${PROJECT_SOURCE_DIR}/src/seq_main.cpp
        ${PROJECT_SOURCE_DIR}/src/suffix_array.cpp
) # C++ sequential source file location
target_include_directories(hpc PRIVATE ${PROJ_INC})
target_compile_options(hpc PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:
        ${HOST_WARN}
        $<$<CONFIG:Release>:${HOST_REL}>
        $<$<CONFIG:Debug>:${HOST_DBG}>
        >
)
target_compile_definitions(hpc PRIVATE MODE_SEQ)

# ===== Sequential build with OpenMP =====
find_package(OpenMP REQUIRED)
add_executable(hpc_omp
        ${PROJECT_SOURCE_DIR}/src/omp_main.cpp
        ${PROJECT_SOURCE_DIR}/src/suffix_array_omp.cpp
) # C++ sequential source file location with OpenMP
target_include_directories(hpc_omp PRIVATE ${PROJ_INC})
target_link_libraries(hpc_omp PRIVATE OpenMP::OpenMP_CXX)
target_compile_options(hpc_omp PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:
        ${HOST_WARN}
        $<$<CONFIG:Release>:${HOST_REL}>
        $<$<CONFIG:Debug>:${HOST_DBG}>
        >
)
target_compile_definitions(hpc_omp PRIVATE MODE_SEQ_OMP)

# ===== Parallel MPI build =====
find_package(MPI REQUIRED)
add_executable(hpc_mpi
        ${PROJECT_SOURCE_DIR}/src/mpi_main.cpp
        ${PROJECT_SOURCE_DIR}/src/suffix_array_mpi.cpp
        ${PROJECT_SOURCE_DIR}/src/suffix_array.cpp
) # C++ parallel source file location with MPI
target_include_directories(hpc_mpi PRIVATE ${PROJ_INC})
target_link_libraries(hpc_mpi PRIVATE MPI::MPI_CXX)
target_compile_options(hpc_mpi PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:
        ${HOST_WARN}
        $<$<CONFIG:Release>:${HOST_REL}>
        $<$<CONFIG:Debug>:${HOST_DBG}>
        >
)
target_compile_definitions(hpc_mpi PRIVATE MODE_MPI)

# ===== Parallel MPI + OpenMP hybrid build =====
add_executable(hpc_mpi_omp
        ${PROJECT_SOURCE_DIR}/src/mpi_omp_main.cpp
        ${PROJECT_SOURCE_DIR}/src/suffix_array_mpi_omp.cpp
        ${PROJECT_SOURCE_DIR}/src/suffix_array.cpp
) # C++ parallel source file location with MPI and OpenMP
target_include_directories(hpc_mpi_omp PRIVATE ${PROJ_INC})
target_link_libraries(hpc_mpi_omp PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)
target_compile_options(hpc_mpi_omp PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:
        ${HOST_WARN}
        $<$<CONFIG:Release>:${HOST_REL}>
        $<$<CONFIG:Debug>:${HOST_DBG}>
        >
)
target_compile_definitions(hpc_mpi_omp PRIVATE MODE_MPI_OMP)

# ===== CUDA GPU build =====
add_executable(hpc_cuda
        ${PROJECT_SOURCE_DIR}/src/cuda_main.cpp
        ${PROJECT_SOURCE_DIR}/src/suffix_array_cuda.cu
        ${PROJECT_SOURCE_DIR}/src/suffix_array.cpp
)
target_include_directories(hpc_cuda PRIVATE ${PROJ_INC})
set_target_properties(hpc_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)
target_compile_options(hpc_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:
        ${HOST_WARN}
        $<$<CONFIG:Release>:${HOST_REL}>
        $<$<CONFIG:Debug>:${HOST_DBG}>
        >
        $<$<COMPILE_LANGUAGE:CUDA>:
        ${CUDA_COMMON} ${CUDA_FAST}
        $<$<CONFIG:Debug>:${CUDA_DBG}>
        >
)
target_compile_definitions(hpc_cuda PRIVATE MODE_CUDA)

# ===== Multi-Stream CUDA GPU build =====
add_executable(hpc_cuda_ms
        ${PROJECT_SOURCE_DIR}/src/cuda_ms_main.cpp
        ${PROJECT_SOURCE_DIR}/src/suffix_array_cuda_ms.cu
        ${PROJECT_SOURCE_DIR}/src/suffix_array.cpp
)
target_include_directories(hpc_cuda_ms PRIVATE ${PROJ_INC})
set_target_properties(hpc_cuda_ms PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)
target_compile_options(hpc_cuda_ms PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:
        ${HOST_WARN}
        $<$<CONFIG:Release>:${HOST_REL}>
        $<$<CONFIG:Debug>:${HOST_DBG}>
        >
        $<$<COMPILE_LANGUAGE:CUDA>:
        ${CUDA_COMMON} ${CUDA_FAST}
        $<$<CONFIG:Debug>:${CUDA_DBG}>
        >
)
target_compile_definitions(hpc_cuda_ms PRIVATE MODE_CUDA_MS)