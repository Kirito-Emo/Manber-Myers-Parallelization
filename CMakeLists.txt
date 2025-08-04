cmake_minimum_required(VERSION 3.15)
project(Manber-Myers-Parallelization LANGUAGES CXX)

# C++17, optimization -O2 and warning
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -Wall")

# Header files
include_directories(
        ${PROJECT_SOURCE_DIR}/include
)

# ===== Sequential build (no OpenMP) =====
set(SA_ARCS "${PROJECT_SOURCE_DIR}/src/suffix_array.cpp")
add_executable(hpc ${PROJECT_SOURCE_DIR}/src/seq_main.cpp ${SA_ARCS}) # C++ sequential source file location

# ===== Sequential build with OpenMP =====
find_package(OpenMP REQUIRED)
set(SA_ARCS_OMP "${PROJECT_SOURCE_DIR}/src/suffix_array_openmp.cpp")
add_executable(hpc_omp ${PROJECT_SOURCE_DIR}/src/seq_main.cpp ${SA_ARCS_OMP}) # C++ sequential source file location with OpenMP
target_link_libraries(hpc_omp OpenMP::OpenMP_CXX)

# ===== Parallel MPI build =====
find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})
add_executable(hpc_mpi
        ${PROJECT_SOURCE_DIR}/src/mpi_main.cpp
        ${PROJECT_SOURCE_DIR}/src/suffix_array_mpi.cpp
        ${SA_ARCS}
) # C++ parallel source file location with MPI
target_link_libraries(hpc_mpi MPI::MPI_CXX)

# ===== Parallel MPI + OpenMP build =====
add_executable(hpc_mpi_omp
        ${PROJECT_SOURCE_DIR}/src/mpi_main.cpp
        ${PROJECT_SOURCE_DIR}/src/suffix_array_mpi.cpp
        ${SA_ARCS}
) # C++ parallel source file location with MPI and OpenMP
target_link_libraries(hpc_mpi_omp MPI::MPI_CXX OpenMP::OpenMP_CXX)