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

# Common source files location
set(SA_ARCS "${PROJECT_SOURCE_DIR}/src/suffix_array.cpp")

# C++ sequential source file location
add_executable(hpc ${PROJECT_SOURCE_DIR}/src/seq_main.cpp ${SA_ARCS})

# C++ parallel MPI source file location
find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})
add_executable(hpc_mpi
        ${PROJECT_SOURCE_DIR}/src/mpi_main.cpp
        ${PROJECT_SOURCE_DIR}/src/suffix_array_mpi.cpp
        ${SA_ARCS}
)
target_link_libraries(hpc_mpi MPI::MPI_CXX)