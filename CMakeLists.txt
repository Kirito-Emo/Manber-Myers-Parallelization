cmake_minimum_required(VERSION 3.15)
project(Manber-Myers-Parallelization LANGUAGES CXX)

# C++17, optimization -O2 and warning
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -Wall")

# Header files location
include_directories(${PROJECT_SOURCE_DIR}/include)

# C++ source files location
file(GLOB SOURCES "${PROJECT_SOURCE_DIR}/src/*.cpp")

# Add executable target
add_executable(hpc ${SOURCES})