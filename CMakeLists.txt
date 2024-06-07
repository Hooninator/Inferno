
cmake_minimum_required(VERSION 3.3)
project(Inferno VERSION 1.0.0 LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -rdc=true") 
set(CMAKE_CUDA_ARCHITECTURES "80")

#cmake_policy(SET CMP0074 OLD)


# Main library
add_library(Inferno src/main.cu)


# Compiler options
target_compile_options(Inferno PRIVATE -Wno-deprecated-declarations)

# Find MPI
find_package(MPI)

# Find CUDA and NVSHMEM 
find_package(CUDA)

set(NVSHMEM_HOME "/global/common/software/nersc9/nvshmem/2.11.0")

# Include directories
target_include_directories(Inferno PUBLIC include)
target_include_directories(Inferno PUBLIC "${NVSHMEM_HOME}/include")

# Link MPI
target_link_libraries(Inferno PUBLIC MPI::MPI_CXX)

# Link NVSHMEM 
target_link_libraries(Inferno PUBLIC "-L${NVSHMEM_HOME}/lib" -lnvshmem -lrt)
set_target_properties(Inferno PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# Link CUDA libs
target_link_libraries(Inferno PUBLIC "-L/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/math_libs/lib64/" -lcublas -lcusparse -lcuda -lcudart)

# Driver
add_executable(run_inferno tests/run_inferno.cu)

set_target_properties(run_inferno PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_link_libraries(run_inferno PUBLIC Inferno)
target_link_libraries(run_inferno PUBLIC MPI::MPI_CXX)
target_link_libraries(run_inferno PUBLIC "-L${NVSHMEM_HOME}/lib"  -lnvshmem -lnvidia-ml -lrt -lcublas -lcusparse -lcuda -lcudart)

target_include_directories(run_inferno PRIVATE include)
target_include_directories(run_inferno PUBLIC "${NVSHMEM_HOME}/include")


