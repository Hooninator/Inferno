cmake_minimum_required(VERSION 3.3)
project(Inferno VERSION 1.0.0 LANGUAGES CXX CUDA)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)


# Main library
add_library(Inferno src/main.cpp)


# Compiler options
target_compile_options(Inferno PRIVATE -Wno-deprecated-declarations)

# Include directories
target_include_directories(Inferno PRIVATE include)


# Find CUDA and UPCXX
find_package(CUDA)
find_package(UPCXX)

# Link UPCXX
target_link_libraries(Inferno PUBLIC UPCXX::upcxx)


# Driver
add_executable(run_inferno tests/run_inferno.cpp)
target_link_libraries(run_inferno PUBLIC UPCXX::upcxx)
target_link_libraries(run_inferno PUBLIC Inferno)
target_include_directories(run_inferno PRIVATE include)


