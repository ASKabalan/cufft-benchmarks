cmake_minimum_required(VERSION 3.18)
project(CUFFT_Bench LANGUAGES CXX CUDA)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find the CUDA toolkit
find_package(CUDAToolkit REQUIRED)

# Include directories
include_directories(include)

# Set source files
set(SOURCES src/cufft_bench.cu)

# Add the executable
add_executable(cufft_bench ${SOURCES})

# Link against CUDA and cuFFT
target_link_libraries(cufft_bench CUDA::cufft)

# Include directories for the executable
target_include_directories(cufft_bench PRIVATE include)
