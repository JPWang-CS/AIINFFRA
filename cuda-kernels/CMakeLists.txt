cmake_minimum_required(VERSION 3.18)
project(cuda-kernels LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90")

# Common headers
include_directories(${CMAKE_SOURCE_DIR}/include)

# Utility: CUDA error checking + tensor helpers
add_library(kernel_utils STATIC ${CMAKE_SOURCE_DIR}/include/cuda_utils.cu)
target_include_directories(kernel_utils PUBLIC ${CMAKE_SOURCE_DIR}/include)

# ---- Kernels ----

# Vector Add (warm up)
add_executable(vector_add vector_add.cu)
target_link_libraries(vector_add kernel_utils)

# GEMM
add_executable(gemm gemm/gemm.cu)
target_link_libraries(gemm kernel_utils)

# Softmax
add_executable(softmax softmax/softmax.cu)
target_link_libraries(softmax kernel_utils)

# LayerNorm
add_executable(layernorm layernorm/layernorm.cu)
target_link_libraries(layernorm kernel_utils)

# Flash Attention
add_executable(flash_attn flash_attention/flash_attn.cu)
target_link_libraries(flash_attn kernel_utils)
