# CMakeLists.txt for Busy GPU
cmake_minimum_required(VERSION 3.18)

# Project information
project(BusyGPU VERSION 1.0.0 LANGUAGES CXX CUDA)

# Version information
set(VERSION "1.0.0")
string(TIMESTAMP BUILD_DATE "%Y-%m-%d")
cmake_host_system_information(RESULT BUILD_OS QUERY OS_NAME)
cmake_host_system_information(RESULT BUILD_ARCH QUERY OS_PLATFORM)

# GPU architectures (SM versions)
# SM 5.2: Maxwell (GTX 9xx, Titan X)
# SM 6.0: Pascal (P100)
# SM 6.1: Pascal (GTX 10xx, Titan Xp)
# SM 7.0: Volta (V100)
# SM 7.5: Turing (RTX 20xx, Titan RTX)
# SM 8.0: Ampere (A100)
# SM 8.6: Ampere (RTX 30xx, A40)
# SM 8.9: Ada Lovelace (RTX 40xx, L40)
# SM 9.0: Hopper (H100)
set(GPU_ARCHS_LIST "52" "60" "61" "70" "75" "80" "86" "89" "90")

# Generate GPU_ARCHS_STR for version info
string(REPLACE ";" "," GPU_ARCHS_STR "${GPU_ARCHS_LIST}")

# Set CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES ${GPU_ARCHS_LIST})

# CUDA compilation flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -use_fast_math")
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)

# Source files
set(SOURCES busy_gpu.cu)

# Executable target
add_executable(busy_gpu ${SOURCES})

# Set C++ standard
set_target_properties(busy_gpu PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 11
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION ON
)

# Add compile definitions
target_compile_definitions(busy_gpu PRIVATE
    VERSION="${VERSION}"
    BUILD_DATE="${BUILD_DATE}"
    BUILD_OS="${BUILD_OS}"
    BUILD_ARCH="${BUILD_ARCH}"
    GPU_ARCHS="${GPU_ARCHS_STR}"
)

# Find CUDA Toolkit
find_package(CUDAToolkit REQUIRED)
target_link_libraries(busy_gpu CUDA::cudart_static)

# Install target
install(TARGETS busy_gpu
    RUNTIME DESTINATION bin
)

# Custom targets
add_custom_target(info
    COMMAND nvidia-smi
    COMMENT "Showing GPU information"
)

# Print configuration information
message(STATUS "Busy GPU Configuration:")
message(STATUS "  Version: ${VERSION}")
message(STATUS "  Build Date: ${BUILD_DATE}")
message(STATUS "  Build OS: ${BUILD_OS}")
message(STATUS "  Build Architecture: ${BUILD_ARCH}")
message(STATUS "  CUDA Architectures: ${GPU_ARCHS_STR}")
message(STATUS "  CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "  CUDA Version: ${CUDAToolkit_VERSION}")
