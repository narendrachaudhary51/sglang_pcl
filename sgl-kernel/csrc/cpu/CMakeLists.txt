cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(sgl_kernel)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python COMPONENTS Interpreter Development.Module ${SKBUILD_SABI_COMPONENT} REQUIRED)

execute_process(
    COMMAND ${Python_EXECUTABLE}
            -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_PY_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS ${TORCH_PY_PREFIX})
list(APPEND CMAKE_PREFIX_PATH ${TORCH_PY_PREFIX}/Torch)
find_package(Torch REQUIRED)

include_directories(
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_INSTALL_PREFIX}/include
    ${Python_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../csrc
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Build all cpp files in current and sub dirs
file(GLOB_RECURSE SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

# Exclude all arch dependent files, then add back only files for this arch
set(ARCH_DIRS "x86_64|aarch64|ppc64")
list(FILTER SOURCES EXCLUDE REGEX "^${CMAKE_CURRENT_SOURCE_DIR}/(${ARCH_DIRS})/")

# Platform-specific source and library directory
set(MY_ARCH_DIR "")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(PLAT_LIB_DIR "/usr/lib/x86_64-linux-gnu")
    set(MY_ARCH_DIR "x86_64")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(PLAT_LIB_DIR "/usr/lib/aarch64-linux-gnu")
    set(MY_ARCH_DIR "aarch64")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "ppc64le|ppc64")
    set(PLAT_LIB_DIR "/usr/lib/powerpc64le-linux-gnu")
    set(MY_ARCH_DIR "ppc64")
else()
    set(PLAT_LIB_DIR "/usr/lib/${CMAKE_SYSTEM_PROCESSOR}-linux-gnu")
endif()
link_directories(${PLAT_LIB_DIR})
if(MY_ARCH_DIR)
    file(GLOB_RECURSE ARCH_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/${MY_ARCH_DIR}/*.cpp")
    list(APPEND SOURCES ${ARCH_SOURCES})
endif()

# Conda library path support
if(DEFINED ENV{CONDA_PREFIX})
    set(CONDA_LIB_DIR "$ENV{CONDA_PREFIX}/lib")
    message(STATUS "Using Conda lib dir: ${CONDA_LIB_DIR}")
    link_directories(${CONDA_LIB_DIR})
    set(CONDA_INCLUDE_DIR "$ENV{CONDA_PREFIX}/include")
    include_directories(${CONDA_INCLUDE_DIR})

    # Look for libnuma in Conda's lib directory
    find_library(NUMA_LIB numa HINTS "${CONDA_LIB_DIR}")
    if(NUMA_LIB)
        message(STATUS "Found libnuma: ${NUMA_LIB}")
    else()
        message(FATAL_ERROR "libnuma not found in Conda environment at ${CONDA_LIB_DIR}\n"
                            "Please install it using: conda install libnuma numactl\n")
    endif()
else()
    if(DEFINED ENV{VIRTUAL_ENV})
        set(VENV_LIB_DIR "$ENV{VIRTUAL_ENV}/lib")
        message(STATUS "Using venv lib dir: ${VENV_LIB_DIR}")
        link_directories(${VENV_LIB_DIR})
        set(VENV_INCLUDE_DIR "$ENV{VIRTUAL_ENV}/include")
        include_directories(${VENV_INCLUDE_DIR})
    endif()
    # Look for libnuma in system env paths
    find_library(NUMA_LIB numa)
    if(NUMA_LIB)
        message(STATUS "Found libnuma: ${NUMA_LIB}")
    else()
        message(FATAL_ERROR "libnuma not found in system environment\n"
                            "Please install it using: apt-get install libnuma numactl\n")
    endif()
endif()


# These kernels still rely on x86-specific AMX/AVX512 implementations.
# Keep them out of Arm64 bootstrap builds until native Arm paths land.
set(SGLANG_CPU_X86_ONLY_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/gemm_int4.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/moe.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/moe_fp8.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/moe_int4.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/moe_int8.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/qkv_proj.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mamba/conv.cpp
)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    add_compile_definitions(SGLANG_CPU_ARM64_SKIP_X86_ONLY_OPS)
    list(REMOVE_ITEM SOURCES ${SGLANG_CPU_X86_ONLY_SOURCES})
endif()

if(NOT DEFINED ENV{SGLANG_CPU_FP8_CVT_FTZ})
    set(ENV{SGLANG_CPU_FP8_CVT_FTZ} "1")
endif()

if("$ENV{SGLANG_CPU_FP8_CVT_FTZ}" STREQUAL "1")
    message(STATUS "Enabling macro: SGLANG_CPU_FP8_CVT_FTZ")
    add_compile_definitions(SGLANG_CPU_FP8_CVT_FTZ)
endif()

if(MY_ARCH_DIR STREQUAL "x86_64")
    add_compile_options(
        -O3
        -Wno-unknown-pragmas
        -march=x86-64-v4
        -mavx512bf16
        -mavx512vnni
        -mamx-tile
        -mamx-bf16
        -mamx-int8
        -fopenmp
    )
else()
    add_compile_options(
        -O3
        -Wno-unknown-pragmas
        -march=native
        -fopenmp
    )
endif()

Python_add_library(common_ops MODULE USE_SABI ${SKBUILD_SABI_VERSION} WITH_SOABI ${SOURCES})
target_link_libraries(common_ops PRIVATE ${TORCH_LIBRARIES} ${NUMA_LIB})
target_include_directories(common_ops PRIVATE ${TORCH_INCLUDE_DIRS})

# ============================================================================
# Optional: libxsmm GEMM backend for MXFP4 (e2m1 weight + e8m0 scale) /
# BF16 activations.
# ----------------------------------------------------------------------------
# Enabled when LIBXSMM_ROOT is set in the environment and the build target is
# x86_64. LIBXSMM_ROOT must point at an install (or build) directory that
# contains include/libxsmm.h and lib/libxsmm.so (or libxsmm.a). When the
# variable is unset, the mxfp4 kernels build with only the default oneDNN/AMX
# tinygemm path and the libxsmm toggle is a no-op.
# ============================================================================
if(DEFINED ENV{LIBXSMM_ROOT} AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(LIBXSMM_ROOT "$ENV{LIBXSMM_ROOT}")
    message(STATUS "libxsmm: enabling mxfp4 GEMM backend (LIBXSMM_ROOT=${LIBXSMM_ROOT})")
    target_compile_definitions(common_ops PRIVATE SGL_KERNEL_HAS_LIBXSMM)
    target_include_directories(common_ops PRIVATE "${LIBXSMM_ROOT}/include")
    find_library(LIBXSMM_LIB
        NAMES xsmm
        HINTS "${LIBXSMM_ROOT}/lib" "${LIBXSMM_ROOT}/lib64"
        NO_DEFAULT_PATH)
    if(NOT LIBXSMM_LIB)
        message(FATAL_ERROR "libxsmm: could not find libxsmm in ${LIBXSMM_ROOT}/lib")
    endif()
    message(STATUS "libxsmm: using libxsmm at ${LIBXSMM_LIB}")
    target_link_libraries(common_ops PRIVATE ${LIBXSMM_LIB})
    # AMX/AVX512 are required by the GEMM kernels libxsmm JITs at runtime;
    # the x86 -march/-mamx flags set globally above supply them on the host.
else()
    message(STATUS "libxsmm: disabled (set LIBXSMM_ROOT to enable mxfp4 GEMM backend)")
endif()

install(TARGETS common_ops
    LIBRARY DESTINATION sgl_kernel
)
