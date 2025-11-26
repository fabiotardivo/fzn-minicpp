cmake_minimum_required(VERSION 3.28)
project(fzn-minicpp CXX)

# C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA
find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_COMPILER ${CUDAToolkit_NVCC_EXECUTABLE})
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "80")
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
string(APPEND CMAKE_CUDA_FLAGS_DEBUG " -g -G")
string(APPEND CMAKE_CUDA_FLAGS_RELWITHDEBINFO " -lineinfo")
# Shhh, it is all good
string(APPEND CMAKE_CUDA_FLAGS " --diag-suppress 177")
string(APPEND CMAKE_CUDA_FLAGS " --diag-suppress 550")
string(APPEND CMAKE_CUDA_FLAGS " --diag-suppress 815")
string(APPEND CMAKE_CUDA_FLAGS " --diag-suppress 830")
string(APPEND CMAKE_CUDA_FLAGS " --diag-suppress 997")
enable_language(CUDA)

# Includes
include_directories(.)
include_directories(${CUDAToolkit_INCLUDE_DIRS})

# Dependency
include(FetchContent)
FetchContent_Declare(
        cxxopts
        GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
        GIT_TAG        master
)
FetchContent_Declare(
        libfzn
        GIT_REPOSITORY https://github.com/fabiotardivo/libfzn.git
        GIT_TAG        main
)
FetchContent_Declare(
        libminicpp
        GIT_REPOSITORY https://github.com/fabiotardivo/libminicpp.git
        GIT_TAG        gpu-dev-smc
)
FetchContent_MakeAvailable(cxxopts)
FetchContent_MakeAvailable(libfzn)
FetchContent_MakeAvailable(libminicpp)

# Sources
set(FZN_CONSTR_SRC
	fzn_constraints/bool_array.cpp
	fzn_constraints/bool_bin.cpp
	fzn_constraints/bool_misc.cpp
	fzn_constraints/int_array.cpp
	fzn_constraints/int_bin.cpp
	fzn_constraints/int_lin.cpp
	fzn_constraints/int_misc.cpp
	fzn_constraints/int_tern.cpp
)
set(GLB_CONSTR_SRC
	global_constraints/cumulative.cpp
	global_constraints/bin_packing.cpp
	global_constraints/stable_matching.cpp
)
set(GPU_CONSTR_SRC
    gpu_constraints/cumulative.cu
    gpu_constraints/bin_packing.cu
    gpu_constraints/all_different.cu
    gpu_constraints/circuit.cu
    gpu_constraints/stable_matching.cu
)
set(GPU_SRC
    libfca/BitMatrix.cu
    libfca/SqrBitMatrix.cu
)
set (FZN_MINICPP_SRC
	fzn_cli_helper.cpp
	fzn_constraints_helper.cpp
	fzn_output_helper.cpp
	fzn_search_helper.cpp
	fzn_statistics_helper.cpp
	fzn_variables_helper.cpp
	main.cpp
)

# Executable
add_executable(fzn-minicpp ${FZN_MINICPP_SRC} ${FZN_CONSTR_SRC} ${GLB_CONSTR_SRC} ${GPU_CONSTR_SRC} ${GPU_SRC})
target_link_libraries(fzn-minicpp cxxopts libfzn libminicpp)
