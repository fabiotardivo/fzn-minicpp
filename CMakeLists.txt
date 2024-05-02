cmake_minimum_required(VERSION 3.22)
project(fzn-minicpp)

# C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
enable_language(CXX)

# CUDA
find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_COMPILER ${CUDAToolkit_NVCC_EXECUTABLE})
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "60;70;80;")
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
string(APPEND CMAKE_CUDA_FLAGS_DEBUG " -G")
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

# External projects
add_subdirectory(libfzn)
add_subdirectory(libminicpp)

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
	global_constraints/table.cpp
	global_constraints/stable_matching.cpp
)
set(GPU_CONSTR_SRC
	gpu_constriants/cumulative.cu
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
add_executable(fzn-minicpp ${FZN_MINICPP_SRC} ${FZN_CONSTR_SRC} ${GLB_CONSTR_SRC} ${GPU_CONSTR_SRC})
target_link_libraries(fzn-minicpp libfzn libminicpp)
