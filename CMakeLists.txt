cmake_minimum_required(VERSION 3.28)
project(fzn-minicpp CXX)

# Build
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

# C++
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Includes
include_directories(.)

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
        GIT_TAG        main
)
set(ONNXRUNTIME_VERSION 1.23.1)
set(ONNXRUNTIME_PLATFORM linux-x64-gpu)
set(ONNXRUNTIME_URL
        https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-${ONNXRUNTIME_PLATFORM}-${ONNXRUNTIME_VERSION}.tgz
)
FetchContent_Declare(
        onnxruntime
        URL ${ONNXRUNTIME_URL}
)
FetchContent_MakeAvailable(cxxopts)
FetchContent_MakeAvailable(libfzn)
FetchContent_MakeAvailable(libminicpp)
FetchContent_MakeAvailable(onnxruntime)

add_library(onnxruntime INTERFACE)
target_include_directories(onnxruntime INTERFACE ${onnxruntime_SOURCE_DIR}/include)
target_link_libraries(onnxruntime INTERFACE ${onnxruntime_SOURCE_DIR}/lib/libonnxruntime.so)

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
)
set (FZN_MINICPP_SRC
	fzn_cli_helper.cpp
	fzn_constraints_helper.cpp
	fzn_output_helper.cpp
	fzn_search_helper.cpp
	fzn_statistics_helper.cpp
	fzn_variables_helper.cpp
)
set (ML_SRC
     ml/OnnxHandler.cpp
)


# Executables
add_executable(fzn-minicpp main.cpp ${FZN_MINICPP_SRC} ${FZN_CONSTR_SRC} ${GLB_CONSTR_SRC})
target_link_libraries(fzn-minicpp cxxopts libfzn libminicpp)

add_executable(fzn-minicpp-ml-solver main_ml_solver.cpp ${FZN_MINICPP_SRC} ${FZN_CONSTR_SRC} ${GLB_CONSTR_SRC} ${ML_SRC})
target_include_directories(fzn-minicpp-ml-solver PRIVATE ${onnxruntime_SOURCE_DIR}/include ml)
target_link_libraries(fzn-minicpp-ml-solver cxxopts libfzn libminicpp
        ${onnxruntime_SOURCE_DIR}/lib/libonnxruntime.so
        ${onnxruntime_SOURCE_DIR}/lib/libonnxruntime_providers_cuda.so
        ${onnxruntime_SOURCE_DIR}/lib/libonnxruntime_providers_shared.so)

add_executable(fzn-minicpp-ml-sampler main_ml_sampler.cpp ${FZN_MINICPP_SRC} ${FZN_CONSTR_SRC} ${GLB_CONSTR_SRC})
target_link_libraries(fzn-minicpp-ml-sampler cxxopts libfzn libminicpp)
