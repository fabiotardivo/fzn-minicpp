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

# Dependencies
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


set(LIBTORCH_DEVICE "cu128")
set(LIBTORCH_VERSION 2.10.0)
set(LIBTORCH_URL
        https://download.pytorch.org/libtorch/${LIBTORCH_DEVICE}/libtorch-shared-with-deps-${LIBTORCH_VERSION}%2B${LIBTORCH_DEVICE}.zip
)

FetchContent_Declare(
        libtorch
        URL ${LIBTORCH_URL}
)

FetchContent_MakeAvailable(cxxopts libfzn libminicpp libtorch)

# Find Torch package
set(CMAKE_PREFIX_PATH ${libtorch_SOURCE_DIR})
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

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

set(FZN_MINICPP_SRC
        fzn_cli_helper.cpp
        fzn_constraints_helper.cpp
        fzn_output_helper.cpp
        fzn_search_helper.cpp
        fzn_statistics_helper.cpp
        fzn_variables_helper.cpp
)

set(ML_SRC
        ml/TorchHandler.cpp
)

# Executables
add_executable(fzn-minicpp main.cpp ${FZN_MINICPP_SRC} ${FZN_CONSTR_SRC} ${GLB_CONSTR_SRC})
target_link_libraries(fzn-minicpp cxxopts libfzn libminicpp)

add_executable(fzn-minicpp-ml-solver main_ml_solver.cpp ${FZN_MINICPP_SRC} ${FZN_CONSTR_SRC} ${GLB_CONSTR_SRC} ${ML_SRC})
target_include_directories(fzn-minicpp-ml-solver PRIVATE ml)
target_link_libraries(fzn-minicpp-ml-solver cxxopts libfzn libminicpp "${TORCH_LIBRARIES}")

add_executable(fzn-minicpp-ml-sampler main_ml_sampler.cpp ${FZN_MINICPP_SRC} ${FZN_CONSTR_SRC} ${GLB_CONSTR_SRC})
target_link_libraries(fzn-minicpp-ml-sampler cxxopts libfzn libminicpp)