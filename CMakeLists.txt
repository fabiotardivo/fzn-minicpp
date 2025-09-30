cmake_minimum_required(VERSION 3.28)
project(fzn-minicpp CXX)

# C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Machine Learning
find_package(pybind11 CONFIG REQUIRED)

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
)
set (FZN_MINICPP_SRC
	fzn_cli_helper.cpp
	fzn_constraints_helper.cpp
	fzn_output_helper.cpp
	fzn_search_helper.cpp
	fzn_statistics_helper.cpp
	fzn_variables_helper.cpp
)

# Executables
add_executable(fzn-minicpp main.cpp ${FZN_MINICPP_SRC} ${FZN_CONSTR_SRC} ${GLB_CONSTR_SRC})
target_link_libraries(fzn-minicpp cxxopts libfzn libminicpp libminicpp pybind11::embed)

add_executable(fzn-minicpp-ml-solver main_ml_solver.cpp ${FZN_MINICPP_SRC} ${FZN_CONSTR_SRC} ${GLB_CONSTR_SRC})
target_link_libraries(fzn-minicpp-ml-solver cxxopts libfzn libminicpp pybind11::embed)

add_executable(fzn-minicpp-ml-sampler main_ml_sampler.cpp ${FZN_MINICPP_SRC} ${FZN_CONSTR_SRC} ${GLB_CONSTR_SRC})
target_link_libraries(fzn-minicpp-ml-sampler cxxopts libfzn libminicpp pybind11::embed)
