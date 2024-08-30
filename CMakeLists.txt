cmake_minimum_required(VERSION 3.22)
project(fzn-minicpp)

# C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
enable_language(CXX)

# Includes
include_directories(.)

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
	global_constraints/bin_packing.cpp
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
add_executable(fzn-minicpp ${FZN_MINICPP_SRC} ${FZN_CONSTR_SRC} ${GLB_CONSTR_SRC})
target_link_libraries(fzn-minicpp libfzn libminicpp)
