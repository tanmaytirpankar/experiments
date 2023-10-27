cmake_minimum_required(VERSION 3.22)
project(experiments)

set(CMAKE_CXX_STANDARD 14)

add_executable(CG_half CG/cg_half.cpp)
add_executable(CG_float CG/cg_float.cpp)
add_executable(CG_double CG/cg_double.cpp)
add_executable(CG_combined_float_double CG/cg_combined_float_double.cpp)

target_link_libraries(CG_half -lmpfr)
