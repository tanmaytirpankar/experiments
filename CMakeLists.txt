cmake_minimum_required(VERSION 3.22)
project(experiments)

set(CMAKE_CXX_STANDARD 14)

add_executable(CG_float CG/cg_float.cpp)
add_executable(CG_double CG/cg_double.cpp)
add_executable(CG_combined CG/cg_combined.cpp)
