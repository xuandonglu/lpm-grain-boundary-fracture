cmake_minimum_required(VERSION 3.21)
project(lpm_fracture_solver LANGUAGES C)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(MKL_INTERFACE lp64)
set(MKL_THREADING sequential CACHE STRING "oneMKL internal threading; OpenMP handles particle loops")
find_package(MKL CONFIG REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS C)
add_executable(cohesive_solver src/native/cohesive_solver.c src/native/pardiso_wrapper.c)
target_link_libraries(cohesive_solver PRIVATE MKL::MKL OpenMP::OpenMP_C)
target_compile_definitions(cohesive_solver PRIVATE PARDISO=pardiso_abi_safe)
if(NOT MSVC)
  target_compile_options(cohesive_solver PRIVATE -O3)
  target_link_libraries(cohesive_solver PRIVATE m)
endif()
# The source uses C99 complex arithmetic. On Windows, use Clang or GCC with
# a oneMKL-compatible target rather than the MSVC C compiler.

