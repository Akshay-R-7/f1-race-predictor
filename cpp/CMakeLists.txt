cmake_minimum_required(VERSION 3.18)
project(racesim CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Threads REQUIRED)
add_library(racesim_core STATIC src/racesim.cpp)
target_include_directories(racesim_core PUBLIC include)
target_link_libraries(racesim_core PUBLIC Threads::Threads)

add_executable(bench bench.cpp)
target_link_libraries(bench racesim_core)

# Python module (needs: pip install pybind11)
option(BUILD_PYTHON "Build the Python module" ON)
if(BUILD_PYTHON)
  # Find whichever Python is active (works with python, python3, py, and virtual environments)
  find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
  execute_process(COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
                  OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(_racesim src/bindings.cpp)
  target_link_libraries(_racesim PRIVATE racesim_core)
  # Put the module straight into the f1sim package (also on Visual Studio, which adds Release/ folders)
  set(OUT ${CMAKE_SOURCE_DIR}/../f1sim)
  set_target_properties(_racesim PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${OUT}
      LIBRARY_OUTPUT_DIRECTORY_RELEASE ${OUT} LIBRARY_OUTPUT_DIRECTORY_DEBUG ${OUT}
      RUNTIME_OUTPUT_DIRECTORY_RELEASE ${OUT})
endif()

# Tests with GoogleTest (downloaded on first configure)
option(BUILD_TESTS "Build the C++ tests" ON)
if(BUILD_TESTS)
  find_package(GTest QUIET)          # use an installed GoogleTest if there is one
  if(NOT GTest_FOUND)                # otherwise download it once
    include(FetchContent)
    FetchContent_Declare(googletest
      URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
  endif()
  enable_testing()
  add_executable(racesim_tests tests/test_racesim.cpp)
  target_link_libraries(racesim_tests racesim_core GTest::gtest_main)
  include(GoogleTest)
  gtest_discover_tests(racesim_tests)
endif()
