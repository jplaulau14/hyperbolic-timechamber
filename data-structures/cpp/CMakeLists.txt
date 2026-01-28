cmake_minimum_required(VERSION 3.14)
project(static_array_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(static_array_test tests/static_array_test.cpp)
target_include_directories(static_array_test PRIVATE src)

add_executable(dynamic_array_test tests/dynamic_array_test.cpp)
target_include_directories(dynamic_array_test PRIVATE src)
