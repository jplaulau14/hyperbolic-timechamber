cmake_minimum_required(VERSION 3.14)
project(static_array_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(static_array_test tests/static_array_test.cpp)
target_include_directories(static_array_test PRIVATE src)

add_executable(dynamic_array_test tests/dynamic_array_test.cpp)
target_include_directories(dynamic_array_test PRIVATE src)

add_executable(linked_list_test tests/linked_list_test.cpp)
target_include_directories(linked_list_test PRIVATE src)

add_executable(stack_test tests/stack_test.cpp)
target_include_directories(stack_test PRIVATE src)

add_executable(queue_test tests/queue_test.cpp)
target_include_directories(queue_test PRIVATE src)

add_executable(hash_map_test tests/hash_map_test.cpp)
target_include_directories(hash_map_test PRIVATE src)

add_executable(binary_heap_test tests/binary_heap_test.cpp)
target_include_directories(binary_heap_test PRIVATE src)

add_executable(binary_search_tree_test tests/binary_search_tree_test.cpp)
target_include_directories(binary_search_tree_test PRIVATE src)
