cmake_minimum_required(VERSION 3.24.4)

project(oc-cuda-arrnd-header
    VERSION 1.0.0
    DESCRIPTION "An N dimensional array header implementation"
    HOMEPAGE_URL "https://github.com/oren12321/oc-cuda-arrnd-header")

enable_language(CUDA)

if (NOT WIN32)
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -Wall")
    endif()
endif()

include(GNUInstallDirs)

set(COMPUTE_CAP "-gencode=arch=compute_50,code=\"sm_50,compute_50\"")
add_library(${PROJECT_NAME} STATIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/oc/cuda-arrnd-header.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/include/oc/cuda-arrnd-header.h)
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})
target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${COMPUTE_CAP}>)

include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.11.0
    FIND_PACKAGE_ARGS NAMES GTest)
FetchContent_MakeAvailable(googletest)

FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.6.1
    FIND_PACKAGE_ARGS)
FetchContent_MakeAvailable(benchmark)

file(GLOB_RECURSE TESTS_SRCS CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/tests/*.cpp")
add_executable(${PROJECT_NAME}_tests ${TESTS_SRCS})
target_include_directories(${PROJECT_NAME}_tests PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(${PROJECT_NAME}_tests ${PROJECT_NAME}::${PROJECT_NAME} GTest::gtest GTest::gtest_main)
set_property(TARGET ${PROJECT_NAME}_tests PROPERTY CXX_STANDARD 17)

file(GLOB_RECURSE BENCHMARK_SRCS CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/benchmark/*.cpp")
add_executable(${PROJECT_NAME}_benchmark ${BENCHMARK_SRCS})
target_include_directories(${PROJECT_NAME}_benchmark PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(${PROJECT_NAME}_benchmark ${PROJECT_NAME}::${PROJECT_NAME} benchmark::benchmark)
set_property(TARGET ${PROJECT_NAME}_benchmark PROPERTY CXX_STANDARD 17)

add_custom_target(run_oc-cuda-arrnd-header_tests COMMAND ${PROJECT_NAME}_tests DEPENDS ${PROJECT_NAME}_tests)
add_custom_target(run_oc-cuda-arrnd-header_benchmark COMMAND ${PROJECT_NAME}_benchmark DEPENDS ${PROJECT_NAME}_benchmark)
