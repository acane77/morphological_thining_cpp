cmake_minimum_required(VERSION 3.16)
project(morphological_thining_cpp)

set(CMAKE_CXX_STANDARD 11)

file(GLOB SRCS *.cpp)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_executable(morphological_thining_cpp ${SRCS})
