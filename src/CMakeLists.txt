cmake_minimum_required(VERSION 3.10)

# set the project name
project(neural-net_bgl)

set(CMAKE_CXX_STANDARD_LIBRARIES "-lboost_serialization")
set(SOURCE nn_bgl.cpp neural-net_bgl.cpp)
set(HEADER ./nn_bgl.cpp.h)


include_directories(.)

# add the executable
add_executable(neural-net_bgl ${SOURCE})
add_executable(makeTrainingSamples makeTrainingSamples.cpp)
