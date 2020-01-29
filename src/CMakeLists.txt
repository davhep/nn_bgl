cmake_minimum_required(VERSION 3.10)

# set the project name
project(neural-net_bgl)

set(CMAKE_CXX_STANDARD_LIBRARIES "-lboost_serialization")

# add the executable
add_executable(neural-net_bgl neural-net_bgl.cpp)
add_executable(makeTrainingSamples makeTrainingSamples.cpp)
