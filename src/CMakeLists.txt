cmake_minimum_required(VERSION 3.10)

# set the project name
project(neural-net_bgl)

set(CMAKE_BUILD_TYPE DEBUG)
set(CMAKE_CXX_STANDARD_LIBRARIES "-lboost_serialization -lboost_program_options")
set(SOURCE_TRAIN nn_bgl.cpp training_data.cpp train_model.cpp)
set(SOURCE_ANALYZE nn_bgl.cpp training_data.cpp analyze_model.cpp)
set(HEADER nn_bgl.h training_data.h)

include_directories(.)

# add the executable
add_executable(train_model ${SOURCE_TRAIN})
add_executable(analyze_model ${SOURCE_ANALYZE})
add_executable(makeTrainingSamples makeTrainingSamples.cpp)

#configure_file(scripts  scripts)
file(COPY run_test.sh DESTINATION .)