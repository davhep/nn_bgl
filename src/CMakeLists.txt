cmake_minimum_required(VERSION 3.10)

# set the project name
project(neural-net_bgl)

set(CMAKE_BUILD_TYPE DEBUG)
set(CMAKE_CXX_STANDARD_LIBRARIES "-lboost_serialization -lboost_program_options")
set(SOURCE_TRAIN nn_bgl.cpp training_data_human.cpp  train_model.cpp)
set(SOURCE_TRAIN_MNIST nn_bgl.cpp  training_data_mnist.cpp train_model_mnist.cpp)
set(SOURCE_ANALYZE nn_bgl.cpp training_data_human.cpp analyze_model.cpp)
set(HEADER nn_bgl.h training_data_human.h training_data_mnist.h)

include_directories(.)

# add the executable
add_executable(train_model ${SOURCE_TRAIN})
add_executable(train_model_mnist ${SOURCE_TRAIN_MNIST})
add_executable(analyze_model ${SOURCE_ANALYZE})
add_executable(makeTrainingSamples makeTrainingSamples.cpp)

#configure_file(scripts  scripts)
file(COPY run_test.sh DESTINATION .)
