# API Documentation

## Overview

This document provides an overview of the main classes and functions in the nn_bgl library.

## Main Classes

### Net Class

The main neural network class that represents a neural network as a directed acyclic graph.

**Key Methods:**
- `transferFunction(double x)` - Transfer function (tanh)
- `transferFunctionDerivative(double x)` - Derivative of transfer function
- `getResults(vector<double> &resultVals)` - Get output values
- `backProp(const vector<double> &targetVals, bool update_weights)` - Backpropagation training

### TrainingData Class

Base class for training data providers.

**Key Methods:**
- `isEof()` - Check if end of file reached
- `getNextInputs(vector<double> &inputVals)` - Get next input values
- `getTargetOutputs(vector<double> &targetOutputVals)` - Get target output values
- `reset()` - Reset to beginning of data

### TrainingDataHuman Class

Implementation for human-generated training data.

**Key Methods:**
- `InitFile(const string filename)` - Initialize from file
- `ReadAllFromFile(...)` - Read all data from file

### TrainingDataMnist Class

Implementation for MNIST dataset training data.

**Key Methods:**
- `InitFile(const string filename_images, const string filename_labels)` - Initialize from MNIST files

## Usage Examples

### Basic Training

```cpp
#include <nn_bgl/nn_bgl.h>
#include <nn_bgl/training_data_human.h>

// Create training data
TrainingDataHuman trainingData;
trainingData.InitFile("train_data.txt");

// Create and train network
Net net;
// ... configure network topology ...
net.backProp(targetVals, true);
```

### MNIST Training

```cpp
#include <nn_bgl/nn_bgl.h>
#include <nn_bgl/training_data_mnist.h>

// Create MNIST training data
TrainingDataMnist trainingData;
trainingData.InitFile("train-images-idx3-ubyte", "train-labels-idx1-ubyte");

// Train network
// ... training loop ...
```

## File Structure

- Headers: `include/nn_bgl/`
- Source: `src/`
- Examples: `examples/`
- Tests: `tests/`
- Documentation: `docs/`

## Building

```bash
mkdir build && cd build
cmake .. && make
```

## Dependencies

- Boost Libraries (serialization, program_options)
- C++17 compatible compiler
