# Neural Network with Boost Graph Library (nn_bgl)

## Overview and Features

This is a C++ implementation of a simple neural network with a unique approach.

The key feature is representation of neural-network as a directed acyclic graph (DAG) of non-linear neurons, unlike common realizations of fully-connected consecutive layers.

So, one can insert and delete neurons and connections one by one.

Information about DAG structure, weight and dynamical coefficients are stored in boost::graph object.

Optimization of weights is realized by a basic error backward propagation way.

Experimental dynamic visualization of model fitting.

## Data Model

In this model, we have to fit the quarter of ring under (x,y) plane:

- 0.5 if 2 < x²+y² < 4
- 0 otherwise

So, we have two inputs x,y and one output.

This model can be easily explored and visualized, but (in fact) - more realistic compared to XOR fitting procedure.

## Dependencies

- build-essential
- boost::program_options
- boost::serialization
- gnuplot (with bash environment)

## Project Structure

```
nn_bgl/
├── include/nn_bgl/     # Header files
├── src/                # Source files
├── scripts/            # Utility scripts
├── tests/              # Test files
├── examples/           # Example usage
├── docs/               # Documentation
├── build/              # Build directory
└── CMakeLists.txt      # Main build configuration
```

## Build Process

```bash
# Create build directory
mkdir build
cd build

# Configure and build
cmake ..
make

# Or build in parallel
make -j$(nproc)
```

## Run Process

### 1. Generate Training Data

```bash
./makeTrainingSamples
```

This generates:
- `topology.txt` - example starting topology file (basic n-layer)
- `train_data.txt` - data for training
- `validate_data.txt` - data for validating

### 2. Train Model

```bash
./train_model
```

Train model for some epochs (100 by default).

In result, .dot and boost::serialization files for model will be created, both for final and best result.

By default, saved serialized final model will be loaded next run and fitting will be continued, so we can iterate by re-run this binary any time.

With `--gnuplot` option, fitting process will be visualized by replotting of gnuplot 3D plot every epoch (UNSTABLE, but PRETTY LOOK).

### 3. Analyze Model

```bash
./analyze_model
```

Calculation of dependencies in fitted model weights, deltas and etc.

## Available Executables

- `train_model` - Train model with human data
- `train_model_mnist` - Train model with MNIST data
- `analyze_model` - Analyze trained model
- `makeTrainingSamples` - Generate training samples

## Development

### Adding New Features

1. Add header files to `include/nn_bgl/`
2. Add source files to `src/`
3. Update `CMakeLists.txt` if needed
4. Update include statements to use `#include <nn_bgl/header.h>`

### Testing

```bash
cd build
make test
```

## License

[Add your license information here]



