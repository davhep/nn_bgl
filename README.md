1. Overview and features.

This is a C++ implementation of simple neural network.

The key feature is representation of neural-network as directed acyclic graph (DAG) of non-linear neurons, unlike to common realizations of full-connected consecutive layers.

So, one can insert adn delete neurons and connections one by one.

Information about DAG structure, weight and dynamical coefficients are strored in boost::graph object.

Optimization of weights is realized by a basic error backward propogation way.

Experimental dynamic visualization of model fitting

2. Data model

In this model, we have to fit the quarter of ring under (x,y) plane:

0.5 if 2 < x^2+y^2  < 4

0 otherwise

So, we have two inputs x,y and one output.

This model can be easily explored and visualized, but (in fact) - more realistic compared to XOR fitting procedure.

3. Dependencies.

build-essential

boost::program_options

boost::serialization

gnuplot (with bash environment)

4. Build process

mkdir build

cmake $(nn_bgl_dir)/src

make

5. Run process

./makeTrainingSamples

To generate:

topology.txt example starting topology file  - which is basic n-layer.

train_data.txt - data for trainig.

validate_data.txt - data for validating.

./train_model

Train model for some (100 by default) epochs.

In result, .dot and boost::serialization files for mode will be created, both for final and best result.

By default, saved serialized final model will be loaded next run and fitting wil be continued, so we can iterate by re-run this binary any time.

with --gnuplot option, fitting process will be visualized by replotting of gnuplot 3D plot every epoch (UNSTABLE, but PRETTY LOOK).

./analyze_model

Calculation of dependecies in fitted model weights, deltas and etc.



