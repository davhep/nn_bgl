#include <nn_bgl/nn_bgl.h>
#include <algorithm>
#include <chrono>
#include <random>
#include <stdexcept>

// ****************** class Net ******************

double Net::transferFunction(double x) const noexcept {
    // tanh - output range [-1.0..1.0]
    return std::tanh(x);
}

double Net::transferFunctionDerivative(double x) const noexcept {
    // tanh derivative
    return 1.0 - x * x;
}

void Net::dump(const std::string& /* label */) {
    // Implementation for debugging (currently commented out)
    // Could be implemented with modern logging library
}

void Net::getResults(std::vector<double>& resultVals) const {
    resultVals.clear();
    resultVals.reserve(output_layer.size());
    
    for (const auto& [vertex, _] : output_layer) {
        resultVals.push_back(m_net_graph[vertex].m_outputVal);
    }
}

void Net::updateNeuronGradients(const std::vector<double>& targetVals) {
    for (size_t i = topo_sorted.size() - 1; i != static_cast<size_t>(-1); --i) {
        // reverse topological order
        auto neuron = &m_net_graph[topo_sorted[i]];
        
        if (DEBUG_LOW) {
            std::cerr << "Update gradients on outputs for neuron " << neuron->tag << std::endl;
        }
        
        double delta = 0.0;
        
        if (neuron->is_output) {
            // ok, we are in last layer of neurons - desired outputs are fixed from output values
            auto output_iter = output_layer.find(topo_sorted[i]);
            if (output_iter != output_layer.end()) {
                delta = targetVals[output_iter->second] - neuron->m_outputVal;
            }
            
            if (DEBUG_LOW) {
                std::cerr << "Out neuron, so update on delta_out targetVals[output_layer.find(topo_sorted[i])] = "
                          << targetVals[output_layer.find(topo_sorted[i])->second]
                          << " neuron->m_outputVal = " << neuron->m_outputVal << " delta= " << delta << std::endl;
            }
        } else {
            typename boost::graph_traits<Graph>::out_edge_iterator ei, ei_end;
            if (DEBUG_LOW) {
                std::cerr << "Update hidden layer neuron" << std::endl;
            }
            
            for (boost::tie(ei, ei_end) = out_edges(topo_sorted[i], m_net_graph); ei != ei_end; ++ei) {
                auto target = boost::target(*ei, m_net_graph);
                delta += m_net_graph[*ei].m_weight * m_net_graph[target].m_gradient;
                
                if (DEBUG_LOW) {
                    std::cerr << "target neuron " << m_net_graph[target].tag
                              << " m_net_graph[*ei].m_weight = " << m_net_graph[*ei].m_weight
                              << " m_net_graph[target].m_gradient = " << m_net_graph[target].m_gradient << std::endl;
                }
            }
        }
        
        neuron->m_gradient = delta * transferFunctionDerivative(neuron->m_outputVal);
        
        if (DEBUG_LOW) {
            std::cerr << "neuron->m_outputVal = " << neuron->m_outputVal << std::endl;
            std::cerr << "delta = " << delta << " transferFunctionDerivative(neuron->m_outputVal) = "
                      << transferFunctionDerivative(neuron->m_outputVal) << std::endl;
            std::cerr << "neuron->tag = " << neuron->tag << " neuron->m_gradient = " << neuron->m_gradient << std::endl;
        }
    }
}

void Net::updateEdgeWeights(bool update_weights) {
    for (size_t i = topo_sorted.size() - 1; i != static_cast<size_t>(-1); --i) {
        // reverse topological order
        // auto neuron = &m_net_graph[topo_sorted[i]]; // Unused variable

        typename boost::graph_traits<Graph>::in_edge_iterator ei, ei_end;
        for (boost::tie(ei, ei_end) = in_edges(topo_sorted[i], m_net_graph); ei != ei_end; ++ei) {
            // auto source = boost::source(*ei, m_net_graph); // Unused variable
            // auto target = boost::target(*ei, m_net_graph); // Unused variable
            m_net_graph[*ei].m_delta_weight = m_net_graph[*ei].rate * 
                m_net_graph[boost::source(*ei, m_net_graph)].m_outputVal * 
                m_net_graph[boost::target(*ei, m_net_graph)].m_gradient;

            if (update_weights) {
                m_net_graph[*ei].m_weight += m_net_graph[*ei].m_delta_weight;
            }
            
            if (DEBUG_LOW) {
                std::cerr << "Wij " << m_net_graph[boost::source(*ei, m_net_graph)].tag
                          << " to " << m_net_graph[boost::target(*ei, m_net_graph)].tag
                          << " updated for " << m_net_graph[*ei].m_delta_weight << std::endl;
            }
        }
    }
}

void Net::backProp(const std::vector<double>& targetVals, bool update_weights) {
    if (targetVals.size() != output_layer.size()) {
        throw std::invalid_argument("Target values size must match output layer size");
    }
    
    // Calculate overall net error (RMS of output neuron errors)
    m_error = 0.0;

    for (const auto& [vertex, _] : output_layer) {
        double delta = targetVals[output_layer.find(vertex)->second] - m_net_graph[vertex].m_outputVal;
        m_error += delta * delta;
        
        if (DEBUG_LOW) {
            std::cerr << "output_layer_element.second = " << output_layer.find(vertex)->second
                      << " output_layer_element.first.tag = " << m_net_graph[vertex].tag << std::endl;
            std::cerr << "targetVals[output_layer.find(vertex)] = " << targetVals[output_layer.find(vertex)->second]
                      << " m_net_graph[vertex].m_outputVal = " << m_net_graph[vertex].m_outputVal << std::endl;
        }
    }
    
    m_error /= static_cast<double>(output_layer.size()); // get average error squared
    m_error = std::sqrt(m_error); // RMS

    // Implement a recent average measurement
    m_recentAverageError = m_error;
    
    if (DEBUG_LOW) {
        std::cerr << "m_recentAverageError = " << m_recentAverageError
                  << " m_recentAverageSmoothingFactor = " << RECENT_AVERAGE_SMOOTHING_FACTOR
                  << " m_error = " << m_error << std::endl;
    }

    // Update neuron deltas and gradients
    updateNeuronGradients(targetVals);
    
    // Update edge weights
    updateEdgeWeights(update_weights);
}



void Net::feedForward(const std::vector<double>& inputVals) {
    if (input_layer.size() != inputVals.size()) {
        throw std::invalid_argument("Input values size must match input layer size");
    }
    
    for (size_t i = 0; i < topo_sorted.size(); i++) {
        auto vertex = topo_sorted[i];
        auto neuron = &m_net_graph[vertex];
        
        if (DEBUG_LOW) {
            std::cerr << "Processing neuron " << neuron->tag << std::endl;
        }
        
        if (neuron->is_input) {
            neuron->m_outputVal = inputVals[neuron->input_signal];
            if (DEBUG_LOW) {
                std::cerr << "inputVals[neuron.input_signal] = " << inputVals[neuron->input_signal] << std::endl;
            }
        } else {
            neuron->m_input_value = 0.0;
            typename boost::graph_traits<Graph>::in_edge_iterator ei, ei_end;
            
            for (boost::tie(ei, ei_end) = in_edges(vertex, m_net_graph); ei != ei_end; ++ei) {
                // auto source = boost::source(*ei, m_net_graph); // Unused variable
                // auto target = boost::target(*ei, m_net_graph); // Unused variable
                neuron->m_input_value += m_net_graph[*ei].m_weight * 
                    m_net_graph[boost::source(*ei, m_net_graph)].m_outputVal;
                
                if (DEBUG_LOW) {
                    std::cerr << "m_net_graph[*ei].m_weight = " << m_net_graph[*ei].m_weight
                              << " m_net_graph[boost::source(*ei, m_net_graph)].m_outputVal = "
                              << m_net_graph[boost::source(*ei, m_net_graph)].m_outputVal
                              << " neuron->m_input_value = " << neuron->m_input_value << std::endl;
                }
            }
            neuron->m_outputVal = transferFunction(neuron->m_input_value);
        }
        
        if (DEBUG_LOW) {
            std::cerr << "neuron->m_outputVal = " << neuron->m_outputVal << std::endl;
        }
    }
}

void Net::on_topology_update() {
    // if we change topology (add/remove edge or vertex), we have to update some secondary information about graph
    
    // we have to recalculate topological order of vertices for correct forward/backward procedure
    topo_sorted.clear();
    boost::topological_sort(m_net_graph, std::front_inserter(topo_sorted));
    
    std::cerr << "A topological ordering: ";
    for (size_t i = 0; i < topo_sorted.size(); i++) {
        std::cerr << m_net_graph[topo_sorted[i]].tag << " ";
    }
    std::cerr << std::endl;
    
    // because of re-numbering vertices after remove vertices, we have to update output/input lists
    input_layer.clear();
    output_layer.clear();
    
    boost::graph_traits<Graph>::vertex_iterator vi, vi_end;
    for (boost::tie(vi, vi_end) = boost::vertices(m_net_graph); vi != vi_end; ++vi) {
        if (m_net_graph[*vi].is_input) {
            input_layer.insert(std::make_pair(*vi, m_net_graph[*vi].input_signal));
        }
        if (m_net_graph[*vi].is_output) {
            output_layer.insert(std::make_pair(*vi, m_net_graph[*vi].output_signal));
        }
    }
}

void Net::createLayeredNetwork(const std::vector<unsigned>& topology) {
    unsigned numLayers = topology.size();
    unsigned neurons_total = 0;
    
    for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum) {
        unsigned numInputs = layerNum == 0 ? 0 : topology[layerNum - 1];
        unsigned first_prev_neuron = neurons_total - numInputs;
        
        for (unsigned neuronNum = 0; neuronNum < topology[layerNum]; ++neuronNum) {
            NeuronP neuron;
            neuron.tag = neurons_total;
            tag_max = neuron.tag;
            
            if (layerNum == 0) {
                // if we are in input layer, storage output tags
                neuron.is_input = true;
                neuron.input_signal = neuronNum;
            }
            
            if (layerNum == numLayers - 1) {
                // if we are in output layer, storage output tags
                neuron.is_output = true;
                neuron.output_signal = neuronNum;
            }
            
            boost::add_vertex(neuron, m_net_graph);

            for (unsigned neuronNum_prev = 0; neuronNum_prev < numInputs; neuronNum_prev++) {
                SinapsP sinaps;
                sinaps.m_weight = static_cast<double>((neuronNum_prev + neuronNum) % 10) / 10.0;
                boost::add_edge(first_prev_neuron + neuronNum_prev, neurons_total, sinaps, m_net_graph);
            }
            neurons_total++;
        }
    }
}

void Net::createWaterfallNetwork(const std::vector<unsigned>& topology) {
    unsigned int total_neurons = std::accumulate(topology.begin(), topology.end(), 0U);
    std::cerr << "Total neurons: " << total_neurons << std::endl;
    
    std::vector<unsigned int> neuron_in;
    
    for (unsigned int neuron_num = 0; neuron_num < total_neurons; neuron_num++) {
        std::cerr << "generating neuron " << neuron_num << std::endl;
        NeuronP neuron;
        neuron.tag = neuron_num;
        boost::add_vertex(neuron, m_net_graph);
        tag_max++;
        
        if (neuron_num < topology[0]) {
            // if we are in input layer, storage output tags
            neuron.input_signal = neuron_num;
        }
        
        if (neuron_num >= (total_neurons - topology.back())) {
            // if we are in output layer, storage output tags
            neuron.output_signal = neuron_num;
        }
        
        if (neuron_num >= topology[0]) {
            // so, we are have to connect non-input neuron to some other neurons
            unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
            std::shuffle(neuron_in.begin(), neuron_in.end(), std::default_random_engine(seed));
            
            unsigned int n_max = 4; // how many neurons output connect to this neuron input
            for (unsigned int n = 0; (n < n_max) && (n < neuron_num); n++) {
                unsigned int input_neuron = neuron_in[n];
                std::cerr << "input neuron = " << input_neuron << std::endl;
                SinapsP sinaps;
                sinaps.m_weight = static_cast<double>(rand() % 100) / 100.0;
                boost::add_edge(input_neuron, neuron_num, sinaps, m_net_graph);
            }
        }
        
        if (neuron_num < (total_neurons - topology.back())) {
            neuron_in.push_back(neuron_num);
        }
    }
    
    // ok, we have to check if some neuron have no outputs, and connect them to some outer neurons
    Graph::vertex_iterator v, vend;
    for (boost::tie(v, vend) = vertices(m_net_graph); v != vend; ++v) {
        auto outer_neuron_check = output_layer.find(*v);
        if (outer_neuron_check != output_layer.end()) continue;
        
        auto out_edges = boost::out_edges(*v, m_net_graph);
        if (out_edges.first == out_edges.second) {
            for (const auto& outer_neuron : output_layer) {
                std::cerr << *v << " " << outer_neuron.first << std::endl;
                SinapsP sinaps;
                sinaps.m_weight = static_cast<double>(rand() % 100) / 100.0;
                boost::add_edge(*v, outer_neuron.first, sinaps, m_net_graph);
            }
        }
    }
}

Net::Net(const std::vector<unsigned>& topology, NetType type_of_network) {
    input_layer.clear();
    output_layer.clear();
    minimal_error = 1e6;
    
    switch (type_of_network) {
        case NetType::LAYERS:
            createLayeredNetwork(topology);
            break;
        case NetType::WATER_FALL:
            createWaterfallNetwork(topology);
            break;
    }
    
    on_topology_update();
}

void Net::save(const std::string& path) const {
    std::ofstream file{path};
    if (file) {
        boost::archive::text_oarchive oa{file};
        oa << *this;
    }
}

void Net::load(const std::string& path) {
    std::ifstream file{path};
    if (file) {
        std::cerr << "Loading Net ..." << std::endl;
        if (DEBUG_LOW) {
            std::cerr << "topo size before: " << topo_sorted.size()
                      << " edges size before: " << boost::num_edges(m_net_graph)
                      << " vertexes size before: " << boost::num_vertices(m_net_graph) << std::endl;
        }
        
        try {
            boost::archive::text_iarchive ia{file};
            m_net_graph.clear(); // because of deserialization add new vertices and edges, w/o removing old ones
            ia >> *this;
        } catch (const std::exception& e) {
            std::cerr << "Loading failed. Exception: " << e.what() << std::endl;
            return;
        }
        
        if (DEBUG_LOW) {
            std::cerr << "topo size after: " << topo_sorted.size()
                      << " edges size after: " << boost::num_edges(m_net_graph)
                      << " vertexes size after: " << boost::num_vertices(m_net_graph) << std::endl;
        }
    }
}

