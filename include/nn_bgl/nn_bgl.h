#pragma once

#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>
#include <map>
#include <numeric>
#include <iterator>
#include <random>
#include <algorithm>
#include <chrono>
#include <memory>
#include <optional>
#include <span>

#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/graph/topological_sort.hpp>
#include <boost/graph/adj_list_serialize.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/deque.hpp>
#include <boost/serialization/map.hpp>
#include <boost/container/detail/iterator.hpp>
#include <boost/container/detail/iterators.hpp>

// Modern C++ constants instead of defines
constexpr bool DEBUG_LOW = false;

enum class NetType : uint8_t { 
    LAYERS, 
    WATER_FALL 
};

struct NeuronP {
    int tag{};
    bool is_input{false};
    bool is_output{false};
    unsigned int input_signal{};    // if connected to corresponding input value
    unsigned int output_signal{};   // if we get from neuron corresponding output value
    double m_input_value{};
    double m_outputVal{};
    double m_gradient{};           // gradient of loss function on input signal of given neuron
    unsigned long age{};           // neuron age, in number of iterations
    
    template<class Archive>
    void serialize(Archive& ar, const unsigned int /* file_version */) {
        ar & tag & is_input & is_output & input_signal & output_signal 
           & m_input_value & m_outputVal & m_gradient;
    }
};

struct SinapsP {
    double m_weight{0.02};
    double m_delta_weight{};
    unsigned long age{};
    double rate{0.02};
    
    template<class Archive>
    void serialize(Archive& ar, const unsigned int /* file_version */) {
        ar & m_weight & m_delta_weight & age;
    }
};

using Graph = boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS, NeuronP, SinapsP>;
using vertex_descriptor = boost::graph_traits<Graph>::vertex_descriptor;
using edge_descriptor = boost::graph_traits<Graph>::edge_descriptor;

class Net {
public:
    explicit Net(const std::vector<unsigned>& topology, NetType type_of_network = NetType::LAYERS);
    
    // Rule of Five
    Net(const Net&) = default;
    Net(Net&&) noexcept = default;
    Net& operator=(const Net&) = default;
    Net& operator=(Net&&) noexcept = default;
    ~Net() = default;
    
    // Core functionality
    void feedForward(const std::vector<double>& inputVals);
    void backProp(const std::vector<double>& targetVals, bool update_weights = true);
    void getResults(std::vector<double>& resultVals) const;
    
    // Getters
    [[nodiscard]] double getRecentAverageError() const noexcept { return m_recentAverageError; }
    [[nodiscard]] double getError() const noexcept { return m_error; }
    [[nodiscard]] double getEta() const noexcept { return eta; }
    [[nodiscard]] double getAlpha() const noexcept { return alpha; }
    [[nodiscard]] unsigned int getTrainingPass() const noexcept { return trainingPass; }
    [[nodiscard]] double getMinimalError() const noexcept { return minimal_error; }
    
    // Setters
    void setEta(double new_eta) noexcept { eta = new_eta; }
    void setAlpha(double new_alpha) noexcept { alpha = new_alpha; }
    void incrementTrainingPass() noexcept { ++trainingPass; }
    
    // Utility functions
    void dump(const std::string& label);
    [[nodiscard]] double transferFunction(double x) const noexcept;
    [[nodiscard]] double transferFunctionDerivative(double x) const noexcept;
    void on_topology_update();
    void save(const std::string& path) const;
    void load(const std::string& path);
    
    // Public members (consider making these private with getters)
    Graph m_net_graph;
    std::map<vertex_descriptor, int> input_layer;
    std::map<vertex_descriptor, int> output_layer;
    std::deque<vertex_descriptor> topo_sorted;
    double eta{0.15};           // overall net learning rate
    double alpha{0.5};          // momentum, multiplier of last deltaWeight, [0.0..n]
    unsigned int trainingPass{};
    double minimal_error{1e6};
    int tag_max{};

private:
    // Private helper functions
    void updateNeuronGradients(const std::vector<double>& targetVals);
    void updateEdgeWeights(bool update_weights);
    void createLayeredNetwork(const std::vector<unsigned>& topology);
    void createWaterfallNetwork(const std::vector<unsigned>& topology);
    
    // Member variables
    double m_error{};
    double m_recentAverageError{};
    
    // Static member
    static constexpr double RECENT_AVERAGE_SMOOTHING_FACTOR = 100.0;
    
    template<class Archive>
    void serialize(Archive& ar, const unsigned int /* file_version */) {
        ar & m_net_graph & eta & input_layer & output_layer & topo_sorted 
           & m_recentAverageError & minimal_error & trainingPass & tag_max;
    }
    
    friend class boost::serialization::access;
};
