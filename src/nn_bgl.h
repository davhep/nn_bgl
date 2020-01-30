#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>
#include <map>  
#include <numeric>


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
#include <iostream>
#include <fstream>

#include <boost/container/detail/iterator.hpp>
#include <boost/container/detail/iterators.hpp>

#include <iterator>

#define debug_high false
#define debug_low false

using namespace std;

struct NeuronP {
   int tag;
   double m_input_value;
   double m_outputVal;
   double m_gradient;
   template<class Archive>
   void serialize(Archive & ar, const unsigned int file_version){
	   ar & tag;
	   ar & m_input_value;
	   ar & m_outputVal;
	   ar & m_gradient;
   }
};


struct SinapsP {
   double m_weight;
   double m_delta_weight;
   template<class Archive>
   void serialize(Archive & ar, const unsigned int file_version){
	   ar & m_weight;
   }
};



typedef boost::adjacency_list<boost::vecS,
        boost::vecS, boost::bidirectionalS,
        NeuronP, SinapsP> Graph;
        
typedef boost::graph_traits<Graph>::vertex_descriptor vertex_descriptor;
typedef boost::graph_traits<Graph>::edge_descriptor edge_descriptor;
        
class Net
{
public:
	Net(const vector<unsigned> &topology);
	void feedForward(const vector<double> &inputVals);
	void backProp(const vector<double> &targetVals);
	void getResults(vector<double> &resultVals) const;
	double getRecentAverageError(void) const { return m_recentAverageError;}
	void dump(std::string label);
	double transferFunction(double x);
	double transferFunctionDerivative(double x);
	double eta = 0.15; // overall net learning rate
    double alpha = 0.5; // momentum, multiplier of last deltaWeight, [0.0..n]
    double minimal_error;

	Graph m_net_graph;
	std::deque<vertex_descriptor> topo_sorted;
	double m_error;
	double m_recentAverageError = 0;
	static double m_recentAverageSmoothingFactor;
	std::map<vertex_descriptor, int> input_layer, output_layer;
	
	void save(const std::string path);
	void load(const std::string path);
	
	template<class Archive>
		void serialize(Archive & ar, const unsigned int file_version);
};
