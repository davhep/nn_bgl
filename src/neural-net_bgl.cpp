#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>
#include <map>  

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

class TrainingData
{
public:
	TrainingData(const string filename);
	bool isEof(void)
	{
		return m_trainingDataFile.eof();
	}
	void getTopology(vector<unsigned> &topology);
	
	// Returns the number of input values read from the file:
	unsigned getNextInputs(vector<double> &inputVals);
	unsigned getTargetOutputs(vector<double> &targetOutputVals);
    void reset(void); 
private:
	ifstream m_trainingDataFile;
};

void TrainingData::reset(void)
{
	m_trainingDataFile.clear();
	m_trainingDataFile.seekg (0);
	string line;
	getline(m_trainingDataFile, line);
}

void TrainingData::getTopology(vector<unsigned> &topology)
{
	string line;
	string label;

	getline(m_trainingDataFile, line);
	stringstream ss(line);
	ss >> label;
	if(this->isEof() || label.compare("topology:") != 0)
	{
		abort();
	}

	while(!ss.eof())
	{
		unsigned n;
		ss >> n;
		topology.push_back(n);
	}
	return;
}

TrainingData::TrainingData(const string filename)
{
	m_trainingDataFile.open(filename.c_str());
}


unsigned TrainingData::getNextInputs(vector<double> &inputVals)
{
    inputVals.clear();

    string line;
    getline(m_trainingDataFile, line);
    stringstream ss(line);

    string label;
    ss >> label;
    if (label.compare("in:") == 0) {
        double oneValue;
        while (ss >> oneValue) {
            inputVals.push_back(oneValue);
        }
    }

    return inputVals.size();
}

unsigned TrainingData::getTargetOutputs(vector<double> &targetOutputVals)
{
    targetOutputVals.clear();

    string line;
    getline(m_trainingDataFile, line);
    stringstream ss(line);

    string label;
    ss>> label;
    if (label.compare("out:") == 0) {
        double oneValue;
        while (ss >> oneValue) {
            targetOutputVals.push_back(oneValue);
        }
    }

    return targetOutputVals.size();
}



// ****************** class Net ******************

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
   template<class Archive>
   void serialize(Archive & ar, const unsigned int file_version){
	   ar & m_weight;
   }
};



typedef boost::adjacency_list<boost::vecS,
        boost::vecS, boost::bidirectionalS,
        NeuronP, SinapsP> Graph;
        
typedef boost::graph_traits<Graph>::vertex_descriptor vertex_descriptor;
        
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

	Graph m_net_graph;
	std::deque<vertex_descriptor> topo_sorted;
	double m_error;
	double m_recentAverageError = 0;
	static double m_recentAverageSmoothingFactor;
	std::map<vertex_descriptor, int> input_layer, output_layer;
	
	void save(const char *path);
	void load(const char *path);
	
	template<class Archive>
		void serialize(Archive & ar, const unsigned int file_version);
};

double Net::transferFunction(double x)
{
	// tanh - output range [-1.0..1.0]
	return tanh(x);
}

double Net::transferFunctionDerivative(double x)
{
	// tanh derivative
	// cout << x << "	" << 1.0 - x * x << "	" << 1/cosh(2*x)/cosh(2*x) << endl;
	//return 1/cosh(1.5*x)/cosh(1.5*x);
	return 1.0 - x * x;
	//return cos(x*3.1415/2);
}

void Net::dump(std::string label)
{/*
	cout << label;
	for(unsigned layerNum = 0; layerNum < m_layers.size(); ++layerNum){
		for(unsigned n = 0; n < m_layers[layerNum].size(); ++n){
			cout << "	layerNum= " << layerNum << "	neuronNum= " << n << "	outputValue= " << m_layers[layerNum][n].getOutputVal();
			for(auto connection: m_layers[layerNum][n].m_outputWeights) cout << "	Weight: " << connection.weight << "	deltaWeight: " << connection.deltaWeight;
			//auto neuron_dump = m_layers[layerNum][n].dump();
			//m_dump.push_back(m_layers[layerNum][n]);
		}
	}*/
}

double Net::m_recentAverageSmoothingFactor = 100.0; // Number of training samples to average over

void Net::getResults(vector<double> &resultVals) const
{
	resultVals.clear();
	for(auto output_layer_element: output_layer)
		resultVals.push_back(m_net_graph[output_layer_element.first].m_outputVal);
}

void Net::backProp(const std::vector<double> &targetVals)
{
	assert(targetVals.size() == output_layer.size());
	
	// Calculate overal net error (RMS of output neuron errors)

	m_error = 0.0;

	for(auto output_layer_element: output_layer)
	{
		double delta = targetVals[ output_layer_element.second ] - m_net_graph[output_layer_element.first].m_outputVal;
		m_error += delta *delta;
		if(debug_low) cout << "targetVals[ output_layer_element.second ] = " << targetVals[ output_layer_element.second ] << "	m_net_graph[output_layer_element.first].m_outputVal= " << m_net_graph[output_layer_element.first].m_outputVal << endl;
	}
	
	m_error /= output_layer.size(); // get average error squared
	m_error = sqrt(m_error); // RMS

	// Implement a recent average measurement:

	m_recentAverageError = 
			(m_recentAverageError * m_recentAverageSmoothingFactor + m_error)
			/ (m_recentAverageSmoothingFactor + 1.0);
    if(debug_low){
		cout << "m_recentAverageError =  (m_recentAverageError * m_recentAverageSmoothingFactor + m_error) / (m_recentAverageSmoothingFactor + 1.0);" << endl;
		cout << m_recentAverageError << "	" << m_recentAverageError << "	" << m_recentAverageSmoothingFactor << "	" << m_error << "	" << m_recentAverageSmoothingFactor << endl;
	}
   	for (int i = topo_sorted.size() - 1; i >= 0; i--){
		//reverse topological order
		auto neuron = &m_net_graph[topo_sorted[i]];
		if (debug_low) cout << "Update gradients on outputs for neuron " << neuron->tag << endl;
		double delta = 0;
	    if(output_layer.find(topo_sorted[i]) != output_layer.end()){
			//ok, we are in last layer of neurons - desired outputs are fixed from output values
			delta = targetVals[output_layer.find(topo_sorted[i])->second] - neuron->m_outputVal;
			if (debug_low) cout << "Out neuron, so update on delta_out targetVals[output_layer.find(topo_sorted[i])] =	" << targetVals[output_layer.find(topo_sorted[i])->second]  << "	neuron->m_outputVal = " << neuron->m_outputVal << "delta= " << delta << endl;			
		}
		else
		{
			typename boost::graph_traits<Graph>::out_edge_iterator ei, ei_end;
			if (debug_low) cout << "Update hidden layer neuron" << endl;
		    for (boost::tie(ei, ei_end) = out_edges(topo_sorted[i], m_net_graph); ei != ei_end; ++ei){
				auto source = boost::source ( *ei, m_net_graph);
				auto target = boost::target ( *ei, m_net_graph );
				delta += m_net_graph[*ei].m_weight * m_net_graph[target].m_gradient;
				if (debug_low) cout << "target neuron " << m_net_graph[target].tag << " sm_net_graph[*ei].m_weight =	" << m_net_graph[*ei].m_weight  << "	m_net_graph[target].m_gradient = " << m_net_graph[target].m_gradient << endl;
			}
		}
		neuron->m_gradient = delta * transferFunctionDerivative(neuron->m_outputVal);
		if (debug_low) cout << "neuron->m_outputVal=	" << neuron->m_outputVal << endl;
		if (debug_low) cout << "delta=	" << delta  << "	transferFunctionDerivative(neuron->m_outputVal)= " << transferFunctionDerivative(neuron->m_outputVal) << endl;
		if (debug_low) cout << "neuron->tag=	" << neuron->tag  << "	neuron->m_gradient= " << neuron->m_gradient << endl;
		
   }
   
   for (int i = topo_sorted.size() - 1; i >= 0; i--){
		//reverse topological order
		auto neuron = &m_net_graph[topo_sorted[i]];
		typename boost::graph_traits<Graph>::in_edge_iterator ei, ei_end;
		    for (boost::tie(ei, ei_end) = in_edges(topo_sorted[i], m_net_graph); ei != ei_end; ++ei){
				auto source = boost::source ( *ei, m_net_graph);
				auto target = boost::target ( *ei, m_net_graph );
				double delta_weight = eta * m_net_graph[source].m_outputVal * m_net_graph[target].m_gradient;
				m_net_graph[*ei].m_weight += delta_weight;
				if (debug_low) cout << "Wij " << m_net_graph[source].tag << "	to " << m_net_graph[target].tag << " updated for " << delta_weight << endl;
			}
	}
}



void Net::feedForward(const vector<double> &inputVals)
{
	assert(input_layer.size() == inputVals.size());
	
	for (int i = 0; i < topo_sorted.size(); i++){
		auto vertex = topo_sorted[i];
		auto neuron = &m_net_graph[vertex];
		if(debug_low) cout << "Processing neuron " << neuron->tag << endl;
		auto input_layer_position = input_layer.find(vertex);
		if(input_layer_position != input_layer.end()){ 	//ok, we are in first layer of neurons - outputs are fixed from input values
			neuron->m_outputVal = inputVals[input_layer_position->second];
			if(debug_low) cout << "inputVals[input_layer_position->second]= " << inputVals[input_layer_position->second] << endl;
		}
		else
		{
			neuron->m_input_value = 0;
		    typename boost::graph_traits<Graph>::in_edge_iterator ei, ei_end;
		    for (boost::tie(ei, ei_end) = in_edges(vertex, m_net_graph); ei != ei_end; ++ei) {
				auto source = boost::source ( *ei, m_net_graph);
				auto target = boost::target ( *ei, m_net_graph );
				neuron->m_input_value += m_net_graph[*ei].m_weight * m_net_graph[source].m_outputVal;
				if(debug_low) cout << "m_net_graph[*ei].m_weight = " << m_net_graph[*ei].m_weight << " m_net_graph[source].m_outputVal= " <<  m_net_graph[source].m_outputVal << " neuron->m_input_value= " << neuron->m_input_value << endl;
				}
			neuron->m_outputVal = transferFunction(neuron->m_input_value);
		}
		if(debug_low) cout << "neuron->m_outputVal = " << neuron->m_outputVal << endl;
   }
}

Net::Net(const vector<unsigned> &topology)
{
	input_layer.clear();
	output_layer.clear();
	
	unsigned numLayers = topology.size();
	unsigned neurons_total = 0;
	for(unsigned layerNum = 0; layerNum < numLayers; ++layerNum){
		unsigned numInputs = layerNum == 0 ? 0 : topology[layerNum - 1];
		unsigned first_prev_neuron = neurons_total - numInputs;
		for(unsigned neuronNum = 0; neuronNum < topology[layerNum]; ++neuronNum){
			NeuronP neuron;
			neuron.tag = neurons_total;
			auto vertex_new = boost::add_vertex(neuron, m_net_graph);
			if(layerNum == 0) //if we are in input layer, storage ouput tags
				input_layer.insert(std::pair<vertex_descriptor, int>(vertex_new, neuronNum));
			if(layerNum == numLayers - 1) //if we are in output layer, storage ouput tags
				output_layer.insert(std::pair<vertex_descriptor, int>(vertex_new, neuronNum));
			for(unsigned neuronNum_prev = 0; neuronNum_prev < numInputs; neuronNum_prev++){
				SinapsP sinaps;
				sinaps.m_weight = double((neuronNum_prev+neuronNum) % 10)/10;
				boost::add_edge(first_prev_neuron+neuronNum_prev, neurons_total, sinaps, m_net_graph);
			 }
			 neurons_total++;
		}
	}
	
	boost::topological_sort(m_net_graph, std::front_inserter(topo_sorted));
    cout << "A topological ordering: ";
    for (int i = 0; i < topo_sorted.size(); i++) cout << m_net_graph[topo_sorted[i]].tag  << " ";
    cout << endl;
}

void Net::save(const char *path){
	std::ofstream file{path};
	if(file){
		boost::archive::text_oarchive oa{file};
		oa << *this;
	}
}

void Net::load(const char *path){
	std::ifstream file{path};
	if(file){
		cout << "Loading Net ..." << endl;
		if(debug_low){
			cout << "topo size before: " << topo_sorted.size()
				<< "	edges size before: " << boost::num_edges(m_net_graph)
				<< "	vertexes size before: " << boost::num_vertices(m_net_graph) << endl;
		}
	    try{
			boost::archive::text_iarchive ia{file};
			m_net_graph.clear();//because of deserialization add new vertices and edges, w/o removing old ones
			ia >> *this;
		}
		catch(int a)
		{
		  cout << "Loading faild. Caught exception number:  " << a << endl;
		  return;
		}
		
		if(debug_low){
			cout << "topo size after: " << topo_sorted.size()
				<< "	edges size after: " << boost::num_edges(m_net_graph)
				<< "	vertexes size after: " << boost::num_vertices(m_net_graph) << endl;
		}
	}
}
template<class Archive>
void Net::serialize(Archive & ar, const unsigned int file_version){
	   ar & m_net_graph;
	   ar & eta;
	   ar & input_layer;
	   ar & output_layer;
	   ar & topo_sorted;
	   //ar & m_recentAverageError;
}


void showVectorVals(string label, vector<double> &v)
{
	cout << label << " ";
	for(unsigned i = 0; i < v.size(); ++i)
	{
		cout << v[i] << " ";
	}
	cout << endl;
}
int main()
{
	TrainingData trainData("trainingData.txt");
	vector<unsigned> topology;
	
	trainData.getTopology(topology);
	Net myNet(topology);
	myNet.load("trained_model_serialized.txt");
	
	vector<double> inputVals, targetVals, resultVals;	
	int trainingPass = 0;
	int epochs_max=100;
	while(trainingPass < epochs_max){
		++trainingPass;
		myNet.eta = 100.0/(trainingPass+1000.0);
		while(!trainData.isEof())
		{
			// Get new input data and feed it forward:
			if(trainData.getNextInputs(inputVals) != topology[0])
				break;
			myNet.feedForward(inputVals);
	
			// Collect the net's actual results:
			myNet.getResults(resultVals);
			// Train the net what the outputs should have been:
			trainData.getTargetOutputs(targetVals);
			if(debug_high)
			{
				cout << "Pass" << trainingPass << endl;
				showVectorVals("Inputs :", inputVals);
				showVectorVals("Outputs:", resultVals);
				showVectorVals("Targets:", targetVals);
			}	
			assert(targetVals.size() == topology.back());	
			myNet.backProp(targetVals);	
		}
	    trainData.reset();
	    cerr << "At epoch " << trainingPass << " Net recent average error: " << myNet.getRecentAverageError() << endl;
    }
    
    ofstream dot_file("trained_model_vizualization.dot");
	boost::dynamic_properties dp;
	dp.property("node_id", get(&NeuronP::tag, myNet.m_net_graph));
    dp.property("label", get(&NeuronP::tag, myNet.m_net_graph));
    dp.property("label", get(&SinapsP::m_weight, myNet.m_net_graph));
	boost::write_graphviz_dp(dot_file, myNet.m_net_graph, dp);
	
	myNet.save("trained_model_serialized.txt");
}
