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
   unsigned m_myIndex;
	double m_gradient;
	NeuronP& operator=(NeuronP other)
	{
		tag = other.tag;
	}
	
    NeuronP& operator=(int other)
	{
		tag = other;
	}
};


struct SinapsP {
   double m_weight;
};



typedef boost::adjacency_list<boost::vecS,
        boost::vecS, boost::bidirectionalS,
        NeuronP, SinapsP> Graph;
        
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
	std::deque<boost::graph_traits<Graph>::vertex_descriptor> topo_sorted;
	double m_error;
	double m_recentAverageError;
	static double m_recentAverageSmoothingFactor;
	std::vector<unsigned> output_neurons;


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
	for(unsigned n = 0; n < output_neurons.size(); ++n) resultVals.push_back(m_net_graph[output_neurons[n]].m_outputVal);
}

void Net::backProp(const std::vector<double> &targetVals)
{
	
	// Calculate overal net error (RMS of output neuron errors)

	m_error = 0.0;

	for(unsigned n = 0; n < output_neurons.size(); ++n)
	{
		double delta = targetVals[n] - m_net_graph[output_neurons[n]].m_outputVal;
		m_error += delta *delta;
	}
	m_error /= output_neurons.size(); // get average error squared
	m_error = sqrt(m_error); // RMS

	// Implement a recent average measurement:

	m_recentAverageError = 
			(m_recentAverageError * m_recentAverageSmoothingFactor + m_error)
			/ (m_recentAverageSmoothingFactor + 1.0);
   
   	for (int i = topo_sorted.size() - 1; i >= 0; i--){
		//reverse topological order
		auto neuron = &m_net_graph[topo_sorted[i]];
		if (debug_low) cout << "Update gradients on outputs for neuron " << neuron->tag << endl;
		double delta = 0;
	    if((topo_sorted.size() - neuron->tag) <= output_neurons.size()){
			//ok, we are in last layer of neurons - desired outputs are fixed from output values
			delta = targetVals[topo_sorted.size() - neuron->tag-1] - neuron->m_outputVal;
			if (debug_low) cout << "Out neuron, so update on delta_out targetVals[topo_sorted.size() - neuron->tag -1] =	" << targetVals[topo_sorted.size() - neuron->tag -1]  << "	neuron->m_outputVal = " << neuron->m_outputVal << "delta= " << delta << endl;			
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
		if (debug_low) cout << "neuron->tag=	" << neuron->tag  << "	neuron->m_gradient= " << neuron->m_gradient << endl;
		if (debug_low) cout << "delta=	" << delta  << "	transferFunctionDerivative(neuron->m_outputVal)= " << transferFunctionDerivative(neuron->m_outputVal) << endl;
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
	for (int i = 0; i < topo_sorted.size(); i++){
		auto neuron = &m_net_graph[topo_sorted[i]];
	    if(neuron->tag < inputVals.size()){
			//ok, we are in first layer of neurons - outputs are fixed from input values
			neuron->m_outputVal = inputVals[neuron->tag];
		}
		else
		{
			neuron->m_input_value = 0;
		    typename boost::graph_traits<Graph>::in_edge_iterator ei, ei_end;
		    for (boost::tie(ei, ei_end) = in_edges(topo_sorted[i], m_net_graph); ei != ei_end; ++ei) {
				auto source = boost::source ( *ei, m_net_graph);
				auto target = boost::target ( *ei, m_net_graph );
				neuron->m_input_value += m_net_graph[*ei].m_weight * m_net_graph[source].m_outputVal;
				}
			neuron->m_outputVal = transferFunction(neuron->m_input_value);
		}
   }
}

Net::Net(const vector<unsigned> &topology)
{
	
	
	unsigned numLayers = topology.size();
	unsigned neurons_total = 0;
	for(unsigned layerNum = 0; layerNum < numLayers; ++layerNum){
		unsigned numInputs = layerNum == 0 ? 0 : topology[layerNum - 1];
		unsigned first_prev_neuron = neurons_total - numInputs;
		for(unsigned neuronNum = 0; neuronNum < topology[layerNum]; ++neuronNum){
			NeuronP neuron;
			neuron.tag = neurons_total;
			boost::add_vertex(neuron, m_net_graph);
			if(layerNum == numLayers - 1) output_neurons.push_back(neuron.tag);//if we are in output layer, storage ouput tags
			for(unsigned neuronNum_prev = 0; neuronNum_prev < numInputs; neuronNum_prev++){
				SinapsP sinaps;
				sinaps.m_weight = double(rand()) / double(RAND_MAX);
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
	vector<double> inputVals, targetVals, resultVals;
	
	int trainingPass = 0;
	int epochs_max=10000;
	std::vector<Net> net_dump;
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
	
			if(epochs_max == 1 && false) //last epoch, lets dump it
			{
				showVectorVals("Dump inputs :", inputVals);
				showVectorVals("Dump outputs:", resultVals);
				showVectorVals("Dump targets:", targetVals);
				myNet.dump("Net_dump: ");
				net_dump.push_back(myNet);
				cout << endl;
			}
		}
	    trainData.reset();
	    cerr << "At epoch " << trainingPass << " Net recent average error: " << myNet.getRecentAverageError() << endl;
    }
    
    ofstream dot_file("automaton.dot");
	boost::dynamic_properties dp;
	dp.property("node_id", get(&NeuronP::tag, myNet.m_net_graph));
    dp.property("label", get(&NeuronP::tag, myNet.m_net_graph));
    dp.property("label", get(&SinapsP::m_weight, myNet.m_net_graph));
	boost::write_graphviz_dp(dot_file, myNet.m_net_graph, dp);
}
