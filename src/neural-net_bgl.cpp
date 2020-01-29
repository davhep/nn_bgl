#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>
#include <map>  
#include <numeric>

#include <iostream>
#include <fstream>

#include <iterator>

#include <./nn_bgl.h>

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
	vector<Net> myNet_dump; //to save model states for different input signals to analyze corellations
	
	vector<double> inputVals, targetVals, resultVals;	
	int trainingPass = 0;
	int epochs_max=1000;
	
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
			if(trainingPass == epochs_max) myNet_dump.push_back(myNet);
		}
	    trainData.reset();
	    cerr << "At epoch " << trainingPass << " Net recent average error: " << myNet.getRecentAverageError() << endl;
	    if(myNet.getRecentAverageError() < myNet.minimal_error){
			myNet.minimal_error = myNet.getRecentAverageError();
			myNet.save("trained_model_serialized_with_min_error.txt");
			ofstream dot_file("trained_model_vizualization_with_min_error.dot");
			boost::dynamic_properties dp;
			dp.property("node_id", get(&NeuronP::tag, myNet.m_net_graph));
		    dp.property("label", get(&NeuronP::tag, myNet.m_net_graph));
		    dp.property("label", get(&SinapsP::m_weight, myNet.m_net_graph));	
			boost::write_graphviz_dp(dot_file, myNet.m_net_graph, dp);
			cout << "minimal error detected, model saved to files" << endl;
	    }
    }
    ofstream dot_file("trained_model_vizualization.dot");
	boost::dynamic_properties dp;
	dp.property("node_id", get(&NeuronP::tag, myNet.m_net_graph));
    dp.property("label", get(&NeuronP::tag, myNet.m_net_graph));
    dp.property("label", get(&SinapsP::m_weight, myNet.m_net_graph));
	boost::write_graphviz_dp(dot_file, myNet.m_net_graph, dp);
	
	myNet.save("trained_model_serialized.txt");
	
	// Ok, now we start to analyze colleced data from dumped states of neural net
	
	// First, let`s analyze edge delta_weights
	
	// Here is some magic iterating over edges
	// I have no simple translation of edge desciptor or edge iterator from myNet to saved dumped_net edge desciptor or edge iterator
	// If i save edge descriptors to vector in model, iterate over number in vector and pick up descriptor from saved model,
	// It will fail - saved descriptor is pointed to actual model state of edge, not saved.
	// I think, the reason is described in official documentaion:
	// ===cite==
	// https://www.boost.org/doc/libs/1_61_0/libs/graph/doc/quick_tour.html
	// An edge descriptor plays the same kind of role as the vertex descriptor object, it is a "black box" provided by the graph type.
	// ===cite==
	// I have no idea how to debug problems of copying and derefencing of "black box"
	// So, we have to choose another way (c) V.I. Lenin
	// vertices (source and target) are simple numbers, so we go
	
	//std::map<std::pair<int,int>, vector<double>> m_deltas;
	boost::graph_traits<Graph>::edge_iterator ei, ei_end;
	for (boost::tie(ei, ei_end) = boost::edges(myNet.m_net_graph); ei != ei_end; ++ei){
			auto source = boost::source ( *ei, myNet.m_net_graph);
			auto target = boost::target ( *ei, myNet.m_net_graph);
			cout << source << "	to	" << target << endl;
			vector<double> m_deltas, weights;
			for(auto dumped_net : myNet_dump){
				std::pair<edge_descriptor,bool> edge_saved = boost::edge(source, target, dumped_net.m_net_graph);
				assert(edge_saved.second == true); //because we _must_ have edge in saved graph, if one exists in current graph
				//cout << dumped_net.m_net_graph[edge_saved.first].m_delta_weight << endl;
				weights.push_back(dumped_net.m_net_graph[edge_saved.first].m_weight);
				m_deltas.push_back(dumped_net.m_net_graph[edge_saved.first].m_delta_weight);
			}
							
			double sum_weights = std::accumulate(weights.begin(), weights.end(), 0.0);
			double mean_weights = sum_weights / weights.size();
			
			double sum_deltaw = std::accumulate(m_deltas.begin(), m_deltas.end(), 0.0);
			double mean_deltaw = sum_deltaw / m_deltas.size();
			
			double sq_sum = std::inner_product(m_deltas.begin(), m_deltas.end(), m_deltas.begin(), 0.0);
			double stdev = std::sqrt(sq_sum / m_deltas.size() - mean_deltaw * mean_deltaw);
			cout << "W= " << mean_weights << " d= " << mean_deltaw << "	d_var=	" << stdev << " d_var/W= " << stdev/mean_weights << endl;
			
	}
}
