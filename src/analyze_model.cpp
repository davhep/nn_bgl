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
#include <./training_data.h>

#define debug_high false
#define debug_low false

void showVectorVals(string label, vector<double> &v)
{
	cout << label << " ";
	for(unsigned i = 0; i < v.size(); ++i)
	{
		cout << v[i] << " ";
	}
	cout << endl;
}

void dumpVectorVals(string label, ofstream &data_dump, vector<double> &v)
{
	data_dump << label << " ";
	for(unsigned i = 0; i < v.size(); ++i)
	{
		data_dump << v[i] << " ";
	}
}

int main()
{
	TrainingData trainData("trainingData.txt");
	vector<unsigned> topology;	
	trainData.getTopology(topology);
	
	Net myNet(topology);
	myNet.load("final_result_serialized.txt");
	cout << "myNet.minimal_error = " << myNet.minimal_error << endl;
	
	vector<double> inputVals, targetVals, resultVals;	
	int trainingPass = 0;
	int epochs_max=1000;
	
	ofstream data_dump("model_vs_practice.txt");
	// for gnuplotting by 
	// splot 'model_vs_practice.txt' u 2:3:5, 'model_vs_practice.txt' u 2:3:7
	boost::graph_traits<Graph>::edge_iterator ei, ei_end;
	std::map<std::pair<int,int>,vector<double>>	m_deltas, weights;
	while(!trainData.isEof()){
		// Get new input data and feed it forward:
		assert(trainData.getNextInputs(inputVals) == topology[0]);
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
		myNet.backProp(targetVals, false); // don not upgrade weigths to avoid model change while analyze

		dumpVectorVals("inputVals	", data_dump, inputVals);
		dumpVectorVals("resultVals	", data_dump, resultVals);
		dumpVectorVals("targetVals	", data_dump, targetVals);
		data_dump << endl;
		
		for (boost::tie(ei, ei_end) = boost::edges(myNet.m_net_graph); ei != ei_end; ++ei){
				auto source = boost::source ( *ei, myNet.m_net_graph);
				auto target = boost::target ( *ei, myNet.m_net_graph);
				std::pair<edge_descriptor,bool> edge_saved = boost::edge(source, target, myNet.m_net_graph);
				assert(edge_saved.second == true); //because we _must_ have edge in saved graph, if one exists in current graph
				//cout << dumped_net.m_net_graph[edge_saved.first].m_delta_weight << endl;
				weights[std::make_pair(source,target)].push_back(myNet.m_net_graph[edge_saved.first].m_weight);
				m_deltas[std::make_pair(source,target)].push_back(myNet.m_net_graph[edge_saved.first].m_delta_weight);
		}
	}

	for (boost::tie(ei, ei_end) = boost::edges(myNet.m_net_graph); ei != ei_end; ++ei){
		auto source = boost::source ( *ei, myNet.m_net_graph);
		auto target = boost::target ( *ei, myNet.m_net_graph);
		auto weights_vec = weights[std::make_pair(source,target)];
	    auto m_deltas_vec = m_deltas[std::make_pair(source,target)];	
		double sum_weights = std::accumulate(weights_vec.begin(), weights_vec.end(), 0.0);
		double mean_weights = sum_weights / weights.size();
		
		double sum_deltaw = std::accumulate(m_deltas_vec.begin(), m_deltas_vec.end(), 0.0);
		double mean_deltaw = sum_deltaw / m_deltas_vec.size();
		
		double sq_sum = std::inner_product(m_deltas_vec.begin(), m_deltas_vec.end(), m_deltas_vec.begin(), 0.0);
		double stdev = std::sqrt(sq_sum / m_deltas_vec.size() - mean_deltaw * mean_deltaw);
		cout << source << " to " << target << " W= " << mean_weights << " d= " << mean_deltaw << "	d_var=	" << stdev << " d_var/W= " << stdev/mean_weights << endl;
	}
}
