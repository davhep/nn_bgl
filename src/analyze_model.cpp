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
	boost::graph_traits<Graph>::edge_iterator ei, ei_end;
	for (boost::tie(ei, ei_end) = boost::edges(myNet.m_net_graph); ei != ei_end; ++ei){
			auto source = boost::source ( *ei, myNet.m_net_graph);
			auto target = boost::target ( *ei, myNet.m_net_graph);
			vector<double> m_deltas, weights;
			while(!trainData.isEof()){
				// Get new input data and feed it forward:
				assert(trainData.getNextInputs(inputVals) == topology[0]);
				myNet.feedForward(inputVals);
				dumpVectorVals("inputVals", data_dump, inputVals);
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
				std::pair<edge_descriptor,bool> edge_saved = boost::edge(source, target, myNet.m_net_graph);
				assert(edge_saved.second == true); //because we _must_ have edge in saved graph, if one exists in current graph
				//cout << dumped_net.m_net_graph[edge_saved.first].m_delta_weight << endl;
				weights.push_back(myNet.m_net_graph[edge_saved.first].m_weight);
				m_deltas.push_back(myNet.m_net_graph[edge_saved.first].m_delta_weight);
			}
			trainData.reset();
			//showVectorVals( "Weights", weights);
			//showVectorVals("m_deltas", m_deltas);
							
			double sum_weights = std::accumulate(weights.begin(), weights.end(), 0.0);
			double mean_weights = sum_weights / weights.size();
			
			double sum_deltaw = std::accumulate(m_deltas.begin(), m_deltas.end(), 0.0);
			double mean_deltaw = sum_deltaw / m_deltas.size();
			
			double sq_sum = std::inner_product(m_deltas.begin(), m_deltas.end(), m_deltas.begin(), 0.0);
			double stdev = std::sqrt(sq_sum / m_deltas.size() - mean_deltaw * mean_deltaw);
			cout << source << " to " << target << " W= " << mean_weights << " d= " << mean_deltaw << "	d_var=	" << stdev << " d_var/W= " << stdev/mean_weights << endl;
	}
}
