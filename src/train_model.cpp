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

void saveModel(Net myNet, std::string basic_name){
	myNet.save(basic_name+"_serialized.txt");
	ofstream dot_file(basic_name+".dot");
	boost::dynamic_properties dp;
	dp.property("node_id", get(&NeuronP::tag, myNet.m_net_graph));
	dp.property("label", get(&NeuronP::tag, myNet.m_net_graph));
	dp.property("label", get(&SinapsP::m_weight, myNet.m_net_graph));	
	boost::write_graphviz_dp(dot_file, myNet.m_net_graph, dp);
} 

int main()
{
	TrainingData trainData("trainingData.txt");
	vector<unsigned> topology;	
	trainData.getTopology(topology);
	
	Net myNet(topology);
	myNet.load("final_result.txt");
	vector<Net> myNet_dump; //to save model states for different input signals to analyze corellations
	
	vector<double> inputVals, targetVals, resultVals;	
	int trainingPass = 0;
	int epochs_max=1000;
	
	while(trainingPass < epochs_max){
		++trainingPass;
		myNet.eta = 100.0/(trainingPass+1000.0);
		while(!trainData.isEof()){
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
			saveModel(myNet, "best_result");
			cout << "minimal error detected, model saved to files" << endl;
	    }
    }
    saveModel(myNet, "final_result");	
}
