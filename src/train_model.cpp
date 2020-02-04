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
	vector<Net> myNet_dump; //to save model states for different input signals to analyze corellations
	
	vector<double> inputVals, targetVals, resultVals;	
	int trainingPass = 0;
	int epochs_max=1000;
	

	
	FILE *gp = popen("gnuplot -persist","w"); // gp - дескриптор канала
	//to dynamically plot and update 3D graph
	//fprintf(gp, "plot sin(x)\n");
	fprintf(gp, "splot 'model_vs_practice_dynamic.txt' u 2:3:5, 'model_vs_practice_dynamic.txt' u 2:3:7\n");
	fflush(gp);
	
	while(trainingPass <= epochs_max){
		++trainingPass;
		myNet.eta = 100.0/(myNet.trainingPass+1000.0);
		
		double epoch_error = 0;
		double epoch_average_error = 0;
		unsigned int epoch_num_in = 0;
		remove("model_vs_practice_dynamic.txt");
		ofstream data_dump("model_vs_practice_dynamic.txt");
	    // for gnuplotting by 
	    // splot 'model_vs_practice.txt' u 2:3:7, 'model_vs_practice.txt' u 2:3:7
	
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
			myNet.backProp(targetVals, !(trainingPass == epochs_max)); //if last ecphc, do not update weight, just calculate error	
			if(trainingPass == epochs_max) myNet_dump.push_back(myNet);			
			epoch_num_in++;
			epoch_error += myNet.getRecentAverageError();
			
			dumpVectorVals("inputVals	", data_dump, inputVals);
			dumpVectorVals("resultVals	", data_dump, resultVals);
			dumpVectorVals("targetVals	", data_dump, targetVals);
			data_dump << endl;
		}
		fprintf(gp, "reread\n");
		fprintf(gp, "replot\n");
		fflush(gp);
		
		epoch_average_error = epoch_error/epoch_num_in;

	    cerr << "At epoch " << trainingPass << " Net recent average error: " << epoch_average_error << endl;
	    if(epoch_average_error < myNet.minimal_error){
			myNet.minimal_error = epoch_average_error;
			saveModel(myNet, "best_result");
			cout << "minimal error detected, model saved to files" << endl;
	    }
	    trainData.reset();
    }
	saveModel(myNet, "final_result");	
}
