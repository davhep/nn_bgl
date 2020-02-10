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

#include <boost/program_options.hpp>
#include <unistd.h>

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

void saveModel(Net myNet, std::string serialized_file,  std::string dot_file_name){
	myNet.save(serialized_file);
	ofstream dot_file(dot_file_name);
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

int main(int argc, char* argv[])
{
	std::string input_file = "final_result_serialized.txt";
	std::string final_result_serialized = "final_result_serialized.txt";
	std::string final_result_dot = "final_result.dot";
	std::string topology_file_name = "topology.txt";
	bool use_gnuplot = false;
	boost::program_options::options_description desc("Allowed options");
	desc.add_options()
	// First parameter describes option name/short name
	// The second is parameter to option
	// The third is description
	("help,h", "print usage message")
	("input_file,if", boost::program_options::value(&input_file), "pathname for pre-trained filed to load and continue")
	("output_final_serialized,ofs", boost::program_options::value(&final_result_serialized), "pathname for final serialized result ")
	("output_final_dot,ofd", boost::program_options::value(&final_result_dot), "pathname prefix for final dot result")
	("gnuplot,gp",  boost::program_options::bool_switch(&use_gnuplot), "use gnuplot dynamical plotting")
	;
    
    boost::program_options::variables_map vm;
    boost::program_options::store(parse_command_line(argc, argv, desc), vm);
    
    
	if (vm.count("help")) {  
		std::cout << desc << "\n";
		return 0;
	}
    if(vm.count("gnuplot")) use_gnuplot = vm["gnuplot"].as<bool>();  
    if(vm.count("input_file")) input_file = vm["input_file"].as<std::string>();
    if(vm.count("output_final_serialized")) final_result_serialized = vm["output_final_serialized"].as<std::string>();
    if(vm.count("output_final_dot")) final_result_dot = vm["output_final_dot"].as<std::string>();
    
	TrainingData trainData("train_data.txt");
	TrainingData validateData("validate_data.txt");
	vector<unsigned> topology;	
	trainData.getTopology(topology_file_name, topology);
	
	Net myNet(topology);
	myNet.load(input_file);
	
	cout << "myNet.minimal_error = " << myNet.minimal_error << endl;
	
	vector<double> inputVals, targetVals, resultVals;	
	int trainingPass = 0;
	int epochs_max = 100;
	
	FILE *gp;
	if(use_gnuplot){
		gp = popen("gnuplot -persist","w"); // gp - дескриптор канала
		//to dynamically plot and update 3D graph
		//fprintf(gp, "plot sin(x)\n");
		fprintf(gp, "set zrange [-0.2:1.0]\n");
		fprintf(gp, "splot 'model_vs_practice_dynamic.txt' u 2:3:5, 'model_vs_practice_dynamic.txt' u 2:3:7\n");
		fflush(gp);
	}
	while(trainingPass <= epochs_max){
		myNet.eta = 100.0/(myNet.trainingPass+1000.0);
		
		double epoch_error = 0;
		double epoch_average_error = 0;
		unsigned int epoch_num_in = 0;
		ofstream data_dump;
		if(use_gnuplot){
			remove("model_vs_practice_dynamic.txt");
			data_dump.open("model_vs_practice_dynamic.txt");
		};
		
	    // for gnuplotting by 
	    // splot 'model_vs_practice.txt' u 2:3:7, 'model_vs_practice.txt' u 2:3:7
	
		while(trainData.get(inputVals, targetVals)){
			// Get new input data and feed it forward:
			assert(inputVals.size() == myNet.input_layer.size());
			assert(targetVals.size() == myNet.output_layer.size());
			myNet.feedForward(inputVals);	
			// Train the net what the outputs should have been:
			myNet.backProp(targetVals, !(trainingPass == epochs_max)); //if last epoch, do not update weight, just calculate error	
			epoch_num_in++;
			epoch_error += myNet.getRecentAverageError();
			if(use_gnuplot){
				// Collect the net's actual results:
				myNet.getResults(resultVals);
				dumpVectorVals("inputVals	", data_dump, inputVals);
				dumpVectorVals("resultVals	", data_dump, resultVals);
				dumpVectorVals("targetVals	", data_dump, targetVals);
				data_dump << endl;
		    }
		}
		
		epoch_average_error = epoch_error/epoch_num_in;
		cerr << "At epoch " << trainingPass << " Net recent average error: " << epoch_average_error;
	    
	    if(epoch_average_error < myNet.minimal_error){
			myNet.minimal_error = epoch_average_error;
			saveModel(myNet, "best_result_serialized.txt",  "best_result.dot");
			//cout << "minimal error detected, model saved to files" << endl;
	    }
		
		epoch_error = 0;
		epoch_num_in = 0;
		while(validateData.get(inputVals, targetVals)){
			// Get new input data and feed it forward:
			assert(inputVals.size() == myNet.input_layer.size());
			assert(targetVals.size() == myNet.output_layer.size());
			myNet.feedForward(inputVals);	
			myNet.backProp(targetVals, false); 
			epoch_num_in++;
			epoch_error += myNet.getRecentAverageError();
		}
		
		epoch_average_error = epoch_error/epoch_num_in;
		cerr << " validate error = " << epoch_average_error << endl;
		
		if(use_gnuplot){
			fprintf(gp, "reread\n");
			fprintf(gp, "replot\n");
			fflush(gp);
			while(!system("test -z \"$(lsof model_vs_practice_dynamic.txt|grep train)\""));
			//while(!system("./check_lsof"));
		}
	    trainData.reset();
	    validateData.reset();
	    ++trainingPass;
    }
	saveModel(myNet, final_result_serialized, final_result_dot);	
}
