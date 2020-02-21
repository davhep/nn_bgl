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

void saveModel(Net myNet, std::string serialized_file,  std::string dot_file_name){
	myNet.save(serialized_file);
	ofstream dot_file(dot_file_name);
	boost::dynamic_properties dp;
	dp.property("node_id", get(&NeuronP::tag, myNet.m_net_graph));
	dp.property("label", get(&NeuronP::tag, myNet.m_net_graph));
	dp.property("label", get(&SinapsP::m_weight, myNet.m_net_graph));	
	boost::write_graphviz_dp(dot_file, myNet.m_net_graph, dp);
} 


template<class container>
double container_mean(container data){
	return std::accumulate(data.begin(), data.end(), 0.0) / data.size();
}
	
template<class container>
double container_deviation(container data){
	double mean_data = container_mean(data);
	double square_sum = std::inner_product(data.begin(), data.end(), data.begin(), 0.0);
	//cout << endl << mean_data << "	xxx " << square_sum << " xxx " << data.size() << " xxx " << square_sum / data.size() - mean_data * mean_data << endl;
	double dispersion = square_sum / data.size() - mean_data * mean_data;
	if(dispersion < 0) dispersion = 0;
	return std::sqrt(dispersion);
}


int main(int argc, char* argv[])
{
	std::string input_file = "final_result_serialized.txt";
	std::string topology_file_name = "topology.txt";
	
	boost::program_options::options_description desc("Allowed options");
	desc.add_options()
	// First parameter describes option name/short name
	// The second is parameter to option
	// The third is description
	("help,h", "print usage message")
	("input_file,if", boost::program_options::value(&input_file), "pathname for pre-trained filed to load and analyze")
	;
	
	boost::program_options::variables_map vm;
    boost::program_options::store(parse_command_line(argc, argv, desc), vm);
    
    if (vm.count("help")) {  
		std::cout << desc << "\n";
		return 0;
	}
	if(vm.count("input_file")) input_file = vm["input_file"].as<std::string>();
	
	TrainingData trainData("train_data.txt");
	vector<unsigned> topology;	
	trainData.getTopology(topology_file_name, topology);
	
	Net myNet(topology);
	cout << "Loading file " << input_file << endl;
	myNet.load(input_file);
	cout << "myNet.minimal_error = " << myNet.minimal_error << endl;
	
	vector<double> inputVals, targetVals, resultVals;	
	int trainingPass = 0;
	int epochs_max=1000;
	
	ofstream data_dump("model_vs_practice.txt");
	// for gnuplotting by 
	// splot 'model_vs_practice.txt' u 2:3:5, 'model_vs_practice.txt' u 2:3:7
	boost::graph_traits<Graph>::edge_iterator ei, ei_end;
	boost::graph_traits<Graph>::vertex_iterator vi, vi_end;
	std::map<std::pair<int,int>,vector<double>>	m_deltas, weights;
	
	double epoch_error = 0;
	double epoch_average_error = 0;
	unsigned int epoch_num_in = 0;
	
	while(!trainData.isEof()){
		// Get new input data and feed it forward:
		trainData.getNextInputs(inputVals);
		if(inputVals.size() != topology[0]) continue;
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
		
		epoch_num_in++;
		epoch_error += myNet.getRecentAverageError();

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
				//weights[std::make_pair(source,target)].push_back(myNet.m_net_graph[edge_saved.first].m_weight);
				weights[std::make_pair(source,target)].push_back(fabs(myNet.m_net_graph[source].m_outputVal) * myNet.m_net_graph[edge_saved.first].m_weight);
				m_deltas[std::make_pair(source,target)].push_back(myNet.m_net_graph[edge_saved.first].m_delta_weight);
		}
	}
	
	cout << "Averaged error = " << epoch_error/epoch_num_in << endl;
	
	//we try to analize and update model

	std::vector<std::pair<int,int>> egdes_to_remove;
	
	//let`s iterate over synapses and  remove useless - with low weights
	for (boost::tie(ei, ei_end) = boost::edges(myNet.m_net_graph); ei != ei_end; ++ei){
		auto source = boost::source ( *ei, myNet.m_net_graph);
		auto target = boost::target ( *ei, myNet.m_net_graph);
	    auto weights_vec = weights[std::make_pair(source,target)];
	    auto m_deltas_vec = m_deltas[std::make_pair(source,target)];
	    	
		cout << source << " to " << target << " W= " << container_mean(weights_vec) << "	W_var= " << container_deviation(weights_vec) << "	d= " << container_mean(m_deltas_vec) << "	d_var=	" << container_deviation(m_deltas_vec) << " d_var/W= " << container_deviation(m_deltas_vec)/container_mean(weights_vec) << endl;
		if(fabs(container_mean(weights_vec)) < 0.05){
			cout << "Removing edge!!!" << endl;
			egdes_to_remove.push_back(std::pair<int,int>(source, target));
		}
	}
	
	for(auto egde_to_remove : egdes_to_remove)	boost::remove_edge(egde_to_remove.first, egde_to_remove.second, myNet.m_net_graph);
	
	//iterate over vertices and remove neurons without output edges
	std::vector<unsigned int> vertices_to_remove;
	for (boost::tie(vi, vi_end) = boost::vertices(myNet.m_net_graph); vi != vi_end; ++vi){
		cout << *vi << endl;
		auto outer_neuron = myNet.output_layer.find(*vi);
		if(outer_neuron != myNet.output_layer.end()) continue;
		typename boost::graph_traits<Graph>::out_edge_iterator ei, ei_end;
		boost::tie(ei, ei_end) = out_edges(*vi, myNet.m_net_graph);
		if(ei == ei_end){
			cout << "Removing vertex " << *vi << endl;
			vertices_to_remove.push_back(*vi);
			//if(*vi == 21) break;
		}
	}
	
	for(int n=0; n< vertices_to_remove.size(); n++){
		boost::clear_vertex(vertices_to_remove[n]-n, myNet.m_net_graph);
		boost::remove_vertex(vertices_to_remove[n]-n, myNet.m_net_graph);
	}
	
	myNet.topo_sort();
	
	for (boost::tie(vi, vi_end) = boost::vertices(myNet.m_net_graph); vi != vi_end; ++vi){
		cout << *vi << "	" << *vi_end << endl;
	}
	
	
	saveModel(myNet, "updated_model.txt", "updated_model.dot");
	
	
}
