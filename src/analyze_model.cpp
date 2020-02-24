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

template<class container>
double container_correlation(container data_x, container data_y){
	double E_x = container_mean(data_x);
	double E_y = container_mean(data_y);
	double E_xx = std::inner_product(data_x.begin(), data_x.end(), data_x.begin(), 0.0)/data_x.size();
	double E_yy = std::inner_product(data_y.begin(), data_y.end(), data_y.begin(), 0.0)/data_y.size();
	double E_xy = std::inner_product(data_x.begin(), data_x.end(), data_y.begin(), 0.0)/data_x.size();
	double correlation = (E_xy-E_x*E_y)/std::sqrt(E_xx-E_x*E_x)/std::sqrt(E_yy-E_y*E_y);
	return correlation;
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
	std::map<edge_descriptor, vector<double>>	m_deltas, weights;
	std::map<vertex_descriptor, vector<double>> out_values;
	
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
				weights[*ei].push_back(myNet.m_net_graph[*ei].m_weight);
				m_deltas[*ei].push_back(myNet.m_net_graph[*ei].m_delta_weight);
		}
		for (boost::tie(vi, vi_end) = boost::vertices(myNet.m_net_graph); vi != vi_end; ++vi){
			out_values[*vi].push_back(myNet.m_net_graph[*vi].m_outputVal);
		}
	}
	
	cout << "Averaged error = " << epoch_error/epoch_num_in << endl;
	
	//we try to analize and update model

	std::vector<edge_descriptor> egdes_to_remove;
	
	//let`s iterate over synapses
	for (boost::tie(ei, ei_end) = boost::edges(myNet.m_net_graph); ei != ei_end; ++ei){
		auto source = boost::source ( *ei, myNet.m_net_graph);
		auto target = boost::target ( *ei, myNet.m_net_graph);
	    auto weights_vec = weights[*ei];
	    auto m_deltas_vec = m_deltas[*ei];
	    	
		cout << source << " to " << target << " W= " << container_mean(weights_vec) << "	W_var= " << container_deviation(weights_vec) << "	d= " << container_mean(m_deltas_vec) << "	d_var=	" << container_deviation(m_deltas_vec) << " d_var/W= " << container_deviation(m_deltas_vec)/container_mean(weights_vec) << endl;
		
		// remove useless - with low weights
		if(fabs(container_mean(weights_vec)) < 0.05){
			cout << "Removing edge!!!" << endl;
			egdes_to_remove.push_back(*ei);
		}
		
		
		//analyze correlation between gradients on the edge and output from other neuron
		for (boost::tie(vi, vi_end) = boost::vertices(myNet.m_net_graph); vi != vi_end; ++vi){
			double correlation = container_correlation(out_values[*vi], m_deltas_vec);
			if(fabs(correlation) > 0.15) cout << "Corellation with vertices: " <<  "| vi= " << *vi << "	corr=	" << correlation << " | " << endl;
		}
	}
	
	for(auto egde_to_remove : egdes_to_remove)	boost::remove_edge(egde_to_remove, myNet.m_net_graph);
	
	//iterate over vertices and remove neurons without output edges
	do{
		boost::tie(vi, vi_end) = boost::vertices(myNet.m_net_graph);
		for (; vi != vi_end; ++vi){
			if(myNet.m_net_graph[*vi].is_output) continue; //output neurons have no output connections at all, just inputs
			typename boost::graph_traits<Graph>::out_edge_iterator ei, ei_end;
			boost::tie(ei, ei_end) = out_edges(*vi, myNet.m_net_graph);
			if(ei == ei_end){
				cout << "Removing vertex " << *vi << "	with tag " << myNet.m_net_graph[*vi].tag << endl;
				boost::clear_vertex(*vi, myNet.m_net_graph);
				boost::remove_vertex(*vi, myNet.m_net_graph);
				//1) after clear - some output connections for over neurons can be deleted
				//2) after remove_vertex - vertices re-numbered and iteration procedure invalidated 
				//doc says: ... If the VertexList template parameter of the adjacency_list was vecS, then all vertex descriptors, edge descriptors, and iterators for the graph are invalidated by this operation. The builtin vertex_index_t property for each vertex is renumbered so that after the operation the vertex indices still form a contiguous range [0, num_vertices(g)). ...
				//so, we have to break vertices iteration and start again
				break;
			}
		}
	}while(vi != vi_end);
	
	myNet.on_topology_update();

	saveModel(myNet, "updated_model.txt", "updated_model.dot");
}
