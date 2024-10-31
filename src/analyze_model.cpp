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
#include <./training_data_human.h>
#include <boost/program_options.hpp>

#define debug_high false
#define debug_low false

template<class num_type>
void showVectorVals(string label, vector<num_type> &v)
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

std::string print_to_width(double input){
	char str[] = "                              ";
	sprintf (str, "%f", input);
	std::string str_printed(str, 11);
	return(str_printed);
}

class dfs_counter_visitor: public boost::default_dfs_visitor {
public:
  dfs_counter_visitor() : vv(new std::unordered_set<vertex_descriptor>()) {}
  template < typename Vertex, typename Graph >
    void discover_vertex(Vertex u, const Graph & g) const
	{
		vv->insert(u);
		return;
    }
   boost::shared_ptr< std::unordered_set<vertex_descriptor> > vv;
};

class parent_checker{
public:
	parent_checker(vertex_descriptor v, Graph g){		
		auto indexmap = boost::get(boost::vertex_index, g);
		auto colormap = boost::make_vector_property_map<boost::default_color_type>(indexmap);
		boost::depth_first_visit(g, v, vis, colormap);
	}
	bool is_parent(vertex_descriptor v){
		return( vis.vv->count(v)>0 );
	}
private:
	dfs_counter_visitor vis;
};

int main(int argc, char* argv[])
{
	std::string input_file = "final_result_serialized.txt";
	std::string final_result_serialized = "final_result_serialized.txt";
	vector<unsigned> topology={2,10,10,1};
		
	boost::program_options::options_description desc("Allowed options");
	desc.add_options()
	// First parameter describes option name/short name
	// The second is parameter to option
	// The third is description
	("help,h", "print usage message")
	("input_file,if", boost::program_options::value(&input_file), "pathname for pre-trained filed to load and analyze")
	("topology,t", boost::program_options::value<std::vector<unsigned> >()->multitoken(), "topology as layer sizes, say 2 10 10 1 default")
	("output_file_serialized,ofs", boost::program_options::value(&final_result_serialized), "pathname for final serialized result ")
	;
	
	boost::program_options::variables_map vm;
    boost::program_options::store(parse_command_line(argc, argv, desc), vm);
    
    if (vm.count("help")) {  
		std::cout << desc << "\n";
		return 0;
	}
	if(vm.count("input_file")) input_file = vm["input_file"].as<std::string>();
	
	if (!vm["topology"].empty()) {
		topology = vm["topology"].as<vector<unsigned> >();
		showVectorVals("topology: ", topology);
	};
	
	if(vm.count("output_file_serialized")) final_result_serialized = vm["output_file_serialized"].as<std::string>();
	
	TrainingDataHuman trainData;
	trainData.InitFile("train_data.txt");
    TrainingDataHuman validateData;
    validateData.InitFile("validate_data.txt");
	
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
	std::map<vertex_descriptor, vector<double>> out_values, m_gradient;
	
	double epoch_error = 0;
	double epoch_average_error = 0;
	unsigned int epoch_num_in = 0;

    vector<std::pair<vector<double>, vector<double>>> input_output_vals;
    vector<std::pair<vector<double>, vector<double>>> input_output_validate_vals;
    trainData.ReadAllFromFile(input_output_vals, myNet.input_layer.size(), myNet.output_layer.size());
    validateData.ReadAllFromFile(input_output_validate_vals, myNet.input_layer.size(), myNet.output_layer.size());
	
    for (auto data : input_output_vals){
        myNet.feedForward(data.first);
		
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

        myNet.backProp(data.second, false); // don not upgrade weigths to avoid model change while analyze
		
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
			m_gradient[*vi].push_back(myNet.m_net_graph[*vi].m_gradient);
		}
	}
	
	cout << "Averaged error = " << epoch_error/epoch_num_in << endl;
	
	//we try to analize and update model
    
    //to iterate over original myNet and modify myNet_modified to avoid inconsistencies in inerations
    Net myNet_modified = myNet;
    
	std::vector<edge_descriptor> egdes_to_remove;

	//let`s iterate over synapses
	
	//potential neuron - no number for vertex, just two inputs - form two neurons and one output - to one neuron
	struct potential_neuron{
		vertex_descriptor input1, input2, output;
		double correlation;
	};
	
	struct potential_edge{
		vertex_descriptor input, output;
		double correlation;
	};

	std::vector<potential_neuron> neurons_to_add;
	std::vector<potential_edge> edge_to_add;
	
	for (boost::tie(ei, ei_end) = boost::edges(myNet.m_net_graph); ei != ei_end; ++ei){
		auto source = boost::source ( *ei, myNet.m_net_graph);
		auto target = boost::target ( *ei, myNet.m_net_graph);
	    auto weights_vec = weights[*ei];
	    auto m_deltas_vec = m_deltas[*ei];
	    	
		cout << source << " to " << target << " W= " << container_mean(weights_vec) << "	W_var= " << container_deviation(weights_vec) << "	d= " << container_mean(m_deltas_vec) << "	d_var=	" << container_deviation(m_deltas_vec) << " d_var/W= " << container_deviation(m_deltas_vec)/container_mean(weights_vec) << endl;
		
		// remove useless - with low weights
		if(fabs(container_mean(weights_vec)) < 0.02){
			cout << "Removing edge!!!" << endl;
            //egdes_to_remove.push_back(*ei);
            //boost::remove_edge(source, target, myNet_modified.m_net_graph);
            //continue;
		}
		
		
		//analyze correlation between gradients on the edge and output from other neuron
				
		parent_checker checker(target, myNet.m_net_graph);
		for (boost::tie(vi, vi_end) = boost::vertices(myNet.m_net_graph); vi != vi_end; ++vi){
			if(checker.is_parent(*vi) || *vi==source ) continue; //if vi is ancestor of edge target or equal to source, do not any calculations
			double correlation = container_correlation(out_values[*vi], m_deltas_vec);
			neurons_to_add.push_back(potential_neuron{*vi,source,target,correlation});
			if(fabs(correlation) > 0) cout << "Corellation with vertices: " <<  "| vi= " << myNet.m_net_graph[*vi].tag << "	corr=	" << correlation << " | " << endl;
		}
	}
	
	std::sort(neurons_to_add.begin(), neurons_to_add.end(), [](potential_neuron a, potential_neuron b) {
		return fabs(a.correlation) > fabs(b.correlation);
	});
	
	for(auto neuron: neurons_to_add)
		cout << myNet_modified.m_net_graph[neuron.input1].tag << "	"  << myNet_modified.m_net_graph[neuron.input2].tag << "	" << 
			myNet_modified.m_net_graph[neuron.output].tag <<  "	"<< neuron.correlation << endl;
	
	int neurons_to_insert = 2;
    for(int n=0; n < -neurons_to_insert; n++){
		NeuronP neuron_new;
		neuron_new.tag = ++myNet_modified.tag_max;
		SinapsP sinaps_in1, sinaps_in2, sinaps_out;
		auto neuron = neurons_to_add[n];
		cout << "Creating neuron with tag " << neuron_new.tag << "	from " << myNet.m_net_graph[neuron.input1].tag << " and " << myNet.m_net_graph[neuron.input2].tag << " to " << myNet.m_net_graph[neuron.output].tag << endl;
		auto vertex_new = boost::add_vertex(neuron_new, myNet_modified.m_net_graph);
		boost::add_edge(neuron.input1, vertex_new, sinaps_in1, myNet_modified.m_net_graph);
		boost::add_edge(neuron.input2, vertex_new, sinaps_in2, myNet_modified.m_net_graph);
		boost::add_edge(vertex_new, neuron.output, sinaps_out, myNet_modified.m_net_graph);
	}

	//analyze correlation between gradients on the edge and output from other neuron
	boost::graph_traits<Graph>::vertex_iterator vi_1, vi_end_1;
	boost::graph_traits<Graph>::vertex_iterator vi_2, vi_end_2;
	for (boost::tie(vi_1, vi_end_1) = boost::vertices(myNet.m_net_graph); vi_1 != vi_end_1; ++vi_1){
		if(myNet.m_net_graph[*vi_1].is_input) continue; //ignore input neurons
		cout << *vi_1 << "||	";
		parent_checker checker(*vi_1, myNet_modified.m_net_graph);
		for (boost::tie(vi_2, vi_end_2) = boost::vertices(myNet.m_net_graph); vi_2 != vi_end_2; ++vi_2){
			if(checker.is_parent(*vi_2)) continue; //if vi_2 is ancestor of vi_1, do not any calculations
			if(boost::edge(*vi_2, *vi_1, myNet.m_net_graph).second) continue; //if edge already exists, do nothing
			double correlation = container_correlation(m_gradient[*vi_1],out_values[*vi_2]);
			if(fabs(correlation) > 0){
				cout << *vi_2 << "	" << print_to_width(correlation) << "|";
				edge_to_add.push_back(potential_edge{*vi_2, *vi_1, correlation});
				
			}
		}
		cout << endl;
	}
	
	std::sort(edge_to_add.begin(), edge_to_add.end(), [](potential_edge a, potential_edge b) {
			return fabs(a.correlation) > fabs(b.correlation);
		});
		
	
	int edges_to_insert = 2;
	for(int n=0; edges_to_insert>0 && n<edge_to_add.size(); n++){
		auto edge = edge_to_add[n];
		parent_checker checker(edge.output, myNet_modified.m_net_graph);
		if(!checker.is_parent(edge.input)){
			SinapsP sinaps_new;
            sinaps_new.age=500;
            sinaps_new.m_weight=0;
			cout << "Adding edge: from " << myNet.m_net_graph[edge.input].tag << "	to " << myNet.m_net_graph[edge.output].tag << endl;;
			boost::add_edge(edge.input, edge.output, sinaps_new, myNet_modified.m_net_graph);
			edges_to_insert--;
		}
	};

    myNet_modified.on_topology_update();




    //more theory based way to add neuron
    //we have to calc(delta_in_1, out_2 * out_3), and add from 2 and 3 to 1
    // delta_in == m_gradient

    neurons_to_add.clear();

    int output_is_higer = myNet.topo_sorted.size() - 1;
    // output neuron must be higher, than output_is_higer at given i1
    // the idea is if at given i1 output_neuron output_is_higer is child of i1, for all next i1 neuron will be child of output_is_higer too, so we can exclude i3 < output_is_higer

    for(int out = myNet_modified.topo_sorted.size() - 1; out >= 0; out --){
        auto v_out = myNet_modified.topo_sorted[out];
        parent_checker checker(v_out, myNet_modified.m_net_graph);
        for (int i1 = myNet_modified.topo_sorted.size() - 1; i1 >= 0; i1--){
            auto v_i1 = myNet_modified.topo_sorted[i1];
            if(checker.is_parent(v_i1)) continue;
            for (int i2 = i1 - 1; i2 >= 0; i2--){
                auto v_i2 = myNet_modified.topo_sorted[i2];
                if(checker.is_parent(v_i2)) continue;
                vector<double> vec2_mult_vec3={};
                for(int n=0; n < out_values[v_i1].size(); n++)
                    vec2_mult_vec3.push_back(out_values[v_i1][n]*out_values[v_i2][n]);
                double correlation_3 = container_correlation(m_gradient[v_out], vec2_mult_vec3);
                std::cout  << i1 << "\t" << i2 << "\t" << v_out << "\t" << correlation_3 << "\n";
                neurons_to_add.push_back(potential_neuron{v_i1, v_i2, v_out, correlation_3});
            }
        }
    }


/*
    for (int i1 = myNet.topo_sorted.size() - 1; i1 >= 0; i1--){
        auto v1 = myNet.topo_sorted[i1];
        for (int i2 = i1 - 1; i2 >= 0; i2--){
            auto v2 = myNet.topo_sorted[i2];
            for (int i3 = output_is_higer; i3 >= 0; i3--){
                auto v3 = myNet.topo_sorted[i3];
                vector<double> vec2_mult_vec3;
                for(int n=0; n < out_values[v2].size(); n++)
                    vec2_mult_vec3.push_back(out_values[v2][n]*out_values[v3][n]);
                double correlation_3 = container_correlation(m_gradient[v1], vec2_mult_vec3);
                std::cout  << i1 << "\t" << i2 << "\t" << i3 << "\t" << correlation_3 << "\n";
                neurons_to_add.push_back(potential_neuron
                                         {myNet.topo_sorted[i2],
                                          myNet.topo_sorted[i3],
                                          myNet.topo_sorted[i1],
                                            correlation_3});

        }
        }
    }
    */

    std::sort(neurons_to_add.begin(), neurons_to_add.end(), [](potential_neuron a, potential_neuron b) {
        return fabs(a.correlation) > fabs(b.correlation);
    });

    std::cout << "sorting complete\n" << std::endl;

    for(auto neuron: neurons_to_add)
        cout << myNet.m_net_graph[neuron.input1].tag << "	"  << myNet.m_net_graph[neuron.input2].tag << "	" <<
            myNet.m_net_graph[neuron.output].tag <<  "	"<< neuron.correlation << endl;

    int neurons_inserted = 0;
    for(int n=0; (neurons_inserted < neurons_to_insert) && (n<neurons_to_add.size()) ; n++){
        Net myNet_modified_temp = myNet_modified;
        //will try to insert neuron to _temp. if result graph is not cyclic, copy updated _temp back to modified
        NeuronP neuron_new;
        neuron_new.tag = ++myNet_modified_temp.tag_max;
        SinapsP sinaps_in1, sinaps_in2, sinaps_out;
        sinaps_in1.age=500;
        sinaps_in2.age=500;
        sinaps_out.age=500;
        sinaps_in1.m_weight=1e-2;
        sinaps_in2.m_weight=1e-2;
        sinaps_out.m_weight=1e-2;
        auto neuron = neurons_to_add[n];
        cout << "Creating neuron with tag " << neuron_new.tag << "	from " << myNet.m_net_graph[neuron.input1].tag << " and " << myNet.m_net_graph[neuron.input2].tag << " to " << myNet.m_net_graph[neuron.output].tag << endl;
        auto vertex_new = boost::add_vertex(neuron_new, myNet_modified_temp.m_net_graph);
        boost::add_edge(neuron.input1, vertex_new, sinaps_in1, myNet_modified_temp.m_net_graph);
        boost::add_edge(neuron.input2, vertex_new, sinaps_in2, myNet_modified_temp.m_net_graph);
        boost::add_edge(vertex_new, neuron.output, sinaps_out, myNet_modified_temp.m_net_graph);
        try{
            myNet_modified_temp.on_topology_update();
        }
        catch (...)
        {
            std::cout << "Error cycling on graph detected !!!" << std::endl;
            saveModel(myNet_modified_temp, final_result_serialized.c_str(), "updated_model_DAG.dot");
            continue;
        }
        //OK, if exception is not catched, graph is DAG, and we actually add this neuron
        cout << "Actually add neuron" << endl;
        myNet_modified = myNet_modified_temp;
        neurons_inserted++;
    }
	

	
	// iterate over vertices and remove neurons without output edges
	// this is the last procedure, because of after vertex removing there is no more way to math original and modified models
    /*
    do{
		boost::tie(vi, vi_end) = boost::vertices(myNet_modified.m_net_graph);
		for (; vi != vi_end; ++vi){
			if(myNet_modified.m_net_graph[*vi].is_output || myNet_modified.m_net_graph[*vi].is_input) continue; //output neurons have no output connections at all, just inputs
			typename boost::graph_traits<Graph>::in_edge_iterator ei, ei_end;
			typename boost::graph_traits<Graph>::out_edge_iterator eo, eo_end;
			boost::tie(ei, ei_end) = in_edges(*vi, myNet_modified.m_net_graph);
			boost::tie(eo, eo_end) = out_edges(*vi, myNet_modified.m_net_graph);
			if((ei == ei_end) || (eo == eo_end)){
				cout << "Removing vertex " << *vi << "	with tag " << myNet_modified.m_net_graph[*vi].tag << endl;
				boost::clear_vertex(*vi, myNet_modified.m_net_graph);
				boost::remove_vertex(*vi, myNet_modified.m_net_graph);
				//1) after clear - some output connections for over neurons can be deleted
				//2) after remove_vertex - vertices re-numbered and iteration procedure invalidated 
				//doc says: ... If the VertexList template parameter of the adjacency_list was vecS, then all vertex descriptors, edge descriptors, and iterators for the graph are invalidated by this operation. The builtin vertex_index_t property for each vertex is renumbered so that after the operation the vertex indices still form a contiguous range [0, num_vertices(g)). ...
				//so, we have to break vertices iteration and start again
				break;
			}
		}
	}while(vi != vi_end);
    */
	myNet_modified.on_topology_update();


	
	saveModel(myNet_modified, final_result_serialized.c_str(), "updated_model.dot");
}
