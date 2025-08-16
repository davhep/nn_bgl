#include <nn_bgl/nn_bgl.h>

// ****************** class Net ******************



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
	cerr <<label;
	for(unsigned layerNum = 0; layerNum < m_layers.size(); ++layerNum){
		for(unsigned n = 0; n < m_layers[layerNum].size(); ++n){
			cerr <<"	layerNum= " << layerNum << "	neuronNum= " << n << "	outputValue= " << m_layers[layerNum][n].getOutputVal();
			for(auto connection: m_layers[layerNum][n].m_outputWeights) cerr <<"	Weight: " << connection.weight << "	deltaWeight: " << connection.deltaWeight;
			//auto neuron_dump = m_layers[layerNum][n].dump();
			//m_dump.push_back(m_layers[layerNum][n]);
		}
	}*/
}

double Net::m_recentAverageSmoothingFactor = 100.0; // Number of training samples to average over

void Net::getResults(vector<double> &resultVals) const
{
	resultVals.clear();
	for(auto output_layer_element: output_layer)
		resultVals.push_back(m_net_graph[output_layer_element.first].m_outputVal);
}

void Net::backProp(const std::vector<double> &targetVals, bool update_weights)
{
	assert(targetVals.size() == output_layer.size());
	
	// Calculate overal net error (RMS of output neuron errors)

	m_error = 0.0;

	for(auto output_layer_element: output_layer)
	{
		double delta = targetVals[ output_layer_element.second ] - m_net_graph[output_layer_element.first].m_outputVal;
		m_error += delta *delta;
		if(debug_low){
		cerr <<"output_layer_element.second= " << output_layer_element.second << "	output_layer_element.first.tag= " << m_net_graph[output_layer_element.first].tag << endl;
		cerr <<"targetVals[ output_layer_element.second ] = " << targetVals[ output_layer_element.second ] << "	m_net_graph[output_layer_element.first].m_outputVal= " << m_net_graph[output_layer_element.first].m_outputVal << endl;
	}
	}
	
	m_error /= output_layer.size(); // get average error squared
	m_error = sqrt(m_error); // RMS

	// Implement a recent average measurement:

	//m_recentAverageError = 
	//		(m_recentAverageError * m_recentAverageSmoothingFactor + m_error)
	//		/ (m_recentAverageSmoothingFactor + 1.0);
	
	m_recentAverageError = m_error;
	
	    if(debug_low){
			cerr <<"m_recentAverageError =  (m_recentAverageError * m_recentAverageSmoothingFactor + m_error) / (m_recentAverageSmoothingFactor + 1.0);" << endl;
			cerr <<m_recentAverageError << "	" << m_recentAverageError << "	" << m_recentAverageSmoothingFactor << "	" << m_error << "	" << m_recentAverageSmoothingFactor << endl;
        }

        //OK, at first stage - update neuron deltas and gradients
	   	for (int i = topo_sorted.size() - 1; i >= 0; i--){
			//reverse topological order
			auto neuron = &m_net_graph[topo_sorted[i]];
            if (debug_low) cerr <<"Update gradients on outputs for neuron " << neuron->tag << endl;
			double delta = 0;
		    
		    if(neuron->is_output){
				//ok, we are in last layer of neurons - desired outputs are fixed from output values
				delta = targetVals[output_layer.find(topo_sorted[i])->second] - neuron->m_outputVal;
				if (debug_low) cerr <<"Out neuron, so update on delta_out targetVals[output_layer.find(topo_sorted[i])] =	" << targetVals[output_layer.find(topo_sorted[i])->second]  << "	neuron->m_outputVal = " << neuron->m_outputVal << "delta= " << delta << endl;			
			}
			else
			{
				typename boost::graph_traits<Graph>::out_edge_iterator ei, ei_end;
				if (debug_low) cerr <<"Update hidden layer neuron" << endl;
			    for (boost::tie(ei, ei_end) = out_edges(topo_sorted[i], m_net_graph); ei != ei_end; ++ei){
					auto source = boost::source ( *ei, m_net_graph);
					auto target = boost::target ( *ei, m_net_graph );
					delta += m_net_graph[*ei].m_weight * m_net_graph[target].m_gradient;
					if (debug_low) cerr <<"target neuron " << m_net_graph[target].tag << " sm_net_graph[*ei].m_weight =	" << m_net_graph[*ei].m_weight  << "	m_net_graph[target].m_gradient = " << m_net_graph[target].m_gradient << endl;
				}
			}
			
			neuron->m_gradient = delta * transferFunctionDerivative(neuron->m_outputVal);
			if (debug_low) cerr <<"neuron->m_outputVal=	" << neuron->m_outputVal << endl;
			if (debug_low) cerr <<"delta=	" << delta  << "	transferFunctionDerivative(neuron->m_outputVal)= " << transferFunctionDerivative(neuron->m_outputVal) << endl;
			if (debug_low) cerr <<"neuron->tag=	" << neuron->tag  << "	neuron->m_gradient= " << neuron->m_gradient << endl;			
	     }
   

        //OK, at second stage - update on edge weights


         for (int i = topo_sorted.size() - 1; i >= 0; i--){
            //reverse topological order
            auto neuron = &m_net_graph[topo_sorted[i]];

            typename boost::graph_traits<Graph>::in_edge_iterator ei, ei_end;
            for (boost::tie(ei, ei_end) = in_edges(topo_sorted[i], m_net_graph); ei != ei_end; ++ei){
				auto source = boost::source ( *ei, m_net_graph);
				auto target = boost::target ( *ei, m_net_graph );
                //m_net_graph[*ei].m_delta_weight = m_net_graph[*ei].rate * m_net_graph[source].m_outputVal * m_net_graph[target].m_gradient;
                m_net_graph[*ei].m_delta_weight = m_net_graph[*ei].rate * m_net_graph[source].m_outputVal * m_net_graph[target].m_gradient;

				if (update_weights) m_net_graph[*ei].m_weight += m_net_graph[*ei].m_delta_weight;
				if (debug_low) cerr <<"Wij " << m_net_graph[source].tag << "	to " << m_net_graph[target].tag << " updated for " << m_net_graph[*ei].m_delta_weight << endl;
			}
        }
}



void Net::feedForward(const vector<double> &inputVals)
{
       assert(input_layer.size() == inputVals.size());
	
	for (int i = 0; i < topo_sorted.size(); i++){
		auto vertex = topo_sorted[i];
		auto neuron = &m_net_graph[vertex];
		if(debug_low) cerr <<"Processing neuron " << neuron->tag << endl;
		
		if(neuron->is_input){
			neuron->m_outputVal = inputVals[neuron->input_signal];
			if(debug_low) cerr <<"inputVals[neuron.input_signal]= " << inputVals[neuron->input_signal] << endl;
		}
		else
		{
			neuron->m_input_value = 0;
		    typename boost::graph_traits<Graph>::in_edge_iterator ei, ei_end;
		    for (boost::tie(ei, ei_end) = in_edges(vertex, m_net_graph); ei != ei_end; ++ei) {
				auto source = boost::source ( *ei, m_net_graph);
				auto target = boost::target ( *ei, m_net_graph );
				neuron->m_input_value += m_net_graph[*ei].m_weight * m_net_graph[source].m_outputVal;
				if(debug_low) cerr <<"m_net_graph[*ei].m_weight = " << m_net_graph[*ei].m_weight << " m_net_graph[source].m_outputVal= " <<  m_net_graph[source].m_outputVal << " neuron->m_input_value= " << neuron->m_input_value << endl;
				}
			neuron->m_outputVal = transferFunction(neuron->m_input_value);
		}
		if(debug_low) cerr <<"neuron->m_outputVal = " << neuron->m_outputVal << endl;
   }
}

void Net::on_topology_update(){
	//if we change topology (add/remove edge or vertex), we have to update some secondary information about graph
	
	//we have to recalculate toloplogical order of vertices for correct forward/backward procedure
	topo_sorted.clear();
	boost::topological_sort(m_net_graph, std::front_inserter(topo_sorted));
	cerr <<"A topological ordering: ";
    for (int i = 0; i < topo_sorted.size(); i++) cerr <<m_net_graph[topo_sorted[i]].tag  << " ";
    cerr <<endl;
    
    //because of re-numbering vertices after remove vertices, we have to update output/input lists
    input_layer.clear();
    output_layer.clear();
    boost::graph_traits<Graph>::vertex_iterator vi, vi_end;
    for (boost::tie(vi, vi_end) = boost::vertices(m_net_graph); vi != vi_end; ++vi){
		if(m_net_graph[*vi].is_input) input_layer.insert(std::pair<vertex_descriptor, int>(*vi, m_net_graph[*vi].input_signal));
		if(m_net_graph[*vi].is_output) output_layer.insert(std::pair<vertex_descriptor, int>(*vi, m_net_graph[*vi].output_signal));
	}
}

Net::Net(const vector<unsigned> &topology, net_type type_of_network)
{
	input_layer.clear();
	output_layer.clear();
	minimal_error = 1e6;
	unsigned numLayers = topology.size();
	unsigned neurons_total = 0;
	switch(type_of_network){
		case(layers):
			for(unsigned layerNum = 0; layerNum < numLayers; ++layerNum){
				unsigned numInputs = layerNum == 0 ? 0 : topology[layerNum - 1];
				unsigned first_prev_neuron = neurons_total - numInputs;
				for(unsigned neuronNum = 0; neuronNum < topology[layerNum]; ++neuronNum){
					NeuronP neuron;
					neuron.tag = neurons_total;
                    tag_max = neuron.tag;
					if(layerNum == 0){
						//if we are in input layer, storage ouput tags
						neuron.is_input = true;
						neuron.input_signal = neuronNum;
					}
					if(layerNum == numLayers - 1){
						//if we are in output layer, storage ouput tags
						neuron.is_output = true;
						neuron.output_signal = neuronNum;
					}
					auto vertex_new = boost::add_vertex(neuron, m_net_graph);

					for(unsigned neuronNum_prev = 0; neuronNum_prev < numInputs; neuronNum_prev++){
						SinapsP sinaps;
						sinaps.m_weight = double((neuronNum_prev+neuronNum) % 10)/10;
						boost::add_edge(first_prev_neuron+neuronNum_prev, neurons_total, sinaps, m_net_graph);
					 }
					 neurons_total++;
				}
			}
		break;
		case(water_fall):
			unsigned int total_neurons = std::accumulate(topology.begin(), topology.end(), 0);
			cerr <<"Total neurons: " << total_neurons << endl;
			std::vector<unsigned int> neuron_in;
			for(int neuron_num = 0; neuron_num < total_neurons; neuron_num++){
					cerr <<"generating neuron " << neuron_num << endl;
					NeuronP neuron;
					neuron.tag = neuron_num;
					auto vertex_new = boost::add_vertex(neuron, m_net_graph);
					tag_max++;
					if(neuron_num < topology[0]) //if we are in input layer, storage ouput tags
						neuron.input_signal = neuron_num;
					if( neuron_num >= (total_neurons - topology.back()) ) //if we are in output layer, storage ouput tags
						neuron.output_signal = neuron_num;
					if(neuron_num >= topology[0]){//so, we are have to connect non-input neuron to some other neurons
						unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
						std::shuffle(neuron_in.begin(), neuron_in.end(), std::default_random_engine(seed));
						int n_max=4;//how many neurons output connect to this neuron input
						for(int n=0;(n< n_max) && (n<neuron_num);n++){
							unsigned int input_neuron = neuron_in[n];
							cerr <<"input neuron= " << input_neuron << endl;
							SinapsP sinaps;
							sinaps.m_weight = double(rand() % 100)/100;
							boost::add_edge(input_neuron, neuron_num, sinaps, m_net_graph);
						}
					}
					if( neuron_num < (total_neurons - topology.back()) ) neuron_in.push_back(neuron_num);				
			}
			//ok, we have to check if some neuron have no outputs, and connect them to some outer neurons
			Graph::vertex_iterator v, vend;
			for (boost::tie(v, vend) = vertices(m_net_graph); v != vend; ++v){
				auto outer_neuron = output_layer.find(*v);
				if(outer_neuron != output_layer.end()) continue;
				auto out_edges = boost::out_edges(*v, m_net_graph);
		        if(out_edges.first == out_edges.second){
					for(auto outer_neuron: output_layer){
						cerr <<*v << "	" << outer_neuron.first << endl;
						SinapsP sinaps;
						sinaps.m_weight = double(rand() % 100)/100;
						boost::add_edge( *v, outer_neuron.first, sinaps, m_net_graph);
					}
				}
			}
		break;
	}
	on_topology_update();
}

void Net::save( std::string path){
	std::ofstream file{path};
	if(file){
		boost::archive::text_oarchive oa{file};
		oa << *this;
	}
}

void Net::load( std::string path){
	std::ifstream file{path};
	if(file){
		cerr <<"Loading Net ..." << endl;
		if(debug_low){
			cerr <<"topo size before: " << topo_sorted.size()
				<< "	edges size before: " << boost::num_edges(m_net_graph)
				<< "	vertexes size before: " << boost::num_vertices(m_net_graph) << endl;
		}
	    try{
			boost::archive::text_iarchive ia{file};
			m_net_graph.clear();//because of deserialization add new vertices and edges, w/o removing old ones
			ia >> *this;
		}
		catch(int a)
		{
		  cerr <<"Loading faild. Caught exception number:  " << a << endl;
		  return;
		}
		
		if(debug_low){
			cerr <<"topo size after: " << topo_sorted.size()
				<< "	edges size after: " << boost::num_edges(m_net_graph)
				<< "	vertexes size after: " << boost::num_vertices(m_net_graph) << endl;
		}
	}
}
template<class Archive>
void Net::serialize(Archive & ar, const unsigned int file_version){
	   ar & m_net_graph;
	   ar & eta;
	   ar & input_layer;
	   ar & output_layer;
	   ar & topo_sorted;
	   ar & m_recentAverageError;
	   ar & minimal_error;
	   ar & trainingPass;
	   ar & tag_max;
}
