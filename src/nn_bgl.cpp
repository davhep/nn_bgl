#include <./nn_bgl.h>

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
	cout << label;
	for(unsigned layerNum = 0; layerNum < m_layers.size(); ++layerNum){
		for(unsigned n = 0; n < m_layers[layerNum].size(); ++n){
			cout << "	layerNum= " << layerNum << "	neuronNum= " << n << "	outputValue= " << m_layers[layerNum][n].getOutputVal();
			for(auto connection: m_layers[layerNum][n].m_outputWeights) cout << "	Weight: " << connection.weight << "	deltaWeight: " << connection.deltaWeight;
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
		cout << "output_layer_element.second= " << output_layer_element.second << "	output_layer_element.first.tag= " << m_net_graph[output_layer_element.first].tag << endl;
		cout << "targetVals[ output_layer_element.second ] = " << targetVals[ output_layer_element.second ] << "	m_net_graph[output_layer_element.first].m_outputVal= " << m_net_graph[output_layer_element.first].m_outputVal << endl;
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
			cout << "m_recentAverageError =  (m_recentAverageError * m_recentAverageSmoothingFactor + m_error) / (m_recentAverageSmoothingFactor + 1.0);" << endl;
			cout << m_recentAverageError << "	" << m_recentAverageError << "	" << m_recentAverageSmoothingFactor << "	" << m_error << "	" << m_recentAverageSmoothingFactor << endl;
		}
	   	for (int i = topo_sorted.size() - 1; i >= 0; i--){
			//reverse topological order
			auto neuron = &m_net_graph[topo_sorted[i]];
			if (debug_low) cout << "Update gradients on outputs for neuron " << neuron->tag << endl;
			double delta = 0;
		    
		    if(output_layer.find(topo_sorted[i]) != output_layer.end()){
				//ok, we are in last layer of neurons - desired outputs are fixed from output values
				delta = targetVals[output_layer.find(topo_sorted[i])->second] - neuron->m_outputVal;
				if (debug_low) cout << "Out neuron, so update on delta_out targetVals[output_layer.find(topo_sorted[i])] =	" << targetVals[output_layer.find(topo_sorted[i])->second]  << "	neuron->m_outputVal = " << neuron->m_outputVal << "delta= " << delta << endl;			
			}
			else
			{
				typename boost::graph_traits<Graph>::out_edge_iterator ei, ei_end;
				if (debug_low) cout << "Update hidden layer neuron" << endl;
			    for (boost::tie(ei, ei_end) = out_edges(topo_sorted[i], m_net_graph); ei != ei_end; ++ei){
					auto source = boost::source ( *ei, m_net_graph);
					auto target = boost::target ( *ei, m_net_graph );
					delta += m_net_graph[*ei].m_weight * m_net_graph[target].m_gradient;
					if (debug_low) cout << "target neuron " << m_net_graph[target].tag << " sm_net_graph[*ei].m_weight =	" << m_net_graph[*ei].m_weight  << "	m_net_graph[target].m_gradient = " << m_net_graph[target].m_gradient << endl;
				}
			}
			
			neuron->m_gradient = delta * transferFunctionDerivative(neuron->m_outputVal);
			if (debug_low) cout << "neuron->m_outputVal=	" << neuron->m_outputVal << endl;
			if (debug_low) cout << "delta=	" << delta  << "	transferFunctionDerivative(neuron->m_outputVal)= " << transferFunctionDerivative(neuron->m_outputVal) << endl;
			if (debug_low) cout << "neuron->tag=	" << neuron->tag  << "	neuron->m_gradient= " << neuron->m_gradient << endl;			
	     }
   
   
	     for (int i = topo_sorted.size() - 1; i >= 0; i--){
			//reverse topological order
			auto neuron = &m_net_graph[topo_sorted[i]];
			typename boost::graph_traits<Graph>::in_edge_iterator ei, ei_end;
		    for (boost::tie(ei, ei_end) = in_edges(topo_sorted[i], m_net_graph); ei != ei_end; ++ei){
				auto source = boost::source ( *ei, m_net_graph);
				auto target = boost::target ( *ei, m_net_graph );
				m_net_graph[*ei].m_delta_weight = eta * m_net_graph[source].m_outputVal * m_net_graph[target].m_gradient;
				if (update_weights) m_net_graph[*ei].m_weight += m_net_graph[*ei].m_delta_weight;
				if (debug_low) cout << "Wij " << m_net_graph[source].tag << "	to " << m_net_graph[target].tag << " updated for " << m_net_graph[*ei].m_delta_weight << endl;
			}
		}		 
}



void Net::feedForward(const vector<double> &inputVals)
{
	assert(input_layer.size() == inputVals.size());
	
	for (int i = 0; i < topo_sorted.size(); i++){
		auto vertex = topo_sorted[i];
		auto neuron = &m_net_graph[vertex];
		if(debug_low) cout << "Processing neuron " << neuron->tag << endl;
		auto input_layer_position = input_layer.find(vertex);
		if(input_layer_position != input_layer.end()){ 	//ok, we are in first layer of neurons - outputs are fixed from input values
			neuron->m_outputVal = inputVals[input_layer_position->second];
			if(debug_low) cout << "inputVals[input_layer_position->second]= " << inputVals[input_layer_position->second] << endl;
		}
		else
		{
			neuron->m_input_value = 0;
		    typename boost::graph_traits<Graph>::in_edge_iterator ei, ei_end;
		    for (boost::tie(ei, ei_end) = in_edges(vertex, m_net_graph); ei != ei_end; ++ei) {
				auto source = boost::source ( *ei, m_net_graph);
				auto target = boost::target ( *ei, m_net_graph );
				neuron->m_input_value += m_net_graph[*ei].m_weight * m_net_graph[source].m_outputVal;
				if(debug_low) cout << "m_net_graph[*ei].m_weight = " << m_net_graph[*ei].m_weight << " m_net_graph[source].m_outputVal= " <<  m_net_graph[source].m_outputVal << " neuron->m_input_value= " << neuron->m_input_value << endl;
				}
			neuron->m_outputVal = transferFunction(neuron->m_input_value);
		}
		if(debug_low) cout << "neuron->m_outputVal = " << neuron->m_outputVal << endl;
   }
}

void Net::topo_sort(){
	topo_sorted.clear();
	boost::topological_sort(m_net_graph, std::front_inserter(topo_sorted));
	cout << "A topological ordering: ";
    for (int i = 0; i < topo_sorted.size(); i++) cout << m_net_graph[topo_sorted[i]].tag  << " ";
    cout << endl;
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
					auto vertex_new = boost::add_vertex(neuron, m_net_graph);
					if(layerNum == 0) //if we are in input layer, storage ouput tags
						input_layer.insert(std::pair<vertex_descriptor, int>(vertex_new, neuronNum));
					if(layerNum == numLayers - 1) //if we are in output layer, storage ouput tags
						output_layer.insert(std::pair<vertex_descriptor, int>(vertex_new, neuronNum));
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
			cout << "Total neurons: " << total_neurons << endl;
			std::vector<unsigned int> neuron_in;
			for(int neuron_num = 0; neuron_num < total_neurons; neuron_num++){
					cout << "generating neuron " << neuron_num << endl;
					NeuronP neuron;
					neuron.tag = neuron_num;
					auto vertex_new = boost::add_vertex(neuron, m_net_graph);
					if(neuron_num < topology[0]) //if we are in input layer, storage ouput tags
						input_layer.insert(std::pair<vertex_descriptor, int>(vertex_new, neuron_num));
					if( neuron_num >= (total_neurons - topology.back()) ) //if we are in output layer, storage ouput tags
						output_layer.insert(std::pair<vertex_descriptor, int>(vertex_new, neuron_num - (total_neurons - topology.back())));
					if(neuron_num >= topology[0]){//so, we are have to connect non-input neuron to some other neurons
						unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
						std::shuffle(neuron_in.begin(), neuron_in.end(), std::default_random_engine(seed));
						int n_max=4;//how many neurons output connect to this neuron input
						for(int n=0;(n< n_max) && (n<neuron_num);n++){
							unsigned int input_neuron = neuron_in[n];
							cout << "input neuron= " << input_neuron << endl;
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
						cout << *v << "	" << outer_neuron.first << endl;
						SinapsP sinaps;
						sinaps.m_weight = double(rand() % 100)/100;
						boost::add_edge( *v, outer_neuron.first, sinaps, m_net_graph);
					}
				}
			}
		break;
	}
		
	topo_sort();

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
		cout << "Loading Net ..." << endl;
		if(debug_low){
			cout << "topo size before: " << topo_sorted.size()
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
		  cout << "Loading faild. Caught exception number:  " << a << endl;
		  return;
		}
		
		if(debug_low){
			cout << "topo size after: " << topo_sorted.size()
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
}
