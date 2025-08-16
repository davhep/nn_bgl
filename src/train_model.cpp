#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>
#include <map>
#include <numeric>
#include <iterator>
#include <string>

#include <nn_bgl/nn_bgl.h>
#include <nn_bgl/training_data_human.h>

#include <boost/program_options.hpp>
#include <unistd.h>

constexpr bool DEBUG_HIGH = false;

template<class num_type>
void showVectorVals(const std::string& label, const std::vector<num_type>& v) {
    std::cout << label << " ";
    for (const auto& val : v) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
}

void saveModel(Net& myNet, const std::string& serialized_file, const std::string& dot_file_name) {
    myNet.save(serialized_file);
    std::ofstream dot_file(dot_file_name);
    if (dot_file.is_open()) {
        boost::dynamic_properties dp;
        dp.property("node_id", boost::get(&NeuronP::tag, myNet.m_net_graph));
        dp.property("label", boost::get(&NeuronP::tag, myNet.m_net_graph));
        dp.property("label", boost::get(&SinapsP::m_weight, myNet.m_net_graph));
        boost::write_graphviz_dp(dot_file, myNet.m_net_graph, dp);
    }
}

void dumpVectorVals(const std::string& label, std::ofstream& data_dump, const std::vector<double>& v) {
    data_dump << label << " ";
    for (const auto& val : v) {
        data_dump << val << " ";
    }
}

int main(int argc, char* argv[]) {
    std::string input_file = "final_result_serialized.txt";
    std::string final_result_serialized = "final_result_serialized1.txt";
    std::string final_result_dot = "final_result.dot";
    bool use_gnuplot = false;
    std::string init_topology = "layers";
    NetType type_of_network = NetType::LAYERS;
    unsigned int epochs_max = 1000;
    std::vector<unsigned> topology = {2, 6, 6, 1};
    double learn_rate = 1.0;

    boost::program_options::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "print usage message")
        ("input_file,if", boost::program_options::value(&input_file), "pathname for pre-trained filed to load and continue")
        ("output_file_serialized,ofs", boost::program_options::value(&final_result_serialized), "pathname for final serialized result ")
        ("output_file_dot,ofd", boost::program_options::value(&final_result_dot), "pathname prefix for final dot result")
        ("gnuplot,gp", boost::program_options::bool_switch(&use_gnuplot), "use gnuplot dynamical plotting")
        ("init_topology,it", boost::program_options::value(&init_topology), "initial topology: \n layers \n water_fall")
        ("epochs_max,em", boost::program_options::value(&epochs_max), "number of epochs to train")
        ("topology,t", boost::program_options::value<std::vector<unsigned>>()->multitoken(), "topology as layer sizes, say 2 10 10 1 default")
        ("learn_rate,lr", boost::program_options::value(&learn_rate), "learn rate fit coefficient");

    boost::program_options::variables_map vm;
    boost::program_options::store(parse_command_line(argc, argv, desc), vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 0;
    }
    
    if (vm.count("gnuplot")) use_gnuplot = vm["gnuplot"].as<bool>();
    if (vm.count("input_file")) input_file = vm["input_file"].as<std::string>();
    if (vm.count("output_file_serialized")) final_result_serialized = vm["output_file_serialized"].as<std::string>();
    if (vm.count("output_file_dot")) final_result_dot = vm["output_file_dot"].as<std::string>();
    if (vm.count("init_topology")) init_topology = vm["init_topology"].as<std::string>();
    
    if (init_topology == "layers") type_of_network = NetType::LAYERS;
    if (init_topology == "water_fall") type_of_network = NetType::WATER_FALL;
    
    if (vm.count("epochs_max")) epochs_max = vm["epochs_max"].as<unsigned int>();
    
    if (!vm["topology"].empty()) {
        topology = vm["topology"].as<std::vector<unsigned>>();
        showVectorVals("topology: ", topology);
    }
	if(vm.count("learn_rate")) learn_rate = vm["learn_rate"].as<double>();

    TrainingDataHuman trainData;
    trainData.initFile("train_data.txt");
	    TrainingDataHuman validateData;
    validateData.initFile("validate_data.txt");
        
    Net myNet(topology, type_of_network);
    Net myNet_minimal(topology, type_of_network);
    saveModel(myNet, "init_serialized.txt", "init.dot");
    myNet.load(input_file);
    myNet.on_topology_update();
    std::cerr << "myNet.minimal_error = " << myNet.getMinimalError() << std::endl;

    std::vector<std::pair<std::vector<double>, std::vector<double>>> input_output_vals;
    std::vector<std::pair<std::vector<double>, std::vector<double>>> input_output_validate_vals;
    std::vector<double> resultVals;
    trainData.readAllFromFile(input_output_vals, static_cast<int>(myNet.input_layer.size()), static_cast<int>(myNet.output_layer.size()));
    validateData.readAllFromFile(input_output_validate_vals, static_cast<int>(myNet.input_layer.size()), static_cast<int>(myNet.output_layer.size()));

    int trainingPass = 0;
    
    FILE* gp = nullptr;
    if (use_gnuplot) {
        gp = popen("gnuplot -persist", "w");
        fprintf(gp, "set zrange [-0.2:1.0]\n");
        fprintf(gp, "splot 'model_vs_practice_dynamic.txt' u 2:3:5, 'model_vs_practice_dynamic.txt' u 2:3:7\n");
        fflush(gp);
    }

    while (trainingPass <= static_cast<int>(epochs_max)) {
        myNet.setEta(100.0 * 1000 * learn_rate / (myNet.getTrainingPass() + 1000.0) / (myNet.getTrainingPass() + 1000.0));

        boost::graph_traits<Graph>::edge_iterator ei, ei_end;
        for (boost::tie(ei, ei_end) = boost::edges(myNet.m_net_graph); ei != ei_end; ++ei) {
            auto age = myNet.m_net_graph[*ei].age++;
            myNet.m_net_graph[*ei].rate = 50.0 * learn_rate / (static_cast<double>(age) + 1000.0);
            auto source = boost::source(*ei, myNet.m_net_graph);
            int source_tag = myNet.m_net_graph[source].tag;
            auto target = boost::target(*ei, myNet.m_net_graph);
            int target_tag = myNet.m_net_graph[target].tag;
            auto weight = myNet.m_net_graph[*ei].m_weight;
            std::cout << "weights " << myNet.getTrainingPass() << " " << source_tag << " " << target_tag << " " << weight << std::endl;
        }

        double epoch_error = 0.0;
        double epoch_average_error = 0.0;
        unsigned int epoch_num_in = 0;
        std::ofstream data_dump;
        bool do_gnuplot = use_gnuplot && !(trainingPass % 10);
        if (do_gnuplot) {
            std::remove("model_vs_practice_dynamic.txt");
            data_dump.open("model_vs_practice_dynamic.txt");
        }

        for (const auto& data : input_output_vals) {
            myNet.feedForward(data.first);
            myNet.backProp(data.second, !(trainingPass == static_cast<int>(epochs_max)));
        }

        epoch_average_error = epoch_error / epoch_num_in;
        
        if (epoch_average_error < myNet.getMinimalError()) {
            myNet_minimal = myNet;
        }
        
        if (!(trainingPass % 10)) {
            epoch_error = 0.0;
            epoch_num_in = 0;
            
            for (const auto& data : input_output_vals) {
                myNet.feedForward(data.first);
                myNet.backProp(data.second, false);
                epoch_num_in++;
                epoch_error += myNet.getRecentAverageError();
            }

            std::cout << "At epoch " << myNet.getTrainingPass() << " Net recent average error: " << epoch_average_error;
            epoch_error = 0.0;
            epoch_num_in = 0;
            for (const auto& data : input_output_validate_vals) {
                myNet.feedForward(data.first);
                myNet.backProp(data.second, false);
                epoch_num_in++;
                epoch_error += myNet.getRecentAverageError();
            }			
            epoch_average_error = epoch_error / epoch_num_in;
            std::cout << " validate error = " << epoch_average_error << std::endl;
			saveModel(myNet, final_result_serialized, final_result_dot);	
	    }

        if (do_gnuplot) {
            fprintf(gp, "reread\n");
            fprintf(gp, "replot\n");
            fflush(gp);
            while (!system("test -z \"$(lsof model_vs_practice_dynamic.txt|grep train)\""));
        }
        trainData.reset();
        validateData.reset();
        
        ++trainingPass;
        		myNet.incrementTrainingPass();
    }
    
    saveModel(myNet, final_result_serialized, final_result_dot);
    saveModel(myNet_minimal, "best_result_serialized.txt", "best_result.dot");
    
    if (gp) {
        pclose(gp);
    }
    
    return 0;
}
