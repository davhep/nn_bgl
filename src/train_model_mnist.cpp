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
#include <nn_bgl/training_data_mnist.h>

#include <boost/program_options.hpp>
#include <unistd.h>

constexpr bool DEBUG_HIGH = true;

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
    std::string final_result_serialized = "final_result_serialized.txt";
    std::string final_result_dot = "final_result.dot";
    bool use_gnuplot = false;
    std::string init_topology = "layers";
    NetType type_of_network = NetType::LAYERS;
    unsigned int epochs_max = 1000;
    std::vector<unsigned> topology = {784, 10};
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

    TrainingDataMnist trainData;
    trainData.initFile("mnist/t10k-images.idx3-ubyte", "mnist/t10k-labels.idx1-ubyte");
    TrainingDataMnist validateData;
    validateData.initFile("mnist/train-images.idx3-ubyte", "mnist/train-labels.idx1-ubyte");
        
    Net myNet(topology, type_of_network);
    saveModel(myNet, "init_serialized.txt", "init.dot");
    myNet.load(input_file);
    myNet.on_topology_update();
    std::cerr << "myNet.minimal_error = " << myNet.getMinimalError() << std::endl;
    
    std::vector<double> inputVals, targetVals, resultVals;
    int trainingPass = 0;
    
    FILE* gp = nullptr;
    if (use_gnuplot) {
        gp = popen("gnuplot -persist", "w");
        fprintf(gp, "set zrange [-0.2:1.0]\n");
        fprintf(gp, "splot 'model_vs_practice_dynamic.txt' u 2:3:5, 'model_vs_practice_dynamic.txt' u 2:3:7\n");
        fflush(gp);
    }
    
    while (trainingPass <= static_cast<int>(epochs_max)) {
        myNet.setEta(100.0 * learn_rate / (myNet.getTrainingPass() + 1000.0));
        
        double epoch_error = 0.0;
        double epoch_average_error = 0.0;
        unsigned int epoch_num_in = 0;
        std::ofstream data_dump;
        bool do_gnuplot = use_gnuplot && !(trainingPass % 10);
        if (do_gnuplot) {
            std::remove("model_vs_practice_dynamic.txt");
            data_dump.open("model_vs_practice_dynamic.txt");
        }

        while (!trainData.isEof()) {
            trainData.getNextInputs(inputVals);
            trainData.getTargetOutputs(targetVals);
            assert(inputVals.size() == myNet.input_layer.size());
            assert(targetVals.size() == myNet.output_layer.size());
            myNet.feedForward(inputVals);
            myNet.backProp(targetVals, !(trainingPass == static_cast<int>(epochs_max)));
            epoch_num_in++;
            epoch_error += myNet.getRecentAverageError();
            
            if (DEBUG_HIGH) {
                dumpVectorVals("inputVals", data_dump, inputVals);
                dumpVectorVals("resultVals", data_dump, resultVals);
                dumpVectorVals("targetVals", data_dump, targetVals);
            }
            
            if (do_gnuplot) {
                myNet.getResults(resultVals);
                dumpVectorVals("inputVals", data_dump, inputVals);
                dumpVectorVals("resultVals", data_dump, resultVals);
                dumpVectorVals("targetVals", data_dump, targetVals);
                data_dump << std::endl;
            }
        }
        
        epoch_average_error = epoch_error / epoch_num_in;
        
        if (epoch_average_error < myNet.getMinimalError()) {
            saveModel(myNet, "best_result_serialized.txt", "best_result.dot");
        }
        
        if (!(trainingPass % 10)) {
            std::cout << "At epoch " << myNet.getTrainingPass() << " Net recent average error: " << epoch_average_error;
            epoch_error = 0.0;
            epoch_num_in = 0;
            
            while (!validateData.isEof()) {
                validateData.getNextInputs(inputVals);
                validateData.getTargetOutputs(targetVals);
                assert(inputVals.size() == myNet.input_layer.size());
                assert(targetVals.size() == myNet.output_layer.size());
                myNet.feedForward(inputVals);
                myNet.backProp(targetVals, false);
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
    
    if (gp) {
        pclose(gp);
    }
    
    return 0;
}
