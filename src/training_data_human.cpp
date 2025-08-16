#include <nn_bgl/training_data_human.h>
#include <iostream>
#include <sstream>
#include <cassert>
#include <algorithm>

TrainingDataHuman::TrainingDataHuman(const std::string& filename) {
    initFile(filename);
}

bool TrainingDataHuman::isEof() {
    return m_trainingDataFile.eof();
}

void TrainingDataHuman::reset() {
    m_trainingDataFile.clear();
    m_trainingDataFile.seekg(0);
}

void TrainingDataHuman::initFile(const std::string& filename) {
    m_filename = filename;
    m_trainingDataFile.open(filename, std::ios::in);
    if (!m_trainingDataFile.is_open()) {
        throw std::runtime_error("Failed to open training data file: " + filename);
    }
}

void TrainingDataHuman::getNextInputs(std::vector<double>& inputVals) {
    inputVals.clear();

    std::string line;
    if (std::getline(m_trainingDataFile, line)) {
        std::istringstream ss(line);
        std::string label;
        ss >> label;

        if (label == "in:") {
            double oneValue;
            while (ss >> oneValue) {
                inputVals.push_back(oneValue);
            }
        }
    }
}

void TrainingDataHuman::getTargetOutputs(std::vector<double>& targetOutputVals) {
    targetOutputVals.clear();

    std::string line;
    if (std::getline(m_trainingDataFile, line)) {
        std::istringstream ss(line);
        std::string label;
        ss >> label;
        
        if (label == "out:") {
            double oneValue;
            while (ss >> oneValue) {
                targetOutputVals.push_back(oneValue);
            }
        }
    }
}

void TrainingDataHuman::readAllFromFile(std::vector<std::pair<std::vector<double>, std::vector<double>>>& input_output_vals, 
                                       int input_size, int output_size) {
    std::vector<double> inputs, outputs;
    
    while (!m_trainingDataFile.eof()) {
        getNextInputs(inputs);
        getTargetOutputs(outputs);
        
        if (inputs.size() == static_cast<size_t>(input_size) && 
            outputs.size() == static_cast<size_t>(output_size)) {
            input_output_vals.emplace_back(inputs, outputs);
        } else {
            std::cout << "FAIL in input/output sizes!!!!" << std::endl;
        }
    }
}

std::string TrainingDataHuman::getStatus() const {
    return "TrainingDataHuman: " + m_filename + 
           (m_trainingDataFile.is_open() ? " (open)" : " (closed)");
}
