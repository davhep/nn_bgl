#pragma once

#include "training_data.h"
#include <vector>
#include <string>
#include <fstream>
#include <map>

class TrainingDataHuman : public TrainingData {
public:
    TrainingDataHuman() = default;
    explicit TrainingDataHuman(const std::string& filename);
    
    // Rule of Five
    TrainingDataHuman(const TrainingDataHuman&) = default;
    TrainingDataHuman(TrainingDataHuman&&) noexcept = default;
    TrainingDataHuman& operator=(const TrainingDataHuman&) = default;
    TrainingDataHuman& operator=(TrainingDataHuman&&) noexcept = default;
    ~TrainingDataHuman() override = default;
    
    // Core interface implementation
    [[nodiscard]] bool isEof() override;
    void getNextInputs(std::vector<double>& inputVals) override;
    void getTargetOutputs(std::vector<double>& targetOutputVals) override;
    void reset() override;
    
    // Additional functionality
    void initFile(const std::string& filename);
    void readAllFromFile(std::vector<std::pair<std::vector<double>, std::vector<double>>>& input_output_vals, 
                        int input_size, int output_size);
    
    // Utility methods
    [[nodiscard]] std::string getStatus() const override;
    [[nodiscard]] bool isFileOpen() const { return m_trainingDataFile.is_open(); }
    
private:
    std::ifstream m_trainingDataFile;
    std::string m_filename;
    
    // Helper methods
    bool parseLine(const std::string& line, std::vector<double>& inputVals, std::vector<double>& targetOutputVals);
};
