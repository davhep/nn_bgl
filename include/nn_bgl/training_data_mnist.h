#pragma once

#include "training_data.h"
#include <vector>
#include <string>
#include <fstream>
#include <cstdint>

class TrainingDataMnist : public TrainingData {
public:
    TrainingDataMnist() = default;
    TrainingDataMnist(const std::string& filename_images, const std::string& filename_labels);
    
    // Rule of Five
    TrainingDataMnist(const TrainingDataMnist&) = default;
    TrainingDataMnist(TrainingDataMnist&&) noexcept = default;
    TrainingDataMnist& operator=(const TrainingDataMnist&) = default;
    TrainingDataMnist& operator=(TrainingDataMnist&&) noexcept = default;
    ~TrainingDataMnist() override = default;
    
    // Core interface implementation
    [[nodiscard]] bool isEof() override;
    void getNextInputs(std::vector<double>& inputVals) override;
    void getTargetOutputs(std::vector<double>& targetOutputVals) override;
    void reset() override;
    
    // Additional functionality
    void initFile(const std::string& filename_images, const std::string& filename_labels);
    
    // Utility methods
    [[nodiscard]] std::string getStatus() const override;
    [[nodiscard]] bool isFileOpen() const { return images.is_open() && labels.is_open(); }
    [[nodiscard]] unsigned int getCurrentIndex() const { return index; }
    [[nodiscard]] unsigned int getMaxIndex() const { return index_max; }
    
    // Configuration
    void setMaxIndex(unsigned int max_idx) { index_max = max_idx; }
    
private:
    std::ifstream images;
    std::ifstream labels;
    unsigned int index_max{60000};
    unsigned int index{0};
    
    // Constants
    static constexpr size_t IMAGE_SIZE = 784;
    static constexpr size_t LABEL_HEADER_SIZE = 8;
    static constexpr size_t IMAGE_HEADER_SIZE = 16;
    
    // Helper methods
    bool readImage(std::vector<double>& inputVals);
    bool readLabel(std::vector<double>& targetOutputVals);
    void validateFileHeaders();
};
