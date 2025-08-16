#pragma once

#include <vector>
#include <string>

class TrainingData {
public:
    virtual ~TrainingData() = default;
    
    // Core interface methods
    [[nodiscard]] virtual bool isEof() = 0;
    virtual void getNextInputs(std::vector<double>& inputVals) = 0;
    virtual void getTargetOutputs(std::vector<double>& targetOutputVals) = 0;
    virtual void reset() = 0;
    
    // Utility methods
    [[nodiscard]] virtual std::string getStatus() const { return "TrainingData base class"; }
    
protected:
    // Protected constructor for base class
    TrainingData() = default;
    
    // Copy constructor and assignment operator
    TrainingData(const TrainingData&) = default;
    TrainingData& operator=(const TrainingData&) = default;
    
    // Move constructor and assignment operator
    TrainingData(TrainingData&&) noexcept = default;
    TrainingData& operator=(TrainingData&&) noexcept = default;
};
