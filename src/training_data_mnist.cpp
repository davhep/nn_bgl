#include <nn_bgl/training_data_mnist.h>
#include <iostream>
#include <stdexcept>
#include <algorithm>



TrainingDataMnist::TrainingDataMnist(const std::string& filename_images, const std::string& filename_labels) {
    initFile(filename_images, filename_labels);
}

bool TrainingDataMnist::isEof() {
    return index >= index_max;
}

void TrainingDataMnist::reset() {
    index = 0;
}

void TrainingDataMnist::initFile(const std::string& filename_images, const std::string& filename_labels) {
    images.open(filename_images, std::ios::binary | std::ios::in);
    labels.open(filename_labels, std::ios::binary | std::ios::in);

    if (images.fail() || labels.fail()) {
        throw std::runtime_error("Failed to open MNIST files: " + filename_images + " or " + filename_labels);
    }
    
    validateFileHeaders();
}

void TrainingDataMnist::getNextInputs(std::vector<double>& inputVals) {
    if (index >= index_max) {
        throw std::runtime_error("Attempting to read beyond available data");
    }
    
    images.seekg(IMAGE_HEADER_SIZE + IMAGE_SIZE * index, std::ios::beg);
    inputVals.clear();
    inputVals.reserve(IMAGE_SIZE);
    
    std::vector<uint8_t> buffer(IMAGE_SIZE);
    images.read(reinterpret_cast<char*>(buffer.data()), IMAGE_SIZE);
    
    if (images.gcount() != static_cast<std::streamsize>(IMAGE_SIZE)) {
        throw std::runtime_error("Failed to read complete image data");
    }

    for (size_t i = 0; i < IMAGE_SIZE; i++) {
        inputVals.push_back(static_cast<double>(buffer[i]) / 255.0);
    }
    index++;
}

void TrainingDataMnist::getTargetOutputs(std::vector<double>& targetOutputVals) {
    if (index > index_max) {
        throw std::runtime_error("Attempting to read beyond available data");
    }
    
    if (targetOutputVals.size() != 10) {
        targetOutputVals.resize(10);
    }
    
    labels.seekg(LABEL_HEADER_SIZE + index - 1, std::ios::beg);
    uint8_t label_value;
    labels.read(reinterpret_cast<char*>(&label_value), 1);
    
    if (labels.gcount() != 1) {
        throw std::runtime_error("Failed to read label data");
    }
    
    std::fill(targetOutputVals.begin(), targetOutputVals.end(), 0.0);
    if (label_value < 10) {
        targetOutputVals[label_value] = 1.0;
    }
}

std::string TrainingDataMnist::getStatus() const {
    return "TrainingDataMnist: index " + std::to_string(index) + "/" + std::to_string(index_max) +
           (isFileOpen() ? " (files open)" : " (files closed)");
}

void TrainingDataMnist::validateFileHeaders() {
    // Basic validation - in a production system, you'd want more thorough validation
    if (!images.is_open() || !labels.is_open()) {
        throw std::runtime_error("MNIST files not properly opened");
    }
}
