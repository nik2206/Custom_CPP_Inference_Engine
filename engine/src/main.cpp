#include <iostream>
#include <fstream>
#include "json.hpp" // nlohmann/json
#include "Tensor.hpp"

using json = nlohmann::json;

int main() {
    std::cout << "--- C++ Inference Engine Initialized ---\n";

    // Make sure the path matches where you dropped the files!
    std::string model_path = "../../models/fp32_model.json";
    std::ifstream file(model_path);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << model_path << "\n";
        std::cerr << "Ensure the Python script has generated the model weights file first!\n";
        return 1;
    }

    json model_json;
    file >> model_json;

    std::cout << "Successfully loaded: " << model_path << "\n";

    // 1. Parse Conv2D Weights
    if (model_json.contains("conv_weight")) {
        Tensor conv_weights;
        // Pull the flat 1D array directly from the JSON
        conv_weights.data = model_json["conv_weight"].get<std::vector<float>>();
        // Hardcode shape based on Architecture Contract (8 filters, 1 channel, 3 height, 3 width)
        conv_weights.shape = {8, 1, 3, 3}; 
        
        std::cout << "\n[SUCCESS] Parsed conv_weight (" << conv_weights.data.size() << " elements)\n";
        std::cout << "First 5 weights: ";
        for(size_t i = 0; i < 5 && i < conv_weights.data.size(); ++i) {
            std::cout << conv_weights.data[i] << " ";
        }
        std::cout << "\n";
    } else {
        std::cerr << "Error: 'conv_weight' not found in JSON.\n";
    }

    // 2. Parse Conv2D Biases
    if (model_json.contains("conv_bias")) {
        Tensor conv_bias;
        conv_bias.data = model_json["conv_bias"].get<std::vector<float>>();
        conv_bias.shape = {8}; 
        std::cout << "[SUCCESS] Parsed conv_bias\n";
    }

    // 3. Parse Linear (Fully Connected) Weights
    if (model_json.contains("fc_weight")) {
        Tensor fc_weights;
        fc_weights.data = model_json["fc_weight"].get<std::vector<float>>();
        fc_weights.shape = {10, 1352}; // 10 output classes, 1352 input features
        std::cout << "[SUCCESS] Parsed fc_weight\n";
    }

    // 4. Parse Linear Biases
    if (model_json.contains("fc_bias")) {
        Tensor fc_bias;
        fc_bias.data = model_json["fc_bias"].get<std::vector<float>>();
        fc_bias.shape = {10}; 
        std::cout << "[SUCCESS] Parsed fc_bias\n";
    }

    std::cout << "\nDay 1 C++ JSON parsing complete!\n";
    return 0;
}