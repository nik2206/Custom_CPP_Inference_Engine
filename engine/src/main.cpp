#include <iostream>
#include <fstream>
#include "json.hpp" // nlohmann/json
#include "Tensor.hpp"

using json = nlohmann::json;

int main() {
    std::cout << "--- C++ Inference Engine Initialized ---\n";

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

    if (model_json.contains("conv1.weight")) {
        Tensor conv1_weights;
        conv1_weights.data = model_json["conv1.weight"]["data"].get<std::vector<float>>();
        conv1_weights.shape = model_json["conv1.weight"]["shape"].get<std::vector<int>>();
        
        std::cout << "\n[SUCCESS] Parsed conv1.weight:\n";
        conv1_weights.print_info();
        
        std::cout << "First 5 weights: ";
        for(size_t i = 0; i < 5 && i < conv1_weights.data.size(); ++i) {
            std::cout << conv1_weights.data[i] << " ";
        }
        std::cout << "\n";
    }

    return 0;
}