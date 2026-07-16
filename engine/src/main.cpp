#include <iostream>
#include <fstream>
#include <iomanip>
#include "json.hpp" // nlohmann/json
// #include "Layers.hpp" // Include the math file here if separated
#include "Layers.hpp"

using json = nlohmann::json;

int main() {
    // 1. Load the FP32 Weights (Day 1 logic)
    std::ifstream weight_file("../../models/fp32_model.json");
    json weights_json;
    weight_file >> weights_json;
    
    std::vector<float> conv_w = weights_json["conv_weight"].get<std::vector<float>>();
    std::vector<float> conv_b = weights_json["conv_bias"].get<std::vector<float>>();
    std::vector<float> fc_w = weights_json["fc_weight"].get<std::vector<float>>();
    std::vector<float> fc_b = weights_json["fc_bias"].get<std::vector<float>>();

    // 2. Load the Validation Set (Day 2 task)
    std::ifstream val_file("../../models/validation_set.json");
    if (!val_file.is_open()) {
        std::cerr << "Cannot find validation_set.json!\n";
        return 1;
    }
    json val_json;
    val_file >> val_json;

    std::cout << "--- Running FP32 Inference on Validation Set ---\n\n";

    // 3. Process each of the 10 test images
    for (const auto& item : val_json) {
        int id = item["id"];
        std::vector<float> input_image = item["data"].get<std::vector<float>>();
        std::vector<float> expected_logits = item["expected_logits"].get<std::vector<float>>();

        // --- THE FORWARD PASS ---
        std::vector<float> x = conv2d(input_image, conv_w, conv_b);
        relu(x);
        x = maxpool2d(x);
        std::vector<float> final_logits = linear(x, fc_w, fc_b);

        // --- COMPARE OUTPUTS ---
        std::cout << "Image ID: " << id << " | Target Label: " << item["label"] << "\n";
        
        bool passed = true;
        for (int i = 0; i < 10; ++i) {
            float diff = std::abs(final_logits[i] - expected_logits[i]);
            // Checking up to the 4th decimal place
            if (diff > 0.0001f) {
                passed = false;
            }
        }

        if (passed) {
            std::cout << "[SUCCESS] C++ Math matches PyTorch exactly!\n\n";
        } else {
            std::cout << "[FAILED] Mismatch detected.\n";
            std::cout << "C++ Output:     PyTorch Output:\n";
            for (int i = 0; i < 10; ++i) {
                std::cout << std::fixed << std::setprecision(6) 
                          << final_logits[i] << "        " 
                          << expected_logits[i] << "\n";
            }
            std::cout << "\n";
            return 1; // Halt execution so he can debug
        }
    }

    std::cout << "DAY 2 C++ TASKS COMPLETE. ALL MATH IS FLAWLESS.\n";
    return 0;
}