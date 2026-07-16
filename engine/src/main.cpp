#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include "json.hpp"
#include "Layers.hpp"

using json = nlohmann::json;

int main() {
    std::cout << "Loading model weights...\n";
    std::ifstream weight_file("../../models/fp32_model.json");
    if (!weight_file.is_open()) {
        std::cerr << "Error: Cannot find fp32_model.json! Make sure you run from the correct directory.\n";
        return 1;
    }
    
    json weights_json;
    weight_file >> weights_json;
    
    std::vector<float> conv_w = weights_json["conv_weight"].get<std::vector<float>>();
    std::vector<float> conv_b = weights_json["conv_bias"].get<std::vector<float>>();
    std::vector<float> fc_w = weights_json["fc_weight"].get<std::vector<float>>();
    std::vector<float> fc_b = weights_json["fc_bias"].get<std::vector<float>>();

    std::cout << "Loading validation set...\n";
    std::ifstream val_file("../../models/validation_set.json");
    if (!val_file.is_open()) {
        std::cerr << "Error: Cannot find validation_set.json!\n";
        return 1;
    }
    
    json val_json;
    val_file >> val_json;

    std::cout << "\n--- Running FP32 Inference on Validation Set ---\n\n";

    bool all_passed = true;

    for (const auto& item : val_json) {
        int id = item["id"];
        int target_label = item["label"];
        std::vector<float> input_image = item["data"].get<std::vector<float>>();
        std::vector<float> expected_logits = item["expected_logits"].get<std::vector<float>>();

        // --- THE FORWARD PASS ---
        std::vector<float> x = conv2d(input_image, conv_w, conv_b);
        relu(x);
        x = maxpool2d(x);
        std::vector<float> final_logits = linear(x, fc_w, fc_b);

        // --- COMPARE OUTPUTS ---
        bool passed = true;
        for (int i = 0; i < 10; ++i) {
            float diff = std::abs(final_logits[i] - expected_logits[i]);
            // TOLERANCE CHECK: 0.001 accommodates the slight drift between raw C++ CPU math and PyTorch
            if (diff > 0.001f) {
                passed = false;
            }
        }

        if (passed) {
            std::cout << "[SUCCESS] Image ID: " << id << " | Target Label: " << target_label << " | Math matched!\n";
        } else {
            all_passed = false;
            std::cout << "[FAILED] Image ID: " << id << " | Mismatch detected.\n";
            std::cout << "C++ Output:     PyTorch Output:    Diff:\n";
            for (int i = 0; i < 10; ++i) {
                float diff = std::abs(final_logits[i] - expected_logits[i]);
                std::cout << std::fixed << std::setprecision(6) 
                          << final_logits[i] << "        " 
                          << expected_logits[i] << "       "
                          << diff << "\n";
            }
            std::cout << "\n";
        }
    }

    std::cout << "\n------------------------------------------------\n";
    if (all_passed) {
        std::cout << "DAY 2 C++ TASKS COMPLETE. ALL MATH IS FLAWLESS.\n";
    } else {
        std::cout << "DAY 2 INCOMPLETE. FIX PRECISION DRIFT.\n";
    }
    
    return 0;
}