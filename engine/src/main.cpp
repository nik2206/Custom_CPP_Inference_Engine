#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include "json.hpp"
#include "Layers.hpp"

using json = nlohmann::json;

int argmax(const std::vector<float>& logits) {
    int best_idx = 0;
    float max_val = logits[0];
    for(int i = 1; i < 10; ++i) {
        if(logits[i] > max_val) {
            max_val = logits[i];
            best_idx = i;
        }
    }
    return best_idx;
}

int main() {
    // 1. Load the INT8 Weights & Parameters
    std::ifstream weight_file("../../models/int8_model.json");
    if (!weight_file.is_open()) {
        std::cerr << "Error: Cannot find int8_model.json!\n";
        return 1;
    }
    json w_json;
    weight_file >> w_json;
    
    // LOAD AS INT8 (Huge memory savings)
    auto conv_w = w_json["conv_weight"].get<std::vector<int8_t>>();
    auto fc_w = w_json["fc_weight"].get<std::vector<int8_t>>();
    
    // Load FP32 Biases
    auto conv_b = w_json["conv_bias"].get<std::vector<float>>();
    auto fc_b = w_json["fc_bias"].get<std::vector<float>>();

    // Load Quantization Parameters
    float conv_scale = w_json["conv_weight_scale"];
    int conv_zp = w_json["conv_weight_zp"];
    float fc_scale = w_json["fc_weight_scale"];
    int fc_zp = w_json["fc_weight_zp"];

    // 2. Load the Validation Set
    std::ifstream val_file("../../models/validation_set.json");
    if (!val_file.is_open()) {
        std::cerr << "Error: Cannot find validation_set.json!\n";
        return 1;
    }
    json val_json;
    val_file >> val_json;

    std::cout << "--- Running W8A32 Inference ---\n\n";

    for (const auto& item : val_json) {
        int id = item["id"];
        auto input_image = item["data"].get<std::vector<float>>();
        auto expected_logits = item["expected_logits"].get<std::vector<float>>();

        // --- THE INT8 FORWARD PASS ---
        std::vector<float> x = conv2d_w8a32(input_image, conv_w, conv_scale, conv_zp, conv_b);
        relu(x);
        x = maxpool2d(x);
        std::vector<float> final_logits = linear_w8a32(x, fc_w, fc_scale, fc_zp, fc_b);

        // --- VERIFY ARGMAX (CLASS PREDICTION) ---
        int cxx_pred = argmax(final_logits);
        int py_pred = argmax(expected_logits);

        std::cout << "Image ID: " << id 
                  << " | PyTorch Pred: " << py_pred 
                  << " | C++ INT8 Pred: " << cxx_pred;
        
        if (cxx_pred == py_pred) {
            std::cout << "  [SUCCESS]\n";
        } else {
            std::cout << "  [FAILED - ACCURACY DROP DETECTED]\n";
        }
    }
    
    std::cout << "\nDAY 3 & 4 C++ TASKS COMPLETE.\n";
    return 0;
}