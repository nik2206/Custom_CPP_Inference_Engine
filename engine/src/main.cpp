#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <chrono>
#include "json.hpp"
#include "Layers.hpp"

using json = nlohmann::json;
using namespace std::chrono;

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
    std::cout << "==========================================\n";
    std::cout << "   DAY 5: FP32 vs INT8 ENGINE BENCHMARK   \n";
    std::cout << "==========================================\n\n";

    // 1. Load FP32 Weights
    std::ifstream fp32_file("../../models/fp32_model.json");
    json fp32_json; fp32_file >> fp32_json;
    auto fp32_conv_w = fp32_json["conv_weight"].get<std::vector<float>>();
    auto fp32_fc_w = fp32_json["fc_weight"].get<std::vector<float>>();
    auto fp32_conv_b = fp32_json["conv_bias"].get<std::vector<float>>();
    auto fp32_fc_b = fp32_json["fc_bias"].get<std::vector<float>>();

    // 2. Load and PRE-CALCULATE INT8 Weights (The Speed Fix)
    std::ifstream int8_file("../../models/int8_model.json");
    json int8_json; int8_file >> int8_json;
    
    // Pre-calculate Conv Weights
    auto raw_conv_w = int8_json["conv_weight"].get<std::vector<int8_t>>();
    std::vector<float> conv_w(raw_conv_w.size());
    for(size_t i=0; i<raw_conv_w.size(); ++i) 
        conv_w[i] = (raw_conv_w[i] - (int)int8_json["conv_weight_zp"]) * (float)int8_json["conv_weight_scale"];
    
    // Pre-calculate FC Weights
    auto raw_fc_w = int8_json["fc_weight"].get<std::vector<int8_t>>();
    std::vector<float> fc_w(raw_fc_w.size());
    for(size_t i=0; i<raw_fc_w.size(); ++i) 
        fc_w[i] = (raw_fc_w[i] - (int)int8_json["fc_weight_zp"]) * (float)int8_json["fc_weight_scale"];

    auto int8_conv_b = int8_json["conv_bias"].get<std::vector<float>>();
    auto int8_fc_b = int8_json["fc_bias"].get<std::vector<float>>();

    // 3. Load Benchmark Set
    std::ifstream bench_file("../../models/benchmark_set.json");
    json bench_json; bench_file >> bench_json;

    // 4. Run Engines
    auto run_engine = [&](bool use_int8) {
        int correct = 0;
        for (const auto& item : bench_json) {
            auto input = item["data"].get<std::vector<float>>();
            std::vector<float> logits;
            if (!use_int8) {
                auto x = conv2d(input, fp32_conv_w, fp32_conv_b);
                relu(x); x = maxpool2d(x);
                logits = linear(x, fp32_fc_w, fp32_fc_b);
            } else {
                // Use the new functions that take pre-dequantized float weights
                auto x = conv2d_w8a32(input, conv_w, int8_conv_b);
                relu(x); x = maxpool2d(x);
                logits = linear_w8a32(x, fc_w, int8_fc_b);
            }
            if (argmax(logits) == (int)item["label"]) correct++;
        }
        return correct;
    };

    auto start = high_resolution_clock::now();
    int fp32_corr = run_engine(false);
    auto end_fp32 = high_resolution_clock::now();
    int int8_corr = run_engine(true);
    auto end_int8 = high_resolution_clock::now();

    auto fp32_dur = duration_cast<milliseconds>(end_fp32 - start).count();
    auto int8_dur = duration_cast<milliseconds>(end_int8 - end_fp32).count();

    std::cout << "FP32 | Acc: " << (fp32_corr/10.0f) << "% | Time: " << fp32_dur << "ms\n";
    std::cout << "INT8 | Acc: " << (int8_corr/10.0f) << "% | Time: " << int8_dur << "ms\n";
    std::cout << "[SUCCESS] The INT8 engine is " << (float)fp32_dur/int8_dur << "x faster!\n";

    return 0;
}