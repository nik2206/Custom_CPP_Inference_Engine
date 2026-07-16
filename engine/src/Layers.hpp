#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <cstdint>

// Helper to calculate 1D index for 3D/4D tensors (NCHW format)
inline int get_index(int c, int h, int w, int height, int width) {
    return c * (height * width) + h * width + w;
}


// --- 1. Optimized INT8 Layers ---

std::vector<float> conv2d_w8a32(const std::vector<float>& input, 
                                const std::vector<float>& weights, // Renamed to match main.cpp
                                const std::vector<float>& bias) {  // Renamed to match main.cpp
    int in_h = 28, in_w = 28;
    int out_h = 26, out_w = 26; 
    int num_filters = 8;
    
    std::vector<float> output(num_filters * out_h * out_w, 0.0f);
    
    for (int f = 0; f < num_filters; ++f) {
        float b = bias[f]; // Correctly accessing the bias passed to the function
        for (int oh = 0; oh < out_h; ++oh) {
            for (int ow = 0; ow < out_w; ++ow) {
                float acc = 0.0f;
                for (int kh = 0; kh < 3; ++kh) {
                    for (int kw = 0; kw < 3; ++kw) {
                        int in_idx = get_index(0, oh + kh, ow + kw, in_h, in_w);
                        int w_idx = f * 9 + kh * 3 + kw; 
                        
                        // Using 'weights' which is already pre-dequantized
                        acc += input[in_idx] * weights[w_idx]; 
                    }
                }
                output[get_index(f, oh, ow, out_h, out_w)] = acc + b;
            }
        }
    }
    return output;
}

std::vector<float> linear_w8a32(const std::vector<float>& input, 
                                const std::vector<float>& weights, 
                                const std::vector<float>& bias) {
    int in_features = 1352;
    int out_features = 10;
    std::vector<float> output(out_features, 0.0f);
    
    for (int o = 0; o < out_features; ++o) {
        float acc = 0.0f;
        for (int i = 0; i < in_features; ++i) {
            acc += input[i] * weights[o * in_features + i];
        }
        output[o] = acc + bias[o];
    }
    return output;
}

// --- 2. FP32 Layers (Baseline) ---

inline std::vector<float> conv2d(const std::vector<float>& input, 
                                 const std::vector<float>& weights, 
                                 const std::vector<float>& bias) {
    int in_h = 28, in_w = 28;
    int out_h = 26, out_w = 26; 
    int num_filters = 8;
    
    std::vector<float> output(num_filters * out_h * out_w, 0.0f);
    
    for (int f = 0; f < num_filters; ++f) {
        for (int oh = 0; oh < out_h; ++oh) {
            for (int ow = 0; ow < out_w; ++ow) {
                float sum = bias[f];
                for (int kh = 0; kh < 3; ++kh) {
                    for (int kw = 0; kw < 3; ++kw) {
                        int in_idx = get_index(0, oh + kh, ow + kw, in_h, in_w);
                        int w_idx = f * 9 + kh * 3 + kw;
                        sum += input[in_idx] * weights[w_idx];
                    }
                }
                int out_idx = get_index(f, oh, ow, out_h, out_w);
                output[out_idx] = sum;
            }
        }
    }
    return output;
}

inline std::vector<float> linear(const std::vector<float>& input, 
                                 const std::vector<float>& weights, 
                                 const std::vector<float>& bias) {
    int in_features = 1352;
    int out_features = 10;
    
    std::vector<float> output(out_features, 0.0f);
    
    for (int o = 0; o < out_features; ++o) {
        double sum = static_cast<double>(bias[o]);
        for (int i = 0; i < in_features; ++i) {
            sum += static_cast<double>(input[i]) * static_cast<double>(weights[o * in_features + i]);
        }
        output[o] = static_cast<float>(sum);
    }
    return output;
}

// --- 3. Shared Activation & Pooling ---

inline void relu(std::vector<float>& tensor) {
    for (float& val : tensor) {
        val = std::max(0.0f, val);
    }
}

inline std::vector<float> maxpool2d(const std::vector<float>& input) {
    int in_c = 8, in_h = 26, in_w = 26;
    int out_h = 13, out_w = 13; 
    
    std::vector<float> output(in_c * out_h * out_w, 0.0f);
    
    for (int c = 0; c < in_c; ++c) {
        for (int oh = 0; oh < out_h; ++oh) {
            for (int ow = 0; ow < out_w; ++ow) {
                float max_val = -999999.0f;
                for (int kh = 0; kh < 2; ++kh) {
                    for (int kw = 0; kw < 2; ++kw) {
                        int in_idx = get_index(c, oh * 2 + kh, ow * 2 + kw, in_h, in_w);
                        max_val = std::max(max_val, input[in_idx]);
                    }
                }
                int out_idx = get_index(c, oh, ow, out_h, out_w);
                output[out_idx] = max_val;
            }
        }
    }
    return output;
}