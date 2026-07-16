#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>

// Helper to calculate 1D index for 3D/4D tensors (NCHW format)
inline int get_index(int c, int h, int w, int height, int width) {
    return c * (height * width) + h * width + w;
}

// 1. Conv2D (1 Input Channel, 8 Filters, 3x3 Kernel, Stride 1)
std::vector<float> conv2d(const std::vector<float>& input, 
                          const std::vector<float>& weights, 
                          const std::vector<float>& bias) {
    int in_h = 28, in_w = 28;
    int out_h = 26, out_w = 26; // 28 - 3 + 1
    int num_filters = 8;
    
    std::vector<float> output(num_filters * out_h * out_w, 0.0f);
    
    for (int f = 0; f < num_filters; ++f) {
        for (int oh = 0; oh < out_h; ++oh) {
            for (int ow = 0; ow < out_w; ++ow) {
                float sum = bias[f];
                // 3x3 Kernel loop
                for (int kh = 0; kh < 3; ++kh) {
                    for (int kw = 0; kw < 3; ++kw) {
                        int in_idx = get_index(0, oh + kh, ow + kw, in_h, in_w);
                        // Weight format: Filter x Channel x 3 x 3
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

// 2. ReLU Activation (In-place)
void relu(std::vector<float>& tensor) {
    for (float& val : tensor) {
        val = std::max(0.0f, val);
    }
}

// 3. MaxPool2D (2x2 Kernel, Stride 2)
std::vector<float> maxpool2d(const std::vector<float>& input) {
    int in_c = 8, in_h = 26, in_w = 26;
    int out_h = 13, out_w = 13; // 26 / 2
    
    std::vector<float> output(in_c * out_h * out_w, 0.0f);
    
    for (int c = 0; c < in_c; ++c) {
        for (int oh = 0; oh < out_h; ++oh) {
            for (int ow = 0; ow < out_w; ++ow) {
                float max_val = -999999.0f;
                // 2x2 pooling window
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

// 4. Linear / Fully Connected (1352 inputs -> 10 outputs)
std::vector<float> linear(const std::vector<float>& input, 
                          const std::vector<float>& weights, 
                          const std::vector<float>& bias) {
    int in_features = 1352; // 8 * 13 * 13
    int out_features = 10;
    
    std::vector<float> output(out_features, 0.0f);
    
    for (int o = 0; o < out_features; ++o) {
        float sum = bias[o];
        for (int i = 0; i < in_features; ++i) {
            // Weight format: Output_Class x Input_Feature
            sum += input[i] * weights[o * in_features + i];
        }
        output[o] = sum;
    }
    return output;
}