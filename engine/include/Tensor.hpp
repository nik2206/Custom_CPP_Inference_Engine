#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <vector>
#include <iostream>

struct Tensor {
    std::vector<float> data;
    std::vector<int> shape; // Layout: {Batch, Channel, Height, Width} (NCHW)

    void print_info() const {
        std::cout << "Tensor Shape: [";
        for (size_t i = 0; i < shape.size(); ++i) {
            std::cout << shape[i] << (i == shape.size() - 1 ? "" : ", ");
        }
        std::cout << "], Total Elements: " << data.size() << "\n";
    }
};

#endif