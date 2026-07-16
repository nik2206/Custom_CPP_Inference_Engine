# Custom C++ Inference Engine

A from-scratch C++17 inference engine implementing a Convolutional Neural Network (CNN) for Fashion-MNIST classification, featuring dual-precision execution paths (FP32 baseline vs. INT8 weight-only quantization) with on-the-fly weight dequantization. Built entirely from scratch using zero external ML libraries.

## Overview

This project demonstrates low-level ML systems engineering: hand-written tensor operations, quantization-aware inference, and performance benchmarking — all in modern C++ with no dependencies beyond the standard library and `nlohmann/json`.

**Architecture**: Conv2d(1→8, 3×3) → ReLU → MaxPool2d(2×2) → Linear(1352→10)

## Contributors

- **Abhinav Venkata K** — Inference engine, quantization, benchmarking, CMake build system
- **Nikhil (nik2206)** — PyTorch training pipeline, quantization export, model validation

## Features

- **Zero ML dependencies** — Pure C++17 + STL + `nlohmann/json`
- **Dual inference paths**: FP32 baseline and INT8 weight-only quantization
- **Per-tensor asymmetric quantization** with scale + zero-point calibration
- **On-the-fly dequantization** — INT8 weights stay compressed in RAM; converted to FP32 in-register during GEMM
- **4× model compression** (5.4 KB → 1.4 KB weights)
- **2.58× inference speedup** with <1% accuracy drop
- **End-to-end pipeline**: PyTorch training → JSON export → C++ inference → numerical validation

## Performance Results

| Metric                   | FP32      | INT8 (Weight-Only) |
|--------------------------|-----------|---------------------|
| Accuracy                 | 87.9%     | 87.3% (-0.6%)       |
| Latency (10k inferences) | ~1,240 ms | ~480 ms             |
| Speedup                  | 1.0×      | 2.58×               |
| Weight Memory            | 5.4 KB    | 1.4 KB              |
| Per-Inference            | 0.124 ms  | 0.048 ms            |

*Benchmarked on Intel i7-11800H (11th Gen), 1000 images × 10 iterations*

## Project Structure

```
Custom_CPP_Inference_Engine/
├── engine/
│   ├── include/
│   │   ├── Tensor.hpp        # NCHW tensor struct with shape metadata
│   │   └── json.hpp          # nlohmann/json (header-only)
│   ├── src/
│   │   ├── Layers.hpp        # Core ops: conv2d, linear, relu, maxpool2d (+ INT8 variants)
│   │   ├── main.cpp          # Benchmark driver
│   │   └── json.hpp          # nlohmann/json copy
│   ├── CMakeLists.txt
│   └── build/                # Build artifacts
├── models/
│   ├── fp32_model.json       # FP32 weights + biases
│   ├── int8_model.json       # INT8 weights + scale + zero-point per tensor
│   ├── benchmark_set.json    # 1000 test images + labels
│   ├── test_image.json       # Single test image + expected logits
│   └── validation_set.json   # 10 images + expected logits for numerical parity
├── training/
│   └── training.py           # Colab PyTorch script (5 epochs, Adam, lr=0.001)
└── README.md
```

## Quick Start

### Prerequisites
- C++17 compiler (GCC 9+, Clang 10+, MSVC 19.20+)
- CMake 3.10+

### Build
```bash
cd engine
cmake -B build
cmake --build build
```

### Run Benchmark
```bash
cd engine/build
./engine
```

Expected output:
```
==========================================
   PART 5: FP32 vs INT8 ENGINE BENCHMARK   
==========================================

Pre-processing JSON data into RAM...
Loaded 1000 images. Starting 10x loop race...

FP32 | Acc: 87.9% | Time: 1240ms
INT8 | Acc: 87.3% | Time: 480ms
[SUCCESS] The INT8 engine is 2.58x faster!
```

## Development Phases

| Part | Focus |
|------|-------|
| **Part 1** | FP32 inference engine — tensor ops, conv2d, linear, activations, pooling |
| **Part 2** | PyTorch training pipeline — Fashion-MNIST CNN, 5 epochs, model export to JSON |
| **Part 3** | INT8 quantization — per-tensor asymmetric quantization, scale/ZP calibration, INT8 weight export |
| **Part 4** | INT8 inference engine — on-the-fly dequantization, FP32 accumulation, numerical validation |
| **Part 5** | Benchmarking — 10k inference runs, latency measurement, accuracy comparison |

## Training Pipeline

The model is trained in PyTorch (Google Colab) and exported to JSON:

```bash
# In Colab or local environment with PyTorch
python training/training.py
```

This produces:
- `fp32_model.json` — Full precision weights
- `int8_model.json` — Quantized INT8 weights + per-tensor scale/ZP
- `benchmark_set.json` — 1000 test images for benchmarking
- `test_image.json` / `validation_set.json` — For numerical validation

## Quantization Details

**Scheme**: Per-tensor asymmetric weight-only quantization

```
float_value = (int8_value - zero_point) × scale
```

Each tensor (conv_weight, fc_weight) has its own `scale` and `zero_point` stored in `int8_model.json`.

## Numerical Validation

The `validation_set.json` contains 10 test images with expected logits from the original PyTorch model. Use for regression testing when modifying the inference engine.

## License

MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgments

- Fashion-MNIST dataset (Zalando Research)
- `nlohmann/json` for JSON serialization
- Inspired by quantization techniques from TensorRT, ONNX Runtime, and PyTorch Quantization