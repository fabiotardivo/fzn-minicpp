#pragma once

#include <torch/script.h>
#include <torch/torch.h>
#include <string>
#include <vector>
#include <memory>

// Singleton class for PyTorch model inference
class TorchHandler
{
public:
    static TorchHandler& getInstance(const std::string& modelPath = "");

    // Get failure probability for a partial assignment
    float runInference(const std::vector<float>& pa) const;

    // Batch inference for multiple partial assignments
    std::vector<float> runInferenceBatch(const std::vector<std::vector<float>>& pasBatch) const;

    void setVerbose(bool verbose) { this->verbose = verbose; }

    // Delete copy/move operations
    TorchHandler(const TorchHandler&) = delete;
    TorchHandler& operator=(const TorchHandler&) = delete;
    TorchHandler(TorchHandler&&) = delete;
    TorchHandler& operator=(TorchHandler&&) = delete;

private:
    explicit TorchHandler(const std::string& modelPath);
    void initialize();

    std::string modelPath;
    bool verbose = false;
    bool modelLoaded = false;

    torch::jit::script::Module model;
    torch::Device device = torch::kCPU;
};
