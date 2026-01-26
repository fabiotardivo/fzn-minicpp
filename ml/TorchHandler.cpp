#include "TorchHandler.h"
#include <iostream>
#include <cstring>

TorchHandler& TorchHandler::getInstance(const std::string& modelPath)
{
    static TorchHandler instance(modelPath);
    return instance;
}

TorchHandler::TorchHandler(const std::string& modelPath)
: modelPath(modelPath), device(torch::kCPU)
{
    if (!modelPath.empty())
    {
        initialize();
    }
}

void TorchHandler::initialize()
{
    if (modelLoaded)
        return;

    try
    {
        // Check for CUDA availability
        if (torch::cuda::is_available())
        {
            device = torch::kCUDA;
            if (verbose)
            {
                std::cout << "[PyTorch] CUDA enabled (GPU acceleration active)" << std::endl;
                std::cout << "[PyTorch] CUDA devices: " << torch::cuda::device_count() << std::endl;
            }
        }
        else if (verbose)
        {
            std::cout << "[PyTorch] Using CPU" << std::endl;
        }

        // Load the TorchScript model
        model = torch::jit::load(modelPath);
        model.to(device);
        model.eval();

        modelLoaded = true;

        if (verbose)
        {
            std::cout << "[PyTorch] Model loaded: " << modelPath << std::endl;
        }
    }
    catch (const c10::Error& e)
    {
        std::cerr << "[ERROR] Failed to load PyTorch model: " << e.what() << std::endl;
        throw;
    }
}

float TorchHandler::runInference(const std::vector<float>& pa) const
{
    if (!modelLoaded)
    {
        throw std::runtime_error("[PyTorch] Model not loaded");
    }

    try
    {
        // Convert to tensor: raw float values (model handles NaN preprocessing internally)
        // Need to copy data because from_blob doesn't own it
        std::vector<float> paCopy = pa;

        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        torch::Tensor rawValues = torch::from_blob(
            paCopy.data(),
                                                   {1, static_cast<long>(pa.size())},
                                                   options
        ).clone().to(device);

        // Run inference
        torch::NoGradGuard no_grad;
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(rawValues);

        auto output = model.forward(inputs).toTensor();

        // Move to CPU and get failure probability (single value)
        output = output.to(torch::kCPU);
        return output.item<float>();
    }
    catch (const c10::Error& e)
    {
        std::cerr << "[ERROR] PyTorch inference failed: " << e.what() << std::endl;
        throw;
    }
}

std::vector<float> TorchHandler::runInferenceBatch(const std::vector<std::vector<float>>& pasBatch) const
{
    if (!modelLoaded)
    {
        throw std::runtime_error("[PyTorch] Model not loaded");
    }

    if (pasBatch.empty())
    {
        return {};
    }

    try
    {
        const size_t batchSize = pasBatch.size();
        const size_t numVars = pasBatch[0].size();

        // Flatten batch into single vector
        std::vector<float> flattenedData;
        flattenedData.reserve(batchSize * numVars);
        for (const auto& pa : pasBatch)
        {
            if (pa.size() != numVars)
            {
                throw std::runtime_error("[PyTorch] Inconsistent variable count in batch");
            }
            flattenedData.insert(flattenedData.end(), pa.begin(), pa.end());
        }

        // Convert to tensor
        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        torch::Tensor rawValues = torch::from_blob(
            flattenedData.data(),
                                                   {static_cast<long>(batchSize), static_cast<long>(numVars)},
                                                   options
        ).clone().to(device);

        // Run inference
        torch::NoGradGuard no_grad;
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(rawValues);

        auto output = model.forward(inputs).toTensor();

        // Move to CPU and convert to vector
        output = output.to(torch::kCPU);

        // Handle both (batch, 1) and (batch,) shapes
        output = output.squeeze();

        std::vector<float> results(batchSize);
        if (batchSize == 1)
        {
            results[0] = output.item<float>();
        }
        else
        {
            std::memcpy(results.data(), output.data_ptr<float>(), batchSize * sizeof(float));
        }

        return results;
    }
    catch (const c10::Error& e)
    {
        std::cerr << "[ERROR] PyTorch batch inference failed: " << e.what() << std::endl;
        throw;
    }
}
