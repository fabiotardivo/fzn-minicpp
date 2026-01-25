#include "OnnxHandler.h"
#include <iostream>
#include <algorithm>
#include <cmath>

OnnxHandler& OnnxHandler::getInstance(const std::string& modelPath)
{
    static OnnxHandler instance(modelPath);
    return instance;
}

OnnxHandler::OnnxHandler(const std::string& modelPath)
: modelPath(modelPath)
{
    if (!modelPath.empty())
    {
        initialize();
    }
}

void OnnxHandler::initialize()
{
    if (modelLoaded)
        return;

    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // Check for CUDA at runtime
    auto availableProviders = Ort::GetAvailableProviders();
    bool cudaAvailable = std::find(availableProviders.begin(),
                                   availableProviders.end(),
                                   "CUDAExecutionProvider") != availableProviders.end();

    if (cudaAvailable)
    {
        try
        {
            OrtCUDAProviderOptions cudaOptions;
            cudaOptions.device_id = 0;
            //cudaOptions.arena_extend_strategy = OrtArenaExtendStrategy::kNextPowerOfTwo;
            cudaOptions.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch::OrtCudnnConvAlgoSearchExhaustive;
            cudaOptions.do_copy_in_default_stream = 1;

            sessionOptions.AppendExecutionProvider_CUDA(cudaOptions);

            if (verbose)
            {
                std::cout << "[ONNX] CUDA enabled (GPU acceleration active)" << std::endl;
            }
        }
        catch (const Ort::Exception& e)
        {
            std::cout << "[WARNING] CUDA initialization failed: " << e.what() << std::endl;
            std::cout << "[ONNX] Falling back to CPU" << std::endl;
        }
    }
    else if (verbose)
    {
        std::cout << "[ONNX] Using CPU" << std::endl;
    }

    session = std::make_unique<Ort::Session>(env, modelPath.c_str(), sessionOptions);
    memoryInfo = std::make_unique<Ort::MemoryInfo>(
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU));

    inputNames = session->GetInputNames();
    outputNames = session->GetOutputNames();

    modelLoaded = true;

    if (verbose)
    {
        std::cout << "[ONNX] Model loaded: " << modelPath << std::endl;
        std::cout << "[ONNX] Input count: " << inputNames.size() << std::endl;
        std::cout << "[ONNX] Output count: " << outputNames.size() << std::endl;
    }
}

float OnnxHandler::runInference(const std::vector<float>& pa) const
{
    if (!modelLoaded)
    {
        throw std::runtime_error("[ONNX] Model not loaded");
    }

    const int64_t batchSize = 1;
    const int64_t numVars = static_cast<int64_t>(pa.size());
    std::vector<int64_t> tensorShape = {batchSize, numVars};

    // SIMPLIFIED: Just pass raw float values directly!
    // ONNX model handles NaN detection, masking, and value shifting internally

    // Create input tensor with raw values (including NaN)
    std::vector<float> rawValues = pa;  // Copy input vector

    Ort::Value valuesTensor = Ort::Value::CreateTensor<float>(
        *memoryInfo,
        rawValues.data(),
        rawValues.size(),
        tensorShape.data(),
        tensorShape.size()
    );

    // Prepare input tensors vector (only one input now!)
    std::vector<Ort::Value> inputTensors;
    inputTensors.emplace_back(std::move(valuesTensor));

    // Convert input names to const char*
    std::vector<const char*> inputNamesCstr;
    inputNamesCstr.reserve(inputNames.size());
    for (const auto& name : inputNames)
        inputNamesCstr.push_back(name.c_str());

    // Convert output names to const char*
    std::vector<const char*> outputNamesCstr;
    outputNamesCstr.reserve(outputNames.size());
    for (const auto& name : outputNames)
        outputNamesCstr.push_back(name.c_str());

    // Run inference with raw values
    auto outputTensors = session->Run(
        Ort::RunOptions{nullptr},
        inputNamesCstr.data(),
        inputTensors.data(),
        inputTensors.size(),
        outputNamesCstr.data(),
        outputNamesCstr.size()
    );

    // Extract failure probability (already sigmoid'd in ONNX model!)
    const float* outputData = outputTensors[0].GetTensorData<float>();
    float probability = outputData[0];

    return probability;
}

float OnnxHandler::getFailureProbability(const std::vector<float>& pa) const
{
    // Same as runInference - now returns probability directly
    return runInference(pa);
}