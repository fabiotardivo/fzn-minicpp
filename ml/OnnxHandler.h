#pragma once

#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>
#include <memory>

// Singleton class for ONNX model inference
class OnnxHandler
{
public:
    static OnnxHandler& getInstance(const std::string& modelPath = "");

    // Get failure probability for a partial assignment
    // Returns probability in [0, 1] that the partial assignment leads to failure
    float runInference(const std::vector<float>& pa) const;

    // Alias for runInference - clearer naming for transformer model
    float getFailureProbability(const std::vector<float>& pa) const;

    void setVerbose(bool verbose) { this->verbose = verbose; }

    // Delete copy/move operations
    OnnxHandler(const OnnxHandler&) = delete;
    OnnxHandler& operator=(const OnnxHandler&) = delete;
    OnnxHandler(OnnxHandler&&) = delete;
    OnnxHandler& operator=(OnnxHandler&&) = delete;

private:
    explicit OnnxHandler(const std::string& modelPath);
    void initialize();

    std::string modelPath;
    bool verbose = false;
    bool modelLoaded = false;

    Ort::Env env{ORT_LOGGING_LEVEL_VERBOSE, "TransformerEnv"};
    Ort::SessionOptions sessionOptions;
    std::unique_ptr<Ort::Session> session;
    std::unique_ptr<Ort::MemoryInfo> memoryInfo;

    std::vector<std::string> inputNames;
    std::vector<std::string> outputNames;
};
