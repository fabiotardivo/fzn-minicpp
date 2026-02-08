// BatchedOnnxInferDual.hpp
#pragma once

#include <onnxruntime_cxx_api.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

class BatchedOnnxInferDual {
public:
  BatchedOnnxInferDual(const std::string& valueOnnxPath,
                       const std::string& variableOnnxPath,
                       bool useCuda)
      : env_(ORT_LOGGING_LEVEL_WARNING, "cpxmlv3"),
        opts_(),
        allocator_() {

    opts_.SetIntraOpNumThreads(1);
    opts_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    if (useCuda) {
      OrtCUDAProviderOptions cuda_opts;
      std::memset(&cuda_opts, 0, sizeof(cuda_opts));
      opts_.AppendExecutionProvider_CUDA(cuda_opts);
    }

    valueSession_ = Ort::Session(env_, valueOnnxPath.c_str(), opts_);
    varSession_   = Ort::Session(env_, variableOnnxPath.c_str(), opts_);

    validateSingleIO(valueSession_, "value");
    validateSingleIO(varSession_, "variable");

    // Cache names (owned by allocator). Copy into std::string for safety.
    {
      Ort::AllocatedStringPtr inName  = valueSession_.GetInputNameAllocated(0, allocator_);
      Ort::AllocatedStringPtr outName = valueSession_.GetOutputNameAllocated(0, allocator_);
      valueInputName_  = inName.get();
      valueOutputName_ = outName.get();
    }
    {
      Ort::AllocatedStringPtr inName  = varSession_.GetInputNameAllocated(0, allocator_);
      Ort::AllocatedStringPtr outName = varSession_.GetOutputNameAllocated(0, allocator_);
      varInputName_  = inName.get();
      varOutputName_ = outName.get();
    }
  }

  // Value-model batched result: probabilities aligned with the domain values used for varIdx.
  struct BatchResult {
    std::vector<int32_t> values;
    std::vector<float> probabilities;
  };

  // Runs the VALUE model in batch:
  // - repeats pa B times
  // - assigns varIdx to each candidate value from var's domain
  // - returns per-row probability from the value ONNX
  template <typename Var>
  BatchResult scoreAllValuesForVar(const std::vector<int32_t>& pa, int varIdx, const Var& var) {
    const int64_t nVars = static_cast<int64_t>(pa.size());
    if (varIdx < 0 || varIdx >= static_cast<int>(nVars)) {
      throw std::runtime_error("varIdx out of range.");
    }

    const int batch = var->size();
    if (batch <= 0) {
      return BatchResult{};
    }

    // Enumerate domain values
    std::vector<int32_t> domainVals;
    domainVals.reserve(static_cast<size_t>(batch));

    const int lo = var->min();
    const int hi = var->max();
    for (int v = lo; v <= hi; ++v) {
      if (var->contains(v)) domainVals.push_back(static_cast<int32_t>(v));
    }

    const int64_t B = static_cast<int64_t>(domainVals.size());
    if (B == 0) {
      return BatchResult{};
    }

    // Build input [B, nVars] by repeating pa
    std::vector<int32_t> input;
    input.resize(static_cast<size_t>(B * nVars));

    std::memcpy(input.data(), pa.data(), static_cast<size_t>(nVars) * sizeof(int32_t));
    for (int64_t r = 1; r < B; ++r) {
      std::memcpy(input.data() + r * nVars, input.data(), static_cast<size_t>(nVars) * sizeof(int32_t));
    }

    // Set chosen var column
    for (int64_t r = 0; r < B; ++r) {
      input[static_cast<size_t>(r * nVars + varIdx)] = domainVals[static_cast<size_t>(r)];
    }

    std::vector<float> probs = runSession(
        valueSession_,
        valueInputName_,
        valueOutputName_,
        input,
        B,
        nVars);

    return BatchResult{std::move(domainVals), std::move(probs)};
  }

  // Runs the VARIABLE model once on a single PA.
  // Returns raw scores [nVars] (higher means riskier).
  std::vector<float> scoreVariables(const std::vector<int32_t>& pa) {
    const int64_t nVars = static_cast<int64_t>(pa.size());
    if (nVars <= 0) {
      return {};
    }

    // Session expects [B, nVars]. Use batch=1.
    return runSession(
        varSession_,
        varInputName_,
        varOutputName_,
        pa,
        1,
        nVars);
  }

private:
  static void validateSingleIO(Ort::Session& sess, const char* tag) {
    if (sess.GetInputCount() != 1 || sess.GetOutputCount() != 1) {
      throw std::runtime_error(std::string("Expected 1 input and 1 output in the ") + tag + " ONNX model.");
    }
  }

  static std::vector<float> runSession(Ort::Session& session,
                                       const std::string& inputName,
                                       const std::string& outputName,
                                       const std::vector<int32_t>& input,
                                       int64_t batch,
                                       int64_t nVars) {
    Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::array<int64_t, 2> shape{batch, nVars};

    const size_t expectedCount = static_cast<size_t>(batch * nVars);
    if (input.size() != expectedCount) {
      // scoreVariables passes pa of size nVars and batch=1, so expectedCount == nVars.
      // scoreAllValuesForVar passes full [B*nVars].
      throw std::runtime_error("Input buffer size does not match batch*nVars.");
    }

    Ort::Value inTensor = Ort::Value::CreateTensor<int32_t>(
        memInfo,
        const_cast<int32_t*>(input.data()),
        input.size(),
        shape.data(),
        shape.size());

    const char* inNames[] = {inputName.c_str()};
    const char* outNames[] = {outputName.c_str()};

    auto outs = session.Run(Ort::RunOptions{nullptr}, inNames, &inTensor, 1, outNames, 1);
    if (outs.size() != 1) {
      throw std::runtime_error("Expected exactly 1 output tensor.");
    }

    Ort::Value& outTensor = outs[0];
    if (!outTensor.IsTensor()) {
      throw std::runtime_error("Output is not a tensor.");
    }

    Ort::TensorTypeAndShapeInfo info = outTensor.GetTensorTypeAndShapeInfo();
    const size_t outCount = info.GetElementCount();

    const float* outData = outTensor.GetTensorData<float>();
    std::vector<float> out;
    out.assign(outData, outData + outCount);
    return out;
  }

  Ort::Env env_;
  Ort::SessionOptions opts_;
  Ort::AllocatorWithDefaultOptions allocator_;

  Ort::Session valueSession_{nullptr};
  Ort::Session varSession_{nullptr};

  std::string valueInputName_;
  std::string valueOutputName_;
  std::string varInputName_;
  std::string varOutputName_;
};
