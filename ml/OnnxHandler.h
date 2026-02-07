// BatchedOnnxInfer.hpp
#pragma once

#include <onnxruntime_cxx_api.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>

class BatchedOnnxInfer {
public:
  BatchedOnnxInfer(const std::string& onnxPath, bool useCuda)
      : env_(ORT_LOGGING_LEVEL_WARNING, "cpxmlv3"),
        opts_(),
        allocator_() {

    opts_.SetIntraOpNumThreads(1);
    opts_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

#ifdef USE_CUDA
    if (useCuda) {
      OrtCUDAProviderOptions cuda_opts;
      std::memset(&cuda_opts, 0, sizeof(cuda_opts));
      opts_.AppendExecutionProvider_CUDA(cuda_opts);
    }
#else
    (void)useCuda;
#endif

    session_ = Ort::Session(env_, onnxPath.c_str(), opts_);

    // Cache names (owned by allocator). Copy into std::string for safety.
    {
      Ort::AllocatedStringPtr inName = session_.GetInputNameAllocated(0, allocator_);
      Ort::AllocatedStringPtr outName = session_.GetOutputNameAllocated(0, allocator_);
      inputName_ = inName.get();
      outputName_ = outName.get();
    }

    // Optional: validate input/output shapes and types once
    if (session_.GetInputCount() != 1 || session_.GetOutputCount() != 1) {
      throw std::runtime_error("Expected 1 input and 1 output in the ONNX model.");
    }
  }

  // Returns probabilities aligned with the domain values used for varIdx.
  // Also returns the domain values used (so caller can map prob -> value).
  struct BatchResult {
    std::vector<int32_t> values;
    std::vector<float> probabilities;
  };

  template <typename Var>
  BatchResult scoreAllValuesForVar(
      const std::vector<int32_t>& pa,
      int varIdx,
      const Var& var) {

    const int64_t nVars = static_cast<int64_t>(pa.size());
    if (varIdx < 0 || varIdx >= static_cast<int>(nVars)) {
      throw std::runtime_error("varIdx out of range.");
    }

    const int batch = var->size();
    if (batch <= 0) {
      return BatchResult{};
    }

    // 1) Enumerate domain values
    std::vector<int32_t> domainVals;
    domainVals.reserve(static_cast<size_t>(batch));

    // Option A: bounded [min..max] with holes (contains)
    // Works even if domain is sparse, but costs O(range).
    if (var->isBound()) {
      const int lo = var->min();
      const int hi = var->max();
      for (int v = lo; v <= hi; ++v) {
        if (var->contains(v)) domainVals.push_back(static_cast<int32_t>(v));
      }
    } else {
      // If unbounded, you must decide a finite enumeration strategy.
      // Here we fail fast.
      throw std::runtime_error("Variable is not bounded. Need a finite domain enumeration strategy.");
    }

    if (static_cast<int>(domainVals.size()) != batch) {
      // If size() disagrees with enumeration, size() might be "dynamic" or contains() costly.
      // We trust enumeration.
      // If you really need exact size preallocation, make Variable expose an iterator/fillValues().
    }

    const int64_t B = static_cast<int64_t>(domainVals.size());
    if (B == 0) {
      return BatchResult{};
    }

    // 2) Preallocate input buffer: [B, nVars]
    std::vector<int32_t> input;
    input.resize(static_cast<size_t>(B * nVars));

    // 3) Copy PA into every row
    // Do the first row then memcpy it into others.
    std::memcpy(input.data(), pa.data(), static_cast<size_t>(nVars) * sizeof(int32_t));
    for (int64_t r = 1; r < B; ++r) {
      std::memcpy(input.data() + r * nVars, input.data(), static_cast<size_t>(nVars) * sizeof(int32_t));
    }

    // 4) Set column varIdx with each domain value
    for (int64_t r = 0; r < B; ++r) {
      input[static_cast<size_t>(r * nVars + varIdx)] = domainVals[static_cast<size_t>(r)];
    }

    // 5) Run ONNX: input int32, output float (probabilities)
    std::vector<float> probs = run(input, B, nVars);

    return BatchResult{std::move(domainVals), std::move(probs)};
  }

private:
  std::vector<float> run(const std::vector<int32_t>& input, int64_t batch, int64_t nVars) {
    Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::array<int64_t, 2> shape{batch, nVars};

    Ort::Value inTensor = Ort::Value::CreateTensor<int32_t>(
        memInfo,
        const_cast<int32_t*>(input.data()),
        input.size(),
        shape.data(),
        shape.size());

    const char* inNames[] = {inputName_.c_str()};
    const char* outNames[] = {outputName_.c_str()};

    auto outs = session_.Run(Ort::RunOptions{nullptr}, inNames, &inTensor, 1, outNames, 1);
    if (outs.size() != 1) {
      throw std::runtime_error("Expected exactly 1 output tensor.");
    }

    Ort::Value& outTensor = outs[0];
    if (!outTensor.IsTensor()) {
      throw std::runtime_error("Output is not a tensor.");
    }

    // Output shape expected: [batch] or [batch,1]
    Ort::TensorTypeAndShapeInfo info = outTensor.GetTensorTypeAndShapeInfo();
    auto outShape = info.GetShape();
    size_t outCount = info.GetElementCount();

    const float* outData = outTensor.GetTensorData<float>();

    std::vector<float> probs;
    probs.assign(outData, outData + outCount);

    // If output is [B,1], flatten already matches element count, so ok.
    // If you want strictly B, you can trim or validate here.
    return probs;
  }

  Ort::Env env_;
  Ort::SessionOptions opts_;
  Ort::AllocatorWithDefaultOptions allocator_;
  Ort::Session session_{nullptr};

  std::string inputName_;
  std::string outputName_;
};
