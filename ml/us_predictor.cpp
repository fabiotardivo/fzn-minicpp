/**
 * @file us_predictor.cpp
 * @brief Implementation of ONNX US Transformer model inference
 */

#include "us_predictor.h"
#include <iostream>
#include <algorithm>

namespace us_predictor {

USPredictor::USPredictor(
    const std::string& model_path,
    bool use_gpu,
    int num_threads
) : env_(ORT_LOGGING_LEVEL_WARNING, "USPredictor"),
    n_vars_(0)  // Will be inferred from first prediction
{
    // Configure session options
    session_options_.SetIntraOpNumThreads(num_threads > 0 ? num_threads : 1);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    
    // Add execution provider
    if (use_gpu) {
        OrtCUDAProviderOptions cuda_options;
        cuda_options.device_id = 0;
        session_options_.AppendExecutionProvider_CUDA(cuda_options);
    }
    
    // Create session
#ifdef _WIN32
    std::wstring model_path_w(model_path.begin(), model_path.end());
    session_ = std::make_unique<Ort::Session>(env_, model_path_w.c_str(), session_options_);
#else
    session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options_);
#endif
    
    // Get input/output names
    // New model expects "values" (int32) and returns "probabilities" (float32)
    input_names_.push_back("values");
    output_names_.push_back("probabilities");
    
    std::cout << "USPredictor initialized" << std::endl;
    std::cout << "  Input: 'values' (int32) - raw values where INT_MIN = unassigned" << std::endl;
    std::cout << "  Output: 'probabilities' (float32) - US probabilities in [0, 1]" << std::endl;
    std::cout << "  Preprocessing: embedded in ONNX (no metadata needed)" << std::endl;
}

USPredictor::~USPredictor() {
    // Cleanup is handled by unique_ptr
}

std::vector<float> USPredictor::predict(const std::vector<int>& values) {
    // Convert to int32_t once (ONNX expects int32)
    std::vector<int32_t> values_int32(values.begin(), values.end());
    
    // Single prediction is just a batch of size 1
    auto result = run_inference(values_int32, 1);
    
    // Infer n_vars from first prediction if not set
    if (n_vars_ == 0) {
        n_vars_ = values.size();
        std::cout << "Inferred n_vars = " << n_vars_ << " from first prediction" << std::endl;
    }
    
    return result;
}

std::vector<float> USPredictor::run_inference(
    const std::vector<int32_t>& batch_values,
    size_t batch_size
) {
    // Infer n_vars if not set
    if (n_vars_ == 0 && batch_size > 0) {
        n_vars_ = batch_values.size() / batch_size;
    }
    
    // Validate input size
    if (batch_values.size() != batch_size * n_vars_) {
        throw std::invalid_argument(
            "batch_values size (" + std::to_string(batch_values.size()) + 
            ") must equal batch_size * n_vars (" + 
            std::to_string(batch_size * n_vars_) + ")"
        );
    }
    
    // Create input tensor shape
    std::vector<int64_t> input_shape = {
        static_cast<int64_t>(batch_size), 
        static_cast<int64_t>(n_vars_)
    };
    
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    
    // Create tensor directly from batch_values (no copy needed!)
    Ort::Value values_tensor = Ort::Value::CreateTensor<int32_t>(
        memory_info,
        const_cast<int32_t*>(batch_values.data()),  // Safe: ONNX won't modify input
        batch_values.size(),
        input_shape.data(),
        input_shape.size()
    );
    
    // Prepare input tensors
    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(std::move(values_tensor));
    
    // Run inference
    auto output_tensors = session_->Run(
        Ort::RunOptions{nullptr},
        input_names_.data(),
        input_tensors.data(),
        input_tensors.size(),
        output_names_.data(),
        output_names_.size()
    );
    
    // Extract output (probabilities, not logits!)
    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    size_t output_size = batch_size * n_vars_;
    
    return std::vector<float>(output_data, output_data + output_size);
}

} // namespace us_predictor
