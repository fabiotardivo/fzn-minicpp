/**
 * @file us_predictor.h
 * @brief C++ inference handler for US Transformer ONNX models
 * 
 * This class provides high-performance batch inference for Unsatisfiable Set
 * prediction using ONNX Runtime.
 * 
 * Key Features:
 * - Simple interface: pass raw values, get probabilities
 * - INT_MIN sentinel for unassigned variables
 * - No metadata files needed (embedded in ONNX)
 * - GPU acceleration support
 * - Returns (value, score) pairs for easy sorting
 * 
 * Typical Usage:
 * @code
 * // 1. Create predictor (metadata auto-loaded from ONNX)
 * USPredictor predictor("model.onnx");
 * 
 * // 2. Create assignment (use INT_MIN for unassigned)
 * std::vector<int> values(n_vars, INT_MIN);
 * values[0] = 5;
 * values[2] = 10;
 * 
 * // 3. Predict for single assignment
 * auto probs = predictor.predict(values);
 * 
 * // 4. Or predict batch for branching (returns value-score pairs)
 * auto candidates = predictor.predict_batch(values, var_idx, my_variable);
 * // candidates[i] = (domain_value, probability)
 * 
 * // 5. Sort by score to choose best value
 * std::sort(candidates.begin(), candidates.end(),
 *           [](const auto& a, const auto& b) { return a.second < b.second; });
 * int best_value = candidates[0].first;  // Value with lowest US probability
 * @endcode
 * 
 * Note: The ONNX model now handles:
 * - Value → token conversion (no need for metadata in C++)
 * - Sigmoid application (output is probabilities, not logits)
 */

#ifndef US_PREDICTOR_H
#define US_PREDICTOR_H

#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <memory>
#include <stdexcept>
#include <climits>

namespace us_predictor {

/**
 * @brief ONNX-based predictor for Unsatisfiable Sets
 * 
 * Simple interface: pass raw values (INT_MIN for unassigned), get probabilities.
 * All preprocessing happens inside the ONNX model.
 */
class USPredictor {
public:
    /**
     * @brief Constructor - loads the ONNX model
     * @param model_path Path to ONNX model (with embedded preprocessing)
     * @param use_gpu Use GPU acceleration (default: false)
     * @param num_threads Number of CPU threads (default: 0 = auto)
     */
    USPredictor(
        const std::string& model_path,
        bool use_gpu = false,
        int num_threads = 0
    );
    
    ~USPredictor();
    
    /**
     * @brief Predict US probabilities for a partial assignment
     * 
     * @param values Partial assignment where INT_MIN = unassigned
     * @return Vector of probabilities [0, 1] for each variable
     * 
     * Example:
     * @code
     * std::vector<int> values(60, INT_MIN);  // All unassigned
     * values[0] = 5;
     * values[2] = 10;
     * 
     * auto probs = predictor.predict(values);
     * // probs[i] = P(variable i is in US | current assignment)
     * @endcode
     */
    std::vector<float> predict(const std::vector<int>& values);
    
    /**
     * @brief Batch inference for branching decisions
     * 
     * Given a partial assignment and a variable to branch on, computes
     * US probabilities for all values in the variable's domain.
     * 
     * The Var type must support:
     * - int size() const          - domain size
     * - int min() const           - minimum value
     * - int max() const           - maximum value  
     * - bool contains(int) const  - check if value is in domain
     * 
     * @tparam Var Variable type from your CP solver
     * @param values Current partial assignment (INT_MIN = unassigned)
     * @param var_idx Index of variable to branch on
     * @param var Variable object with domain information
     * @return Vector of (value, probability) pairs for each domain value
     * 
     * Example:
     * @code
     * std::vector<int> values(60, INT_MIN);
     * values[0] = 5;
     * values[2] = 10;
     * 
     * // Branch on variable 1
     * auto result = predictor.predict_batch(values, 1, my_variable);
     * // result[i] = (domain_value, P(var_idx in US | values ∪ {var_idx = domain_value}))
     * 
     * // Sort by probability (descending)
     * std::sort(result.begin(), result.end(), 
     *           [](const auto& a, const auto& b) { return a.second > b.second; });
     * @endcode
     */
    template<typename Var>
    std::vector<std::pair<int, float>> predict_batch(
        const std::vector<int>& values,
        int var_idx,
        const Var& var
    );
    
    /**
     * @brief Get number of variables (inferred from first prediction)
     */
    size_t get_n_vars() const { return n_vars_; }

private:
    // ONNX Runtime objects
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;
    Ort::AllocatorWithDefaultOptions allocator_;
    
    // Model configuration (inferred from ONNX)
    size_t n_vars_;
    
    // Input/Output names
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    
    /**
     * @brief Run inference on value batch
     * @param batch_values Raw values (INT_MIN for unassigned)
     * @param batch_size Number of samples in batch
     * @return Probabilities [0, 1] for each variable in each sample
     */
    std::vector<float> run_inference(
        const std::vector<int32_t>& batch_values,
        size_t batch_size
    );
};

// ============================================================================
// Template Implementation
// ============================================================================

template<typename Var>
std::vector<std::pair<int, float>> USPredictor::predict_batch(
    const std::vector<int>& values,
    int var_idx,
    const Var& var
) {
    // Validate inputs
    if (n_vars_ > 0 && values.size() != n_vars_) {
        throw std::invalid_argument(
            "values size (" + std::to_string(values.size()) + 
            ") must match n_vars (" + std::to_string(n_vars_) + ")"
        );
    }
    
    if (var_idx < 0 || static_cast<size_t>(var_idx) >= values.size()) {
        throw std::invalid_argument(
            "var_idx (" + std::to_string(var_idx) + 
            ") must be in [0, " + std::to_string(values.size()) + ")"
        );
    }
    
    // Get domain size
    int domain_size = var.size();
    if (domain_size <= 0) {
        throw std::invalid_argument(
            "Variable domain size must be positive, got " + 
            std::to_string(domain_size)
        );
    }
    
    // Build batch of values for all domain values (use int32_t to avoid copy in run_inference)
    std::vector<int32_t> batch_values;
    batch_values.reserve(domain_size * values.size());
    
    std::vector<int> domain_values;
    domain_values.reserve(domain_size);
    
    // Iterate through domain
    int min_val = var.min();
    int max_val = var.max();
    
    for (int value = min_val; value <= max_val; ++value) {
        if (!var.contains(value)) {
            continue;
        }
        
        domain_values.push_back(value);
        
        // Create assignment with this value
        for (size_t i = 0; i < values.size(); ++i) {
            if (static_cast<int>(i) == var_idx) {
                batch_values.push_back(static_cast<int32_t>(value));
            } else {
                batch_values.push_back(static_cast<int32_t>(values[i]));
            }
        }
    }
    
    size_t actual_batch_size = domain_values.size();
    if (actual_batch_size == 0) {
        throw std::runtime_error("Variable domain is empty");
    }
    
    // Run batch inference (returns probabilities, not logits!)
    std::vector<float> probs = run_inference(batch_values, actual_batch_size);
    
    // Build result as vector of (value, probability) pairs
    std::vector<std::pair<int, float>> result;
    result.reserve(actual_batch_size);
    
    for (size_t i = 0; i < actual_batch_size; ++i) {
        size_t idx = i * values.size() + var_idx;
        result.push_back({domain_values[i], probs[idx]});
    }
    
    return result;
}

} // namespace us_predictor

#endif // US_PREDICTOR_H
