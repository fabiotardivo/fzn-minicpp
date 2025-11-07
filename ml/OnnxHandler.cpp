#include "OnnxHandler.h"

#include <iomanip>
#include <iostream>
#include <format>
#include <fstream>
#include <string>


// --- Singleton instance initialization ---
OnnxHandler* OnnxHandler::instance = nullptr;

OnnxHandler& OnnxHandler::get_instance()
{
    if (!instance)
    {
        throw std::runtime_error(
            "OnnxHandler instance not created. Call create_instance() first.");
    }
    return *instance;
}

void OnnxHandler::create_instance(const std::string & model_path, const float max_val,const size_t pa_length,const float scale_ratio,const float out_of_scale_marker, const DistanceType distance_type, const bool use_bitmask){
    if (!instance)
    {
        instance = new OnnxHandler(model_path, max_val, pa_length, scale_ratio, out_of_scale_marker, distance_type,use_bitmask);
    }
}

OnnxHandler::OnnxHandler(const std::string & model_path, const float max_val, const size_t pa_length, const float scale_ratio, const float out_of_scale_marker, const DistanceType distance_type, const bool use_bitmask)
{

    m_cfg.max_val = max_val;
    m_cfg.pa_length = pa_length;
    m_cfg.scale_ratio = scale_ratio;
    m_cfg.out_of_scale_marker = out_of_scale_marker;
    m_cfg.distance_type = distance_type;
    m_cfg.use_bitmask = use_bitmask;

    if (!m_cfg.use_bitmask)
    {
        if (m_cfg.scale_ratio <= 1.0)
        {
            m_cfg.scale_ratio = 5.0 / 4.0; // set >1.0 to reserve headroom (e.g. 5.0/4.0)
            m_cfg.out_of_scale_marker = 1.0;
            // This allows all the values to be within valid range if bitmask is not used
        }
    }
    m_model_path = model_path;
    initialize_onnx_model();
}


void OnnxHandler::initialize_onnx_model()
{
    if (m_model_loaded)
        return;

    try
    {
        m_session_options.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_ALL);

        /*#ifdef _WIN32
            // Windows way
            _putenv_s("ORT_CUDA_USE_CUDNN", "0");
            _putenv_s("CUDA_LAUNCH_BLOCKING", "1");  // optional, forces sync errors
        #else
            // Linux / WSL / macOS way
            setenv("ORT_CUDA_USE_CUDNN", "0", 1);
            setenv("CUDA_LAUNCH_BLOCKING", "1", 1);  // optional
        #endif*/

        // Add this line to show warnings (2) to complete verbose (0) (only errors
        // and above will be shown)
        // m_session_options.SetLogSeverityLevel(0);


#ifdef USE_CUDA
        try
        {
            OrtCUDAProviderOptions cuda_options;
            m_session_options.AppendExecutionProvider_CUDA(cuda_options);
            if (m_verbose)
            {
                ArgumentParser::get_instance().get_output_stream()
                    << "[ONNX] CUDA execution provider enabled via USE_CUDA."
                    << std::endl;
            }
        }
        catch (const Ort::Exception& e)
        {
            ArgumentParser::get_instance().get_output_stream()
                << "[WARNING][ONNX] Failed to enable CUDA, defaulting to CPU: "
                << e.what() << std::endl;
        }
#else
        std::cout << "[ONNX] Compiled without CUDA (USE_CUDA not defined), using CPU."
            << std::endl;
#endif

        m_session = std::make_unique<Ort::Session>(m_env, m_model_path.c_str(),
                                                   m_session_options);
        m_allocator = std::make_unique<Ort::AllocatorWithDefaultOptions>();
        m_memory_info = std::make_unique<Ort::MemoryInfo>(
            Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU));

        // Fixed: No allocator argument
        m_input_names = m_session->GetInputNames();
        m_output_names = m_session->GetOutputNames();

        m_model_loaded = true;
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error(std::string("Failed to load ONNX model: ") + e.what());
    }

    if (m_verbose)
    {
        std::cout << "[ONNX] Model loaded: " << m_model_path << std::endl;

        // Print model input and output details
        const auto input_names = m_session->GetInputNames();
        const auto output_names = m_session->GetOutputNames();

        std::cout << "[ONNX] Model Inputs:\n";
        for (size_t i = 0; i < input_names.size(); ++i)
        {
            const auto& name = input_names[i];
            auto type_info = m_session->GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            const auto element_type = tensor_info.GetElementType();
            auto shape = tensor_info.GetShape();

            std::cout << "  Name: " << name << "\n";
            std::cout << "  Type: " << element_type << "\n";
            std::cout << "  Shape: [";
            for (size_t j = 0; j < shape.size(); ++j)
            {
                std::cout << shape[j];
                if (j < shape.size() - 1)
                    std::cout << ", ";
            }
            std::cout << "]\n";
        }

        std::cout << "[ONNX] Model Outputs:\n";
        for (size_t i = 0; i < output_names.size(); ++i)
        {
            const auto& name = output_names[i];
            auto type_info = m_session->GetOutputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            const auto element_type = tensor_info.GetElementType();
            auto shape = tensor_info.GetShape();

            std::cout << "  Name: " << name << "\n";
            std::cout << "  Type: " << element_type << "\n";
            std::cout << "  Shape: [";
            for (size_t j = 0; j < shape.size(); ++j)
            {
                std::cout << shape[j];
                if (j < shape.size() - 1)
                    std::cout << ", ";
            }
            std::cout << "]\n";
        }
        std::cout << "[ONNX] Model successfully printed." << std::endl;
    }
}


std::vector<float> OnnxHandler::run_inference(const std::vector<float>& pa,
                                              const std::vector<float>& bitmask) const
{
    if (!m_model_loaded)
    {
        throw std::runtime_error("[ONNX] Model not loaded before inference.");
    }


    if (pa.size() != bitmask.size())
    {
        throw std::invalid_argument("values and mask must have same length");
    }


    auto& session = *m_session;
    const auto& memory_info = *m_memory_info;

    const int64_t N = static_cast<int64_t>(pa.size());
    std::vector<int64_t> tensor_shape = {1, N};

    // Value tensor
    std::vector<float> variables_float(pa.begin(),
                                       pa.end());
    Ort::Value values_tensor = Ort::Value::CreateTensor<float>(
        memory_info, variables_float.data(), variables_float.size(),
        tensor_shape.data(), tensor_shape.size());

    // Mask tensor
    std::vector<float> bitmask_float(bitmask.begin(), bitmask.end());
    Ort::Value mask_tensor = Ort::Value::CreateTensor<float>(
        memory_info, bitmask_float.data(), bitmask_float.size(),
        tensor_shape.data(), tensor_shape.size());

    // Inputs in expected order: values first, mask second
    std::vector<Ort::Value> input_tensors;
    input_tensors.emplace_back(std::move(values_tensor));
    input_tensors.emplace_back(std::move(mask_tensor));

    // --- FIX #2: verify input-name count and order ---
    if (m_input_names.size() != input_tensors.size())
        throw std::runtime_error("[ONNX] Number of model inputs does not match provided tensors.");


    // Convert input/output names to const char* arrays
    std::vector<const char*> input_names_cstr;
    input_names_cstr.reserve(m_input_names.size());
    for (const auto& name : m_input_names)
    {
        input_names_cstr.push_back(name.c_str());
    }
    std::vector<const char*> output_names_cstr;
    output_names_cstr.reserve(m_output_names.size());
    for (const auto& name : m_output_names)
    {
        output_names_cstr.push_back(name.c_str());
    }

    // Run the model
    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr}, input_names_cstr.data(), input_tensors.data(),
        input_tensors.size(), output_names_cstr.data(), output_names_cstr.size());

    // Get output tensor data
    const float* output_data = output_tensors[0].GetTensorMutableData<float>();
    const auto shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape(); // [1, seq_len]
    const size_t seq_len = static_cast<size_t>(shape[1]);
    std::vector<float> reconstructed(output_data, output_data + seq_len);
    return reconstructed;   // already normalized floats in [0,1]
}


float OnnxHandler::get_score(const std::string& pa, bool toClean) const
{
    const auto [fst, snd] = preprocess_pa_inference(pa, toClean);
    const auto reconstructed_assignment = run_inference(fst, snd);
    return get_distance(fst, reconstructed_assignment, snd);
}

float OnnxHandler::get_score(std::vector<float> const & pa) const
{
    const auto [fst, snd] = normalize_pa_inference(pa);
    const auto reconstructed_assignment = run_inference(fst, snd);
    return get_distance(fst, reconstructed_assignment, snd);
}

std::string OnnxHandler::clean_pa(const std::string& pa)
{
    // Split into trimmed lines
    std::vector<std::string> lines;
    std::istringstream stream(pa);
    std::string line;
    while (std::getline(stream, line))
    {
        if (!line.empty())
        {
            lines.push_back(line);
        }
    }

    // Validate line count
    if (lines.size() != 2)
    {
        throw std::runtime_error("[ERROR] Single output must contain exactly two lines.");
    }

    // Validate ending patterns
    if (lines[0].size() < 2 || lines[1].size() < 2 ||
        lines[0].substr(lines[0].size() - 2) != ",1" ||
        lines[1].substr(lines[1].size() - 2) != ",0")
    {
        throw std::runtime_error("[ERROR] First line of output must end with ',1' and second line with ',0'.");
    }

    // Remove trailing ",0" from the second line
    return lines[1].substr(0, lines[1].size() - 2);
}


std::pair<std::vector<float>, std::vector<float>> OnnxHandler::preprocess_pa_inference(
    const std::string& pa, bool toClean) const
{
    return normalize_string_inference(toClean ? clean_pa(pa) : pa);
}

std::string OnnxHandler::preprocess_pa_train(const std::string& pa) const
{
    const auto cleaned_pa = clean_pa(pa);
    return normalize_string_training(cleaned_pa);
}


std::pair<std::vector<float>, std::vector<float>> OnnxHandler::normalize_string_inference(
    const std::string& pa) const
{
    std::vector<float> result;
    std::vector<float> bitmask;
    const double scale_value = m_cfg.max_val * m_cfg.scale_ratio;

    result.reserve(m_cfg.pa_length); // small heuristic to reduce reallocations
    bitmask.reserve(m_cfg.pa_length);

    for_each_comma_token(pa, [&](const std::string& token)
    {
        const auto parsed = parse_token_double(token);
        if (!parsed.has_value())
        {
            result.push_back(static_cast<float>(m_cfg.out_of_scale_marker));
            if (m_cfg.use_bitmask)
            {
                bitmask.push_back(0.0f);
            }
            else
            {
                bitmask.push_back(1.0f);
            }
        }
        else
        {
            float const scaled = (*parsed) / scale_value;
            //scaled = std::max(scaled, 1.0f);
            result.push_back(scaled);
            bitmask.push_back(1.0f);
        }
    });

    return {std::move(result), std::move(bitmask)};
}

std::pair<std::vector<float>, std::vector<float>> OnnxHandler::normalize_pa_inference(std::vector<float> const &  pa) const
{
    std::vector<float> result;
    std::vector<float> bitmask;
    const double scale_value = m_cfg.max_val * m_cfg.scale_ratio;

    result.reserve(m_cfg.pa_length); // small heuristic to reduce reallocations
    bitmask.reserve(m_cfg.pa_length);

    for(auto const & val : pa)
    {
        if (std::isnan(val))
        {
            result.push_back(static_cast<float>(m_cfg.out_of_scale_marker));
            if (m_cfg.use_bitmask)
            {
                bitmask.push_back(0.0f);
            }
            else
            {
                bitmask.push_back(1.0f);
            }
        }
        else
        {
            auto const scaled = val / scale_value;
            result.push_back(scaled);
            bitmask.push_back(1.0f);
        }
    }

    return {std::move(result), std::move(bitmask)};
}


std::string OnnxHandler::normalize_string_training(
    const std::string& pa) const
{
    const double scale_value = m_cfg.max_val * m_cfg.scale_ratio;
    std::string out;
    out.reserve(pa.size()); // rough reservation

    for_each_comma_token(pa, [&](const std::string& token)
    {
        auto parsed = parse_token_double(token);
        if (!parsed.has_value() && m_cfg.use_bitmask)
        {
            out += "-,";
        }
        else
        {
            double scaled;
            if (parsed.has_value())
            {
                scaled = (*parsed) / scale_value;
            }
            else
            {
                scaled = m_cfg.out_of_scale_marker;
            }

            // Format as float with 6 decimal places
            std::ostringstream oss;
            oss.setf(std::ios::fixed);
            oss.precision(6);
            oss << static_cast<float>(scaled);
            out += oss.str();
            out += ",";
        }
    });

    if (!out.empty() && out.back() == ',') out.pop_back();
    return out;
}

bool OnnxHandler::is_missing_or_nan_token(const std::string& tok)
{
    if (tok.empty()) return true;

    // trim whitespace (simple)
    const auto first = tok.find_first_not_of(" \t\n\r");
    const auto last = tok.find_last_not_of(" \t\n\r");
    if (first == std::string::npos) return true;
    const std::string trimmed = tok.substr(first, last - first + 1);

    if (trimmed == "-") return true;

    // lowercase the token to check nan variants
    std::string low;
    low.reserve(trimmed.size());
    std::ranges::transform(trimmed, std::back_inserter(low),
                           [](const unsigned char c) { return static_cast<char>(std::tolower(c)); });

    return (low == "nan");
}

std::optional<double> OnnxHandler::parse_token_double(const std::string& tok)
{
    // If token is missing/nan marker -> treat as invalid
    if (is_missing_or_nan_token(tok)) return std::nullopt;

    // Trim whitespace
    auto first = tok.find_first_not_of(" \t\n\r");
    auto last = tok.find_last_not_of(" \t\n\r");
    if (first == std::string::npos) return std::nullopt;
    const std::string trimmed = tok.substr(first, last - first + 1);

    try
    {
        const double val = std::stod(trimmed);
        if (std::isnan(val)) return std::nullopt;
        return val;
    }
    catch (...)
    {
        return std::nullopt;
    }
}

// Iterate comma-separated tokens and call the provided token handler for each token.
// token_handler receives the raw token string (not trimmed) and may inspect/parse it.
void OnnxHandler::for_each_comma_token(const std::string& s,
                                       const std::function<void(const std::string&)>& token_handler)
{
    std::stringstream ss(s);
    std::string token;
    while (std::getline(ss, token, ','))
    {
        token_handler(token);
    }
}


float OnnxHandler::get_distance(const std::vector<float>& a, const std::vector<float>& b,
                                const std::vector<float>& bitmask) const
{
    switch (m_cfg.distance_type)
    {
    case DistanceType::CATEGORICAL:
    case DistanceType::EUCLIDEAN:
        return numerical_distance(a, b, bitmask);

    case DistanceType::LEVENSHTEIN:
        return levenshtein_distance(a, b, bitmask);

    default:
        throw std::runtime_error("Unknown distance type.");
    }
}


std::vector<int> OnnxHandler::denormalize_pa(const std::vector<float>& v, const std::vector<float>& bitmask) const
{
    std::vector<int> denormalized_pa;
    const auto scale_value = m_cfg.max_val * m_cfg.scale_ratio;
    const auto max_value = static_cast<int>(m_cfg.max_val);
    const auto denormalized_nan = max_value + 1;
    const bool use_bitmask = m_cfg.use_bitmask;
    auto index = 0;

    for (const float x : v)
    {
        if ((use_bitmask && bitmask[index] == 1.0) || !use_bitmask)
        {
            auto to_insert = static_cast<int>(std::round(x * scale_value));
            if (to_insert > max_value)
                to_insert = denormalized_nan;
            denormalized_pa.push_back(to_insert);
        }
        index++;
    }
    return denormalized_pa;
}

std::string OnnxHandler::pa_to_string(const std::vector<int>& v) const
{
    std::string return_string = "";
    const auto max_value = static_cast<int>(m_cfg.max_val);

    for (const int x : v)
    {
        if (x >= max_value)
        {
            return_string += "-,";
        }
        return_string += std::to_string(x) + ",";
    }

    if (!return_string.empty() && return_string.back() == ',')
        return_string.pop_back();

    return return_string;
}


float OnnxHandler::levenshtein_distance(const std::vector<float>& input, const std::vector<float>& reconstructed,
                                        const std::vector<float>& bitmask) const
{
    const std::string A = pa_to_string(denormalize_pa(input, bitmask));
    const std::string B = pa_to_string(denormalize_pa(reconstructed, bitmask));

    const auto n = A.size();
    const auto m = B.size();

    // Create a 2D DP table
    std::vector<std::vector<int>> dp(n + 1, std::vector<int>(m + 1));

    // Base cases: distance from empty string
    for (int i = 0; i <= n; ++i) dp[i][0] = i;
    for (int j = 0; j <= m; ++j) dp[0][j] = j;

    // Fill the table
    for (int i = 1; i <= n; ++i)
    {
        for (int j = 1; j <= m; ++j)
        {
            const int cost = (A[i - 1] == B[j - 1]) ? 0 : 1;

            dp[i][j] = std::min({
                dp[i - 1][j] + 1, // Deletion
                dp[i][j - 1] + 1, // Insertion
                dp[i - 1][j - 1] + cost // Substitution
            });
        }
    }

    return static_cast<float>(dp[n][m]);
}


float OnnxHandler::numerical_distance(const std::vector<float>& input, const std::vector<float>& reconstructed,
                                      const std::vector<float>& bitmask) const
{
    const std::vector<int> A = denormalize_pa(input, bitmask);
    const std::vector<int> B = denormalize_pa(reconstructed, bitmask);


    if (A.size() != B.size() || A.empty())
    {
        throw std::runtime_error("Error: Arrays must be non-empty and of equal length.\n");
    }

    double total = 0.0;
    size_t n = A.size();

    for (size_t i = 0; i < n; ++i)
    {
        if (m_cfg.distance_type == DistanceType::CATEGORICAL)
        {
            total += (A[i] != B[i]) ? 1.0 : 0.0;
        }
        else if (m_cfg.distance_type == DistanceType::EUCLIDEAN)
        {
            total += std::abs(A[i] - B[i]);
        }
    }

    return static_cast<float>(total / n); // average value
}

void OnnxHandler::process_training_data(const std::string& inputFile, const std::string& outputFile, const std::string& domain_name) const
{
    std::ifstream input(inputFile);
    if (!input)
    {
        throw std::runtime_error("Error: could not open " + inputFile + "\n");
    }

    // Open output file in write mode (creates if not exists, overwrites if exists)
    std::ofstream output(outputFile, std::ios::out | std::ios::trunc);
    if (!output)
    {
        throw std::runtime_error("Error: could not open " + outputFile + "\n");
    }

    std::string distance_type;
    switch (m_cfg.distance_type)
    {
    case DistanceType::CATEGORICAL:
        distance_type = "categorical";
        break;
    case DistanceType::EUCLIDEAN:
        distance_type = "euclidean";
        break;
    case DistanceType::LEVENSHTEIN:
        distance_type = "levenshtein";
        break;
    default:
        throw std::runtime_error("Error: unknown distance type\n");
    }

    output << m_cfg.max_val << "," << m_cfg.pa_length << "," << m_cfg.scale_ratio << "," << m_cfg.out_of_scale_marker <<
        "," << distance_type << "," << domain_name << std::endl;

    std::string line;
    while (std::getline(input, line))
    {
        std::string cleaned = normalize_string_training(line);
        output << cleaned << "\n";
    }

    input.close();
    output.close();
}
