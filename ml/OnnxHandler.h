#pragma once
#include <onnxruntime_cxx_api.h>
#include <optional>
#include <string>


enum DistanceType
{
    CATEGORICAL, ///< Categorical distance, each position is either a yes or no match. Depending on if we are using the bitmask or not the nan counts as a mismatch or is ignored.
    EUCLIDEAN, ///< Numerical distance, each position is measure as |reconstructed_value - real_value|. Depending on if we are using the bitmask or not the nan counts as a mismatch or is ignored.
    LEVENSHTEIN ///< All the values are aggregated and create a string, then Levenshtein is used. Depending on if we are using the bitmask or not the nan counts as a mismatch or is ignored.
};

inline
DistanceType distanceFromString(std::string const & s)
{
    static const std::unordered_map<std::string, DistanceType> map
    {
            {"categorical", DistanceType::CATEGORICAL},
            {"euclidean", DistanceType::EUCLIDEAN},
            {"levenshtein", DistanceType::LEVENSHTEIN}
    };

    auto it = map.find(s);
    if (it != map.end()) return it->second;
    throw std::invalid_argument("Invalid distance: " + s);
}


// ---------------------------
// Configurable parameters
// ---------------------------
struct InputConfig
{
    float max_val = 120.0; // maximum possible raw value in your domain
    size_t pa_length = 120; // Number of variables in the problems and length of the pa
    float scale_ratio = 1.0; // set >1.0 to reserve headroom (e.g. 5.0/4.0)
    float out_of_scale_marker = -1; // value to use for missing/out-of-scale entries
    DistanceType distance_type = DistanceType::CATEGORICAL; // distance computation method
    bool use_bitmask = true; // whether to use bitmasking for missing values

};


class OnnxHandler
{
public:

    InputConfig m_cfg; ///< Configuration parameters for input preprocessing.

    /**
     * \brief Get the singleton instance of OnnxHandler.
     * \return Reference to the singleton instance.
     */
    static OnnxHandler& get_instance();

    /**
     * \brief Create the singleton instance of OnnxHandler.
     */

    static void create_instance(const std::string& model_path, float max_val, size_t pa_length, float scale_ratio, float out_of_scale_marker, DistanceType
                                distance_type, bool
                                use_bitmask);
    OnnxHandler(const std::string& model_path, float max_val, size_t pa_length, float scale_ratio,
                float out_of_scale_marker, DistanceType distance_type, bool use_bitmask);

    /**
     * \brief Get the score for a given Partial Assignment using the using native C++ code.
     * \param pa The partial assignment to evaluate.
     * \return The heuristic score for the partial assignment.
     */
    [[nodiscard]] float get_score(const std::string& pa, bool toClean = true) const;
    [[nodiscard]] float get_score(std::vector<float> const & pa) const;

    void process_training_data(const std::string& inputFile, const std::string& outputFile, const std::string& domain_name) const;


    /** \brief Deleted copy constructor (singleton pattern). */
    OnnxHandler(const OnnxHandler&) = delete;

    /** \brief Deleted copy assignment operator (singleton pattern). */
    OnnxHandler& operator=(const OnnxHandler&) = delete;

    /** \brief Deleted move constructor (singleton pattern). */
    OnnxHandler(OnnxHandler&&) = delete;

    /** \brief Deleted move assignment operator (singleton pattern). */
    OnnxHandler& operator=(OnnxHandler&&) = delete;




private:
    /**
     * \brief Private constructor for singleton pattern.
     */
    OnnxHandler(float max_val, size_t pa_length, float scale_ratio, float out_of_scale_marker,
                DistanceType distance_type,
                bool use_bitmask);

    static OnnxHandler* instance; ///< Singleton instance pointer


    bool m_verbose = false;
    ///< Verbose mode for debugging and logging.

    std::string m_model_path = "models/model.onnx";
    //ArgumentParser::get_instance().get_GNN_model_path(); ///< Path to the GNN model

    ///// --- ONNX Runtime inference components ---
    Ort::Env m_env{
        ORT_LOGGING_LEVEL_ERROR,
        "GraphNNEnv"
    }; ///< ONNX Runtime environment for GNN inference.
    Ort::SessionOptions m_session_options; ///< ONNX Runtime session options.
    std::unique_ptr<Ort::Session>
    m_session; ///< Pointer to the ONNX Runtime session.
    std::unique_ptr<Ort::AllocatorWithDefaultOptions>
    m_allocator; ///< Allocator for ONNX Runtime memory management.
    std::unique_ptr<Ort::MemoryInfo>
    m_memory_info; ///< Memory info for ONNX Runtime tensors.

    std::vector<std::string>
    m_input_names; ///< Names of the input nodes for the ONNX model.
    std::vector<std::string>
    m_output_names; ///< Names of the output nodes for the ONNX model.

    bool m_model_loaded =
        false; ///< Indicates whether the ONNX model has been loaded.


    /**
     * \brief Initializes the ONNX Runtime model for NN inference.
     *
     * Sets up the ONNX Runtime environment, session options, loads the GNN model,
     * and prepares input/output names and memory information required for
     * inference. This function should be called before performing any inference
     * with the model.
     */
    void initialize_onnx_model();

    /**
     * \brief Runs ONNX Runtime inference on the provided Partial Assignment.
     *
     * \param pa The Partial Assignment to evaluate.
     * \param bitmask
     * \return The reconstructed vector of float to use to calculate the score.
     */
    [[nodiscard]] std::vector<float> run_inference(const std::vector<float>& pa, const std::vector<float>& bitmask) const;


    /**
       * \brief Cleans the input Partial Assignment string by removing extra spaces and ensuring proper formatting.
       * \param pa The input Partial Assignment as a string.
       * \return A cleaned Partial Assignment string.
       */
    static std::string clean_pa(const std::string& pa);


    /** \brief Function to preprocess the partial assignment before passing it to the model.
     *
     *  \param pa The input partial assignment as a string.
     * \return A vector of floats representing the preprocessed partial assignment.
 */
    [[nodiscard]] std::pair<std::vector<float>, std::vector<float>> preprocess_pa_inference(const std::string& pa, bool toClean = true) const;

    /** \brief Function to preprocess the partial assignment string for training.
     *
     *  \param pa The input partial assignment as a string.
     * \return A normalized partial assignment string suitable for training.
     */
    [[nodiscard]] std::string preprocess_pa_train(const std::string& pa) const;



    /**
     * \brief Preprocess a Partial Assignment string for inference.
     * \param pa The input Partial Assignment as a string.
     * \return A vector of floats representing the normalized Partial Assignment.
     */
    [[nodiscard]] std::pair<std::vector<float>, std::vector<float>> normalize_string_inference(const std::string& pa) const;
    [[nodiscard]] std::pair<std::vector<float>, std::vector<float>> normalize_pa_inference(std::vector<float> const & pa) const;

    /**
     * \brief Preprocess a Partial Assignment string for training.
     * \param pa The input Partial Assignment as a string.
     * \return A normalized Partial Assignment string suitable for training.
     */
    [[nodiscard]] std::string normalize_string_training(const std::string& pa) const;
    static bool is_missing_or_nan_token(const std::string& tok);
    static std::optional<double> parse_token_double(const std::string& tok);
    static void for_each_comma_token(const std::string& s, const std::function<void(const std::string&)>& token_handler);

    /** \brief Compute the Numerical distance between two vectors of floats.
    * \param input The first vector.
    * \param reconstructed The second vector.
    * \param bitmask The bitmask vector indicating valid positions.
    * \return The Numerical distance between the two vectors.
    */
    [[nodiscard]] float numerical_distance(const std::vector<float>& input, const std::vector<float>& reconstructed, const std::vector<float>& bitmask) const;


    /** \brief Compute the Levenshtein distance between two vectors of floats.
    * \param input The first vector.
    * \param reconstructed The second vector.
    * \param bitmask The bitmask vector indicating valid positions.
    * \return The Levenshtein distance between the two vectors.
    */
    [[nodiscard]] float levenshtein_distance(const std::vector<float>& input, const std::vector<float>& reconstructed, const std::vector<float>& bitmask) const;



    /** \brief Compute the distance between two vectors of floats (depending on the chosen method).
    * \param input The first vector.
    * \param reconstructed The second vector.
    * \param bitmask The bitmask vector indicating valid positions.
    * \return The distance between the two vectors.
    */
    [[nodiscard]] float get_distance(const std::vector<float>& input, const std::vector<float>& reconstructed, const std::vector<float>& bitmask) const;
    [[nodiscard]] std::vector<int> denormalize_pa(const std::vector<float>& v, const std::vector<float>& bitmask) const;
    [[nodiscard]] std::string pa_to_string(const std::vector<int>& v) const;
};
