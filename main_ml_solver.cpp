#include <ml/TorchHandler.h>
#include <Parser.h>
#include <Printer.h>
#include <solver.hpp>
#include <search.hpp>
#include "fzn_constraints_helper.h"
#include "fzn_output_helper.h"
#include "fzn_search_helper.h"
#include "fzn_statistics_helper.h"
#include "fzn_variables_helper.h"
#include <libfca/Slice.hpp>

#include <ml/utils.h>
#include <sstream>

int main(int argc, char * argv[])
{
    using namespace std;

    // Parse options
    std::string fzn;
    std::string model;
    std::string rank = "worst";
    int lookahead = 0;
    int batchSize = 32;  // Batch size for inference

    cxxopts::Options optsParser("fzn-minicpp-ml", "A C++ MiniZinc solver based on MiniCP.");
    optsParser.custom_help("[Options]");
    optsParser.positional_help("<FlatZinc>");
    optsParser.add_options()
        ("a", "Print all solutions", cxxopts::value<bool>())
        ("n", "Stop search after <n> solutions", cxxopts::value<unsigned int>())
        ("s", "Print search statistics", cxxopts::value<bool>())
        ("t", "Stop search after <t> ms", cxxopts::value<unsigned int>())
        ("model", "Machine learning model in TorchScript format (.pt)", cxxopts::value<std::string>(model))
        ("rank", "Criteria to rank scores: best, avg, worst (Default = worst)", cxxopts::value<std::string>(rank))
        ("lookahead", "Lookahead depth (Default = 0)", cxxopts::value<int>(lookahead))
        ("batch-size", "Batch size for ML inference (Default = 32)", cxxopts::value<int>(batchSize))
        ("fzn", "FlatZinc", cxxopts::value<std::string>(fzn))
        ("h,help", "Print usage");
    optsParser.parse_positional({"fzn"});

    auto args = optsParser.parse(argc, argv);

    if ((args.count("h") == 0) and (not fzn.empty()) and (not model.empty()))
    {
        // Create Statistics
        SearchStatistics stats;
        stats.setStartTime();

        // Create Solver
        CPSolver::Ptr solver = Factory::makeSolver();

        // FlatZinc parsing
        Fzn::Parser fznParser;
        Fzn::Model const & fznModel = fznParser.parse(fzn);

        // Variables and Printer
        FznVariablesHelper varsHelper(solver, fznModel);
        Fzn::Printer fznPrinter;
        varsHelper.makeBoolVariables(fznModel.bool_vars, &fznPrinter);
        varsHelper.makeIntVariables(fznModel.int_vars, &fznPrinter);

        // Constraints
        FznConstraintHelper constrsHelper(solver, varsHelper);
        auto const isConsistent = constrsHelper.makeConstraints(fznModel);

        // Create Search
        FznSearchHelper searchHelper(solver, varsHelper);

        // Load ML evaluator with PyTorch
        ML::RankType rankType = ML::rankFromString(rank);
        TorchHandler::getInstance(model).setVerbose(true);

        // Update batch size if provided
        if (args["batch-size"].count() != 0)
        {
            batchSize = args["batch-size"].as<int>();
        }

        // Batched ML evaluation function
        ML::EvalFunctionType eval_fun = [&](int varIdx, ML::IntVars const & vars)
        {
            auto const & base_pa = ML::getPartialAssignment(vars);
            auto const & base_pas = ML::getAllPartialAssignments(varIdx, vars, base_pa);
            auto const & pas = ML::getLookahead(lookahead, vars, base_pas);

            // Convert list to vector for batching (if pas is a list)
            std::vector<std::vector<float>> pasVec(pas.begin(), pas.end());

            std::vector<float> scores;
            scores.reserve(pasVec.size());

            // Batch inference
            if (pasVec.size() <= static_cast<size_t>(batchSize))
            {
                // Single batch - process all at once
                scores = TorchHandler::getInstance(model).runInferenceBatch(pasVec);
            }
            else
            {
                // Multiple batches - process in chunks
                for (size_t i = 0; i < pasVec.size(); i += batchSize)
                {
                    size_t end = std::min(i + batchSize, pasVec.size());
                    std::vector<std::vector<float>> batch(pasVec.begin() + i, pasVec.begin() + end);

                    auto batchScores = TorchHandler::getInstance(model).runInferenceBatch(batch);
                    scores.insert(scores.end(), batchScores.begin(), batchScores.end());
                }
            }

            auto result = ML::getScoreVal(varIdx, pas, scores, rankType);
            return result;
        };

        DFSearch search(solver, searchHelper.getMLSearchStrategy(fznModel, eval_fun));
        FznStatisticsHelper::hookToSearch(stats, search);

        // Search limits
        Limit searchLimits = FznSearchHelper::makeSearchLimits(fznModel, args);

        // Output
        FznOutputHelper outputHelper(fznPrinter, cout, fznModel, args);
        outputHelper.hookToSearch(search);

        // Launch Search
        stats.setSearchStartTime();
        if (isConsistent)
        {
            if (fznModel.solve_type == "satisfy")
            {
                search.solve(stats, searchLimits);
            }
            else if (fznModel.solve_type == "minimize")
            {
                Objective::Ptr obj = Factory::minimize(varsHelper.getObjectiveVar());
                obj->onFailure([&](){ stats.incrTighteningFail();});
                search.optimize(obj, stats, searchLimits);
            }
            else if (fznModel.solve_type == "maximize")
            {
                Objective::Ptr obj = Factory::maximize(varsHelper.getObjectiveVar());
                obj->onFailure([&](){ stats.incrTighteningFail();});
                search.optimize(obj, stats, searchLimits);
            }
            else
            {
                throw std::runtime_error("Unknown problem type");
            }
        }
        else
        {
            stats.setCompleted();
        }

        stats.setSearchEndTime();

        // Final output
        outputHelper.printFinalOutput(stats.getCompleted(), stats.getSolutions());

        // Statistics output
        if (args["s"].count() != 0)
        {
            FznStatisticsHelper::printStatistics(stats, solver, fznModel, search, cout);
        }

        exit(EXIT_SUCCESS);
    }
    else
    {
        std::cout << optsParser.help();
        exit(EXIT_FAILURE);
    }
}