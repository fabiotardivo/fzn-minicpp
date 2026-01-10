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
#include <ml/OnnxHandler.h>
#include <ml/utils.h>
#include <sstream>

int main(int argc, char * argv[])
{
    using namespace std;

    // Parse options
    std::string fzn;
    std::string model;
    bool masked = true;
    std::string distance = "categorical";
    std::string rank = "worst";
    int lookahead = 0;
    float scale_ratio = 1.0; // set >1.0 to reserve headroom (e.g. 5.0/4.0)
    float out_of_scale_marker = -1; // value to use for missing/out-of-scale entries
    cxxopts::Options optsParser("fzn-minicpp-ml", "A C++ MiniZinc solver based on MiniCP.");
    optsParser.custom_help("[Options]");
    optsParser.positional_help("<FlatZinc>");
    optsParser.add_options()
        ("a", "Print all solutions", cxxopts::value<bool>())
        ("n", "Stop search after <n> solutions", cxxopts::value<unsigned int>())
        ("s", "Print search statistics", cxxopts::value<bool>())
        ("t", "Stop search after <t> ms", cxxopts::value<unsigned int>())
        ("model", "Machine learning model in ONNX format", cxxopts::value<std::string>(model))
        ("masked", "Ignore unassigned variables (Default = True)", cxxopts::value<bool>(masked))
        ("distance", "Criteria to evaluate reconstruction: categorical, euclidean, levenshtein (Default = categorical).", cxxopts::value<std::string>(distance))
        ("rank", "Criteria to rank scores: best, avg, worst (Default = best)", cxxopts::value<std::string>(rank))
        ("lookahead", "Lookahead depth (Default = 0)", cxxopts::value<int>(lookahead))
        ("scale", "Scale ratio (Default = 1.0)", cxxopts::value<float>(scale_ratio))
        ("out-of-scale", "Out of scale marker (Default = -1.0)", cxxopts::value<float>(out_of_scale_marker))
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

        // Load ML evaluator
        ML::IntVars const & intDecVars = searchHelper.getIntDecisionalVars(fznModel);
        size_t pa_length = intDecVars.size(); // Number of variables in the problems and length of the pa
        float max_val = 0; // maximum possible raw value in your domain
        for(auto const & var : intDecVars)
            max_val = std::max(max_val,static_cast<float>(var->max()));
        DistanceType const distance_type = distanceFromString(distance); // distance computation method
        OnnxHandler::create_instance(model, max_val, pa_length, scale_ratio, out_of_scale_marker, distance_type, masked);
        OnnxHandler const & onnx_handler = OnnxHandler::get_instance();
        ML::RankType rankType = ML::rankFromString(rank);

        ML::EvalFunctionType eval_fun = [&](int varIdx, ML::IntVars const & vars)
        {
            auto const & base_pa = ML::getPartialAssignment(vars);
            auto const & base_pas = ML::getAllPartialAssignments(varIdx,vars, base_pa);
            auto const & pas = ML::getLookahead(lookahead, vars, base_pas);
            std::vector<float> scores;
            for (auto const & pa : pas)
            {
                //ML::printPartialAssignment(pa);
                scores.push_back(onnx_handler.get_score(pa));
            }
            auto result = ML::getScoreVal(varIdx,pas,scores,rankType);
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
