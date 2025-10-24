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

int main(int argc, char * argv[])
{
    using namespace std;

    // Parse options
    std::string fzn;
    std::string ml_model;
    cxxopts::Options optsParser("fzn-minicpp-ml", "A C++ MiniZinc solver based on MiniCP.");
    optsParser.custom_help("[Options]");
    optsParser.positional_help("<FlatZinc>");
    optsParser.add_options()
        ("a", "Print all solutions", cxxopts::value<bool>())
        ("n", "Stop search after <n> solutions", cxxopts::value<unsigned int>())
        ("s", "Print search statistics", cxxopts::value<bool>())
        ("t", "Stop search after <t> ms", cxxopts::value<unsigned int>())
        ("m,model", "Machine learning model", cxxopts::value<std::string>(ml_model))
        ("fzn", "FlatZinc", cxxopts::value<std::string>(fzn))
        ("h,help", "Print usage");
    optsParser.parse_positional({"fzn"});

    auto args = optsParser.parse(argc, argv);

    if ((args.count("h") == 0) and (not fzn.empty()) and (not ml_model.empty()))
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

        // Load ML evaluator
        std::function<float(std::vector<float> const &)> ml_eval_fun = [&](std::vector<float> const & pa) -> float
        {
            int vIdx = -1;
            for (int i = 0; i < pa.size(); i += 1)
            {
                if (not std::isnan(pa[i]))
                {
                   vIdx = i;
                   break;
                }
            }
            assert(vIdx >=0);
            float val = pa[vIdx];
            float score = static_cast<float>(pa.size() * vIdx + val);
            //Fca::Slice<float>::print(pa.data(), pa.data()+pa.size(), "%f");
            //std::cout << score << std::endl;
            //std::cout << score << std::endl;
            return score;
        };

        // Create Search
        FznSearchHelper searchHelper(solver, varsHelper);
        DFSearch search(solver, searchHelper.getSearchStrategy(fznModel, ml_eval_fun));
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
