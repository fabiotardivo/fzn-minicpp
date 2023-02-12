#include <libfzn/Parser.h>
#include <libfzn/Printer.h>
#include <libminicpp/solver.hpp>
#include <libminicpp/search.hpp>
#include <libfca/Timer.hpp>

#include "fzn_cli_helper.h"
#include "fzn_constraints_helper.h"
#include "fzn_output_helper.h"
#include "fzn_search_helper.h"
#include "fzn_statistics_helper.h"
#include "fzn_variables_helper.h"

int main(int argc, char * argv[])
{
    using namespace std;

    // Parse options
    FznCliHelper fzn_cli_helper;
    auto args = fzn_cli_helper.parseArgs(argc, argv);

    if ((args.count("h") == 0) and (args.count("fzn") == 1))
    {
        // Create Statistics
        SearchStatistics stats;
        stats.setStartTime();

        // Create Solver
        CPSolver::Ptr solver = Factory::makeSolver();

        // FlatZinc parsing
        Fzn::Parser fzn_parser;
        Fzn::Model const & fzn_model = fzn_parser.parse(args["fzn"].as<std::string>().c_str());

        // Variables and Printer
        FznVariablesHelper fzn_vars_helper(solver, fzn_model);
        Fzn::Printer fzn_printer;
        fzn_vars_helper.makeBoolVariables(fzn_model.bool_vars, fzn_printer);
        fzn_vars_helper.makeIntVariables(fzn_model.int_vars, fzn_printer);

        // Constraints
        FznConstraintHelper fzn_constr_helper(solver, fzn_vars_helper);
        bool const inconsistency_detected = not fzn_constr_helper.makeConstraints(fzn_model);

        // Create Search
        FznSearchHelper fzn_search_helper(solver, fzn_vars_helper);
        DFSearch search(solver, fzn_search_helper.getSearchStrategy(fzn_model));
        FznStatisticsHelper::hookToSearch(stats, search);

        // Search limits
        Limit search_limits = FznSearchHelper::makeSearchLimits(fzn_model, args);

        // Output
        FznOutputHelper fzn_output_helper(fzn_printer, cout, fzn_model, args);
        fzn_output_helper.hookToSearch(search);

        // Launch Search
        stats.setSearchStartTime();
        if (inconsistency_detected)
        {
            stats.setCompleted();
        }
        else
        {
            if (fzn_model.solve_type == "satisfy")
            {
                search.solve(stats, search_limits);
            }
            else if (fzn_model.solve_type == "minimize")
            {
                Objective::Ptr obj = Factory::minimize(fzn_vars_helper.getObjectiveVar());
                obj->onFailure([&](){ stats.incrTighteningFail();});
                search.optimize(obj, stats, search_limits);
            }
            else if (fzn_model.solve_type == "maximize")
            {
                Objective::Ptr obj = Factory::maximize(fzn_vars_helper.getObjectiveVar());
                obj->onFailure([&](){ stats.incrTighteningFail();});
                search.optimize(obj, stats, search_limits);
            }
            else
            {
                throw std::runtime_error("Unknown problem type");
            }
        }
        stats.setSearchEndTime();

        // Final output
        fzn_output_helper.printFinalOutput(stats.getCompleted(), stats.getSolutions());

        // Statistics output
        if (args["s"].count() != 0)
        {
            FznStatisticsHelper::printStatistics(stats, solver, fzn_model, search, cout);
        }

        exit(EXIT_SUCCESS);
    }
    else
    {
        std::cout << fzn_cli_helper.getHelp();
        exit(EXIT_FAILURE);
    }
}
