#include <Parser.h>
#include <Printer.h>

#include <pybind11/embed.h>

#include <solver.hpp>
#include <search.hpp>

#include "fzn_constraints_helper.h"
#include "fzn_output_helper.h"
#include "fzn_search_helper.h"
#include "fzn_statistics_helper.h"
#include "fzn_variables_helper.h"

#include "ml_sampler.h"

namespace py = pybind11;

int main(int argc, char * argv[])
{
    using namespace std;

    // Parse options
    cxxopts::Options optsParser("fzn-minicpp-ml", "A C++ MiniZinc solver based on MiniCP.");
    optsParser.custom_help("[Options]");
    optsParser.positional_help("<FlatZinc>");
    optsParser.add_options()
        ("a", "Print all solutions", cxxopts::value<bool>())
        ("n", "Stop search after <n> solutions", cxxopts::value<unsigned int>())
        ("s", "Print search statistics", cxxopts::value<bool>())
        ("t", "Stop search after <t> ms", cxxopts::value<unsigned int>())
        ("p,producers", "Number of producers", cxxopts::value<unsigned int>())
        ("m,model", "Machine Learning model", cxxopts::value<std::string>())
        ("fzn", "FlatZinc", cxxopts::value<std::string>())
        ("h,help", "Print usage");
    optsParser.parse_positional({"fzn"});

    auto args = optsParser.parse(argc, argv);

    if ((args.count("h") == 0) and (args.count("fzn") == 1))
    {
        // FlatZinc file
        auto const & fzn = args["fzn"].as<std::string>();

        // Producers
        if (args.count("p") == 1)
        {
            // Launch producers
            bool stop = false;
            std::mutex outMutex;
            auto const nProducers = args["p"].as<unsigned int>();
            std::vector<std::thread> producers;
            producers.reserve(nProducers);
            for (auto pIdx = 0; pIdx < nProducers; pIdx += 1)
            {
                producers.emplace_back(ML::Producer, pIdx, std::ref(fzn), std::ref(outMutex), std::ref(stop));
            }

            // Timeout
            auto const timeout = args["t"].as<unsigned int>();
            std::this_thread::sleep_for(std::chrono::milliseconds(timeout));
            stop = true;
            for (auto & p : producers)
            {
                if (p.joinable())
                {
                    p.join();
                }
            }
        }
        else if (args.count("m") == 1)
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

            // Read python file
            static py::scoped_interpreter guard{};

            py::object scope = py::eval_file(args["m"].as<std::string>());
            py::function getVarVal = py::globals()["getVarVal"].cast<py::function>();

            // Create Search
            FznSearchHelper searchHelper(solver, varsHelper);
            DFSearch search(solver, searchHelper.getSearchStrategy(fznModel, getVarVal));
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
    else
    {
        std::cout << optsParser.help();
        exit(EXIT_FAILURE);
    }
}
