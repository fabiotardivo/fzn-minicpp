#pragma once

#include <sstream>
#include <cxxopts/cxxopts.hpp>
#include <libfzn/Printer.h>
#include <libfzn/Model.h>
#include <libminicpp/search.hpp>

class FznOutputHelper
{
    private:
        constexpr static char const solution_separator[] = "----------";
        constexpr static char const search_completed_with_solution_separator[] = "==========";
        constexpr static char const search_completed_without_solution_separator[] = "=====UNSATISFIABLE=====";
        constexpr static char const search_interrupted_without_solution_separator[] = "=====UNKNOWN=====";
        Fzn::Printer const & fzn_printer;
        bool print_all_solutions;
        std::stringstream last_solution;
        std::ostream & output_stream;

    public:
        FznOutputHelper(Fzn::Printer const & fzn_printer, std::ostream & output_stream, Fzn::Model const & fzn_model, cxxopts::ParseResult const & args);
        template<typename Search>
        void hookToSearch(Search & search);
        void printFinalOutput(bool completed, unsigned int solutions);
};


template<typename Search>
void FznOutputHelper::hookToSearch(Search & search)
{
    using namespace std;
    search.onSolution([&]()
    {
        last_solution.str("");
        fzn_printer.print_outputs(last_solution);
        if (print_all_solutions)
        {
            output_stream << last_solution.str();
            output_stream << solution_separator << endl;
        }
    });
}
