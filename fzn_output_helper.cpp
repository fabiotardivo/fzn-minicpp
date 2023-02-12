
#include "fzn_output_helper.h"

FznOutputHelper::FznOutputHelper(Fzn::Printer const & fzn_printer, std::ostream & output_stream, Fzn::Model const & fzn_model, cxxopts::ParseResult const & args):
    fzn_printer(fzn_printer), output_stream(output_stream)
{
    if (fzn_model.solve_type == "satisfy")
    {
        print_all_solutions = true;
    }
    else
    {
        print_all_solutions = args["a"].count() != 0;
    }
}

void FznOutputHelper::printFinalOutput(bool completed, unsigned int solutions)
{
    using namespace std;
    if (not print_all_solutions)
    {
        output_stream << last_solution.str();
        output_stream << solution_separator << endl;
    }
    if (completed)
    {
        if (solutions > 0)
        {
            output_stream << search_completed_with_solution_separator << endl;
        }
        else
        {
            output_stream << search_completed_without_solution_separator << endl;
        }
    }
    else if (solutions == 0)
    {
        output_stream << search_interrupted_without_solution_separator << endl;
    }
}
