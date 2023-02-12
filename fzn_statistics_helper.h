#pragma once

#include <libminicpp/search.hpp>
#include <libfzn/Model.h>
#include "fzn_variables_helper.h"

class FznStatisticsHelper
{
    public:
        static void hookToSearch(SearchStatistics & stats, DFSearch & search);
        static void printStatistics(SearchStatistics const & stats,
                                    CPSolver::Ptr solver,
                                    Fzn::Model const & fzn_model,
                                    DFSearch const & search,
                                    std::ostream & os);
};
