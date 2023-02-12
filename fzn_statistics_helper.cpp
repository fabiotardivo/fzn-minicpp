#include "fzn_statistics_helper.h"

void FznStatisticsHelper::hookToSearch(SearchStatistics & stats, DFSearch & search)
{
    search.onSolution([&](){stats.incrSolutions();});
    search.onBranch([&](){ stats.incrNodes();});
    search.onFailure([&](){stats.incrFailures();});
}

void FznStatisticsHelper::printStatistics(SearchStatistics const & stats,
                                          CPSolver::Ptr solver,
                                          Fzn::Model const & fzn_model,
                                          DFSearch const & search,
                                          std::ostream & os)
{
    using namespace std;

    os << std::fixed << std::setprecision(3)
       << "%%%mzn-stat: initTime=" << stats.getInitTime() << std::endl
       << "%%%mzn-stat: solveTime=" << stats.getSolveTime() << std::endl
       << "%%%mzn-stat: solutions=" << stats.getSolutions() << std::endl
       << "%%%mzn-stat: variables=" << fzn_model.int_vars.size() + fzn_model.bool_vars.size() << std::endl
       << "%%%mzn-stat: propagators=" << fzn_model.constraints.size() << std::endl
       << "%%%mzn-stat: propagations=" << solver->getPropagations() << std::endl
       << "%%%mzn-stat: nodes=" << 1 + stats.getNodes() + stats.getFailures() - stats.getTighteningFail() << std::endl
       << "%%%mzn-stat: failures=" << stats.getFailures()  - stats.getTighteningFail() << std::endl
       << "%%%mzn-stat: peakDepth=" << search.getPeakDepth() << std::endl
       << "%%%mzn-stat-end" << std::endl;
}
