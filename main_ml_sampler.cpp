#include <thread>

#include <Parser.h>
#include <solver.hpp>
#include <search.hpp>

#include "fzn_constraints_helper.h"
#include "fzn_search_helper.h"
#include "fzn_variables_helper.h"
#include "ml_sampler.h"

int main(int argc, char * argv[])
{
    using namespace std;

    // Parse options
    int sTimeout = std::numeric_limits<int>::max();
    int nSamplers = std::thread::hardware_concurrency();
    std::string fzn;
    cxxopts::Options optsParser("fzn-minicpp", "A C++ MiniZinc solver based on MiniCP.");
    optsParser.custom_help("[Options]");
    optsParser.positional_help("<FlatZinc>");
    optsParser.add_options()
        ("t,timeout", "Stop search after <t> s", cxxopts::value(sTimeout))
        ("s,samplers", "Number of samplers", cxxopts::value(nSamplers))
        ("fzn", "FlatZinc", cxxopts::value(fzn))
        ("h,help", "Print usage");
    optsParser.parse_positional({"fzn"});

    auto args = optsParser.parse(argc, argv);

    if ((args.count("h") == 0) and (not fzn.empty()))
    {
        // Launch samplers
        bool stop = false;
        std::mutex outMutex;
        std::vector<std::thread> sThreads;
        sThreads.reserve(nSamplers);
        for (auto sIdx = 0; sIdx < nSamplers; sIdx += 1)
        {
            sThreads.emplace_back(ML::Sampler, sIdx, std::ref(fzn), std::ref(outMutex), std::ref(stop));
        }

        // Timeout
        std::this_thread::sleep_for(std::chrono::seconds (sTimeout));
        stop = true;
        for (auto & t : sThreads)
        {
            if (t.joinable()) t.join();
        }
    }
    else
    {
        std::cout << optsParser.help();
        exit(EXIT_FAILURE);
    }
}
