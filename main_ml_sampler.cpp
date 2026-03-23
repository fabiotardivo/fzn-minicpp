#include <thread>
#include <fstream>

#include <Parser.h>
#include <solver.hpp>

#include "ml/Sampler.h"
#include "ml/Common.h"

int main(int argc, char * argv[])
{
     // Parse options
    int timeout = std::numeric_limits<int>::max();
    int nSamplers = static_cast<int>(std::thread::hardware_concurrency());
    unsigned long long maxFailures = std::numeric_limits<unsigned long long>::max();
    bool fixedSeed = false;
    double maxTemperature = 1.0;
    std::string fznPath;
    std::string outPath;
    cxxopts::Options optsParser("fzn-minicpp", "A C++ MiniZinc sampler based on MiniCPP.");
    optsParser.custom_help("[Options]");
    optsParser.positional_help("<FlatZinc>");
    optsParser.add_options()
        ("t,timeout", "Stop search after <t> s", cxxopts::value(timeout))
        ("s,samplers", "Number of samplers", cxxopts::value(nSamplers))
        ("seed", "Fixed random seed (default: false)", cxxopts::value(fixedSeed))
        ("failures", "Stop search after 'arg' failures",cxxopts::value(maxFailures))
        ("temperature", "Sampling temperature: 0=deterministic heuristic, 1=proportional, inf=uniform random",
        cxxopts::value(maxTemperature))
        ("o,output", "Output file path", cxxopts::value(outPath))
        ("fzn", "FlatZinc file path", cxxopts::value(fznPath))
        ("h,help", "Print usage");
    optsParser.parse_positional({"fzn"});

    auto args = optsParser.parse(argc, argv);

    if ((args.count("h") == 0) and (not outPath.empty()) and (not fznPath.empty()))
    {
        // Open output file
        auto outFile = openFile(outPath);
        std::mutex outMutex;

        // Launch samplers
        bool stop = false;
        std::vector<std::thread> sThreads;
        sThreads.reserve(nSamplers);
        for (auto sIdx = 0; sIdx < nSamplers; sIdx += 1)
        {
            unsigned long long budget;
            // Exponential spread: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512...
            // With 64 samplers and maxFailures=1000, you get:
            // many samplers at depth 1-10, fewer at depth 100-1000
            if (maxFailures == std::numeric_limits<unsigned long long>::max())
                budget = 1ULL << std::min((int)sIdx, 15);  // cap at 2^15 = 32768
            else {
                // spread exponentially between 1 and maxFailures
                double t = (double)sIdx / (double)(nSamplers - 1);
                budget = std::max(1ULL, (unsigned long long)std::pow((double)maxFailures, t));
            }

            // spread temperature: low (near heuristic) to high (near random)
            double temperature = maxTemperature * (sIdx + 1) / nSamplers;


            sThreads.emplace_back(
                ML::Sampler,
                sIdx,
                fixedSeed,
                temperature,
                std::ref(fznPath),
                std::ref(outFile),
                std::ref(outMutex),
                budget,  // spread budgets,
                std::ref(stop));
        }

        // Timeout
        std::this_thread::sleep_for(std::chrono::seconds (timeout));
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