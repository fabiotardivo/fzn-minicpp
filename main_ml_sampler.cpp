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
    std::string fznPath;
    std::string outPath;
    cxxopts::Options optsParser("fzn-minicpp", "A C++ MiniZinc sampler based on MiniCPP.");
    optsParser.custom_help("[Options]");
    optsParser.positional_help("<FlatZinc>");
    optsParser.add_options()
        ("t,timeout", "Stop search after <t> s", cxxopts::value(timeout))
        ("s,samplers", "Number of samplers", cxxopts::value(nSamplers))
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
            sThreads.emplace_back(
                ML::Sampler,
                sIdx,
                std::ref(fznPath),
                std::ref(outFile),
                std::ref(outMutex),
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