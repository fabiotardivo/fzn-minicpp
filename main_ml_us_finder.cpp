#include <thread>
#include <fstream>
#include <Parser.h>
#include <solver.hpp>

#include "fzn_search_helper.h"
#include "ml/USFinder.h"

std::string readFile(std::string const & fPath) {
    std::ifstream in(fPath, std::ios::binary);
    if (!in) throw std::runtime_error("Failed to open the file: " + fPath);

    in.seekg(0, std::ios::end);
    std::streamoff const size = in.tellg();
    if (size < 0) throw std::runtime_error("Failed to scan the file: " + fPath);
    in.seekg(0, std::ios::beg);

    std::string data;
    data.resize(static_cast<std::size_t>(size));
    if (size > 0) in.read(data.data(), size);

    if (!in && size > 0)  throw std::runtime_error("Failed to read the file: " + fPath);
    return data;
}

std::string_view getNextLine(std::string_view& sv) {

    std::string_view line;

    if (not sv.empty())
    {
        auto pos = sv.find('\n');
        line = (pos == std::string_view::npos) ? sv: sv.substr(0, pos);
        sv.remove_prefix(pos == std::string_view::npos ? sv.size() : pos + 1);
    }

    return line;
}

int main(int argc, char * argv[])
{
    using namespace std;

    // Parse options
    int timeout = std::numeric_limits<int>::max();
    int nFinders = std::thread::hardware_concurrency();
    int nAttempts = 2;
    std::string fzn;
    std::string paPath;
    std::string outPath;
    cxxopts::Options optsParser("fzn-minicpp", "A C++ MiniZinc solver based on MiniCP.");
    optsParser.custom_help("[Options]");
    optsParser.positional_help("<FlatZinc>");
    optsParser.add_options()
        ("t,timeout", "Stop search after <t> s", cxxopts::value(timeout))
        ("f,finders", "Number of finders", cxxopts::value(nFinders))
        ("pa", "Partial assignments file path", cxxopts::value(paPath))
        ("o,output", "Output file path", cxxopts::value(outPath))
        ("a,attempts", "Attempts to find MUS ", cxxopts::value(nAttempts))
        ("fzn", "FlatZinc", cxxopts::value(fzn))
        ("h,help", "Print usage");
    optsParser.parse_positional({"fzn"});

    auto args = optsParser.parse(argc, argv);

    if ((args.count("h") == 0) and (not paPath.empty()) and (not outPath.empty()) and (not fzn.empty()))
    {
        // Read partial assignments file
        std::string paString = readFile(paPath);
        std::string_view paStringView(paString);

        // Skip first 2 header lines
        getNextLine(paStringView);
        getNextLine(paStringView);

        // Build a vector of lines
        std::vector<std::string_view> paLines;
        while (not paStringView.empty())
        {
            std::string_view line = getNextLine(paStringView);
            if (not line.empty())
            {
                paLines.emplace_back(line);
            }
        }

        if (paLines.empty()) throw std::runtime_error("No partial assignments");
        if (paLines.size() % 2 != 0) throw std::runtime_error("Odd number of partial assignments");

        // Launch finders
        bool stop = false;
        auto outFile = openFile(outPath);
        std::mutex outMutex;
        std::vector<std::thread> fThreads;
        fThreads.reserve(nFinders);
        int const pairsPerThread = (paLines.size() / 2 + nFinders - 1) / nFinders;
        int const linesPerThread = pairsPerThread * 2;
        for (auto fIdx = 0; fIdx < nFinders; fIdx += 1)
        {
            int const start = fIdx * linesPerThread;
            int const end = std::min(start + linesPerThread, static_cast<int>(paLines.size()));
            std::span<std::string_view const> fLines(paLines.begin() + start,paLines.begin() + end);

            fThreads.emplace_back(ML::USFinder,
                                    fIdx,
                                    nAttempts,
                                    fLines,
                                    std::ref(fzn),
                                    std::ref(outFile),
                                    std::ref(outMutex),
                                    std::ref(stop));
        }

        // Timeout
        std::this_thread::sleep_for(std::chrono::seconds(timeout));
        stop = true;
        for (auto & t : fThreads)
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
