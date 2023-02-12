#pragma once

#include "cxxopts/cxxopts.hpp"

class FznCliHelper
{
    private:
        cxxopts::Options options_parser;

    public:
        FznCliHelper();
        cxxopts::ParseResult parseArgs(int argc, char * argv[]);
        std::string getHelp();
};