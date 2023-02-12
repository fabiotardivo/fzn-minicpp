#include "fzn_cli_helper.h"

FznCliHelper::FznCliHelper() :
        options_parser("fzn-minicpp", "A C++ MiniZinc solver based on MiniCP.")
{
    options_parser.custom_help("[Options]");
    options_parser.positional_help("<FlatZinc>");
    options_parser.add_options()
            ("a", "Print all solutions", cxxopts::value<bool>())
            ("n", "Stop search after 'arg' solutions", cxxopts::value<unsigned int>())
            ("s", "Print search statistics", cxxopts::value<bool>())
            ("t", "Stop search after 'arg' ms", cxxopts::value<unsigned int>())
            ("fzn", "FlatZinc", cxxopts::value<std::string>())
            ("h,help", "Print usage");
    options_parser.parse_positional({"fzn"});
}

cxxopts::ParseResult FznCliHelper::parseArgs(int argc, char ** argv)
{
    return options_parser.parse(argc, argv);
}

std::string FznCliHelper::getHelp()
{
    return options_parser.help();
}

