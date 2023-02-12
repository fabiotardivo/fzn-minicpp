#include "fzn_search_helper.h"

FznSearchHelper::FznSearchHelper(CPSolver::Ptr solver, FznVariablesHelper & fvh) :
    solver(solver), fvh(fvh)
{}

std::function<Branches(void)> FznSearchHelper::getSearchStrategy(Fzn::Model const & fzn_model)
{
    using namespace std;

    std::vector<std::function<Branches(void)>> search_strategy;
    for (auto const & search_annotation: fzn_model.search_strategy)
    {
        if (holds_alternative<Fzn::basic_search_annotation_t>(search_annotation))
        {
            auto const & basic_search_annotation = get<Fzn::basic_search_annotation_t>(search_annotation);
            auto basic_search_strategy = makeBasicSearchStrategy(basic_search_annotation);
            search_strategy.emplace_back(std::move(basic_search_strategy));
        }
        else if (holds_alternative<Fzn::array_search_annotation_t>(search_annotation))
        {
            auto const & array_search_annotation = get<Fzn::array_search_annotation_t>(search_annotation);
            auto const & pred_identifier = get<0>(array_search_annotation);
            auto const & basic_search_annotations = get<1>(array_search_annotation);
            if (pred_identifier == "seq_search")
            {
                for (auto const & basic_search_annotation : basic_search_annotations)
                {
                    auto basic_search_strategy = makeBasicSearchStrategy(basic_search_annotation);
                    search_strategy.emplace_back(std::move(basic_search_strategy));
                }
            }
            else
            {
                throw std::runtime_error("Unsupported search annotation");
            }
        }
        else
        {
            throw std::runtime_error("Unknown search annotation");
        }
    }

    // Default search
    auto const & bool_var_sel = makeVariableSelection<vector<var<bool>::Ptr>, var<bool>::Ptr>("first_fail");
    auto const & bool_val_sel = makeValueSelection<var<bool>::Ptr>("indomain_min");
    auto bool_search_strategy = [=](){return bool_val_sel(solver, bool_var_sel(fvh.getAllBoolVars()));};
    search_strategy.emplace_back(std::move(bool_search_strategy));
    auto const & int_var_sel = makeVariableSelection<vector<var<int>::Ptr>, var<int>::Ptr>("first_fail");
    auto const & int_val_sel = makeValueSelection<var<int>::Ptr>("indomain_min");
    auto int_search_strategy = [=]() {return int_val_sel(solver, int_var_sel(fvh.getAllIntVars()));};
    search_strategy.emplace_back(std::move(int_search_strategy));

    return land(search_strategy);
}

Limit FznSearchHelper::makeSearchLimits(Fzn::Model const & fzn_model, cxxopts::ParseResult const & args)
{
    auto const max_solutions = FznSearchHelper::getMaxSolutions(fzn_model, args);
    auto const max_time_ms = FznSearchHelper::getMaxSearchTime(args);

    return [=](SearchStatistics const & search_statistics)
    {
        return search_statistics.getSolutions() >= max_solutions or search_statistics.getRunningTime() * 1000 >= max_time_ms;
    };
}

std::function<Branches(void)> FznSearchHelper::makeBasicSearchStrategy(Fzn::basic_search_annotation_t const & basic_search_annotation)
{
    using namespace std;

    auto const & pred_identifier = get<0>(basic_search_annotation);
    auto const & var_expr = get<1>(basic_search_annotation);
    auto const & annotations = get<2>(basic_search_annotation);

    if (pred_identifier == "int_search")
    {
        using int_var_t = var<int>::Ptr;
        using array_int_var_t = vector<int_var_t>;

        auto const & var_sel = makeVariableSelection<array_int_var_t, int_var_t>(annotations.at(0).first);
        auto const & val_sel = makeValueSelection<int_var_t>(annotations.at(1).first);

        // Decision variables
        array_int_var_t array_int_var;
        if (holds_alternative<Fzn::basic_var_expr_t>(var_expr))
        {
            Fzn::constraint_arg_t const & array_int_vars_id{get<Fzn::basic_var_expr_t>(var_expr)};
            array_int_var = fvh.getArrayIntVars(array_int_vars_id);
        }
        else if (holds_alternative<vector<Fzn::basic_var_expr_t>>(var_expr))
        {
            Fzn::constraint_arg_t const & array_int_vars_ids{get<vector<Fzn::basic_var_expr_t>>(var_expr)};
            array_int_var = fvh.getArrayIntVars(array_int_vars_ids);
        }
        else
        {
            throw runtime_error("Unrecognized search variables");
        }
        return [=](){return val_sel(solver, var_sel(array_int_var));};
    }
    else if (pred_identifier == "bool_search")
    {
        using bool_var_t = var<bool>::Ptr;
        using array_bool_var_t = vector<bool_var_t>;

        auto const & var_sel = makeVariableSelection<array_bool_var_t, bool_var_t>(annotations.at(0).first);
        auto const & val_sel = makeValueSelection<bool_var_t>(annotations.at(1).first);

        // Decision variables
        array_bool_var_t array_bool_var;
        if (holds_alternative<Fzn::basic_var_expr_t>(var_expr))
        {
            Fzn::constraint_arg_t const & array_bool_vars_id{get<Fzn::basic_var_expr_t>(var_expr)};
            array_bool_var = fvh.getArrayBoolVars(array_bool_vars_id);
        }
        else if (holds_alternative<vector<Fzn::basic_var_expr_t>>(var_expr))
        {
            Fzn::constraint_arg_t const & array_bool_vars_id{get<vector<Fzn::basic_var_expr_t>>(var_expr)};
            array_bool_var = fvh.getArrayBoolVars(array_bool_vars_id);
        }
        else
        {
            throw runtime_error("Unrecognized search variables");
        }
        return [=](){return val_sel(solver, var_sel(array_bool_var));};
    }
    else
    {
        stringstream msg;
        msg << "Unsupported search annotation selection : " << pred_identifier;
        throw runtime_error(msg.str());
    }
}

unsigned int
FznSearchHelper::getMaxSolutions(Fzn::Model const & fzn_model, cxxopts::ParseResult const & args)
{
    if (fzn_model.solve_type == "satisfy")
    {
        if (args["a"].count() != 0)
        {
            return UINT_MAX;
        }
        else if (args["n"].count() != 0)
        {
            return args["n"].as<unsigned int>();
        }
        else
        {
            return 1;
        }
    }
    else
    {
        return UINT_MAX;
    }
}

unsigned int
FznSearchHelper::getMaxSearchTime( cxxopts::ParseResult const & args)
{
    if (args["t"].count() == 0)
    {
        return UINT_MAX;
    }
    else
    {
        return args["t"].as<unsigned int>();
    }
}
