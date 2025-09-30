#include <pybind11/stl.h>

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

std::function<Branches(void)> FznSearchHelper::getSearchStrategy(Fzn::Model const & fzn_model, pybind11::object const & ml_eval_fun)
{
    using namespace std;

    std::vector<std::function<Branches(void)>> search_strategy;
    if (fzn_model.search_strategy.size() == 1)
    {
        auto const search_annotation = fzn_model.search_strategy[0];
        if (holds_alternative<Fzn::basic_search_annotation_t>(search_annotation))
        {
            auto const & basic_search_annotation = get<Fzn::basic_search_annotation_t>(search_annotation);
            return makeBasicSearchStrategy(basic_search_annotation, ml_eval_fun);
        }
        else
        {
            throw std::runtime_error("Unknown search annotation");
        }
    }
    else
    {
        throw std::runtime_error("Unsupported search annotation");
    }
}

std::function<Branches(void)> FznSearchHelper::getSampleStrategy(Fzn::Model const & fzn_model)
{
    using namespace std;

    std::vector<std::function<Branches(void)>> search_strategy;
    if (fzn_model.search_strategy.size() == 1)
    {
        auto const search_annotation = fzn_model.search_strategy[0];
        if (holds_alternative<Fzn::basic_search_annotation_t>(search_annotation))
        {
            auto const & basic_search_annotation = get<Fzn::basic_search_annotation_t>(search_annotation);
            return makeBasicSampleStrategy(basic_search_annotation);
        }
        else
        {
            throw std::runtime_error("Unknown search annotation");
        }
    }
    else
    {
        throw std::runtime_error("Unsupported search annotation");
    }
}

std::vector<var<int>::Ptr> FznSearchHelper::getIntDecisionalVars(Fzn::Model const & fzn_model)
{
    using namespace std;

    if (fzn_model.search_strategy.size() == 1)
    {
        auto const & search_annotation =  fzn_model.search_strategy[0];
        if (holds_alternative<Fzn::basic_search_annotation_t>(search_annotation))
        {

            auto const & basic_search_annotation = get<Fzn::basic_search_annotation_t>(search_annotation);
            auto const & pred_identifier = get<0>(basic_search_annotation);
            auto const & var_expr = get<1>(basic_search_annotation);

            if (pred_identifier == "int_search")
            {
                return getIntDecisionalVars(var_expr);
            }
            else
            {
                return {};
            }
        }
        else
        {
            throw runtime_error("Unsupported search annotation");
        }
    }
    else
    {
        throw runtime_error("Unsupported search annotation");
    }
}

std::vector<var<int>::Ptr> FznSearchHelper::getIntDecisionalVars(Fzn::var_expr_t var_expr)
{
    using namespace std;

    using int_var_t = var<int>::Ptr;
    using array_int_var_t = vector<int_var_t>;

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
        throw std::runtime_error("Unrecognized search variables");
    }
    return array_int_var;
}

std::vector<var<bool>::Ptr> FznSearchHelper::getBoolDecisionalVars(Fzn::var_expr_t var_expr)
{
    using namespace std;

    using bool_var_t = var<bool>::Ptr;
    using array_bool_var_t = vector<bool_var_t>;

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
        throw std::runtime_error("Unrecognized search variables");
    }
    return array_bool_var;
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

        array_int_var_t array_int_var = getIntDecisionalVars(var_expr);
        return [=](){return val_sel(solver, var_sel(array_int_var));};
    }
    else if (pred_identifier == "bool_search")
    {
        using bool_var_t = var<bool>::Ptr;
        using array_bool_var_t = vector<bool_var_t>;

        auto const & var_sel = makeVariableSelection<array_bool_var_t, bool_var_t>(annotations.at(0).first);
        auto const & val_sel = makeValueSelection<bool_var_t>(annotations.at(1).first);

        // Decision variables
        array_bool_var_t array_bool_var = getBoolDecisionalVars(var_expr);
        return [=](){return val_sel(solver, var_sel(array_bool_var));};
    }
    else
    {
        stringstream msg;
        msg << "Unsupported search annotation selection : " << pred_identifier;
        throw runtime_error(msg.str());
    }
}

std::function<Branches(void)> FznSearchHelper::makeBasicSearchStrategy(Fzn::basic_search_annotation_t const & basic_search_annotation, pybind11::object const & ml_eval_fun)
{
    using namespace std;

    auto const & pred_identifier = get<0>(basic_search_annotation);
    auto const & var_expr = get<1>(basic_search_annotation);
    auto const & annotations = get<2>(basic_search_annotation);

    if (pred_identifier == "int_search")
    {
        using int_var_t = var<int>::Ptr;
        using array_int_var_t = vector<int_var_t>;

        // Decision variables
        array_int_var_t array_int_var = getIntDecisionalVars(var_expr);
        auto const nVars = static_cast<int>(array_int_var.size());
        auto const solver = array_int_var[0]->getSolver();

        return [=]()
        {
            std::vector<float> pa;
            for (auto varIdx = 0; varIdx < nVars; varIdx += 1)
            {
                auto const & var = array_int_var[varIdx];
                pa.push_back(var->isBound() ? var->min() : NAN);
            }

            int_var_t bestVar = nullptr;
            auto bestVal = INT_MAX;
            float bestScore = std::numeric_limits<float>::max();
            for (auto varIdx = 0; varIdx < nVars; varIdx += 1)
            {
                auto const &var = array_int_var[varIdx];
                if (not var->isBound())
                {
                    std::vector<float> toEval = pa;
                    auto const minVal = var->min();
                    auto const maxVal = var->max();
                    for (auto val = minVal; val <= maxVal; val += 1)
                    {
                        if (var->contains(val))
                        {
                            toEval[varIdx] = val;
                            auto const score = ml_eval_fun(toEval).cast<float>();
                            if (score < bestScore)
                            {
                                bestScore = score;
                                bestVal = val;
                                bestVar = var;
                            }

                        }
                    }
                }
            }
            return indomain_fixed(solver, bestVar, bestVal);
        };
    }
    else
    {
        stringstream msg;
        msg << "Unsupported search annotation : " << pred_identifier;
        throw runtime_error(msg.str());
    }
}


std::function<Branches(void)> FznSearchHelper::makeBasicSampleStrategy(Fzn::basic_search_annotation_t const & basic_search_annotation)
{
    using namespace std;

    auto const & pred_identifier = get<0>(basic_search_annotation);
    auto const & var_expr = get<1>(basic_search_annotation);
    auto const & annotations = get<2>(basic_search_annotation);

    if (pred_identifier == "int_search")
    {
        using int_var_t = var<int>::Ptr;
        using array_int_var_t = vector<int_var_t>;

        std::string const var_sel_name = "random";
        std::string const val_sel_name = "indomain_random";
        auto const & var_sel = makeVariableSelection<array_int_var_t, int_var_t>(var_sel_name);
        auto const & val_sel = makeValueSelection<int_var_t>(val_sel_name);

        // Decision variables
        array_int_var_t array_int_var = getIntDecisionalVars(var_expr);
        return [=](){return val_sel(solver, var_sel(array_int_var));};
    }
    else
    {
        stringstream msg;
        msg << "Unsupported search annotation : " << pred_identifier;
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