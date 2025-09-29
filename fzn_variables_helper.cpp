#include "fzn_variables_helper.h"


FznVariablesHelper::FznVariablesHelper(CPSolver::Ptr solver, Fzn::Model const & fzn_model) :
        solver(solver),
        fzn_model(fzn_model)
{}

void FznVariablesHelper::makeBoolVariables(std::unordered_map<std::string, Fzn::Var> const & fzn_bool_vars, Fzn::Printer * fzn_printer)
{
    using namespace std;
    using namespace Fzn;

    // Bool variables
    for (auto const & entry : fzn_bool_vars)
    {
        auto const & identifier = entry.first;
        auto const & fzn_bool_var = entry.second;

        var<bool>::Ptr bool_var = Factory::makeBoolVar(solver);
        bool_vars.emplace(identifier, bool_var);

        if (fzn_printer != nullptr)
        {
            for (auto const & annotation : fzn_bool_var.annotations)
            {
                if (annotation.first == "output_var")
                {
                    fzn_printer->add_output<bool>(identifier, [=]() -> int {return bool_var->min();});
                }
            }
        }
    }

    // Arrays of bool variables
    for (auto const & entry : fzn_model.array_bool_vars)
    {
        auto const & identifier = entry.first;
        auto const & array_bool_var = entry.second;
        if (fzn_printer != nullptr)
        {
            for (auto const & annotation : array_bool_var.annotations)
            {
                if (annotation.first == "output_array")
                {
                    auto indices = get<vector<Fzn::int_range_t>>(annotation.second.at(0));
                    vector<function<bool()>> callbacks;
                    for (auto const & bool_var_id : array_bool_var.variables)
                    {
                        auto & bool_var = bool_vars.at(string(bool_var_id));
                        callbacks.emplace_back([=](){return bool_var->min();});
                    }
                    fzn_printer->add_output<bool>(identifier, indices, callbacks);
                }
            }
        }
    }
}

void FznVariablesHelper::makeIntVariables(std::unordered_map<std::string, Fzn::Var> const & fzn_int_vars, Fzn::Printer * fzn_printer)
{
    using namespace std;
    using namespace Fzn;

    // Integer variables
    for (auto const & entry : fzn_int_vars)
    {
        auto const & identifier = entry.first;
        auto const & fzn_int_var = entry.second;

        var<int>::Ptr int_var;
        if (std::holds_alternative<int_range_t>(fzn_int_var.domain))
        {
            auto const & int_range = std::get<int_range_t>(fzn_int_var.domain);
            int_var = Factory::makeIntVar(solver, int_range.first, int_range.second);
        }
        else if (std::holds_alternative<int_set_t>(fzn_int_var.domain))
        {
            auto const & int_set = any_cast<int_set_t>(fzn_int_var.domain);
            int_var = Factory::makeIntVar(solver, int_set);
        }
        else
        {
            throw runtime_error("Unexpected integer variable type");
        }
        int_vars.emplace(identifier, int_var);

        if (fzn_printer != nullptr)
        {
            for (auto const & annotation : fzn_int_var.annotations)
            {
                if (annotation.first == "output_var")
                {
                    fzn_printer->add_output<int>(identifier, [=]() -> int {return int_var->min();});
                }
            }
        }
    }

    // Arrays of integer variables
    for (auto const & entry : fzn_model.array_int_vars)
    {
        auto const & identifier = entry.first;
        auto const & array_int_var = entry.second;

        if (fzn_printer != nullptr)
        {
            for (auto const & annotation : array_int_var.annotations)
            {
                if (annotation.first == "output_array")
                {
                    auto indices = get<vector<Fzn::int_range_t>>(annotation.second.at(0));
                    vector<function<int()>> callbacks;
                    for (auto const & int_var_id : array_int_var.variables)
                    {
                        auto & int_var = int_vars.at(string(int_var_id));
                        callbacks.emplace_back([=](){return int_var->min();});
                    }
                    fzn_printer->add_output<int>(identifier, indices, callbacks);
                }
            }
        }
    }
}

bool FznVariablesHelper::getBool(const Fzn::constraint_arg_t & arg)
{
    using namespace std;
    if (std::holds_alternative<bool>(arg))
    {
        return get<bool>(arg);
    }
    else
    {
        throw runtime_error("Expected boolean value");
    }
}

int FznVariablesHelper::getInt(const Fzn::constraint_arg_t & arg)
{
    using namespace std;
    if (std::holds_alternative<int>(arg))
    {
        return get<int>(arg);
    }
    else
    {
        throw runtime_error("Expected boolean value");
    }
}

var<bool>::Ptr FznVariablesHelper::getBoolVar(Fzn::constraint_arg_t const & fzn_arg)
{
    using namespace std;

    if (holds_alternative<Fzn::identifier_t>(fzn_arg))
    {
        string const bool_var_id(get<Fzn::identifier_t>(fzn_arg));
        return bool_vars.at(bool_var_id);
    }
    else if (holds_alternative<bool>(fzn_arg))
    {
        bool const bool_val = get<bool>(fzn_arg);
        string const bool_var_id(bool_val ? "true" : "false");
        if (bool_vars.count(bool_var_id) == 0)
        {
            bool_vars.emplace(bool_var_id, Factory::makeBoolVar(solver, bool_val));
        }
        return bool_vars.at(bool_var_id);
    }
    else
    {
        throw runtime_error("Expected a boolean variable");
    }
}

var<int>::Ptr FznVariablesHelper::getIntVar(Fzn::constraint_arg_t const & fzn_arg)
{
    using namespace std;

    if (holds_alternative<Fzn::identifier_t>(fzn_arg))
    {
        string const int_var_id(get<Fzn::identifier_t>(fzn_arg));
        return int_vars.at(int_var_id);
    }
    else if (holds_alternative<int>(fzn_arg))
    {
        int const int_val = get<int>(fzn_arg);
        string const int_val_id = to_string(int_val);
        if (int_vars.count(int_val_id) == 0)
        {
            int_vars.emplace(int_val_id, Factory::makeIntVar(solver, int_val, int_val));
        }
        return int_vars.at(int_val_id);
    }
    else
    {
        throw runtime_error("Expected an integer variable");
    }
}

var<int>::Ptr FznVariablesHelper::getObjectiveVar() const
{
    using namespace std;

    if (fzn_model.objective_var != "")
    {
        return int_vars.at(string(fzn_model.objective_var));
    }
    else
    {
        throw runtime_error("No objective variable");
    }
}

std::vector<bool> FznVariablesHelper::getArrayBool(const Fzn::constraint_arg_t & fzn_arg)
{
    using namespace std;
    if(std::holds_alternative<Fzn::identifier_t>(fzn_arg))
    {
        string const array_bool_id(get<Fzn::identifier_t>(fzn_arg));
        return fzn_model.array_bool_consts.at(array_bool_id);
    }
    else if (std::holds_alternative<vector<bool>>(fzn_arg))
    {
        return get<vector<bool>>(fzn_arg);
    }
    else
    {
        throw runtime_error("Expected array of boolean value");
    }
}

std::vector<int> FznVariablesHelper::getArrayInt(const Fzn::constraint_arg_t & fzn_arg)
{
    using namespace std;
    if(std::holds_alternative<Fzn::identifier_t>(fzn_arg))
    {
        string const array_int_id(get<Fzn::identifier_t>(fzn_arg));
        return fzn_model.array_int_consts.at(array_int_id);
    }
    else if (std::holds_alternative<vector<int>>(fzn_arg))
    {
        return get<vector<int>>(fzn_arg);
    }
    else
    {
        throw runtime_error("Expected array of integer values");
    }
}

std::vector<var<bool>::Ptr> FznVariablesHelper::getArrayBoolVars(Fzn::constraint_arg_t const & fzn_arg)
{
    using namespace std;

    if (holds_alternative<std::monostate>(fzn_arg))
    {
        return std::vector<var<bool>::Ptr>();
    }
    else if (holds_alternative<Fzn::identifier_t>(fzn_arg))
    {
        string const array_bool_vars_id(get<Fzn::identifier_t>(fzn_arg));
        auto const & array_bool_var = fzn_model.array_bool_vars.at(array_bool_vars_id).variables;
        return getArrayBoolVars(array_bool_var);
    }
    else if (holds_alternative<vector<Fzn::identifier_t>>(fzn_arg))
    {
        auto const & array_bool_var = get<vector<Fzn::identifier_t>>(fzn_arg);
        return getArrayBoolVars(array_bool_var);
    }
    else
    {
        throw runtime_error("Expected an array of boolean variables");
    }
}

std::vector<var<int>::Ptr> FznVariablesHelper::getArrayIntVars(Fzn::constraint_arg_t const & fzn_arg)
{
    using namespace std;

    if (holds_alternative<std::monostate>(fzn_arg))
    {
        return std::vector<var<int>::Ptr>();
    }
    else if (holds_alternative<Fzn::identifier_t>(fzn_arg))
    {
        string const array_int_vars_id(get<Fzn::identifier_t>(fzn_arg));
        auto const & array_int_var = fzn_model.array_int_vars.at(array_int_vars_id).variables;
        return getArrayIntVars(array_int_var);
    }
    else if (holds_alternative<vector<Fzn::identifier_t>>(fzn_arg))
    {
        auto const & array_int_var = get<vector<Fzn::identifier_t>>(fzn_arg);
        return getArrayIntVars(array_int_var);
    }
    else
    {
        throw runtime_error("Expected an array of integer variables");
    }
}

std::vector<var<bool>::Ptr> FznVariablesHelper::getAllBoolVars() const
{
    std::vector<var<bool>::Ptr> all_bool_vars(bool_vars.size());
    transform(bool_vars.begin(), bool_vars.end(), all_bool_vars.begin(), [](auto const & id_var) { return id_var.second;});
    return all_bool_vars;
}

std::vector<var<int>::Ptr> FznVariablesHelper::getAllIntVars() const
{
    std::vector<var<int>::Ptr> all_int_vars(int_vars.size());
    transform(int_vars.begin(), int_vars.end(), all_int_vars.begin(), [](auto const & id_var) { return id_var.second;});
    return all_int_vars;
}

std::vector<var<bool>::Ptr> FznVariablesHelper::getArrayBoolVars(std::vector<Fzn::identifier_t> const & fzn_array_bool_vars_ids)
{
    using namespace std;
    auto const array_size = fzn_array_bool_vars_ids.size();
    std::vector<var<bool>::Ptr> array_bool_vars(array_size);
    for (auto i = 0; i < array_size; i += 1)
    {
        Fzn::constraint_arg_t const bool_var_id{fzn_array_bool_vars_ids.at(i)};
        array_bool_vars.at(i) = getBoolVar(bool_var_id);
    }
    return array_bool_vars;
}

std::vector<var<int>::Ptr> FznVariablesHelper::getArrayIntVars(std::vector<Fzn::identifier_t> const & fzn_array_int_vars_ids)
{
    using namespace std;
    auto const array_size = fzn_array_int_vars_ids.size();
    std::vector<var<int>::Ptr> array_int_vars(array_size);
    for (auto i = 0; i < array_size; i += 1)
    {
        Fzn::constraint_arg_t const bool_var_id{fzn_array_int_vars_ids.at(i)};
        array_int_vars.at(i) = getIntVar(bool_var_id);
    }
    return array_int_vars;
}