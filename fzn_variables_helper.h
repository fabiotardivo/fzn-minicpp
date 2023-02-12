#pragma once

#include <string>
#include <unordered_map>

#include <libfzn/Model.h>
#include <libfzn/Printer.h>
#include <libminicpp/intvar.hpp>

class FznVariablesHelper
{
    private:
        CPSolver::Ptr solver;
        Fzn::Model const & fzn_model;
        std::unordered_map<std::string, var<int>::Ptr> int_vars;
        std::unordered_map<std::string, var<bool>::Ptr> bool_vars;

    public:
        FznVariablesHelper(CPSolver::Ptr solver, Fzn::Model const & fzn_model);
        void makeBoolVariables(std::unordered_map<std::string, Fzn::Var> const & fzn_bool_vars, Fzn::Printer & fzn_printer);
        void makeIntVariables(std::unordered_map<std::string, Fzn::Var> const & fzn_int_vars, Fzn::Printer & fzn_printer);
        static bool getBool(Fzn::constraint_arg_t const & fzn_arg);
        static int getInt(Fzn::constraint_arg_t const & fzn_arg);
        var<bool>::Ptr getBoolVar(Fzn::constraint_arg_t const & fzn_arg);
        var<int>::Ptr getIntVar(Fzn::constraint_arg_t const & fzn_arg);
        var<int>::Ptr getObjectiveVar() const;
        std::vector<bool> getArrayBool(Fzn::constraint_arg_t const & fzn_arg);
        std::vector<int> getArrayInt(Fzn::constraint_arg_t const & fzn_arg);
        std::vector<var<bool>::Ptr> getArrayBoolVars(Fzn::constraint_arg_t const & fzn_arg);
        std::vector<var<int>::Ptr> getArrayIntVars(Fzn::constraint_arg_t const & fzn_arg);
        std::vector<var<bool>::Ptr> getAllBoolVars() const;
        std::vector<var<int>::Ptr> getAllIntVars() const;
    private:
        std::vector<var<bool>::Ptr> getArrayBoolVars(std::vector<Fzn::identifier_t> const & fzn_array_bool_vars_ids);
        std::vector<var<int>::Ptr> getArrayIntVars(std::vector<Fzn::identifier_t> const & fzn_array_int_vars_ids);
};
