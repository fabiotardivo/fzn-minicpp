#include <libminicpp/constraint.hpp>
#include <libminicpp/table.hpp>
#include "fzn_constraints_helper.h"
#include "fzn_constraints/bool_array.hpp"
#include "fzn_constraints/bool_bin.hpp"
#include "fzn_constraints/bool_misc.hpp"
#include "fzn_constraints/int_array.hpp"
#include "fzn_constraints/int_bin.hpp"
#include "fzn_constraints/int_tern.hpp"
#include "fzn_constraints/int_lin.hpp"
#include "fzn_constraints/int_misc.hpp"
#include "global_constraints/cumulative.hpp"
#include "global_constraints/table.hpp"
#include "global_constraints/stable_matching.hpp"
#include "gpu_constriants/cumulative.cuh"

using backward_implication_t = std::function<void()>;

FznConstraintHelper::FznConstraintHelper(CPSolver::Ptr solver, FznVariablesHelper & fvh) :
        solver(std::move(solver)), fvh(fvh)
{
    addIntConstraintsBuilders();
    addBoolConstraintsBuilders();
    addGlobalConstraintsBuilders();
}

bool FznConstraintHelper::makeConstraints(Fzn::Model const & fzn_model)
{
    using namespace std;

    for (auto const & fzn_constraint : fzn_model.constraints)
    {
        string fzn_constraint_id(fzn_constraint.identifier);
        if (constriants_builders.count(fzn_constraint_id) == 1)
        {
            TRYFAIL
                Constraint::Ptr constraint = constriants_builders.at(fzn_constraint_id)(fzn_constraint.arguments, fzn_constraint.annotations);
                solver->post(constraint);
            ONFAIL
                return false;
            ENDFAIL
        }
        else
        {
            stringstream msg;
            msg << "Constraint not supported : " << fzn_constraint_id;
            throw runtime_error(msg.str());
        }        
    }
    return true;
}

void FznConstraintHelper::addIntConstraintsBuilders()
{
    using namespace std;

    // Array
    constriants_builders.emplace("array_int_element", [&](vector<Fzn::constraint_arg_t> const & args, vector<Fzn::annotation_t> const & anns) -> Constraint::Ptr {
        auto const & b = fvh.getIntVar(args.at(0));
        auto const & as = fvh.getArrayInt(args.at(1));
        auto const & c = fvh.getIntVar(args.at(2));
        auto const & _b = new (solver) IntVarViewOffset(b, -1);
        return new (solver) Element1DBasic(as, _b, c);
    });

    constriants_builders.emplace("array_var_int_element", [&](vector<Fzn::constraint_arg_t> const & args, vector<Fzn::annotation_t> const & anns) -> Constraint::Ptr {
        auto const & b = fvh.getIntVar(args.at(0));
        auto const & as = fvh.getArrayIntVars(args.at(1));
        auto const & c = fvh.getIntVar(args.at(2));
        auto const & _b = new (solver) IntVarViewOffset(b, -1);
        return new (solver) Element1DVar(as, _b, c);
    });

    constriants_builders.emplace("array_int_maximum", [&](vector<Fzn::constraint_arg_t> const & args, vector<Fzn::annotation_t> const & anns) -> Constraint::Ptr {
        auto const & m = fvh.getIntVar(args.at(0));
        auto const & x = fvh.getArrayIntVars(args.at(1));
        return new (solver) array_int_maximum(m, x);
    });

    constriants_builders.emplace("array_int_minimum", [&](vector<Fzn::constraint_arg_t> const & args, vector<Fzn::annotation_t> const & anns) -> Constraint::Ptr {
        auto const & m = fvh.getIntVar(args.at(0));
        auto const & x = fvh.getArrayIntVars(args.at(1));
        return new (solver) array_int_minimum(m, x);
    });

    // Binary
    constriants_builders.emplace("int_eq", [&](vector<Fzn::constraint_arg_t> const & args, vector<Fzn::annotation_t> const & anns) -> Constraint::Ptr {
        auto const & a = fvh.getIntVar(args.at(0));
        auto const & b = fvh.getIntVar(args.at(1));
        return  new (solver) EQBinBC(a,b,0);
    });

    constriants_builders.emplace("int_ne", [&](vector<Fzn::constraint_arg_t> const & args, vector<Fzn::annotation_t> const & anns) -> Constraint::Ptr {
        auto const & a = fvh.getIntVar(args.at(0));
        auto const & b = fvh.getIntVar(args.at(1));
        return new (solver) NEQBinBC(a,b,0);
    });

    constriants_builders.emplace("int_le", [&](vector<Fzn::constraint_arg_t> const & args, vector<Fzn::annotation_t> const & anns) -> Constraint::Ptr {
        auto const & a = fvh.getIntVar(args.at(0));
        auto const & b = fvh.getIntVar(args.at(1));
        return new (solver) LessOrEqual(a,b);
    });

    constriants_builders.emplace("int_lt", [&](vector<Fzn::constraint_arg_t> const & args, vector<Fzn::annotation_t> const & anns) -> Constraint::Ptr {
        auto const & a = fvh.getIntVar(args.at(0));
        auto const & b = fvh.getIntVar(args.at(1));
        auto const & _b = new (solver) IntVarViewOffset(b, -1);
        return new (solver) LessOrEqual(a,_b);
    });

    constriants_builders.emplace("int_abs", [&](vector<Fzn::constraint_arg_t> const & args, vector<Fzn::annotation_t> const & anns) -> Constraint::Ptr {
        auto const & a = fvh.getIntVar(args.at(0));
        auto const & b = fvh.getIntVar(args.at(1));
        return new (solver) int_abs(a, b);
    });

    // Ternary
    constriants_builders.emplace("int_div", [&](vector<Fzn::constraint_arg_t> const & args, vector<Fzn::annotation_t> const & anns) -> Constraint::Ptr {
        auto const & a = fvh.getIntVar(args.at(0));
        auto const & b = fvh.getIntVar(args.at(1));
        auto const & c = fvh.getIntVar(args.at(2));
        return new (solver) int_div(a, b, c);
     });

    constriants_builders.emplace("int_max", [&](vector<Fzn::constraint_arg_t> const & args, vector<Fzn::annotation_t> const & anns) -> Constraint::Ptr {
        auto const & a = fvh.getIntVar(args.at(0));
        auto const & b = fvh.getIntVar(args.at(1));
        auto const & c = fvh.getIntVar(args.at(2));
        return new (solver) int_max(a, b, c);
    });

    constriants_builders.emplace("int_min", [&](vector<Fzn::constraint_arg_t> const & args, vector<Fzn::annotation_t> const & anns) -> Constraint::Ptr {
        auto const & a = fvh.getIntVar(args.at(0));
        auto const & b = fvh.getIntVar(args.at(1));
        auto const & c = fvh.getIntVar(args.at(2));
        return new (solver) int_min(a, b, c);
    });

    constriants_builders.emplace("int_mod", [&](vector<Fzn::constraint_arg_t> const & args, vector<Fzn::annotation_t> const & anns) -> Constraint::Ptr {
        auto const & a = fvh.getIntVar(args.at(0));
        auto const & b = fvh.getIntVar(args.at(1));
        auto const & c = fvh.getIntVar(args.at(2));
        return new (solver) int_mod(a, b, c);
    });

    constriants_builders.emplace("int_plus", [&](vector<Fzn::constraint_arg_t> const & args, vector<Fzn::annotation_t> const & anns) -> Constraint::Ptr {
        auto const & a = fvh.getIntVar(args.at(0));
        auto const & b = fvh.getIntVar(args.at(1));
        auto const & c = fvh.getIntVar(args.at(2));
        return new (solver) EQTernBC(c,a,b);
    });

    constriants_builders.emplace("int_pow", [&](vector<Fzn::constraint_arg_t> const & args, vector<Fzn::annotation_t> const & anns) -> Constraint::Ptr {
        auto const & a = fvh.getIntVar(args.at(0));
        auto const & b = fvh.getIntVar(args.at(1));
        auto const & c = fvh.getIntVar(args.at(2));
        return new (solver) int_pow(a,b,c);
    });

    constriants_builders.emplace("int_times", [&](vector<Fzn::constraint_arg_t> const & args, vector<Fzn::annotation_t> const & anns) -> Constraint::Ptr {
        auto const & a = fvh.getIntVar(args.at(0));
        auto const & b = fvh.getIntVar(args.at(1));
        auto const & c = fvh.getIntVar(args.at(2));
        return new (solver) int_times(a,b,c);
    });

    // Misc
    constriants_builders.emplace("set_in", [&](vector<Fzn::constraint_arg_t> const & args, vector<Fzn::annotation_t> const & anns) -> Constraint::Ptr {
        auto const & x = fvh.getIntVar(args.at(0));
        auto const & s = fvh.getArrayInt(args.at(1));
        return new (solver) set_in(x, s);
    });

    // Linear
    constriants_builders.emplace("int_lin_eq", [&](vector<Fzn::constraint_arg_t> const & args, vector<Fzn::annotation_t> const & anns) -> Constraint::Ptr {
        auto const & as = fvh.getArrayInt(args.at(0));
        auto const & bs = fvh.getArrayIntVars(args.at(1));
        auto const & c = FznVariablesHelper::getInt(args.at(2));
        auto const & sum = makeLinSum(as, bs);
        return new (solver) EQc(sum, c);
    });

    constriants_builders.emplace("int_lin_ne", [&](vector<Fzn::constraint_arg_t> const & args, vector<Fzn::annotation_t> const & anns) -> Constraint::Ptr {
        auto const & as = fvh.getArrayInt(args.at(0));
        auto const & bs = fvh.getArrayIntVars(args.at(1));
        auto const & c = FznVariablesHelper::getInt(args.at(2));
        auto const & sum = makeLinSum(as, bs);
        return new (solver) NEQc(sum, c);
    });

    constriants_builders.emplace("int_lin_le", [&](vector<Fzn::constraint_arg_t> const & args, vector<Fzn::annotation_t> const & anns) -> Constraint::Ptr {
        auto const & as = fvh.getArrayInt(args.at(0));
        auto const & bs = fvh.getArrayIntVars(args.at(1));
        auto const & c = FznVariablesHelper::getInt(args.at(2));
        auto const & sum = makeLinSum(as, bs);
        return new (solver) LEQc(sum, c);
    });

    // Reification
    constriants_builders.emplace("int_eq_reif", [&](vector<Fzn::constraint_arg_t> const & args, vector<Fzn::annotation_t> const & anns) -> Constraint::Ptr {
        auto const & a = fvh.getIntVar(args.at(0));
        auto const & b = fvh.getIntVar(args.at(1));
        auto const & r = fvh.getBoolVar(args.at(2));
        auto const & eq = new (solver) EQBinBC(a,b,0); // a = b
        auto const & neq = new (solver) NEQBinBC(a,b,0); // a != b
        backward_implication_t bi = [=]() {
            if (a->isBound() and b->isBound() and a->min() == b->min())
            {
                r->assign(true);
            }
            else if (a->max() < b->min() or b->max() < a->min())
            {
                r->assign(false);
            }
        };
        return new (solver) int_bin_reif(a,b,r,eq,neq,bi);
     });

    constriants_builders.emplace("int_ne_reif", [&](vector<Fzn::constraint_arg_t> const & args, vector<Fzn::annotation_t> const & anns) -> Constraint::Ptr {
        auto const & a = fvh.getIntVar(args.at(0));
        auto const & b = fvh.getIntVar(args.at(1));
        auto const & r = fvh.getBoolVar(args.at(2));
        auto const & neq = new (solver) NEQBinBC(a,b,0); // a != b
        auto const & eq = new (solver) EQBinBC(a,b,0); // a = b
        backward_implication_t bi = [=]() {
            if (a->max() < b->min() or b->max() < a->min())
            {
                r->assign(true);
            }
            else if (a->isBound() and b->isBound() and a->min() == b->min())
            {
                r->assign(false);
            }
        };
        return new (solver) int_bin_reif(a,b,r,neq,eq,bi);
    });

    constriants_builders.emplace("int_le_reif", [&](vector<Fzn::constraint_arg_t> const & args, vector<Fzn::annotation_t> const & anns) -> Constraint::Ptr {
        auto const & a = fvh.getIntVar(args.at(0));
        auto const & b = fvh.getIntVar(args.at(1));
        auto const & r = fvh.getBoolVar(args.at(2));
        var<int>::Ptr const & _b = new (solver) IntVarViewOffset(b, 1);
        auto const & le = new (solver) LessOrEqual(a,b); // a <= b
        auto const & gt = new (solver) LessOrEqual(_b,a); // a > b
        backward_implication_t bi = [=]() {
            if (a->max() <= b->min())
            {
                r->assign(true);
            }
            else if (a->min() > b->max())
            {
                r->assign(false);
            }
        };
        return new (solver) int_bin_reif(a,b,r,le,gt,bi);
     });

    constriants_builders.emplace("int_lt_reif", [&](vector<Fzn::constraint_arg_t> const & args, vector<Fzn::annotation_t> const & anns) -> Constraint::Ptr {
        auto const & a = fvh.getIntVar(args.at(0));
        auto const & b = fvh.getIntVar(args.at(1));
        auto const & r = fvh.getBoolVar(args.at(2));
        var<int>::Ptr const & _b = new (solver) IntVarViewOffset(b, -1);
        auto const & lt = new (solver) LessOrEqual(a,_b); // a < b
        auto const & ge = new (solver) LessOrEqual(b,a); // a >= b
        backward_implication_t bi = [=]() {
            if (a->max() < b->min())
            {
                r->assign(true);
            }
            else if (a->min() >= b->max())
            {
                r->assign(false);
            }
        };
        return new (solver) int_bin_reif(a,b,r,lt,ge,bi);
    });

    constriants_builders.emplace("int_lin_eq_reif", [&](vector<Fzn::constraint_arg_t> const & args, vector<Fzn::annotation_t> const & anns) -> Constraint::Ptr {
        auto const & as = fvh.getArrayInt(args.at(0));
        auto const & bs = fvh.getArrayIntVars(args.at(1));
        auto const & c = FznVariablesHelper::getInt(args.at(2));
        auto const & r = fvh.getBoolVar(args.at(3));
        auto const & sum = makeLinSum(as, bs);
        auto const & lin_eq =  new (solver) EQc(sum, c); // as[0] * bs[0] + ... + as[n] * bs[n] = c
        auto const & lin_neq =  new (solver) NEQc(sum, c); // as[0] * bs[0] + ... + as[n] * bs[n] != c
        backward_implication_t bi = [=]() {
            if (sum->isBound() and sum->min() == c)
            {
                r->assign(true);
            }
            else if (sum->max() < c or c < sum->min())
            {
                r->assign(false);
            }
        };
        return new (solver) int_lin_reif(bs,r,lin_eq,lin_neq,bi);
    });

    constriants_builders.emplace("int_lin_ne_reif", [&](vector<Fzn::constraint_arg_t> const & args, vector<Fzn::annotation_t> const & anns) -> Constraint::Ptr {
        auto const & as = fvh.getArrayInt(args.at(0));
        auto const & bs = fvh.getArrayIntVars(args.at(1));
        auto const & c = FznVariablesHelper::getInt(args.at(2));
        auto const & r = fvh.getBoolVar(args.at(3));
        auto const & sum = makeLinSum(as, bs);
        auto const & lin_neq = new (solver) NEQc(sum, c); // as[0] * bs[0] + ... + as[n] * bs[n] != c
        auto const & lin_eq = new (solver) EQc(sum, c); // as[0] * bs[0] + ... + as[n] * bs[n] == c
        backward_implication_t bi = [=]() {
            if (sum->max() < c or c < sum->min())
            {
                r->assign(true);
            }
            else if (sum->isBound() and sum->min() == c)
            {
                r->assign(false);
            }
        };
        return new (solver) int_lin_reif(bs, r, lin_neq, lin_eq, bi);
    });

    constriants_builders.emplace("int_lin_le_reif", [&](vector<Fzn::constraint_arg_t> const & args, vector<Fzn::annotation_t> const & anns) -> Constraint::Ptr {
        auto const & as = fvh.getArrayInt(args.at(0));
        auto const & bs = fvh.getArrayIntVars(args.at(1));
        auto const & c = FznVariablesHelper::getInt(args.at(2));
        auto const & r = fvh.getBoolVar(args.at(3));
        auto const & sum = makeLinSum(as, bs);
        auto const & _sum = new (solver) IntVarViewOpposite(sum);
        auto const & _c = -(c + 1);
        auto const & lin_le = new (solver) LEQc(sum, c); // as[0] * bs[0] + ... + as[n] * bs[n] <= c
        auto const & lin_gt = new (solver) LEQc(_sum, _c); // as[0] * bs[0] + ... + as[n] * bs[n] > c
        backward_implication_t bi = [=]() {
            if (sum->max() <= c)
            {
                r->assign(true);
            }
            else if (sum->min() > c)
            {
                r->assign(false);
            }
        };
        return new (solver) int_lin_reif(bs, r, lin_le, lin_gt, bi);
    });

    // Implication
    constriants_builders.emplace("int_eq_imp", [&](vector<Fzn::constraint_arg_t> const & args, vector<Fzn::annotation_t> const & anns) -> Constraint::Ptr {
        auto const & a = fvh.getIntVar(args.at(0));
        auto const & b = fvh.getIntVar(args.at(1));
        auto const & r = fvh.getBoolVar(args.at(2));
        auto const & eq = new (solver) EQBinBC(a,b,0); // a = b
        backward_implication_t bi = [=]() {
            if (a->max() < b->min() or b->max() < a->min())
            {
                r->assign(false);
            }
        };
        return new (solver) int_bin_imp(a,b,r,eq,bi);
    });

    constriants_builders.emplace("int_ne_imp", [&](vector<Fzn::constraint_arg_t> const & args, vector<Fzn::annotation_t> const & anns) -> Constraint::Ptr {
        auto const & a = fvh.getIntVar(args.at(0));
        auto const & b = fvh.getIntVar(args.at(1));
        auto const & r = fvh.getBoolVar(args.at(2));
        auto const & neq = new (solver) NEQBinBC(a,b,0); // a != b
        backward_implication_t bi = [=]() {
            if (a->isBound() and b->isBound() and a->min() == b->min())
            {
                r->assign(false);
            }
        };
        return new (solver) int_bin_imp(a,b,r,neq,bi);
    });

    constriants_builders.emplace("int_le_imp", [&](vector<Fzn::constraint_arg_t> const & args, vector<Fzn::annotation_t> const & anns) -> Constraint::Ptr {
        auto const & a = fvh.getIntVar(args.at(0));
        auto const & b = fvh.getIntVar(args.at(1));
        auto const & r = fvh.getBoolVar(args.at(2));
        var<int>::Ptr const & _b = new (solver) IntVarViewOffset(b, 1);
        auto const & le = new (solver) LessOrEqual(a,b); // a <= b
        backward_implication_t bi = [=]() {
            if (a->min() > b->max())
            {
                r->assign(false);
            }
        };
        return new (solver) int_bin_imp(a,b,r,le,bi);
    });

    constriants_builders.emplace("int_lt_imp", [&](vector<Fzn::constraint_arg_t> const & args, vector<Fzn::annotation_t> const & anns) -> Constraint::Ptr {
        auto const & a = fvh.getIntVar(args.at(0));
        auto const & b = fvh.getIntVar(args.at(1));
        auto const & r = fvh.getBoolVar(args.at(2));
        var<int>::Ptr const & _b = new (solver) IntVarViewOffset(b, -1);
        auto const & lt = new (solver) LessOrEqual(a,_b); // a < b
        backward_implication_t bi = [=]() {
            if (a->min() >= b->max())
            {
                r->assign(false);
            }
        };
        return new (solver) int_bin_imp(a,b,r,lt,bi);
    });

    constriants_builders.emplace("int_lin_eq_imp", [&](vector<Fzn::constraint_arg_t> const & args, vector<Fzn::annotation_t> const & anns) -> Constraint::Ptr {
        auto const & as = fvh.getArrayInt(args.at(0));
        auto const & bs = fvh.getArrayIntVars(args.at(1));
        auto const & c = FznVariablesHelper::getInt(args.at(2));
        auto const & r = fvh.getBoolVar(args.at(3));
        auto const & sum = makeLinSum(as, bs);
        auto const & lin_eq =  new (solver) EQc(sum, c); // as[0] * bs[0] + ... + as[n] * bs[n] = c
        backward_implication_t bi = [=]() {
            if (sum->max() < c or c < sum->min())
            {
                r->assign(false);
            }
        };
        return new (solver) int_lin_imp(bs,r,lin_eq,bi);
    });

    constriants_builders.emplace("int_lin_ne_imp", [&](vector<Fzn::constraint_arg_t> const & args, vector<Fzn::annotation_t> const & anns) -> Constraint::Ptr {
        auto const & as = fvh.getArrayInt(args.at(0));
        auto const & bs = fvh.getArrayIntVars(args.at(1));
        auto const & c = FznVariablesHelper::getInt(args.at(2));
        auto const & r = fvh.getBoolVar(args.at(3));
        auto const & sum = makeLinSum(as, bs);
        auto const & lin_neq = new (solver) NEQc(sum, c); // as[0] * bs[0] + ... + as[n] * bs[n] != c
        backward_implication_t bi = [=]() {
            if (sum->isBound() and sum->min() == c)
            {
                r->assign(false);
            }
        };
        return new (solver) int_lin_imp(bs, r, lin_neq, bi);
    });

    constriants_builders.emplace("int_lin_le_imp", [&](vector<Fzn::constraint_arg_t> const & args, vector<Fzn::annotation_t> const & anns) -> Constraint::Ptr {
        auto const & as = fvh.getArrayInt(args.at(0));
        auto const & bs = fvh.getArrayIntVars(args.at(1));
        auto const & c = FznVariablesHelper::getInt(args.at(2));
        auto const & r = fvh.getBoolVar(args.at(3));
        auto const & sum = makeLinSum(as, bs);
        auto const & _sum = new (solver) IntVarViewOpposite(sum);
        auto const & _c = -(c + 1);
        auto const & lin_le = new (solver) LEQc(sum, c); // as[0] * bs[0] + ... + as[n] * bs[n] <= c
        backward_implication_t bi = [=]() {
            if (sum->min() > c)
            {
                r->assign(false);
            }
        };
        return new (solver) int_lin_imp(bs, r, lin_le, bi);
    });

}

void FznConstraintHelper::addBoolConstraintsBuilders()
{
    using namespace std;

    // Array
    constriants_builders.emplace("array_bool_element", [&](vector<Fzn::constraint_arg_t> const & args, std::vector<Fzn::annotation_t> const & anns) -> Constraint::Ptr {
        auto const & b = fvh.getIntVar(args.at(0));
        auto const & as = fvh.getArrayBool(args.at(1));
        auto const & c = fvh.getBoolVar(args.at(2));
        auto const & _b = new (solver) IntVarViewOffset(b, -1);
        vector<int> _as(as.size());
        transform(as.begin(), as.end(), _as.begin(), [&](bool b) -> int {return static_cast<int>(b);});
        return new (solver) Element1DBasic(_as, _b, c);
    });

    constriants_builders.emplace("array_var_bool_element", [&](vector<Fzn::constraint_arg_t> const & args, std::vector<Fzn::annotation_t> const & anns) -> Constraint::Ptr {
        auto const & b = fvh.getIntVar(args.at(0));
        auto const & as = fvh.getArrayBoolVars(args.at(1));
        auto const & c = fvh.getBoolVar(args.at(2));
        auto const & _b = new (solver) IntVarViewOffset(b, -1);
        return new (solver) Element1DVar(as, _b, c);
    });

    constriants_builders.emplace("array_bool_xor", [&](vector<Fzn::constraint_arg_t> const & args, std::vector<Fzn::annotation_t> const & anns) -> Constraint::Ptr {
        auto const & as =  fvh.getArrayBoolVars(args.at(0));
        return new (solver) array_bool_xor(as);
    });

    // Binary
    constriants_builders.emplace("bool_not", [&](vector<Fzn::constraint_arg_t> const & args, std::vector<Fzn::annotation_t> const & anns) -> Constraint::Ptr {
        auto const & a = fvh.getBoolVar(args.at(0));
        auto const & b = fvh.getBoolVar(args.at(1));
        return new (solver) NEQBinBC(a,b,0);
    });

    constriants_builders.emplace("bool_xor", [&](vector<Fzn::constraint_arg_t> const & args, std::vector<Fzn::annotation_t> const & anns) -> Constraint::Ptr {
        auto const & a = fvh.getBoolVar(args.at(0));
        auto const & b = fvh.getBoolVar(args.at(1));
        return new (solver) bool_xor(a,b);
    });

    constriants_builders.emplace("bool_eq", [&](vector<Fzn::constraint_arg_t> const & args, std::vector<Fzn::annotation_t> const & anns) -> Constraint::Ptr {
        auto const & a = fvh.getBoolVar(args.at(0));
        auto const & b = fvh.getBoolVar(args.at(1));
        return new (solver) EQBinBC(a,b,0);
    });

    constriants_builders.emplace("bool_le", [&](vector<Fzn::constraint_arg_t> const & args, std::vector<Fzn::annotation_t> const & anns) -> Constraint::Ptr {
        auto const & a = fvh.getBoolVar(args.at(0));
        auto const & b = fvh.getBoolVar(args.at(1));
        return new (solver) LessOrEqual(a,b);
    });

    constriants_builders.emplace("bool_lt", [&](vector<Fzn::constraint_arg_t> const & args, std::vector<Fzn::annotation_t> const & anns) -> Constraint::Ptr {
        auto const & a = fvh.getBoolVar(args.at(0));
        auto const & b = fvh.getBoolVar(args.at(1));
        auto const & _b = new (solver) IntVarViewOffset(b, -1);
        return new (solver) LessOrEqual(a,_b);
    });

    // Misc
    constriants_builders.emplace("bool2int", [&](vector<Fzn::constraint_arg_t> const & args, std::vector<Fzn::annotation_t> const & anns) -> Constraint::Ptr {
        auto const & a = fvh.getBoolVar(args.at(0));
        auto const & b = fvh.getIntVar(args.at(1));
        return new (solver) EQBinBC(a,b,0);
    });

    constriants_builders.emplace("bool_clause", [&](vector<Fzn::constraint_arg_t> const & args, std::vector<Fzn::annotation_t> const & anns) -> Constraint::Ptr {
        auto const & as = fvh.getArrayBoolVars(args.at(0));
        auto const & bs = fvh.getArrayBoolVars(args.at(1));
        return new (solver) bool_clause(as,bs);
    });

    // Linear
    constriants_builders.emplace("bool_lin_eq", [&](vector<Fzn::constraint_arg_t> const & args, std::vector<Fzn::annotation_t> const & anns) -> Constraint::Ptr {
        auto const & as = fvh.getArrayInt(args.at(0));
        auto const & bs = fvh.getArrayBoolVars(args.at(1));
        auto const & c = FznVariablesHelper::getInt(args.at(2));
        auto const & sum = makeLinSum(as, bs);
        return new (solver) EQc(sum, c);
    });

    constriants_builders.emplace("bool_lin_le", [&](vector<Fzn::constraint_arg_t> const & args, std::vector<Fzn::annotation_t> const & anns) -> Constraint::Ptr {
        auto const & as = fvh.getArrayInt(args.at(0));
        auto const & bs = fvh.getArrayBoolVars(args.at(1));
        auto const & c = FznVariablesHelper::getInt(args.at(2));
        auto const & sum = makeLinSum(as, bs);
        return new (solver) LEQc(sum, c);
    });

    // Reification
    constriants_builders.emplace("array_bool_and_reif", [&](vector<Fzn::constraint_arg_t> const & args, std::vector<Fzn::annotation_t> const & anns) -> Constraint::Ptr {
        auto const & as =  fvh.getArrayBoolVars(args.at(0));
        auto const & r = fvh.getBoolVar(args.at(1));
        auto const & array_and = new (solver) array_bool_and(as);
        auto const & array_nand = new (solver) array_bool_nand(as);
        backward_implication_t bi = [=]() {
            if (all_of(as.begin(), as.end(), [&](var<bool>::Ptr const & b) -> bool {return b->isTrue();}))
            {
                r->assign(true);
            }
            else if (any_of(as.begin(), as.end(), [&](var<bool>::Ptr const & b) -> bool {return b->isFalse();}))
            {
                r->assign(false);
            }
        };
        return new (solver) array_bool_reif(as,r,array_and,array_nand,bi);
    });

    constriants_builders.emplace("array_bool_or_reif", [&](vector<Fzn::constraint_arg_t> const & args, std::vector<Fzn::annotation_t> const & anns) -> Constraint::Ptr {
        auto const & as =  fvh.getArrayBoolVars(args.at(0));
        auto const & r = fvh.getBoolVar(args.at(1));
        auto const & array_or = new (solver) array_bool_or(as);
        auto const & array_nor = new (solver) array_bool_nor(as);
        backward_implication_t bi = [=]() {
            if (any_of(as.begin(), as.end(), [&](var<bool>::Ptr const & b) -> bool {return b->isTrue();}))
            {
                r->assign(true);
            }
            else if (all_of(as.begin(), as.end(), [&](var<bool>::Ptr const & b) -> bool {return b->isFalse();}))
            {
                r->assign(false);
            }
        };
        return new (solver) array_bool_reif(as,r,array_or,array_nor,bi);
    });

    constriants_builders.emplace("bool_and_reif", [&](vector<Fzn::constraint_arg_t> const & args, std::vector<Fzn::annotation_t> const & anns) -> Constraint::Ptr {
        auto const & a = fvh.getBoolVar(args.at(0));
        auto const & b = fvh.getBoolVar(args.at(1));
        auto const & r = fvh.getBoolVar(args.at(2));
        auto const & logic_and = new (solver) bool_and(a,b); // a /\ b
        auto const & logic_nand = new (solver) bool_nand(a,b); // -(a /\ b)
        backward_implication_t bi = [=]() {
            if (a->isTrue() and b->isTrue())
            {
                r->assign(true);
            }
            else if (a->isFalse() or b->isFalse())
            {
                r->assign(false);
            }
        };
        return new (solver) bool_bin_reif(a,b,r,logic_and,logic_nand, bi);
    });

    constriants_builders.emplace("bool_or_reif", [&](vector<Fzn::constraint_arg_t> const & args, std::vector<Fzn::annotation_t> const & anns) -> Constraint::Ptr {
        auto const & a = fvh.getBoolVar(args.at(0));
        auto const & b = fvh.getBoolVar(args.at(1));
        auto const & r = fvh.getBoolVar(args.at(2));
        auto const & logic_or = new (solver) bool_or(a,b); // a /\ b
        auto const & logic_nor = new (solver) bool_nor(a,b); // -(a /\ b)
        backward_implication_t bi = [=]() {
            if (a->isTrue() or b->isTrue())
            {
                r->assign(true);
            }
            else if (a->isFalse() and b->isFalse())
            {
                r->assign(false);
            }
        };
        return new (solver) bool_bin_reif(a,b,r,logic_or,logic_nor, bi);
    });

    constriants_builders.emplace("bool_xor_reif", [&](vector<Fzn::constraint_arg_t> const & args, std::vector<Fzn::annotation_t> const & anns) -> Constraint::Ptr {
        auto const & a = fvh.getBoolVar(args.at(0));
        auto const & b = fvh.getBoolVar(args.at(1));
        auto const & r = fvh.getBoolVar(args.at(2));
        auto const & logic_xor = new (solver) bool_xor(a,b); // a + b
        auto const & logic_xnor = new (solver) bool_nxor(a,b); // -(a + b)
        backward_implication_t bi = [=]() {
            if ((a->isFalse() and b->isTrue()) or (a->isTrue() and b->isFalse()))
            {
                r->assign(true);
            }
            else if ((a->isFalse() and b->isFalse()) or (a->isTrue() and b->isTrue()))
            {
                r->assign(false);
            }
        };
        return new (solver) bool_bin_reif(a,b,r,logic_xor,logic_xnor,bi);
    });

    constriants_builders.emplace("bool_eq_reif", [&](vector<Fzn::constraint_arg_t> const & args, std::vector<Fzn::annotation_t> const & anns) -> Constraint::Ptr {
        auto const & a = fvh.getBoolVar(args.at(0));
        auto const & b = fvh.getBoolVar(args.at(1));
        auto const & r = fvh.getBoolVar(args.at(2));
        auto const & eq = new (solver) EQBinBC(a,b,0); // a = b
        auto const & neq = new (solver) NEQBinBC(a,b,0); // a != b
        backward_implication_t bi = [=]() {
            if ((a->isTrue() and b->isTrue()) or (a->isFalse() and b->isFalse()))
            {
                r->assign(true);
            }
            else if ((a->isTrue() and b->isFalse()) or (a->isFalse() and b->isTrue()))
            {
                r->assign(false);
            }
        };
        return new (solver) int_bin_reif(a,b,r,eq,neq,bi);
    });

    constriants_builders.emplace("bool_le_reif", [&](vector<Fzn::constraint_arg_t> const & args, std::vector<Fzn::annotation_t> const & anns) -> Constraint::Ptr {
        auto const & a = fvh.getBoolVar(args.at(0));
        auto const & b = fvh.getBoolVar(args.at(1));
        auto const & r = fvh.getBoolVar(args.at(2));
        auto const & _b = new (solver) IntVarViewOffset(b, -1);
        auto const & le = new (solver) LessOrEqual(a,b); // a <= b
        auto const & gt = new (solver) LessOrEqual(_b,a); // a > b
        backward_implication_t bi = [=]() {
            if (b->isTrue() or (a->isFalse() and b->isFalse()))
            {
                r->assign(true);
            }
            else if (a->isTrue() and b->isFalse())
            {
                r->assign(false);
            }
        };
        return new (solver) int_bin_reif(a,b,r,le,gt,bi);
    });

    constriants_builders.emplace("bool_lt_reif", [&](vector<Fzn::constraint_arg_t> const & args, std::vector<Fzn::annotation_t> const & anns) -> Constraint::Ptr {
        auto const & a = fvh.getBoolVar(args.at(0));
        auto const & b = fvh.getBoolVar(args.at(1));
        auto const & r = fvh.getBoolVar(args.at(2));
        auto const & _b = new (solver) IntVarViewOffset(b, -1);
        auto const & lt = new (solver) LessOrEqual(a,_b); // a < b
        auto const & ge = new (solver) LessOrEqual(b,a); // a >= b
        backward_implication_t bi = [=]() {
            if (a->isFalse() and b->isTrue())
            {
                r->assign(true);
            }
            else if (b->isFalse())
            {
                r->assign(false);
            }
        };
        return new (solver) int_bin_reif(a,b,r,lt,ge,bi);
    });

    // Implication
    constriants_builders.emplace("array_bool_and_imp", [&](vector<Fzn::constraint_arg_t> const & args, std::vector<Fzn::annotation_t> const & anns) -> Constraint::Ptr {
        auto const & as =  fvh.getArrayBoolVars(args.at(0));
        auto const & r = fvh.getBoolVar(args.at(1));
        auto const & array_and = new (solver) array_bool_and(as);
        backward_implication_t bi = [=]() {
            if (any_of(as.begin(), as.end(), [&](var<bool>::Ptr const & b) -> bool {return b->isFalse();}))
            {
                r->assign(false);
            }
        };
        return new (solver) array_bool_imp(as,r,array_and,bi);
    });

    constriants_builders.emplace("array_bool_or_imp", [&](vector<Fzn::constraint_arg_t> const & args, std::vector<Fzn::annotation_t> const & anns) -> Constraint::Ptr {
        auto const & as =  fvh.getArrayBoolVars(args.at(0));
        auto const & r = fvh.getBoolVar(args.at(1));
        auto const & array_or = new (solver) array_bool_or(as);
        backward_implication_t bi = [=]() {
            if (all_of(as.begin(), as.end(), [&](var<bool>::Ptr const & b) -> bool {return b->isFalse();}))
            {
                r->assign(false);
            }
        };
        return new (solver) array_bool_imp(as,r,array_or,bi);
    });

    constriants_builders.emplace("array_bool_xor_imp", [&](vector<Fzn::constraint_arg_t> const & args, std::vector<Fzn::annotation_t> const & anns) -> Constraint::Ptr {
        auto const & as =  fvh.getArrayBoolVars(args.at(0));
        auto const & r = fvh.getBoolVar(args.at(1));
        auto const & array_xor = new (solver) array_bool_xor(as);
        backward_implication_t bi = [=]() {
            auto nbVarBound = count_if(as.begin(), as.end(), [&](var<bool>::Ptr const & b) -> bool {return b->isBound();});
            auto nbVarTrue = count_if(as.begin(), as.end(), [&](var<bool>::Ptr const & b) -> bool {return b->isTrue();});
            if (nbVarBound == as.size() and nbVarTrue % 2 == 0)
            {
                r->assign(false);
            }
        };
        return new (solver) array_bool_imp(as,r,array_xor,bi);
    });

    constriants_builders.emplace("bool_and_imp", [&](vector<Fzn::constraint_arg_t> const & args, std::vector<Fzn::annotation_t> const & anns) -> Constraint::Ptr {
        auto const & a = fvh.getBoolVar(args.at(0));
        auto const & b = fvh.getBoolVar(args.at(1));
        auto const & r = fvh.getBoolVar(args.at(2));
        auto const & logic_and = new (solver) bool_and(a,b); // a /\ b
        backward_implication_t bi = [=]() {
            if (a->isFalse() or b->isFalse())
            {
                r->assign(false);
            }
        };
        return new (solver) bool_bin_imp(a,b,r,logic_and, bi);
    });

    constriants_builders.emplace("bool_or_imp", [&](vector<Fzn::constraint_arg_t> const & args, std::vector<Fzn::annotation_t> const & anns) -> Constraint::Ptr {
        auto const & a = fvh.getBoolVar(args.at(0));
        auto const & b = fvh.getBoolVar(args.at(1));
        auto const & r = fvh.getBoolVar(args.at(2));
        auto const & logic_or = new (solver) bool_or(a,b); // a /\ b
        backward_implication_t bi = [=]() {
            if (a->isFalse() and b->isFalse())
            {
                r->assign(false);
            }
        };
        return new (solver) bool_bin_imp(a,b,r,logic_or, bi);
    });

    constriants_builders.emplace("bool_xor_imp", [&](vector<Fzn::constraint_arg_t> const & args, std::vector<Fzn::annotation_t> const & anns) -> Constraint::Ptr {
        auto const & a = fvh.getBoolVar(args.at(0));
        auto const & b = fvh.getBoolVar(args.at(1));
        auto const & r = fvh.getBoolVar(args.at(2));
        auto const & logic_xor = new (solver) EQBinBC(a,b,0); // a + b
        backward_implication_t bi = [=]() {
            if ((a->isFalse() and b->isFalse()) or (a->isTrue() and b->isTrue()))
            {
                r->assign(false);
            }
        };
        return new (solver) bool_bin_imp(a,b,r,logic_xor,bi);
    });

    constriants_builders.emplace("bool_eq_imp", [&](vector<Fzn::constraint_arg_t> const & args, std::vector<Fzn::annotation_t> const & anns) -> Constraint::Ptr {
        auto const & a = fvh.getBoolVar(args.at(0));
        auto const & b = fvh.getBoolVar(args.at(1));
        auto const & r = fvh.getBoolVar(args.at(2));
        auto const & eq = new (solver) EQBinBC(a,b,0); // a = b
        backward_implication_t bi = [=]() {
            if ((a->isTrue() and b->isFalse()) or (a->isFalse() and b->isTrue()))
            {
                r->assign(false);
            }
        };
        return new (solver) int_bin_imp(a,b,r,eq,bi);
    });

    constriants_builders.emplace("bool_le_imp", [&](vector<Fzn::constraint_arg_t> const & args, std::vector<Fzn::annotation_t> const & anns) -> Constraint::Ptr {
        auto const & a = fvh.getBoolVar(args.at(0));
        auto const & b = fvh.getBoolVar(args.at(1));
        auto const & r = fvh.getBoolVar(args.at(2));
        auto const & le = new (solver) LessOrEqual(a,b); // a <= b
        backward_implication_t bi = [=]() {
            if (a->isTrue() and b->isFalse())
            {
                r->assign(false);
            }
        };
        return new (solver) int_bin_imp(a,b,r,le,bi);
    });

    constriants_builders.emplace("bool_lt_imp", [&](vector<Fzn::constraint_arg_t> const & args, std::vector<Fzn::annotation_t> const & anns) -> Constraint::Ptr {
        auto const & a = fvh.getBoolVar(args.at(0));
        auto const & b = fvh.getBoolVar(args.at(1));
        auto const & r = fvh.getBoolVar(args.at(2));
        auto const & _b = new (solver) IntVarViewOffset(b, -1);
        auto const & lt = new (solver) LessOrEqual(a,_b); // a < b
        backward_implication_t bi = [=]() {
            if (b->isFalse())
            {
                r->assign(false);
            }
        };
        return new (solver) int_bin_imp(a,b,r,lt,bi);
    });
}

void FznConstraintHelper::addGlobalConstraintsBuilders()
{
    using namespace std;

    constriants_builders.emplace("minicpp_all_different", [&](vector<Fzn::constraint_arg_t> const & args, vector<Fzn::annotation_t> const & anns) -> Constraint::Ptr {
        auto const & x = fvh.getArrayIntVars(args.at(0));
        return new (solver) AllDifferentAC(x);
    });

    constriants_builders.emplace("minicpp_circuit", [&](vector<Fzn::constraint_arg_t> const & args, vector<Fzn::annotation_t> const & anns) -> Constraint::Ptr {
        auto const x = fvh.getArrayIntVars(args.at(0));
        return new (solver) Circuit(x);
    });

    constriants_builders.emplace("minicpp_cumulative", [&](vector<Fzn::constraint_arg_t> const & args, vector<Fzn::annotation_t> const & anns) -> Constraint::Ptr {
        auto s = fvh.getArrayIntVars(args.at(0));
        auto p = fvh.getArrayInt(args.at(1));
        auto h = fvh.getArrayInt(args.at(2));
        auto c = FznVariablesHelper::getInt(args.at(3));

        bool const gpu = count_if(anns.begin(), anns.end(), [](Fzn::annotation_t const & ann) -> bool {return ann.first == "gpu";});
        if (gpu)
        {
            return new (solver) CumulativeGPU(s,p,h,c);
        }
        else
        {
            return new (solver) Cumulative(s,p,h,c);
        }
    });

    constriants_builders.emplace("minicpp_table_int", [&](vector<Fzn::constraint_arg_t> const & args, vector<Fzn::annotation_t> const & anns) -> Constraint::Ptr {
        auto x = fvh.getArrayIntVars(args.at(0));
        auto t = fvh.getArrayInt(args.at(1));
        assert(t.size() % x.size() == 0);
        auto const tuple_size = x.size();
        auto const tuple_count = t.size() / tuple_size;
        vector<vector<int>> _t;
        for(auto i = 0; i < tuple_count; i += 1)
        {
            auto const tBegin = t.begin() + (i * tuple_size);
            auto const tEnd = tBegin + tuple_size;
            _t.emplace_back(tBegin, tEnd);
        }

        bool const uniud = count_if(anns.begin(), anns.end(), [](Fzn::annotation_t const & ann) -> bool {return ann.first == "uniud";});
        if (uniud)
        {
            return new (solver) Table(x, _t);
        }
        else
        {
            return new (solver) TableCT(x, _t);
        }
    });

    constriants_builders.emplace("minicpp_stable_matching", [&](vector<Fzn::constraint_arg_t> const & args, vector<Fzn::annotation_t> const & anns) -> Constraint::Ptr {
        auto m = fvh.getArrayIntVars(args.at(0));
        auto w = fvh.getArrayIntVars(args.at(1));
        assert(m.size() == w.size());

        auto const n = m.size();
        auto pmFlat = fvh.getArrayInt(args.at(2));
        auto pwFlat = fvh.getArrayInt(args.at(3));

        vector<vector<int>> pm;
        for (auto mIdx = 0; mIdx < n; mIdx += 1)
        {
            auto const tBegin = pmFlat.begin() + (mIdx * n);
            auto const tEnd = tBegin + n;
            pm.emplace_back(tBegin, tEnd);
        }

        vector<vector<int>> pw;
        for (auto wIdx = 0; wIdx < n; wIdx += 1)
        {
            auto const tBegin = pwFlat.begin() + (wIdx * n);
            auto const tEnd = tBegin + n;
            pw.emplace_back(tBegin, tEnd);
        }

        bool const uniud = count_if(anns.begin(), anns.end(), [](Fzn::annotation_t const & ann) -> bool {return ann.first == "uniud";});
        if (uniud)
        {
            return new (solver) StableMatching(m, w, pm, pw);
        }
        else
        {
            throw std::runtime_error("Missing search annotation on stable_matching constraint: \"::uniud\"");
        }
    });
}
