#pragma once

#include <libminicpp/intvar.hpp>

class int_lin_imp : public Constraint
{
    protected:
        std::vector<var<int>::Ptr> _bs;
        var<bool>::Ptr _r;
        Constraint::Ptr _pfi;      //Positive Forward Implication
        std::function<void()> _bi; //Backward Implication

    public:
        int_lin_imp(std::vector<var<int>::Ptr> const & bs, var<bool>::Ptr r, Constraint::Ptr pfi, std::function<void()> & bi);
        void post() override;
        void propagate() override;
};

class int_lin_reif : public int_lin_imp
{
    protected:
        Constraint::Ptr _nfi;      //Negative Forward Implication

    public:
        int_lin_reif(std::vector<var<int>::Ptr> const & bs, var<bool>::Ptr r, Constraint::Ptr pfi, Constraint::Ptr nfi, std::function<void()> & bi);
        void propagate() override;
};
