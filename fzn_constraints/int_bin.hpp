#pragma once

#include <libminicpp/intvar.hpp>

class int_bin : public Constraint
{
    protected:
        var<int>::Ptr _a;
        var<int>::Ptr _b;

    public:
        int_bin(var<int>::Ptr a, var<int>::Ptr b);
        void post() override;
};

class int_bin_imp : public int_bin
{
    protected:
        var<bool>::Ptr _r;
        Constraint::Ptr _pfi;      //Positive Forward Implication
        std::function<void()> _bi; //Backward Implication

    public:
        int_bin_imp(var<int>::Ptr a, var<int>::Ptr b, var<bool>::Ptr r, Constraint::Ptr pfi, std::function<void()> & bi);
        void post() override;
        void propagate() override;
};

class int_bin_reif : public int_bin_imp
{
    protected:
        Constraint::Ptr _nfi;      //Negative Forward Implication

    public:
        int_bin_reif(var<int>::Ptr a, var<int>::Ptr b, var<bool>::Ptr r, Constraint::Ptr pfi, Constraint::Ptr nfi, std::function<void()> & bi);
        void propagate() override;
};

class int_abs : public int_bin
{
    public:
        int_abs(var<int>::Ptr const a, var<int>::Ptr const b) : int_bin(a,b) {};
        void propagate() override;
};
