#pragma once

#include <libminicpp/intvar.hpp>

class bool_bin : public Constraint
{
    protected:
        var<bool>::Ptr _a;
        var<bool>::Ptr _b;

    public:
        bool_bin(var<bool>::Ptr a, var<bool>::Ptr b);
        void post() override;
};

class bool_bin_imp : public bool_bin
{
    protected:
        var<bool>::Ptr _r;
        Constraint::Ptr _pfi;      //Positive Forward Implication
        std::function<void()> _bi; //Backward Implication

    public:
        bool_bin_imp(var<bool>::Ptr a, var<bool>::Ptr b, var<bool>::Ptr r, Constraint::Ptr pfi, std::function<void()> bi);
        void post() override;
        void propagate() override;
};

class bool_bin_reif : public bool_bin_imp
{
    protected:
        Constraint::Ptr _nfi;      //Negative Forward Implication

    public:
        bool_bin_reif(var<bool>::Ptr a, var<bool>::Ptr b, var<bool>::Ptr r, Constraint::Ptr pfi, Constraint::Ptr nfi, std::function<void()> bi);
        void propagate() override;
};

class bool_and : public bool_bin
{
    public:
        bool_and(var<bool>::Ptr a, var<bool>::Ptr b) : bool_bin(a,b) {};
        void propagate() override;
};

class bool_nand : public bool_bin
{
    public:
        bool_nand(var<bool>::Ptr a, var<bool>::Ptr b) : bool_bin(a,b) {};
        void propagate() override;
};

class bool_or : public bool_bin
{
    public:
        bool_or(var<bool>::Ptr a, var<bool>::Ptr b) : bool_bin(a,b) {};
        void propagate() override;
};

class bool_nor : public bool_bin
{
    public:
        bool_nor(var<bool>::Ptr a, var<bool>::Ptr b) : bool_bin(a,b) {};
        void propagate() override;
};

class bool_xor : public bool_bin
{
    public:
        bool_xor(var<bool>::Ptr a, var<bool>::Ptr b) : bool_bin(a,b) {};
        void propagate() override;
};

class bool_nxor : public bool_bin
{
    public:
        bool_nxor(var<bool>::Ptr a, var<bool>::Ptr b) : bool_bin(a,b) {};
        void propagate() override;
};
