#pragma once

#include <libminicpp/intvar.hpp>

class array_bool : public Constraint
{
    protected:
        std::vector<var<bool>::Ptr> _as;

    public:
        array_bool(std::vector<var<bool>::Ptr> const & as);
        void post() override;
};

class array_bool_imp : public array_bool
{
    protected:
        var<bool>::Ptr _r;
        Constraint::Ptr _pfi;      //Positive Forward Implication
        std::function<void()> _bi; //Backward Implication

    public:
        array_bool_imp(std::vector<var<bool>::Ptr> const & as, var<bool>::Ptr r, Constraint::Ptr pfi, std::function<void()> bi);
        void post() override;
        void propagate() override;
};

class array_bool_reif : public array_bool_imp
{
    protected:
        Constraint::Ptr _nfi;      //Negative Forward Implication

    public:
        array_bool_reif(std::vector<var<bool>::Ptr> const & as, var<bool>::Ptr r, Constraint::Ptr pfi, Constraint::Ptr nfi, std::function<void()> bi);
        void propagate() override;
};

class array_bool_and : public array_bool
{
    public:
        array_bool_and(std::vector<var<bool>::Ptr> const & as): array_bool(as) {};
        void propagate() override;
};

class array_bool_nand : public array_bool
{
    public:
        array_bool_nand(std::vector<var<bool>::Ptr> const & as): array_bool(as) {};
        void propagate() override;
};

class array_bool_or : public array_bool
{
    public:
        array_bool_or(std::vector<var<bool>::Ptr> const & as) : array_bool(as) {};
        void propagate() override;
};

class array_bool_nor : public array_bool
{
    public:
        array_bool_nor(std::vector<var<bool>::Ptr> const & as) : array_bool(as) {};
        void propagate() override;
};

class array_bool_xor : public array_bool
{
    public:
        array_bool_xor(std::vector<var<bool>::Ptr> const & as) : array_bool(as) {};
        void propagate() override;
};
