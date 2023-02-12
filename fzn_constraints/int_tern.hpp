#pragma once

#include <libminicpp/intvar.hpp>

void calMulBounds(int aMin, int aMax, int bMin, int bMax, int & min, int & max);
void calDivBounds(int aMin, int aMax, int bMin, int bMax, int & min, int & max);
void calPowBounds(int aMin, int aMax, int bMin, int bMax, int & min, int & max);
void calPowBounds(int aMin, int aMax, double bVal, int & min, int & max);

class int_tern : public Constraint
{
    protected:
        var<int>::Ptr _a;
        var<int>::Ptr _b;
        var<int>::Ptr _c;

    public:
        int_tern(var<int>::Ptr const a, var<int>::Ptr const b, var<int>::Ptr const c);
        void post() override;
};

class int_div : public int_tern
{
    public:
        int_div(var<int>::Ptr const a, var<int>::Ptr const b, var<int>::Ptr const c): int_tern(a,b,c) {}
        void propagate() override;
};

class int_max : public int_tern
{
    public:
        int_max(var<int>::Ptr const a, var<int>::Ptr const b, var<int>::Ptr const c) : int_tern(a,b,c) {};
        void propagate() override;
};

class int_min : public int_tern
{
    public:
        int_min(var<int>::Ptr const a, var<int>::Ptr const b, var<int>::Ptr const c) : int_tern(a,b,c) {};
        void propagate() override;
};

class int_mod : public int_tern
{
    public:
        int_mod(var<int>::Ptr const a, var<int>::Ptr const b, var<int>::Ptr const c) : int_tern(a,b,c) {};
        void propagate() override;
};

class int_pow : public int_tern
{
    public:
        int_pow(var<int>::Ptr const a, var<int>::Ptr const b, var<int>::Ptr const c) : int_tern(a,b,c) {};
        void propagate() override;
};

class int_times : public int_tern
{
    public:
        int_times(var<int>::Ptr const a, var<int>::Ptr const b, var<int>::Ptr const c) : int_tern(a,b,c) {};
        void propagate() override;
};
