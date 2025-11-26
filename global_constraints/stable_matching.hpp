#pragma once

#include <varitf.hpp>
#include <queue>


class StableMatching : public Constraint
{
    //"Private" struct
    struct constraintCall{
    // 0: removeValue
    // 1: deltaMin
    // 2: deltaMax
    // 3: inst
    int function;

    int ij;
    // The man or woman affected

    int a;
    // The value (index) to be removed by removeValue

    int isMan;
    // True if removeValue or inst are working on a man, False if they're working on a woman

    // Constructor
    constraintCall(int function, int ij, int a, int isMan)
        : function(function), ij(ij), a(a), isMan(isMan) {}
    };

    // Constraint private data structures
    protected:
        std::vector<var<int>::Ptr> _x;
        std::vector<var<int>::Ptr> _y;
        std::vector<std::vector<int>> const _xpl;
        std::vector<std::vector<int>> const _ypl;
        int *_xPy;
        int *_yPx;
        int _n;
        std::vector<trail<int>> _xub;
        std::vector<trail<int>> _xlb;
        std::vector<trail<int>> _yub;
        std::vector<trail<int>> _ylb;
        std::queue<constraintCall> _callQueue;
        std::vector<trail<int>> _x_old_sizes;
        std::vector<trail<int>> _y_old_sizes;

    // Constraint methods
    public:
        StableMatching(std::vector<var<int>::Ptr> & m, std::vector<var<int>::Ptr> & w, std::vector<std::vector<int>> const & pm, std::vector<std::vector<int>> const & pw);
        void post() override;
        void propagate() override;
    
    protected:
        void buildReverseMatrix(std::vector<std::vector<int>> zpl, int *zPz);
        void fillQueue();
        void propagateOnQueue();
        void functionDispatcher();
        void removeValue(int i, int a, int isMan);
        void deltaMin(int i);
        void deltaMax(int j);
        void inst(int i, int isMan);
        void init();
};


