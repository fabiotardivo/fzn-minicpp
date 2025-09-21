#pragma once

#include <functional>
#include "constraint.hpp"
#include "libfca/Vector.hpp"
#include "libfca/Utils.hpp"

class BinPackingLoad : public Constraint
{
    public:
        BinPackingLoad(std::vector<var<int>::Ptr> const & load, std::vector<var<int>::Ptr> const & bin, std::vector<int> const & w);
        void post() override;
        void propagate() override;

    protected:
        // Standard propagator
        struct ItemInfo {int index; int weight;};

        template <typename T>
        class VectorView
        {
            private:
                std::vector<T> const & v;
                int idxToHide;
            public:
                explicit VectorView(std::vector<T> const & v) : v(v), idxToHide(INT_MAX) {};
                VectorView(std::vector<T> const & v, int idx) : v(v), idxToHide(idx) {};
                std::size_t size() const {return v.size() - (idxToHide != INT_MAX);};
                T const & at(std::size_t idx) const {return v.at(idx + (idx >= idxToHide));};
        };

        std::vector<var<int>::Ptr> const _load;
        std::vector<var<int>::Ptr> _bin;
        std::vector<int> _weights;
        int nBins;
        int nItems;
        int sum_weights;
        int sum_min_loads;
        int sum_max_loads;
        std::vector<int> bin_sizes;
        std::vector<int> load_sizes;
        std::vector<int> required_loads;
        std::vector<int> possible_loads;
        std::vector<int> packed_items;
        std::vector<std::vector<ItemInfo>> candidate_items;
        std::vector<int> unassigned_weights;

        void calculateNumBins();
        void calculateRequiredPossibleCandidates();
        void propagateLoadCoherence();
        void calculateSumLoads();
        void propagateBasicLoadTightening();
        void propagateBasicItemEliminationCommitment();
        void checkPackableBins();
        void propagateKnapsackLoadTightening();
        void propagateKnapsackItemEliminationCommitment();
        void saveBinLoadSizes();
        bool checkBinLoadSizeChanges();
        void checkLowerbounds();
        std::tuple<bool,int,int> noSum(VectorView<ItemInfo> const & items_info, int alpha, int beta);
        static int sumWeights(VectorView<ItemInfo> const & items_info);


        // Reductions
        static int const REDUCTIONS_COUNT = 3;
        std::vector<int> weightsBaseReduction;
        std::vector<int> weightsCurrentReduction;
        std::array<int,REDUCTIONS_COUNT> deltaReductions;
        int capacityBaseReduction;

        void prepareReductions();


        // Dual-Feasible Functions
        static int const DFFS_COUNT = 6;

        static int fCCM1(int w, int l, int c);
        static int fMT(int w, int l, int c);
        static int fBJ1(int w, int l, int c);
        static int fVB2Base(int w, int l, int c);
        static int fVB2(int w, int l, int c);
        static int fFS1(int w, int l, int c);
        static int fRAD2Base(int w, int l, int c);
        static int fRAD2(int w, int l, int c);


        // Lambdas
        static int const LAMBDA_SAMPLES  = 256;
        struct LambdaRange {int min; int max;};

        static LambdaRange lCCM1(int c) {return {1, c / 2};};
        static LambdaRange lMT(int c)   {return {0, c / 2};}; // The 0 value is included to calculate the L0 bound
        static LambdaRange lBJ1(int c)  {return {1, c};};
        static LambdaRange lVB2(int c)  {return {2, c};};
        static LambdaRange lFS1(int c)  {return {1, 100};};
        static LambdaRange lRAD2(int c) {return {c / 4 + 1, c / 3};};
        static LambdaRange sanitizeLambdaRange(LambdaRange lambda, int nWeights, int maxWeight);


        // Lowerbounds
        static int calcL2(std::vector<int> const & weights, int capacity);
        template<int f(int,int,int)>
        static int calcDffLowerboundSingleLambda(std::vector<int> const & weights, int capacity, int lambda)
        {
            using namespace Fca::Utils::Math;

            // Transform and sum the weights
            int nWeights = static_cast<int>(weights.size());
            int sumTransformedWeights = 0;
            for (int wIdx = 0; wIdx < nWeights; wIdx += 1)
            {
                sumTransformedWeights += f(weights[wIdx], lambda, capacity);
            }

            // Transform the capacity
            int const transformedCapacity = f(capacity, lambda, capacity);

            // Lowerbound
            return ceilDivPosInt(sumTransformedWeights, transformedCapacity);
        }
        template<int f(int,int,int), LambdaRange l(int)>
        static int calcDffLowerbound(std::vector<int> const & weights, int capacity, int nWeights, int maxWeight, bool sanitize = false)
        {
            using namespace Fca::Utils::Math;

            int fLowerbound = 0;
            LambdaRange lRange = l(capacity);
            lRange = sanitize ? sanitizeLambdaRange(lRange,nWeights, maxWeight) : lRange;
            int lStep = ceilDivPosInt(lRange.max - lRange.min + 1, LAMBDA_SAMPLES + 1);
            for (int lambda = lRange.min + lStep; lambda < lRange.max; lambda += lStep)
            {
                int lowerbound =calcDffLowerboundSingleLambda<f>(weights, capacity, lambda);
                fLowerbound = std::max(fLowerbound, lowerbound);
            }
            return fLowerbound;
        }
};