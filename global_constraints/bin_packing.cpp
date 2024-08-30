#include "bin_packing.hpp"

#include "libfca/Timer.hpp"
#include "libfca/Slice.hpp"
#include <algorithm>

using namespace Fca;

BinPackingLoad::BinPackingLoad(std::vector<var<int>::Ptr> const & load, std::vector<var<int>::Ptr> const & bin, std::vector<int> const & w) :
        Constraint(load[0]->getSolver()),
        _load(load),
        _bin(),
        _weights(w)
{
    setPriority(CLOW);

    nBins = static_cast<int>(_load.size());
    nItems = static_cast<int>(bin.size());

    // Workaround MiniZinc indices starting from 1...
    int minBinIdx = INT_MAX;
    for (int i = 0; i < nItems; i += 1)
    {
        minBinIdx = std::min(minBinIdx, bin.at(i)->min());
    }
    for (int i = 0; i < nItems; i += 1)
    {
        auto b = new IntVarViewOffset(bin.at(i), -minBinIdx);
        _bin.push_back(b);
    }

    // Sort items and weights by decreasing weights
    std::vector<std::pair<var<int>::Ptr,int>> item_weights;
    for (int i  = 0; i < nItems; i += 1)
    {
        item_weights.emplace_back(_bin.at(i), _weights.at(i));
    }
    std::sort(item_weights.begin(), item_weights.end(), [](auto & lhs, auto & rhs) {return lhs.second >= rhs.second;});

    for (int i = 0; i < nItems; i += 1)
    {
        _bin.at(i) = item_weights.at(i).first;
        _weights.at(i) = item_weights.at(i).second;
    }

    // Sum weights
    sum_weights = 0;
    for (auto iIdx = 0; iIdx < nItems; iIdx += 1)
    {
        sum_weights += _weights.at(iIdx);
    }

    // Sizes
    bin_sizes.resize(nItems, 0);
    load_sizes.resize(nBins, 0);

    // Possible / Required loads
    required_loads.resize(nBins, 0);
    possible_loads.resize(nBins, 0);

    // Packed / Candidate items
    packed_items.resize(nBins);
    candidate_items.resize(nBins);
    for (auto bIdx = 0; bIdx < nBins; bIdx += 1)
    {
        candidate_items.at(bIdx).reserve(nItems);
    }
    // Reductions
    weightsBaseReduction.resize(nItems + nBins);
    weightsCurrentReduction.resize(nItems + nBins);
}

void BinPackingLoad::post()
{
    for (auto const & l : _load)
    {
        l->propagateOnBoundChange(this);
    }

    for (auto const & b : _bin)
    {
        b->propagateOnDomainChange(this);
    }

    propagate();
}

void BinPackingLoad::propagate()
{
    calculateNumBins();
    saveBinLoadSizes();
    calculateRequiredPossibleCandidates();
    propagateLoadCoherence();
    calculateSumLoads();
    propagateBasicLoadTightening();
    propagateBasicItemEliminationCommitment();
    if (not checkBinLoadSizeChanges())
    {
        checkPackableBins();
        propagateKnapsackLoadTightening();
        propagateKnapsackItemEliminationCommitment();

        if (not checkBinLoadSizeChanges())
        {
            prepareReductions();
            checkLowerbounds();
        }
    }
}

void BinPackingLoad::calculateNumBins()
{
    nBins = 0;
    for (auto iIdx = 0; iIdx < nItems; iIdx += 1)
    {
       nBins = std::max(nBins, _bin.at(iIdx)->max());
    }
    nBins += 1;
}

void BinPackingLoad::calculateRequiredPossibleCandidates()
{
    // Clear values
    for (auto bIdx = 0; bIdx < nBins; bIdx += 1)
    {
        required_loads.at(bIdx) = 0;
        possible_loads.at(bIdx) = 0;
        packed_items.at(bIdx) = 0;
        candidate_items.at(bIdx).clear();
    }

    // Calculate Required / Possible loads and candidate items
    for (auto iIdx = 0; iIdx < nItems; iIdx += 1)
    {
        auto const iWeight = _weights.at(iIdx);
        if (not _bin.at(iIdx)->isBound())
        {
            auto const minBinIdx = _bin.at(iIdx)->min();
            auto const maxBinIdx = _bin.at(iIdx)->max();
            for (auto bIdx = minBinIdx; bIdx <= maxBinIdx; bIdx += 1)
            {
                if (_bin.at(iIdx)->containsBase(bIdx))
                {
                    ItemInfo const iInfo = {iIdx, iWeight};
                    possible_loads.at(bIdx) += iWeight;
                    candidate_items.at(bIdx).push_back(iInfo);
                }
            }
        }
        else
        {
            auto const bIdx = _bin.at(iIdx)->min();
            required_loads.at(bIdx) += iWeight;
            possible_loads.at(bIdx) += iWeight;
            packed_items.at(bIdx) += 1;
        }
    }
}

void BinPackingLoad::propagateLoadCoherence()
{
    // Filter loads
    for (auto bIdx = 0; bIdx < nBins; bIdx += 1)
    {
        auto const bRequiredLoad = required_loads.at(bIdx);
        auto const bPossibleLoad = possible_loads.at(bIdx);
        _load.at(bIdx)->updateBounds(bRequiredLoad, bPossibleLoad);
    }
}

void BinPackingLoad::calculateSumLoads()
{
    // Clear values
    sum_min_loads = 0;
    sum_max_loads = 0;

    // Calculate sum loads
    for (auto bIdx = 0; bIdx < nBins; bIdx += 1)
    {
        auto const bMinLoad = _load.at(bIdx)->min();
        auto const bMaxLoad = _load.at(bIdx)->max();
        sum_min_loads += bMinLoad;
        sum_max_loads += bMaxLoad;
    }
}

void BinPackingLoad::propagateBasicLoadTightening()
{
    for (auto bIdx = 0; bIdx < nBins; bIdx += 1)
    {
        auto const bMinLoad = _load.at(bIdx)->min();
        auto const bMaxLoad = _load.at(bIdx)->max();
        auto const bLowerboundLoad = sum_weights - (sum_max_loads - bMaxLoad);
        auto const bUpperboundLoad = sum_weights - (sum_min_loads - bMinLoad);
        _load.at(bIdx)->updateBounds(bLowerboundLoad, bUpperboundLoad);
    }
}

void BinPackingLoad::propagateBasicItemEliminationCommitment()
{
    for (auto bIdx = 0; bIdx < nBins; bIdx += 1)
    {
        if (not candidate_items.at(bIdx).empty())
        {
            auto const bMinLoad = _load.at(bIdx)->min();
            auto const bMaxLoad = _load.at(bIdx)->max();
            auto const bRequiredLoad = required_loads.at(bIdx);
            auto const bPossibleLoad = possible_loads.at(bIdx);
            auto const n_candidate_items = candidate_items.at(bIdx).size();
            for (auto ciIdx = 0; ciIdx < n_candidate_items; ciIdx += 1)
            {
                auto const iIdx = candidate_items.at(bIdx).at(ciIdx).index;
                auto const iWeight = candidate_items.at(bIdx).at(ciIdx).weight;
                if (bRequiredLoad + iWeight > bMaxLoad)
                {
                    _bin.at(iIdx)->remove(bIdx);
                }
                if (bPossibleLoad - iWeight < bMinLoad)
                {
                    _bin.at(iIdx)->assign(bIdx);
                }
            }
        }
    }
}

void BinPackingLoad::checkPackableBins()
{
    for (auto bIdx = 0; bIdx < nBins; bIdx += 1)
    {
        auto const alpha = _load.at(bIdx)->min() - required_loads.at(bIdx);
        auto const beta = _load.at(bIdx)->max() - required_loads.at(bIdx);
        auto const &[no_sum, alpha_prime, beta_prime] = noSum(VectorView(candidate_items.at(bIdx)), alpha, beta);
        if (no_sum)
        {
            failNow();
        }
    }
}

void BinPackingLoad::propagateKnapsackLoadTightening()
{
    for (auto bIdx = 0; bIdx < nBins; bIdx += 1)
    {
        auto alpha = _load.at(bIdx)->min() - required_loads.at(bIdx);
        auto [no_sum, alpha_prime, beta_prime] = noSum(VectorView(candidate_items.at(bIdx)), alpha, alpha);
        if (no_sum)
        {
            _load.at(bIdx)->removeBelow(required_loads.at(bIdx) + beta_prime);
        }
        alpha = _load.at(bIdx)->max() - required_loads.at(bIdx);
        std::tie(no_sum, alpha_prime, beta_prime) = noSum(VectorView(candidate_items.at(bIdx)), alpha, alpha);
        if (no_sum)
        {
            _load.at(bIdx)->removeAbove(required_loads.at(bIdx) + alpha_prime);
        }
    }
}

void BinPackingLoad::propagateKnapsackItemEliminationCommitment()
{
    for (auto bIdx = 0; bIdx < nBins; bIdx += 1)
    {
        auto const & candidates = candidate_items.at(bIdx);
        int const nCandidates = (int) candidates.size();
        for (auto cIdx = 0; cIdx < nCandidates; cIdx += 1)
        {
            auto const iIdx = candidates.at(cIdx).index;
            int no_sum;
            int alpha_prime;
            int beta_prime;

            // Check if commit
            int alpha = _load.at(bIdx)->min() - required_loads.at(bIdx);
            int beta = _load.at(bIdx)->max() - required_loads.at(bIdx);
            VectorView<ItemInfo> const otherCandidates(candidate_items.at(bIdx), cIdx);
            std::tie(no_sum, alpha_prime, beta_prime) = noSum(otherCandidates, alpha, beta);
            if (no_sum)
            {
                _bin.at(iIdx)->assign(bIdx);
            }

            // Check if eliminate
            auto const iWeight = _weights.at(iIdx);
            alpha -= iWeight;
            beta -= iWeight;
            std::tie(no_sum, alpha_prime, beta_prime) = noSum(otherCandidates, alpha, beta);
            if (no_sum)
            {
                _bin.at(iIdx)->remove(bIdx);
            }
        }
    }
}

void BinPackingLoad::checkLowerbounds()
{
    using namespace Fca;
    using namespace Fca::Utils::Math;

    int lowerbound = 0;
    int const nWeights = static_cast<int>(weightsBaseReduction.size());

    for (int rIdx = 0; rIdx < REDUCTIONS_COUNT; rIdx += 1)
    {
        // Calculate reduction parameters
        int const & delta = deltaReductions[rIdx];
        int const capacity = capacityBaseReduction + delta;

        if (lowerbound <= nBins and capacity > 0)
        {
            // Calculate reduction
            int notZeroCount = 0;
            int maxWeight = 0;
            std::vector<int> & weights = weightsCurrentReduction;
            for (int wIdx = 0; wIdx < nWeights; wIdx += 1)
            {
                int weight = weightsBaseReduction[wIdx] + (wIdx < nBins ? delta : 0);
                notZeroCount += weight != 0;
                maxWeight = std::max(maxWeight, weight);
                weights[wIdx] = weight;
            }

            // Standard lowerbound
            //std::sort(weights.begin(), weights.end(), std::greater<int>());
            //lowerbound = calcL2(weights, capacity);
            //if (lowerbound > nBins) failNow();


            lowerbound = std::max(lowerbound, calcDffLowerbound<fCCM1,lCCM1>(weights, capacity, notZeroCount, maxWeight));
            if (lowerbound > nBins) failNow();

            lowerbound = std::max(lowerbound, calcDffLowerbound<fMT,lMT>(weights, capacity, notZeroCount, maxWeight));
            if (lowerbound > nBins) failNow();

            lowerbound = std::max(lowerbound, calcDffLowerbound<fBJ1,lBJ1>(weights, capacity, notZeroCount, maxWeight));
            if (lowerbound > nBins) failNow();

            lowerbound = std::max(lowerbound, calcDffLowerbound<fVB2,lVB2>(weights, capacity, notZeroCount, maxWeight, true));
            if (lowerbound > nBins) failNow();

            lowerbound = std::max(lowerbound, calcDffLowerbound<fFS1,lFS1>(weights, capacity, notZeroCount, maxWeight, true));
            if (lowerbound > nBins) failNow();

            lowerbound = std::max(lowerbound, calcDffLowerbound<fRAD2,lRAD2>(weights, capacity, notZeroCount, maxWeight));
            if (lowerbound > nBins) failNow();

        }
    }
}

void BinPackingLoad::saveBinLoadSizes()
{
    for (auto iIdx = 0; iIdx < nItems; iIdx += 1)
    {
        bin_sizes.at(iIdx) = _bin.at(iIdx)->size();
    }
    for (auto bIdx = 0; bIdx < nBins; bIdx += 1)
    {
        load_sizes.at(bIdx) = _load.at(bIdx)->size();
    }
}

bool BinPackingLoad::checkBinLoadSizeChanges()
{
    for (auto iIdx = 0; iIdx < nItems; iIdx += 1)
    {
        if (bin_sizes.at(iIdx) > _bin.at(iIdx)->size())
        {
            return true;
        }
    }
    for (auto iIdx = 0; iIdx < nBins; iIdx += 1)
    {
        if (load_sizes.at(iIdx) >_load.at(iIdx)->size())
        {
            return true;
        }
    }
    return false;
}

std::tuple<bool, int, int> BinPackingLoad::noSum(VectorView<ItemInfo> const & items_info, int alpha, int beta)
{
    if (alpha <= 0 or beta >= sumWeights(items_info))
    {
        return std::make_tuple(false,0,0);
    }

    int const card = static_cast<int>(items_info.size() - 1);
    int sum_a = 0;
    int sum_b = 0;
    int sum_c = 0;
    int k = 0;
    int k_prime = 0;
    while (sum_c + items_info.at(card - k_prime).weight < alpha)
    {
        sum_c += items_info.at(card - k_prime).weight;
        k_prime += 1;
    }
    sum_b = items_info.at(card - k_prime).weight;
    while (sum_a < alpha and sum_b <= beta)
    {
        sum_a += items_info.at(k).weight;
        k += 1;
        if (sum_a < alpha)
        {
            k_prime -= 1;
            assert(k_prime >= 0);
            sum_b += items_info.at(card - k_prime).weight;
            sum_c -= items_info.at(card - k_prime).weight;
            while (sum_a + sum_c >= alpha)
            {
                k_prime -= 1;
                assert(k_prime >= 0);
                sum_c -= items_info.at(card - k_prime).weight;
                sum_b += items_info.at(card - k_prime).weight - items_info.at(card - k_prime - k - 1).weight;
            }
        }
    }
    return std::make_tuple(sum_a < alpha, sum_a + sum_c, sum_b);
}

int BinPackingLoad::sumWeights(VectorView<ItemInfo> const & items_info)
{
    int result = 0;
    auto const nItems = items_info.size();
    for (auto iIdx = 0; iIdx < nItems; iIdx += 1)
    {
        result += items_info.at(iIdx).weight;
    }
    return result;
}

void BinPackingLoad::prepareReductions()
{
    // Reset values
    std::fill(weightsBaseReduction.begin(), weightsBaseReduction.end(), 0);

    // R0
    capacityBaseReduction = 0;
    for (auto bIdx = 0; bIdx < nBins; bIdx += 1)
    {
        capacityBaseReduction = std::max(capacityBaseReduction, _load.at(bIdx)->max());
    }
    for (auto bIdx = 0; bIdx < nBins; bIdx += 1)
    {
        weightsBaseReduction.at(bIdx) = capacityBaseReduction - _load.at(bIdx)->max();
    }
    for (auto iIdx = 0; iIdx < nItems; iIdx += 1)
    {
        bool const iAssigned = _bin.at(iIdx)->isBound();
        auto const iWeight = _weights.at(iIdx);
        if (iAssigned)
        {
            auto const bIdx = _bin.at(iIdx)->min();
            weightsBaseReduction.at(bIdx) += iWeight;
        }
        else
        {
            weightsBaseReduction.at(nBins + iIdx) = iWeight;
        }
    }

    // Bump filled bins
    for (auto bIdx = 0; bIdx < nBins; bIdx += 1)
    {
        if (candidate_items.at(bIdx).empty())
        {
            weightsBaseReduction.at(bIdx) = capacityBaseReduction;
        }
    }

    // RMin, RMax
    int smallestVirtualWeight = INT_MAX;
    for (auto bIdx = 0; bIdx < nBins; bIdx += 1)
    {
        smallestVirtualWeight = std::min(smallestVirtualWeight, weightsBaseReduction.at(bIdx));
    }
    deltaReductions[0] = -smallestVirtualWeight;
    deltaReductions[1] = 0;
    deltaReductions[2] = capacityBaseReduction - 2 * smallestVirtualWeight + 1;
}

BinPackingLoad::LambdaRange BinPackingLoad::sanitizeLambdaRange(LambdaRange lambda, int nWeights, int maxWeight)
{
    if (nWeights * maxWeight != 0)
    {
        int lMax = std::min(INT_MAX / (nWeights * maxWeight), lambda.max);
        return {lambda.min, lMax};
    }
    else
    {
        return {0,-1};
    }
}

int BinPackingLoad::calcL2(std::vector<int> const & weights, int capacity)
{
    // References:
    // - A New Algorithm for Optimal Bin Packing
    // - https://site.unibo.it/operations-research/en/research/bpplib-a-bin-packing-problem-library/scip_arcflow.rar
    int maxLB = 0;
    int courLB;
    bool cont = true;
    int N = static_cast<int>(weights.size());
    int C = capacity;
    int const * W = weights.data();
    int mark = N - 1;
    int seuil;
    int sum;
    while (cont) {
        seuil = W[mark];
        sum = 0;
        for (int i = 0; i < N; i++) {
            if (W[i] > C - seuil) {
                sum += C;
            }
            if (W[i] < seuil) {
                sum += 0;
            }
            if (W[i] >= seuil && W[i] <= C - seuil) {
                sum += W[i];
            }
        }
        courLB = (sum - 1) / C + 1;
        if (mark == 0 || W[mark - 1]>(C / 2)) {
            cont = false;
        }
        if (courLB > maxLB) {
            maxLB = courLB;
        }
        mark--;
    }
    return maxLB;
}

int BinPackingLoad::fMT(int w, int l, int c)
{
    // Branch-less f0 transformation

    // Conditions
    int const c0 = w < l;
    int const c1 = l <= w and w <= c - l;
    int const c2 = c - l < w;
    assert(c0 + c1 + c2 == 1);

    // Values
    int const v0 = 0;
    int const v1 = w;
    int const v2 = c;

    return (c0 * v0) + (c1 * v1) + (c2 * v2);
}

int BinPackingLoad::fRAD2Base(int w, int l, int c)
{
    // Conditions
    int const c0 = w < l;
    int const c1 = l <= w and w <= c - 2 * l;
    int const c2 = c - 2 * l < w and w < 2 * l;

    // Values
    int const v0 = 0;
    int const v1 = c / 3;
    int const v2 = c / 2;

    return (c0 * v0) + (c1 * v1) + (c2 * v2);
}

int BinPackingLoad::fRAD2(int w, int l, int c)
{
    // Conditions
    int const c0 = w < 2 * l;
    int const c1 = 2 * l <= w;
    assert(c0 + c1 == 1);

    // Values
    int const v0 = fRAD2Base(w, l, c);
    int const v1 = c - fRAD2Base(c - w, l, c);

    return (c0 * v0) + (c1 * v1);
}

int BinPackingLoad::fVB2Base(int w, int l, int c)
{
    using namespace Fca::Utils::Math;
    int v = ceilDivPosInt(l * w, c);
    return v > 0 ? v - 1 : 0;
}

int BinPackingLoad::fVB2(int w, int l, int c)
{
    using namespace Fca::Utils::Math;

    int const c0 = 2 * w > c;
    int const c1 = 2 * w == c;
    int const c2 = 2 * w < c;

    int const t0 = fVB2Base(c, l, c);
    int const t1 = fVB2Base(w, l, c);
    int const t2 = fVB2Base(c - w, l, c);

    int const v0 = 2 * t0 - 2 * t2;
    int const v1 = t0;
    int const v2 = 2 * t1;

    return (c0 * v0) + (c1 * v1) + (c2 * v2);
}

int BinPackingLoad::fCCM1(int w, int l, int c)
{
    // Branch-less fCMM transformation

    // Conditions
    int const c0 = 2 * w > c;  // x > c / 2
    int const c1 = 2 * w == c; // x == c / 2
    int const c2 = 2 * w < c;  // x < c / 2
    assert(c0 + c1 + c2 == 1);

    // Values
    int const v0 = 2 * ((c / l) - ((c - w) / l));
    int const v1 = c / l;
    int const v2 = 2 * (w / l);

    return (c0 * v0) + (c1 * v1) + (c2 * v2);
}

int BinPackingLoad::fFS1(int w, int l, int c)
{
    // Branch-less fFS1 transformation

    // Conditions
    int c0 = w * (l + 1) % c == 0;
    int c1 = w * (l + 1) % c != 0;
    assert(c0 + c1 == 1);

    // Values
    int v0 = w * l;
    int v1 = ((w * (l + 1)) / c) * c;

    return (c0 * v0) + (c1 * v1);
}

int BinPackingLoad::fBJ1(int w, int l, int c)
{
    using namespace Fca;

    // Branch-less fBJ1 transformation

    // Auxiliary values
    int const p = l - (c % l);

    // Conditions
    int const c0 = w % l <= c % l;
    int const c1 = w % l > c % l;

    // Values
    int const v0 = (w / l) * p;
    int const v1 = (w / l) * p + (w % l) - (c % l);

    return (c0 * v0) + (c1 * v1);
}



