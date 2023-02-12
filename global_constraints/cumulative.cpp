
#include "cumulative.hpp"

Cumulative::Cumulative(std::vector<var<int>::Ptr> & s, std::vector<int> const & p, std::vector<int> const & h, int c) :
    Constraint(s[0]->getSolver()), nActivities(s.size()), c(c), s(s), p(p), h(h), si(s.size())
{
    setPriority(CLOW);
}

void Cumulative::post()
{
    intervals.reserve(MAX_INTERVALS_PER_ACTIVITY_PAIR * nActivities * nActivities);

    for (auto const & v : s)
    {
        v->propagateOnBoundChange(this);
    }
}

void Cumulative::initStartIntervals(int nActivities, var<int>::Ptr const * s, StartInterval * si)
{
    for (auto i = 0; i < nActivities; i += 1)
    {
        auto const & v = s[i];
        si[i] = {v->changed(), v->min(), v->max()};
    }
}

void Cumulative::calcIntervals()
{

    intervals.clear();

    Interval intervalsToTest[MAX_INTERVALS_PER_ACTIVITY_PAIR];
    for (auto i = 0; i < nActivities; i += 1)
    {
        int const pi = p.at(i);
        bool siChanged = si.at(i).changed;
        int const siMin = si.at(i).min;
        int const siMax = si.at(i).max;
        int const eiMax = siMax + pi;

        for (int j = 0; j < nActivities; j += 1)
        {
            bool sjChanged = si.at(j).changed;
            //if (siChanged or sjChanged)
            {
                int const pj = p.at(j);
                int const sjMin = si.at(j).min;
                int const sjMax = si.at(j).max;
                int const ejMin = sjMin + pj;
                int const ejMax = sjMax + pj;

                // Case 1
                intervalsToTest[0] = {siMin, ejMin};
                intervalsToTest[1] = {siMin, ejMax};
                intervalsToTest[2] = {siMax, ejMin};
                intervalsToTest[3] = {siMax, ejMax};

                // Case 2
                intervalsToTest[4] = {siMin, sjMin + ejMax - siMin};
                intervalsToTest[5] = {siMin, sjMin + ejMax - siMax};
                intervalsToTest[6] = {siMax, sjMin + ejMax - siMin};
                intervalsToTest[7] = {siMax, sjMin + ejMax - siMax};

                // Case 3
                intervalsToTest[8] = {siMin + eiMax - ejMin, ejMin};
                intervalsToTest[9] = {siMin + eiMax - ejMin, ejMax};
                intervalsToTest[10] = {siMin + eiMax - ejMax, ejMin};
                intervalsToTest[11] = {siMin + eiMax - ejMax, ejMax};

                for (auto k = 0; k < MAX_INTERVALS_PER_ACTIVITY_PAIR; k += 1)
                {
                    if (intervalsToTest[k].t1 < intervalsToTest[k].t2)
                    {
                        intervals.push_back(intervalsToTest[k]);
                    }
                }
            }
        }
    }
}

void Cumulative::propagate()
{
    using namespace std;

    initStartIntervals(nActivities, s.data(), si.data());
    calcIntervals();

    auto const nIntervals = intervals.size();
    for (auto j = 0; j < nIntervals; j +=1)
    {
        int const t1 = intervals.at(j).t1;
        int const t2 = intervals.at(j).t2;

        int w = 0;
        for (auto a = 0; a < nActivities; a += 1)
        {
            int const ha = h[a];
            int const saMin = s.at(a)->min();
            int const saMax = s.at(a)->max();
            int const pa = p[a];
            int const eaMin = saMin + pa;
            int const eaMax = saMax + pa;
            int const ls = max(0, min(eaMin, t2) - max(saMin, t1));
            int const rs = max(0, min(eaMax, t2) - max(saMax, t1));
            int const mi = min(ls, rs);
            w += ha * mi;
        }

        if (w <= c * (t2 - t1))
        {
            for (auto a = 0; a < nActivities; a += 1)
            {
                int const ha = h[a];
                int const saMin = s.at(a)->min();
                int const saMax = s.at(a)->max();
                int const pa = p[a];
                int const eaMin = saMin + pa;
                int const eaMax = saMax + pa;
                int const ls = max(0, min(eaMin, t2) - max(saMin, t1));
                int const rs = max(0, min(eaMax, t2) - max(saMax, t1));
                int const mi = min(ls, rs);
                int const avail = c * (t2 - t1) - w + ha * mi;
                if (avail < ha * ls)
                {
                    si.at(a).min = max(si.at(a).min, t2 - (avail / ha));
                }
                if (avail < ha * rs)
                {
                    si.at(a).max = min(si.at(a).max, t1 + (avail / ha) - pa);
                }
            }
        }
        else
        {
            failNow();
        }
    }

    for (auto i = 0; i < nActivities; i += 1)
    {
        s.at(i)->removeBelow(si.at(i).min);
        s.at(i)->removeAbove(si.at(i).max);
    }
}