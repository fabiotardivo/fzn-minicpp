#pragma once

#include <libminicpp/varitf.hpp>

#define MAX_INTERVALS_PER_ACTIVITY_PAIR 12

// References:
// - Constraint-Based Scheduling (ISBN: 978-1-4615-1479-4)
// - A New Characterization of Relevant Intervals for Energetic Reasoning (DOI: 10.1007/978-3-319-10428-7_22)

class Cumulative : public Constraint
{
    public:
        struct Interval
        {
            int t1;
            int t2;
        };
        struct StartInterval
        {
            bool changed;
            int min; // Earliest Start Time
            int max; // Latest Start Time
        };

    protected:
        int const  nActivities;
        int const c; // Capacity
        std::vector<var<int>::Ptr> const s;
        std::vector<StartInterval> si;
        std::vector<int> const p; // Processing time
        std::vector<int> const h; // Height
        std::vector<Interval> intervals;

    public:
        Cumulative(std::vector<var<int>::Ptr> & s, std::vector<int> const & p, std::vector<int> const & h, int c);
        void post() override;
        void propagate() override;
    protected:
        static void initStartIntervals(int nActivities, var<int>::Ptr const * s, StartInterval * si);
    private:
        void calcIntervals();
};


