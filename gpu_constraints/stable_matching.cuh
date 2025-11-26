#pragma once

#include <varitf.hpp>


class StableMatchingGPU : public Constraint
{
    // Constraint private data structures
    protected:
        std::vector<var<int>::Ptr> _x;
        std::vector<var<int>::Ptr> _y;
        std::vector<std::vector<int>> const _xpl_vector;
        std::vector<std::vector<int>> const _ypl_vector;
        int _n;
        std::vector<trail<int>> _old_min_men_trail;
        std::vector<trail<int>> _old_max_men_trail;
        std::vector<trail<int>> _old_min_women_trail;
        std::vector<trail<int>> _old_max_women_trail;
        cudaStream_t _stream;
        int _n_SMP;
        trail<int> _propagation_counter_trail;
        int _propagation_counter;
        std::vector<trail<int>> _x_old_sizes;
        std::vector<trail<int>> _y_old_sizes;
        //Host pointers
        uint32_t *_x_domain, *_y_domain;
        int *_xpl, *_ypl;
        int *_xPy, *_yPx;
        int *_stack_mod_men, *_stack_mod_women, *_stack_mod_min_men;
        int *_old_min_men, *_old_max_men, *_old_min_women, *_old_max_women;
        int *_max_men, *_min_women, *_max_women;
        int *_length_min_men_stack, *_new_length_min_men_stack;
        //Device pointers
        uint32_t *_d_x_domain, *_d_y_domain;
        int *_d_xpl, *_d_ypl;
        int *_d_xPy, *_d_yPx;
        int *_d_stack_mod_men, *_d_stack_mod_women, *_d_stack_mod_min_men;
        int *_d_old_min_men, *_d_old_max_men, *_d_old_min_women, *_d_old_max_women;
        int *_d_max_men, *_d_min_women, *_d_max_women;
        int *_d_length_min_men_stack, *_d_new_length_min_men_stack;
        int *_d_new_stack_mod_min_men, *_d_array_min_mod_men;


    // Constraint methods
    public:
        StableMatchingGPU(std::vector<var<int>::Ptr> & m, std::vector<var<int>::Ptr> & w, std::vector<std::vector<int>> const & pm, std::vector<std::vector<int>> const & pw);
        void post() override;
        void propagate() override;

    protected:
        void buildReverseMatrix(std::vector<std::vector<int>> zpl, int *zPz);
        void copyPreferenceMatrix(std::vector<std::vector<int>> zpl, int *zPz);
        void dumpDomainsToBitset(std::vector<var<int>::Ptr> vars, uint32_t* dom, int* old_mins, int* old_maxes, std::vector<trail<int>> old_sizes, int* min_person, int* max_person);
        int getBitHost(uint32_t* bitmap, int index);
        int getDomainBitHost(uint32_t* bitmap, int row, int column);
        void updateHostData();
        void getBlockNumberAndDimension(int n_threads, int *block_size, int *n_blocks);
        void iterateFun2();
        void fillStacks(int* _length_men_stack, int* _length_women_stack);
};


