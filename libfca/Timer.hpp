#pragma once

#include <string>
#include <unordered_map>
#include <chrono>

class TimingData
{
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = std::chrono::time_point<Clock>;

    public:
        TimePoint begin_time;
        double total_run_time;
        int count;

    public:
        TimingData() : begin_time(), total_run_time(0.0), count(0) {}
};

class Timer
{
    private:
        static Timer * _timer;
        TimingData program_data;
        std::unordered_map<std::string, TimingData> functions_data;

    public:
        static void begin(std::string const & function_name = "Anonymous Function" );
        static void end(std::string const & function_name   = "Anonymous Function", bool print_elapsed_time = false);
        static std::string summary();

    private:
        Timer();
        static Timer * instance();
};