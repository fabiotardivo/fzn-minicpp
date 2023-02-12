#include "Timer.hpp"

#include <sstream>
#include <string>
#include <iostream>
#include <unordered_map>

using namespace std;
using namespace std::chrono;

Timer* Timer::_timer = nullptr;

Timer::Timer()
{
    program_data.begin_time = high_resolution_clock::now();
    program_data.count = 1;
}

Timer * Timer::instance()
{
    if (_timer == nullptr)
    {
        _timer = new Timer();
    }
    return _timer;
}

void Timer::begin(std::string const & func_name)
{
	auto it = instance()->functions_data.find(func_name);
	if (it == instance()->functions_data.end())
    {
        TimingData timing_data;
		instance()->functions_data.insert(pair<std::string, TimingData>(func_name, timing_data));
		it = instance()->functions_data.find(func_name);
	}
	it->second.begin_time = high_resolution_clock::now();
}

void Timer::end(std::string const & func_name, bool print_elapsed_time)
{
	auto it = instance()->functions_data.find(func_name);
	if (it != instance()->functions_data.end())
    {
        duration<double, std::milli> elapsed_time = high_resolution_clock::now() - it->second.begin_time;
		it->second.total_run_time += elapsed_time.count();
		it->second.count++;
        if (print_elapsed_time)
        {
            std::cout << func_name << ": " << elapsed_time.count() << "ms" << std::endl;
        }
	}
    else
    {
		std::cerr << "Error: This function '" << func_name << "' is not defined" << std::endl;
	}
}

std::string Timer::summary()
{
    duration<double, std::milli> elapsed_time = std::chrono::high_resolution_clock::now() - Timer::instance()->program_data.begin_time;;
	Timer::instance()->program_data.total_run_time = elapsed_time.count();

	std::stringstream ss;

	static const string func_name      = "Function Name";
	static const int func_name_size    = max((int) func_name.length(), 30);

	static const string total_time     = "Total";
	static const int total_time_size   = max((int) total_time.length(), 15);

    static const string average_time   = "Average";
    static const int average_time_size = max((int) average_time.length(),15);

	static const string called_times   = "Be Called";
	const int called_times_size        = max((int) called_times.length(), 15);

	static const string percentage     = "Percentage";
	const int percentage_size          = max((int)percentage.length(), 15);

	ss << "%% +-------------------" << endl;
	ss << "%% | Profiling Summery" << endl;
	ss << "%% +-------------------" << endl;
	ss << "%% | ";
	ss.width(func_name_size);
	ss << std::left << func_name << " |";
	ss.width( average_time_size );
	ss << std::right << average_time << " |";
    ss.width( total_time_size );
    ss << std::right << total_time << " |";
	ss.width( called_times_size );
	ss << std::right << called_times << " |";
	ss.width( percentage_size );
	ss << std::right << percentage << " |" << std::endl;

	double total_running_time = Timer::instance()->program_data.total_run_time;

	ss << "%% | ";
	ss.width( func_name_size );
	ss << std::left << "Total Run Time" << " |";
	ss.width( average_time_size - sizeof("ms") );
	ss << std::right << total_running_time << " ms |";
    ss.width( total_time_size - sizeof("ms") );
    ss << std::right << total_running_time << " ms |";
	ss.width( called_times_size - sizeof("times") );
	ss << std::right << 1 << " times |";
	ss.width( percentage_size - sizeof("%") );
	ss << std::right << 100.000 << " % |" << endl;

	for (auto it = instance()->functions_data.begin(); it != instance()->functions_data.end(); it++)
    {
		ss << "%% | ";
		ss.width( func_name_size );
		ss << std::left << it->first << " |";
        ss.width( average_time_size - sizeof("ms") );
        ss << std::right<< it->second.total_run_time / it->second.count << " ms |";
		ss.width( total_time_size - sizeof("ms") );
		ss << std::right<< it->second.total_run_time << " ms |";
		ss.width( called_times_size - sizeof("times") );
		ss << std::right<< it->second.count << " times |";
		ss.width( percentage_size - sizeof("%") );
		ss << std::right << int(100000 * it->second.total_run_time / total_running_time ) / 1000.0 << " % |" << endl;
	}

	return ss.str();
}