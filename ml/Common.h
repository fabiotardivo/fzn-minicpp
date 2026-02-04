#pragma once

#include <string>
#include <fstream>
#include <stdexcept>
#include <ranges>

inline
std::ofstream openFile(std::string const & filePath)
{
    if (filePath.empty())
    {
        throw std::invalid_argument("File path is empty");
    }

    std::ofstream file(filePath);
    if (!file.is_open())
    {
        throw std::runtime_error("Could not open file: " + filePath);
    }

    return file;
}

template<typename Var>
void writeBounds(std::vector<Var> const & vars, int const recordSize, std::ostream & outFile)
{
    int const nVars = vars.size();
    int const padding = recordSize - nVars;

    // Minimums
    outFile << vars[0]->min();
    for(auto vIdx = 1; vIdx < nVars; vIdx += 1)
    {
        outFile << "," << vars[vIdx]->min();
    }
    for (auto i = 0; i < padding; i += 1)
    {
        outFile << ",";
    }
    outFile << std::endl;

    // Maximums
    outFile << vars[0]->max();
    for(auto vIdx = 1; vIdx < nVars; vIdx += 1)
    {
        outFile << "," << vars[vIdx]->max();
    }
    for (auto i = 0; i < padding; i += 1)
    {
        outFile << ",";
    }
    outFile << std::endl;
}


class PARecord : public std::vector<float>
{
    public:
        explicit
        PARecord(std::vector<float> const & record) : std::vector<float>(record){}
        PARecord(int size, float value) : std::vector<float>(size, value){}

        std::span<float> getPA()
        {
            assert(size() > 2);
            return std::span<float>(data(), size() -1);
        }

        bool isConsistent() const
        {
            assert(size() > 1);
            return static_cast<bool>(back());
        }

        template<typename Var>
        void from(std::vector<Var> const & pa, bool const isConsistent)
        {
            assert(size() == pa.size() + 1);
            std::copy(pa.begin(), pa.end(), begin());
            back() = static_cast<float>(isConsistent);
        }

        int countAssignedVars()
        {
            assert(size() > 1);
            int const result = static_cast<int>(std::ranges::count_if(getPA(), [](float const & x){return not std::isnan(x);}));
            return result;
        }
};

class USRecord : public std::vector<float>
{
    public:
        explicit
        USRecord(std::vector<float> const & record) : std::vector<float>(record){}
        USRecord(int size, float value) : std::vector<float>(size, value){}

        std::span<float> getPA()
        {
            assert(size() > 1);
            assert(size() % 2 == 0);
            return {data(), size() / 2};
        }

        std::span<float> getUS()
        {
            assert(size() > 1);
            assert(size() % 2 == 0);
            auto const halfSize = size() / 2;
            return {data() + halfSize, halfSize};
        }

        void from(std::span<float> const & pa, std::vector<float> const & us)
        {
            assert(size() == pa.size() + us.size());
            std::copy(pa.begin(), pa.end(), begin());
            std::copy(us.begin(), us.end(), begin() + pa.size());
        }
};

class RecordsBuffer
{
        float * const buffer;
        int const capacity;
        int const recordSize;
        int size;

    public:
        RecordsBuffer(int const capacity, int const recordSize) :
            buffer(new float[capacity * recordSize]),
            capacity(capacity),
            recordSize(recordSize),
            size(0) {}

        ~RecordsBuffer() { delete[] buffer; }

    private:
        std::span<float> getRecord(int const rIdx) const
        {
            assert(rIdx <= size); // Include append case
            std::span<float> record(buffer + (rIdx * recordSize), recordSize);
            return record;
        }

        static
        void dump(float const x, std::ostream & outStream)
        {
            if (not std::isnan(x))
            {
                outStream << static_cast<int>(x);
            }
        }

        void dump(std::ostream & outStream)
        {
            for (auto rIdx = 0; rIdx < size; rIdx += 1)
            {
                auto const & record = getRecord(rIdx);
                dump(record[0], outStream);
                for(auto i = 1; i < recordSize; i += 1)
                {
                    outStream << ",";
                    dump(record[i], outStream);
                }
                outStream << std::endl;
            }
        }

        bool isFull() const
        {
            return size >= capacity;
        }

    public:
        void add(std::vector<float> const & record)
        {
            assert(not isFull());
            assert(record.size() == recordSize);
            std::copy(record.begin(), record.end(), getRecord(size).begin());
            size += 1;
        }

        void dump(std::mutex & outMutex, std::ostream & outStream)
        {
            std::lock_guard<std::mutex> lock(outMutex);
            dump(outStream);
            std::flush(outStream);
        }

        void safeAdd(std::vector<float> const & record, std::mutex & outMutex, std::ostream & outStream)
        {
            if (isFull())
            {
               dump(outMutex, outStream);
            }
            add(record);
        }
};