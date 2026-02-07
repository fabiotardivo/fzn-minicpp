#pragma once

#include <string>
#include <fstream>
#include <stdexcept>
#include <ranges>

static constexpr int UNASSIGNED_VALUE = INT_MIN;

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
    std::flush(outFile);
}

inline
void printInt(int const x, std::ostream & outStream)
{
    if (x != UNASSIGNED_VALUE)
    {
        outStream << x;
    }
}

inline
void printInts(std::span<int> const s, std::ostream & outStream)
{
    int const size = static_cast<int>(s.size());
    printInt(s[0], outStream);
    for (auto vIdx = 1; vIdx < size; vIdx += 1)
    {
        outStream << ",";
        printInt(s[vIdx], outStream);
    }
}

class PARecord : public std::vector<int>
{
    public:
        explicit
        PARecord(std::vector<int> const & record) : std::vector<int>(record){}
        PARecord(int size, int value) : std::vector<int>(size, value){}

        std::span<int> getPA()
        {
            assert(size() > 2);
            return std::span<int>(data(), size() -1);
        }

        bool isInconsistent() const
        {
            assert(size() > 1);
            return static_cast<bool>(back());
        }

        template<typename Var>
        void from(std::vector<Var> const & pa, bool const isInconsistent)
        {
            assert(size() == pa.size() + 1);
            for(auto vIdx = 0; vIdx < pa.size(); vIdx += 1)
            {
                at(vIdx) = pa[vIdx]->isBound() ? pa[vIdx]->min() : UNASSIGNED_VALUE;
            }
            back() = static_cast<int>(isInconsistent);
        }

        static
        void printPA(std::span<int> const & pa, std::ostream & outStream)
        {
            outStream << "PA = ";
            printInts(pa, outStream);
        }

        void print(std::ostream & outStream)
        {
            printPA(getPA(), outStream);
            outStream << " | ";
            outStream << "INC = " << isInconsistent() << std::endl;
        }

        int countAssignedVars()
        {
            assert(size() > 1);
            int const result = static_cast<int>(std::ranges::count_if(getPA(), [](int const & x){return x != UNASSIGNED_VALUE;}));
            return result;
        }
};

class USRecord : public std::vector<int>
{
    public:
        explicit
        USRecord(std::vector<int> const & record) : std::vector<int>(record){}
        USRecord(int size, int value) : std::vector<int>(size, value){}

        std::span<int> getPA()
        {
            assert(size() > 1);
            assert(size() % 2 == 0);
            return {data(), size() / 2};
        }

        std::span<int> getUS()
        {
            assert(size() > 1);
            assert(size() % 2 == 0);
            auto const halfSize = size() / 2;
            return {data() + halfSize, halfSize};
        }

        void from(std::span<int> const & pa, std::vector<int> const & us)
        {
            assert(size() == pa.size() + us.size());
            std::memcpy(data(), pa.data(), pa.size() * sizeof(int));
            std::memcpy(data() + pa.size(), us.data(), us.size() * sizeof(int));
        }

        static
        void printPA(std::span<int> const & pa, std::ostream & outStream)
        {
            outStream << "PA = ";
            printInts(pa, outStream);
        }

        static
        void printUS(std::span<int> const & us, std::ostream & outStream)
        {
            outStream << "US = ";
            printInts(us, outStream);
        }

        void print(std::ostream & outStream)
        {
            printPA(getPA(), outStream);
            outStream << " | ";
            printUS(getUS(), outStream);
            outStream << std::endl;
        }
};

class RecordsBuffer
{
        int * const buffer;
        int const capacity;
        int const recordSize;
        int size;

    public:
        RecordsBuffer(int const capacity, int const recordSize) :
            buffer(new int[capacity * recordSize]),
            capacity(capacity),
            recordSize(recordSize),
            size(0) {}

        ~RecordsBuffer() { delete[] buffer; }

    private:
        std::span<int> getRecord(int const rIdx) const
        {
            assert(rIdx <= size); // Include append case
            std::span<int> record(buffer + (rIdx * recordSize), recordSize);
            return record;
        }

        void dump(std::ostream & outStream)
        {
            for (auto rIdx = 0; rIdx < size; rIdx += 1)
            {
                auto const & record = getRecord(rIdx);
                printInts(record, outStream);
                outStream << std::endl;
            }
            std::flush(outStream);
            size = 0;
        }

        bool isFull() const
        {
            return size >= capacity;
        }

    public:
        void add(std::vector<int> const & record)
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

        void safeAdd(std::vector<int> const & record, std::mutex & outMutex, std::ostream & outStream)
        {
            if (isFull())
            {
               dump(outMutex, outStream);
            }
            add(record);
        }
};