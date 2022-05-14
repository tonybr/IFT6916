#include <string>
#include <iostream>

#include "include/benchmark.hpp"
#include "include/generators.hpp"

int main(int argc, char* argv[])
{
    if(argc != 6)
    {
        std::cerr << "\nUsage: benchmark <numRepetitions> <threadsPerBlock> <numBlocks> <deviceId> <printHeader>\n";
        
        return 1;
    }
    
    // Creates the benchmark
    Benchmark b(std::stoi(argv[1]), std::stoi(argv[2]), std::stoi(argv[3]));
    
    b.setDevice(std::stoi(argv[4]));
    
    if(argv[5][0] == 'T')
    {
        b.printHeader();
    }
    
    // Runs the tests
    b.sum("SumMWC32a_A", MWC32a_A(1234));
    b.sum("SumMWC32a_B", MWC32a_B(1234));
    b.sum("SumMWC32b_A", MWC32b_A(1234));
    b.sum("SumMWC32b_B", MWC32b_B(1234));
    b.sum("SumMWC32c_A", MWC32c_A(1234));
    
    b.sum("SumMWC31_A", MWC31_A(1234));
    b.sum("SumMWC31_B", MWC31_B(1234));
    
    cudaDeviceReset();
    
    return 0;
}
