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
    b.sum("SumPhilox_4_32_10", Philox_4_32_10(0xb5ad4eceda1ce2a9));
    
    b.sum("SumSquares", Squares(0x548c9decbce65297));
    
    b.sum("SumMSWS", MSWS(0xb5ad4eceda1ce2a9));
    
    cudaDeviceReset();
    
    return 0;
}
