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
    b.sum("SumLFSR88_A", LFSR88_A(1234));
    b.sum("SumLFSR88_B", LFSR88_B(1234));
    b.sum("SumLFSR88_C", LFSR88_C(1234));
    b.sum("SumLFSR88_D", LFSR88_D(1234));
    
    b.sum("SumLFSR113_A", LFSR113_A(1234));
    b.sum("SumLFSR113_B", LFSR113_B(1234));
    b.sum("SumLFSR113_C", LFSR113_C(1234));
    b.sum("SumLFSR113_D", LFSR113_D(1234));
    b.sum("SumLFSR113_E", LFSR113_E(1234));
    
    b.sum("SumLFSR258_A", LFSR258_A(1234));
    b.sum("SumLFSR258_B", LFSR258_B(1234));
    b.sum("SumLFSR258_C", LFSR258_C(1234));
    b.sum("SumLFSR258_D", LFSR258_D(1234));
    
    cudaDeviceReset();
    
    return 0;
}
