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
    b.sum("SumMRG31k3p_A", MRG31k3p_A(1234));
    b.sum("SumMRG31k3p_B", MRG31k3p_B(1234));
    b.sum("SumMRG31k3p_C", MRG31k3p_C(1234));
    b.sum("SumMRG31k3p_D", MRG31k3p_D(1234));
    
    b.sum("SumMRG32k3a_B", MRG32k3a_B(1234));
    b.sum("SumMRG32k3a_C", MRG32k3a_C(1234));
    b.sum("SumMRG32k3a_A", MRG32k3a_A(1234));
    
    cudaDeviceReset();
    
    return 0;
}
