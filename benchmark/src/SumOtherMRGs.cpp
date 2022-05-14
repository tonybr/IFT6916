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
    b.sum("SumMRG31a_C", MRG31a_C(1234));
    b.sum("SumMRG31b_A", MRG31b_A(1234));
    b.sum("SumMRG31b_C", MRG31b_C(1234));
    b.sum("SumMRG31c_A", MRG31c_A(1234));
    b.sum("SumMRG31c_C", MRG31c_C(1234));
    b.sum("SumMRG31d_C", MRG31d_C(1234));
    b.sum("SumMRG31e_A", MRG31e_A(1234));
    b.sum("SumMRG31e_C", MRG31e_C(1234));
    b.sum("SumMRG31e_J", MRG31e_J(1234));
    b.sum("SumMRG31e_K", MRG31e_K(1234));
    b.sum("SumMRG31e_L", MRG31e_L(1234));
    
    cudaDeviceReset();
    
    return 0;
}
