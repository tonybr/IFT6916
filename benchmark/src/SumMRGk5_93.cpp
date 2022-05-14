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
    b.sum("SumMRGk5_93_A", MRGk5_93_A(1234));
    b.sum("SumMRGk5_93_B", MRGk5_93_B(1234));
    b.sum("SumMRGk5_93_C", MRGk5_93_C(1234));
    b.sum("SumMRGk5_93_D", MRGk5_93_D(1234));
    b.sum("SumMRGk5_93_E", MRGk5_93_E(1234));
    b.sum("SumMRGk5_93_F", MRGk5_93_F(1234));
    b.sum("SumMRGk5_93_G", MRGk5_93_G(1234));
    b.sum("SumMRGk5_93_H", MRGk5_93_H(1234));
    b.sum("SumMRGk5_93_I", MRGk5_93_I(1234));
    
    cudaDeviceReset();
    
    return 0;
}
