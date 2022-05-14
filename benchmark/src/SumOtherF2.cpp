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
    b.sum("SumXorshift7_A", Xorshift7_A(1234));
    b.sum("SumXorshift7_B", Xorshift7_B(1234));
    
    b.sum("SumXoshiro128Plus_A", Xoshiro128Plus_A(1234));
    b.sum("SumXoshiro128Plus_B", Xoshiro128Plus_B(1234));
    
    b.sum("SumXoshiro128PlusPlus_A", Xoshiro128PlusPlus_A(1234));
    b.sum("SumXoshiro128PlusPlus_B", Xoshiro128PlusPlus_B(1234));
    
    b.sum("SumXoroshiro128Plus", Xoroshiro128Plus(0x4a52b13125e31d57));
    
    b.sum("SumXoshiro256Plus", Xoshiro256Plus(0x4a52b13125e31d57));
    
    b.sum("SumXoshiro256PlusPlus", Xoshiro256PlusPlus(0x4a52b13125e31d57));
    
    cudaDeviceReset();
    
    return 0;
}
