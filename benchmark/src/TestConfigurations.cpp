#include <string>
#include <iostream>

#include "include/benchmark.hpp"
#include "include/generators.hpp"

#include "cuda_runtime_api.h"

int getCoresPerSM(cudaDeviceProp p)
{
    int coresPerSM = 0;
    
    switch(p.major)
    {
    case 3:
        coresPerSM = 192;
        break;
        
    case 5:
        coresPerSM = 128;
        break;
        
    case 6:
        if(p.minor == 1 || p.minor == 2)
            coresPerSM = 128;
        else if(p.minor == 0)
            coresPerSM = 64;
        break;
        
    case 7:
        if(p.minor == 0 || p.minor == 5)
            coresPerSM = 64;
        break;
        
    case 8:
        if(p.minor == 0)
            coresPerSM = 64;
        else if(p.minor == 6)
            coresPerSM = 128;
        break;
    }
    
    return coresPerSM;
}

void testConfiguration(int numReps, int deviceId, int threadsPerBlock, int numBlocks)
{
    Benchmark b(numReps, threadsPerBlock, numBlocks);
    
    b.setDevice(deviceId);
    
    b.sum("SumMRG31c_C", MRG31c_C(1234));
    b.sum("SumXoshiro128Plus_A", Xoshiro128Plus_A(1234));
}

int main(int argc, char* argv[])
{
    if(argc != 3)
    {
        std::cerr << "\nUsage: benchmark <numRepetitions> <deviceId>\n";
        
        return 1;
    }
    
    int numReps = std::stoi(argv[1]);
    int deviceId = std::stoi(argv[2]);
    
    // Prints the properties of the device
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, deviceId);
    
    int numSMs = p.multiProcessorCount;
    
    std::cout << "numSMs= " << numSMs;
    std::cout << " majorVersion= " << p.major;
    std::cout << " minorVersion= " << p.minor;
    std::cout << " coresPerSM= " << getCoresPerSM(p);
    std::cout << "\n";
    
    // Runs the tests
    testConfiguration(numReps, deviceId, 128, 8 * numSMs);
    
    testConfiguration(numReps, deviceId, 512, numSMs);
    testConfiguration(numReps, deviceId, 512, 2 * numSMs);
    
    testConfiguration(numReps, deviceId, 1024, numSMs);
    testConfiguration(numReps, deviceId, 1024, 2 * numSMs);
    testConfiguration(numReps, deviceId, 1024, 3 * numSMs);
    
    cudaDeviceReset();
    
    return 0;
}
