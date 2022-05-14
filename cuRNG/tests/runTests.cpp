#include <cstdint>
#include <iostream>
#include <cassert>

#include "../include/cuRNG/cuRNG.hpp"

template<class T>
void testGenerateUniform01(float expectedVariate, T streams[], uint32_t numStreams, int numVariates)
{
    T streamHost;
    float resultHost;
    
    float resultDevice;
    float* resultsDevice;
    
    cudaMalloc((void**)&resultsDevice, numVariates * sizeof(float));
    
    cudaMemcpy(&streamHost, streams + numStreams - 1, sizeof(T), cudaMemcpyDeviceToHost);
    
    // Generates the uniform variates
    generateUniform01(streams, resultsDevice, numStreams, numVariates, 1, 1024);
    
    // Transfers the last uniform variates onto the host
    cudaMemcpy(&resultDevice, resultsDevice + numVariates - 1, sizeof(float), cudaMemcpyDeviceToHost);
    
    for(int i = 0; i < numVariates / numStreams; i++)
    {
        resultHost = streamHost.nextUniform01();
    }
    
    std::cout << "Expected: " << expectedVariate << ". Host: " << resultHost << ". Device: " << resultDevice << ".\n";
    
    // Compares the results
    assert(resultHost == expectedVariate);
    assert(resultDevice == expectedVariate);
    
    cudaFree(resultsDevice);
}

template<class T>
void testStreamCreator(float expectedVariates[3])
{
    T* streamsDevice;
    
    cudaMalloc((void**)&streamsDevice, 512 * sizeof(T));
    
    StreamCreator<T> streamCreator(1234, 2);
    
    // Creates 512 streams
    streamCreator.create(streamsDevice, 512, 1, 128);
    
    // Tests the first two uniform variates of the last stream
    testGenerateUniform01<T>(expectedVariates[0], streamsDevice, 512, 512);
    testGenerateUniform01<T>(expectedVariates[1], streamsDevice, 512, 1024);
    
    // Creates another 512 streams
    streamCreator.create(streamsDevice, 512, 1, 128);
    
    // Tests the first uniform variate of the last stream
    testGenerateUniform01<T>(expectedVariates[2], streamsDevice, 512, 512);
    
    // Resets the stream creator
    streamCreator.reset();
    
    // Re-creates the first 512 streams
    streamCreator.create(streamsDevice, 512, 1, 128);
    
    // Tests the first uniform variate of the last stream again
    testGenerateUniform01<T>(expectedVariates[0], streamsDevice, 512, 512);
    
    cudaFree(streamsDevice);
}

int main(int argc, char* argv[])
{
    float expectedVariates[3][3] = {{161337908 * 4.65661287308e-10f, 590232186 * 4.65661287308e-10f, 22227907 * 4.65661287308e-10f},
                                    {2684418972 * 2.32830643654e-10f, 2967496468 * 2.32830643654e-10f, 2814735826 * 2.32830643654e-10f},
                                    {724277888 * 2.32830643654e-10f, 2052166183 * 2.32830643654e-10f, 1776076970 * 2.32830643654e-10f}};
    
    std::cout << "MRG31\n";
    testStreamCreator<StreamMRG31>(expectedVariates[0]);
    
    std::cout << "\nMWC32\n";
    testStreamCreator<StreamMWC32>(expectedVariates[1]);
    
    std::cout << "\nXoshiro128++\n";
    testStreamCreator<StreamXoshiro128PlusPlus>(expectedVariates[2]);
    
    std::cout << "\nTests were successful.\n";
    
    return 0;
}
