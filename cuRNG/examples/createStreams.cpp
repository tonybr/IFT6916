#include <string>
#include <iostream>
#include <stdexcept>

#include "../include/cuRNG/cuRNG.hpp"

void assertCuda(cudaError_t status)
{
    if(status != cudaSuccess)
    {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

template<class T>
void createStreams(std::string type, int numReps, int numStreams, int numBlocks, int threadsPerBlock, uint32_t seed)
{
    cudaEvent_t end;
    cudaEvent_t start;
    
    T* streams_device;
    
    // Allocations
    assertCuda(cudaMalloc((void**)&streams_device, numStreams * sizeof(T)));
    
    cudaEventCreate(&end);
    cudaEventCreate(&start);
    
    StreamCreator<T> streamCreator(seed);
    
    // Prints the parameters
    std::cout << type << " ";
    std::cout << numBlocks << " ";
    std::cout << threadsPerBlock << " ";
    
    for(int i = 0; i < numReps; i++)
    {
        float elapsedTime;
        
        // Creates the streams
        cudaEventRecord(start);
        
        streamCreator.create(streams_device, numStreams, numBlocks, threadsPerBlock);
        
        cudaEventRecord(end);
        cudaEventSynchronize(end);
        
        cudaEventElapsedTime(&elapsedTime, start, end);
        
        // Prints the statistics
        std::cout << elapsedTime << " ";
        
        streamCreator.reset();
    }
    
    std::cout << "\n";
    
    // Deallocations
    assertCuda(cudaFree(streams_device));
    
    cudaEventDestroy(end);
    cudaEventDestroy(start);
}

int main(int argc, char* argv[])
{
    if(argc != 3)
    {
        std::cerr << "\nUsage: createStreams <numRepetitions> <numStreams>\n";
        
        return 1;
    }
    
    int numReps = std::stoi(argv[1]);
    int numStreams = std::stoi(argv[2]);
    
    // Prints the header
    std::cout << "type numBlocks threadsPerBlock";
    
    for(int i = 0; i < numReps; i++)
    {
        std::cout << " elapsedTime" << i;
    }
    
    std::cout << "\n";
    
    // Prints the results
    createStreams<StreamMRG31>("MRG31", numReps, numStreams,  60,   32, 1234);
    createStreams<StreamMRG31>("MRG31", numReps, numStreams,  60,   64, 1234);
    createStreams<StreamMRG31>("MRG31", numReps, numStreams,  80,   32, 1234);
    createStreams<StreamMRG31>("MRG31", numReps, numStreams,  80,   64, 1234);
    createStreams<StreamMRG31>("MRG31", numReps, numStreams,  80,  128, 1234);
    createStreams<StreamMRG31>("MRG31", numReps, numStreams,  80,  256, 1234);
    createStreams<StreamMRG31>("MRG31", numReps, numStreams,  80,  512, 1234);
    createStreams<StreamMRG31>("MRG31", numReps, numStreams,  80, 1024, 1234);
    createStreams<StreamMRG31>("MRG31", numReps, numStreams, 160,  512, 1234);
    createStreams<StreamMRG31>("MRG31", numReps, numStreams, 320,  256, 1234);
    
    createStreams<StreamMWC32>("MWC32", numReps, numStreams,  60,   32, 1234);
    createStreams<StreamMWC32>("MWC32", numReps, numStreams,  60,   64, 1234);
    createStreams<StreamMWC32>("MWC32", numReps, numStreams,  80,   32, 1234);
    createStreams<StreamMWC32>("MWC32", numReps, numStreams,  80,   64, 1234);
    createStreams<StreamMWC32>("MWC32", numReps, numStreams,  80,  128, 1234);
    createStreams<StreamMWC32>("MWC32", numReps, numStreams,  80,  256, 1234);
    createStreams<StreamMWC32>("MWC32", numReps, numStreams,  80,  512, 1234);
    createStreams<StreamMWC32>("MWC32", numReps, numStreams,  80, 1024, 1234);
    createStreams<StreamMWC32>("MWC32", numReps, numStreams, 160,  512, 1234);
    createStreams<StreamMWC32>("MWC32", numReps, numStreams, 320,  256, 1234);
    
    createStreams<StreamXoshiro128PlusPlus>("Xoshiro128++", numReps, numStreams,  60,   32, 1234);
    createStreams<StreamXoshiro128PlusPlus>("Xoshiro128++", numReps, numStreams,  60,   64, 1234);
    createStreams<StreamXoshiro128PlusPlus>("Xoshiro128++", numReps, numStreams,  80,   32, 1234);
    createStreams<StreamXoshiro128PlusPlus>("Xoshiro128++", numReps, numStreams,  80,   64, 1234);
    createStreams<StreamXoshiro128PlusPlus>("Xoshiro128++", numReps, numStreams,  80,  128, 1234);
    createStreams<StreamXoshiro128PlusPlus>("Xoshiro128++", numReps, numStreams,  80,  256, 1234);
    createStreams<StreamXoshiro128PlusPlus>("Xoshiro128++", numReps, numStreams,  80,  512, 1234);
    createStreams<StreamXoshiro128PlusPlus>("Xoshiro128++", numReps, numStreams,  80, 1024, 1234);
    createStreams<StreamXoshiro128PlusPlus>("Xoshiro128++", numReps, numStreams, 160,  512, 1234);
    createStreams<StreamXoshiro128PlusPlus>("Xoshiro128++", numReps, numStreams, 320,  256, 1234);
    
    return 0;
}
