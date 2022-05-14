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

void printStatistics(cudaEvent_t& start, cudaEvent_t& end, float results[], int length)
{
    float elapsedTime;
    
    double variance = 0;
    double sum = 0;
    
    // Calculates the elapsed time
    cudaEventElapsedTime(&elapsedTime, start, end);
    
    // Calculates the variance
    for(int i = 0; i < length; i++)
    {
        variance += results[i] * results[i];
        sum += results[i];
    }
    
    variance = (variance - sum * sum / length) / (length - 1);
    
    // Calculates the average
    double average = sum / length;
    
    // Prints the statistics
    std::cout << elapsedTime << " ";
    std::cout << variance << " ";
    std::cout << average << " ";
}

/* Kernels */

template<class T>
__device__ float simulateInventory(T& streamSales, T& streamShipment, int numDays)
{
    int numItems = 600;
    float sumProfit = -6000;
    
    #ifdef UNROLL_1
        #pragma unroll 1
    #endif
    for (int i = 0; i < numDays; i++)
    {
        // Number of sales
        int numSales = min((int)(streamSales.nextUniform01() * 31), numItems);
        
        numItems -= numSales;
        
        // Revenues from the sales
        sumProfit += numSales * 39.95f;
        
        // Restocking of the inventory
        if (numItems < 200 && streamShipment.nextUniform01() < 0.70f)
        {
            // Cost of goods
            sumProfit -= (600 - numItems) * 10;
            
            numItems = 600;
        }
    }
    
    return sumProfit / numDays;
}

// Case A: 2 streams per simulation, from global memory
template<class T>
__global__ void simulateInventoryCaseA(T streams[], float results[], int numSims, int numDays)
{
    int numThreads = gridDim.x * blockDim.x;
    int resultIndex = blockIdx.x * blockDim.x + threadIdx.x;
    
    #ifdef UNROLL_1
        #pragma unroll 1
    #endif
    while(resultIndex < numSims)
    {
        // Copies the streams to register memory
        T streamSales = streams[2 * resultIndex];
        T streamShipment = streams[2 * resultIndex + 1];
        
        // Simulates the inventory
        results[resultIndex] = simulateInventory(streamSales, streamShipment, numDays);
        
        resultIndex += numThreads;
    }
}

// Case B: 2 streams per thread, from global memory
template<class T>
__global__ void simulateInventoryCaseB(T streams[], float results[], int numSims, int numDays)
{
    int numThreads = gridDim.x * blockDim.x;
    int resultIndex = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Copies the streams to register memory
    T streamSales = streams[2 * resultIndex];
    T streamShipment = streams[2 * resultIndex + 1];
    
    #ifdef UNROLL_1
        #pragma unroll 1
    #endif
    while(resultIndex < numSims)
    {
        // Simulates the inventory
        results[resultIndex] = simulateInventory(streamSales, streamShipment, numDays);
        
        resultIndex += numThreads;
        
        // Advances to the next substreams
        streamSales.nextSubstream();
        streamShipment.nextSubstream();
    }
}

// Case C: 2 streams per thread, created locally
template<class T>
__global__ void simulateInventoryCaseC(uint32_t seed, float results[], int numSims, int numDays)
{
    T streams[2];
    
    int numThreads = gridDim.x * blockDim.x;
    int resultIndex = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Creates the streams locally
    createStreams(streams, 2, seed, 2 * resultIndex);
    
    #ifdef UNROLL_1
        #pragma unroll 1
    #endif
    while(resultIndex < numSims)
    {
        // Simulates the inventory
        results[resultIndex] = simulateInventory(streams[0], streams[1], numDays);
        
        resultIndex += numThreads;
        
        // Advances to the next substreams
        streams[0].nextSubstream();
        streams[1].nextSubstream();
    }
}

/* Proxies */

template<class T, class P>
void runCase(T kernel, P streams, float results[], int numSims, int numDays, int numBlocks, int threadsPerBlock)
{
    cudaEvent_t end;
    cudaEvent_t start;
    
    // Allocations
    float* results_host = new float[numSims];
    
    cudaEventCreate(&end);
    cudaEventCreate(&start);
    
    // Runs the simulation
    cudaEventRecord(start);
    
    (*kernel) <<< numBlocks, threadsPerBlock >>> (streams, results, numSims, numDays);
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    assertCuda(cudaMemcpy(results_host, results, numSims * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Prints the statistics
    printStatistics(start, end, results_host, numSims);
    
    // Deallocations
    delete[] results_host;
    
    cudaEventDestroy(end);
    cudaEventDestroy(start);
}

template<class T>
void runSimulation(std::string type, int numReps, int numSims, int numDays, int numBlocks, int threadsPerBlock, uint32_t seed)
{
    T* streams_device;
    float* results_device;
    
    // Allocations
    int numThreads = numBlocks * threadsPerBlock;
    
    assertCuda(cudaMalloc((void**)&streams_device, 2 * numSims * sizeof(T)));
    assertCuda(cudaMalloc((void**)&results_device, numSims * sizeof(float)));
    
    StreamCreator<T> streamCreator(seed);
    
    // Prints the parameters
    std::cout << type << " ";
    std::cout << numBlocks << " ";
    std::cout << threadsPerBlock << " ";
    
    // Case A
    for(int i = 0; i < numReps; i++)
    {
        // Creates the streams
        streamCreator.create(streams_device, 2 * numSims, numBlocks, threadsPerBlock);
        
        // Runs the simulation
        runCase(simulateInventoryCaseA<T>, streams_device, results_device, numSims, numDays, numBlocks, threadsPerBlock);
        
        streamCreator.reset();
    }
    
    // Case B
    for(int i = 0; i < numReps; i++)
    {
        streamCreator.create(streams_device, 2 * numThreads, numBlocks, threadsPerBlock);
        
        runCase(simulateInventoryCaseB<T>, streams_device, results_device, numSims, numDays, numBlocks, threadsPerBlock);
        
        streamCreator.reset();
    }
    
    // Case C
    for(int i = 0; i < numReps; i++)
    {
        runCase(simulateInventoryCaseC<T>, seed, results_device, numSims, numDays, numBlocks, threadsPerBlock);
    }
    
    std::cout << "\n";
    
    // Deallocations
    assertCuda(cudaFree(streams_device));
    assertCuda(cudaFree(results_device));
}

int main(int argc, char* argv[])
{
    if(argc != 4)
    {
        std::cerr << "\nUsage: simulateInventory <numRepetitions> <numSimulations> <numDays>\n";
        
        return 1;
    }
    
    int numReps = std::stoi(argv[1]);
    int numSims = std::stoi(argv[2]);
    int numDays = std::stoi(argv[3]);
    
    // Prints the header
    std::cout << "type numBlocks threadsPerBlock";
    
    for(int i = 1; i <= 3; i++)
    {
        for(int j = 0; j < numReps; j++)
        {
            std::cout << " elapsedTime" << i << j;
            std::cout << " variance" << i << j;
            std::cout << " average" << i << j;
        }
    }
    
    std::cout << "\n";
    
    // Prints the results
    runSimulation<StreamMRG31>("MRG31", numReps, numSims, numDays,  80,   64, 1234);
    runSimulation<StreamMRG31>("MRG31", numReps, numSims, numDays,  80,  128, 1234);
    runSimulation<StreamMRG31>("MRG31", numReps, numSims, numDays,  80,  256, 1234);
    runSimulation<StreamMRG31>("MRG31", numReps, numSims, numDays,  80,  512, 1234);
    runSimulation<StreamMRG31>("MRG31", numReps, numSims, numDays,  80, 1024, 1234);
    runSimulation<StreamMRG31>("MRG31", numReps, numSims, numDays, 160,  512, 1234);
    runSimulation<StreamMRG31>("MRG31", numReps, numSims, numDays, 160, 1024, 1234);
    runSimulation<StreamMRG31>("MRG31", numReps, numSims, numDays, 320,  256, 1234);
    runSimulation<StreamMRG31>("MRG31", numReps, numSims, numDays, 320,  512, 1234);
    
    runSimulation<StreamMWC32>("MWC32", numReps, numSims, numDays,  80,   64, 1234);
    runSimulation<StreamMWC32>("MWC32", numReps, numSims, numDays,  80,  128, 1234);
    runSimulation<StreamMWC32>("MWC32", numReps, numSims, numDays,  80,  256, 1234);
    runSimulation<StreamMWC32>("MWC32", numReps, numSims, numDays,  80,  512, 1234);
    runSimulation<StreamMWC32>("MWC32", numReps, numSims, numDays,  80, 1024, 1234);
    runSimulation<StreamMWC32>("MWC32", numReps, numSims, numDays, 160,  512, 1234);
    runSimulation<StreamMWC32>("MWC32", numReps, numSims, numDays, 160, 1024, 1234);
    runSimulation<StreamMWC32>("MWC32", numReps, numSims, numDays, 320,  256, 1234);
    runSimulation<StreamMWC32>("MWC32", numReps, numSims, numDays, 320,  512, 1234);
    
    runSimulation<StreamXoshiro128PlusPlus>("Xoshiro128++", numReps, numSims, numDays,  80,   64, 1234);
    runSimulation<StreamXoshiro128PlusPlus>("Xoshiro128++", numReps, numSims, numDays,  80,  128, 1234);
    runSimulation<StreamXoshiro128PlusPlus>("Xoshiro128++", numReps, numSims, numDays,  80,  256, 1234);
    runSimulation<StreamXoshiro128PlusPlus>("Xoshiro128++", numReps, numSims, numDays,  80,  512, 1234);
    runSimulation<StreamXoshiro128PlusPlus>("Xoshiro128++", numReps, numSims, numDays,  80, 1024, 1234);
    runSimulation<StreamXoshiro128PlusPlus>("Xoshiro128++", numReps, numSims, numDays, 160,  512, 1234);
    runSimulation<StreamXoshiro128PlusPlus>("Xoshiro128++", numReps, numSims, numDays, 160, 1024, 1234);
    runSimulation<StreamXoshiro128PlusPlus>("Xoshiro128++", numReps, numSims, numDays, 320,  256, 1234);
    runSimulation<StreamXoshiro128PlusPlus>("Xoshiro128++", numReps, numSims, numDays, 320,  512, 1234);
    
    return 0;
}
