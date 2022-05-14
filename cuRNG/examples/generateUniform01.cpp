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

/* For cuRNG */

template<class T>
void generateUniform01_cuRNG(std::string type, int numReps, int numVariates, int numBlocks, int threadsPerBlock, uint32_t seed)
{
    cudaEvent_t end;
    cudaEvent_t start;
    
    T* streams_device;
    float* results_device;
    
    int numThreads = numBlocks * threadsPerBlock;
    
    // Allocations
    float* results_host = new float[numVariates];
    
    assertCuda(cudaMalloc((void**)&streams_device, numThreads * sizeof(T)));
    assertCuda(cudaMalloc((void**)&results_device, numVariates * sizeof(float)));
    
    cudaEventCreate(&end);
    cudaEventCreate(&start);
    
    StreamCreator<T> streamCreator(seed);
    
    // Prints the parameters
    std::cout << type << " ";
    std::cout << numBlocks << " ";
    std::cout << threadsPerBlock << " ";
    
    for(int i = 0; i < numReps; i++)
    {
        // Creates the streams
        streamCreator.create(streams_device, numThreads, numBlocks, threadsPerBlock);
        
        // Generates the variates
        cudaEventRecord(start);
        
        generateUniform01(streams_device, results_device, numThreads, numVariates, numBlocks, threadsPerBlock);
        
        cudaEventRecord(end);
        cudaEventSynchronize(end);
        
        assertCuda(cudaMemcpy(results_host, results_device, numVariates * sizeof(float), cudaMemcpyDeviceToHost));
        
        // Prints the statistics
        printStatistics(start, end, results_host, numVariates);
        
        streamCreator.reset();
    }
    
    std::cout << "\n";
    
    // Deallocations
    delete[] results_host;
    
    assertCuda(cudaFree(streams_device));
    assertCuda(cudaFree(results_device));
    
    cudaEventDestroy(end);
    cudaEventDestroy(start);
}

int main(int argc, char* argv[])
{
    if(argc != 3)
    {
        std::cerr << "\nUsage: generateUniform01 <numRepetitions> <numVariates>\n";
        
        return 1;
    }
    
    int numReps = std::stoi(argv[1]);
    int numVariates = std::stoi(argv[2]);
    
    // Prints the header
    std::cout << "type numBlocks threadsPerBlock";
    
    for(int i = 0; i < numReps; i++)
    {
        std::cout << " elapsedTime" << i;
        std::cout << " variance" << i;
        std::cout << " average" << i;
    }
    
    std::cout << "\n";
    
    // Prints the results
    generateUniform01_cuRNG<StreamMRG31>("MRG31", numReps, numVariates,  64,  64, 1234);
    generateUniform01_cuRNG<StreamMRG31>("MRG31", numReps, numVariates,  64, 512, 1234);
    generateUniform01_cuRNG<StreamMRG31>("MRG31", numReps, numVariates, 512, 128, 1234);
    
    generateUniform01_cuRNG<StreamMWC32>("MWC32", numReps, numVariates,  64,  64, 1234);
    generateUniform01_cuRNG<StreamMWC32>("MWC32", numReps, numVariates,  64, 512, 1234);
    generateUniform01_cuRNG<StreamMWC32>("MWC32", numReps, numVariates, 512, 128, 1234);
    
    generateUniform01_cuRNG<StreamXoshiro128PlusPlus>("Xoshiro128++", numReps, numVariates,  64,  64, 1234);
    generateUniform01_cuRNG<StreamXoshiro128PlusPlus>("Xoshiro128++", numReps, numVariates,  64, 512, 1234);
    generateUniform01_cuRNG<StreamXoshiro128PlusPlus>("Xoshiro128++", numReps, numVariates, 512, 128, 1234);
    
    return 0;
}
