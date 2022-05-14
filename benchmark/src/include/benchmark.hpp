#ifndef BENCHMARK
#define BENCHMARK

#include <string>
#include <stdexcept>
#include <iostream>

/*********************
 **** Decorators *****
 *********************/

template<class T>
class ToInt
{
    T generator_;
    
public:
    typedef uint32_t ReturnType;
    
    __host__ __device__ ToInt()
    {
    }
    
    __host__ __device__ ToInt(T generator)
    {
        generator_ = generator;
    }
    
    __host__ __device__ uint32_t next()
    {
        return (uint32_t)generator_.next();
    }
    
    __host__ __device__ void nextStream()
    {
        generator_.nextStream();
    }
    
    __host__ __device__ uint64_t mod()
    {
        return generator_.mod();
    }
};

template<class T>
class ToFloat
{
    T generator_;
    
public:
    typedef float ReturnType;
    
    __host__ __device__ ToFloat()
    {
    }
    
    __host__ __device__ ToFloat(T generator)
    {
        generator_ = generator;
    }
    
    __host__ __device__ float next()
    {
        return (float)generator_.next() * generator_.div();
    }
    
    __host__ __device__ void nextStream()
    {
        generator_.nextStream();
    }
    
    __host__ __device__ uint64_t mod()
    {
        return generator_.mod();
    }
};

template<class T>
class ToDouble
{
    T generator_;
    
public:
    typedef double ReturnType;
    
    __host__ __device__ ToDouble()
    {
    }
    
    __host__ __device__ ToDouble(T generator)
    {
        generator_ = generator;
    }
    
    __host__ __device__ double next()
    {
        return (double)generator_.next() * generator_.div();
    }
    
    __host__ __device__ void nextStream()
    {
        generator_.nextStream();
    }
    
    __host__ __device__ uint64_t mod()
    {
        return generator_.mod();
    }
};

/*********************
 ****** Kernels ******
 *********************/

template<class T, class U>
__global__ void sumDevice(T streams[], U results[])
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Copies the stream in register memory
    T localStream = streams[id];
    
    U sum = 0;
    
    #ifdef UNROLL_1
        #pragma unroll 1
    #endif
    for(int i = 0; i < 1000000; i++)
    {
        sum += localStream.next();
    }
    
    results[id] = sum;
}

template<class T, class U>
__global__ void generateDevice(T streams[], U results[])
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Copies the stream in register memory
    T localStream = streams[id];
    
    U lastNumber;
    
    #ifdef UNROLL_1
        #pragma unroll 1
    #endif
    for(int i = 0; i < 1000000; i++)
    {
        lastNumber = localStream.next();
    }
    
    results[id] = lastNumber;
}

/*********************
 ******* Proxy *******
 *********************/

class Benchmark
{
    int numReps_;
    
    int threadsPerBlock_;
    int numBlocks_;
    
    void assertCuda(cudaError_t status)
    {
        if(status != cudaSuccess)
        {
            throw std::runtime_error(cudaGetErrorString(status));
        }
    }
    
    template<class U>
    void printStatistics(cudaEvent_t& start, cudaEvent_t& end, U results[], int length)
    {
        float elapsedTime;
        
        U variance = 0;
        U sum = 0;
        
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
        U average = sum / length;
        
        // Prints the statistics
        std::cout << elapsedTime << " ";
        std::cout << variance << " ";
        std::cout << average << " ";
    }
    
    template<class T>
    void run(std::string name, void (*func)(T[], typename T::ReturnType[]), T stream)
    {
        T* streamsHost;
        T* streamsDevice;
        
        typename T::ReturnType* resultsHost;
        typename T::ReturnType* resultsDevice;
        
        cudaEvent_t start;
        cudaEvent_t end;
        
        int numThreads = numBlocks_ * threadsPerBlock_;
        
        // Creates the streams
        streamsHost = new T[numThreads];
        
        for(int i = 0; i < numThreads; i++)
        {
            streamsHost[i] = stream;
            
            stream.nextStream();
        }
        
        // Allocations
        resultsHost = new typename T::ReturnType[numThreads];
        
        cudaMalloc((void**)&streamsDevice, numThreads * sizeof(T));
        cudaMalloc((void**)&resultsDevice, numThreads * sizeof(typename T::ReturnType));
        
        assertCuda(cudaMemcpy(streamsDevice, streamsHost, numThreads * sizeof(T), cudaMemcpyHostToDevice));
        
        cudaEventCreate(&end);
        cudaEventCreate(&start);
        
        // Prints the parameters
        std::cout << name << " ";
        std::cout << stream.mod() << " ";
        std::cout << numBlocks_ << " ";
        std::cout << threadsPerBlock_ << " ";
        
        for(int i = 0; i < numReps_; i++)
        {
            cudaEventRecord(start);
            
            (*func) <<< numBlocks_, threadsPerBlock_ >>> (streamsDevice, resultsDevice);
            
            cudaEventRecord(end);
            cudaEventSynchronize(end);
            
            assertCuda(cudaMemcpy(resultsHost, resultsDevice, numThreads * sizeof(typename T::ReturnType), cudaMemcpyDeviceToHost));
            
            // Prints the statistics
            printStatistics(start, end, resultsHost, numThreads);
        }
        
        std::cout << "\n";
        
        // Deallocations
        delete[] streamsHost;
        delete[] resultsHost;
        
        assertCuda(cudaFree(streamsDevice));
        assertCuda(cudaFree(resultsDevice));
        
        cudaEventDestroy(end);
        cudaEventDestroy(start);
    }
    
public:
    Benchmark(int numReps, int threadsPerBlock, int numBlocks)
    {
        numReps_ = numReps;
        
        threadsPerBlock_ = threadsPerBlock;
        numBlocks_ = numBlocks;
    }
    
    void setDevice(int deviceId)
    {
        assertCuda(cudaSetDevice(deviceId));
    }
    
    void printHeader()
    {
        std::cout << "name modulus numBlocks threadsPerBlock";
        
        for(int i = 0; i < numReps_; i++)
        {
            std::cout << " elapsedTime" << i;
            std::cout << " variance" << i;
            std::cout << " average" << i;
        }
        
        std::cout << "\n";
    }
    
    template<class T>
    void sum(std::string type, T stream)
    {
        run("I_" + type, &sumDevice, ToInt<T>(stream));
        run("F_" + type, &sumDevice, ToFloat<T>(stream));
        run("D_" + type, &sumDevice, ToDouble<T>(stream));
    }
};

#endif
