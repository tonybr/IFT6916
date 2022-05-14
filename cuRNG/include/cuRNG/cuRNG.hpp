#ifndef cuRNG_cuRNG
#define cuRNG_cuRNG

#include <mutex>
#include <memory>
#include <cstdint>

/*********************
 ***** Constants *****
 *********************/

// To modify the coefficients of the generators "MRG31" and "MWC32", it is necessary to
// change the following constants as well as the matrices in the file "utilities.hpp"
// and the vectors in the member functions "nextStream" and "nextSubstream" below.

// To modify the moduli of the generators "MRG31" and "MWC32", more alterations are needed,
// as the algorithm for the modulo operation will most likely require more calculations.

#define cuRNG_MRG31_a1 208960
#define cuRNG_MRG31_a2 214748364
#define cuRNG_MRG31_a3 2861914534

// For MWC32, the code assumes that a1 is greater than zero.

#define cuRNG_MWC32_a1 457939846
#define cuRNG_MWC32_a2 106294984
#define cuRNG_MWC32_a3 295771115

#include "utilities.hpp"

/*********************
 *** Documentation ***
 *********************/

/*! \mainpage Tutorial
 *   
 *   *cuRNG* is a header-only library written in C++ that provides classes for the creation of streams, and substreams, of uniform random numbers on CUDA-enabled GPUs.\n It also offers pseudo-random number generators that were meticulously optimized for this platform.\n\n
 *   
 *   # Introduction to streams and substreams
 *   A pseudo-random number generator is an algorithm that generates a sequence of integers that appears uniformly distributed upon examination, starting from an initial value(s), which is called the "seed". More exactly, a pseudo-random number generator calculates each term of the resulting sequence by maintaining an array of unsigned integers in memory, which is called the "state": the array is initialized based on the seed; then, for each term, the array is re-calculated by applying a recurrence relation, and the next term is derived from the array. Also, mathematically, the integers appear uniformly distributed in the interval `[0, m)` where `m`, which is called the "modulus", depends on the generator.
 *   
 *   However, to create independent sequences of integers, e.g., to run simulations in parallel or to simulate independent random variables, we must find seeds such that the same state will not be generated twice across the sequences. In fact, once we encounter the same state in two sequences, the following terms are also identical, which clearly introduces correlation between the sequences. As a solution, cuRNG offers classes that implement the essential algorithms to create up to `2^32` unique sequences, which are called "streams", of length `2^92` to `2^96` depending on the generator. It can also divide streams into up to `2^32` subsequences, which are called "substreams".
 *   
 *   Another reason to create streams of uniform random numbers is to compare similar stochastic models by using common random numbers (CRNs) as in **[1]**.
 *   
 *   More information about streams can also be found in **[2]** and **[3]**.\n\n
 *   
 *   # Characteristics of the generators
 *   There are three pseudo-random number generators in *cuRNG*: *MRG31*, *MWC32*, and *Xoshiro128*++. The first and the second generators should not be used for now (see README.md), and the third generator was proposed and tested by Blackman and Vigna in **[4]**. The table below summarizes the characteristics of the generators in *cuRNG*. We note that the moduli of *MWC32* and *Xoshiro128*++ are `2^32` but that the modulus of *MRG31* is equal to `2^31 - 1`. As a consequence, *MWC32* and *Xoshiro128*++ yield streams of `32`-bit integers, whereas *MRG31* yields streams of `31`-bit integers. We also note that the state of each generator requires only four `32`-bit integers in memory.
 *   
 *   | Generator | Modulus | Size (bytes) | Stream length | Substream length | 
 *   |--|--|--|--|--|
 *   | *MRG31* | `2^31 - 1` | `16` | `2^92` | `2^60` |
 *   | *MWC32* | `2^32` | `16` | `2^95` | `2^63` |
 *   | *Xoshiro128*++ | `2^32` | `16` | `2^96` | `2^64` |
 *   \n
 *   
 *   # Simple examples
 *   The following program creates `100000` streams of type *StreamMRG31* with the seed `1234`. Then, it generates `20000000` uniform variates of type *float* in `[0, 1)` on the device and transfers the results onto the host.\n The steps are in the comments: to create streams from the host, we apply the steps `2`, `3`, `4`, and `7`; and, to generate uniform variates, we apply the steps `1`, `3`, `5`, `6`, and `7`.
 *   
 *       #include "cuRNG.hpp"
 *       
 *       #define NUM_VARIATES   20000000
 *       #define NUM_STREAMS   100000
 *       
 *       int main(int argc, char* argv[])
 *       {
 *           float* resultsDevice;
 *           StreamMRG31* streamsDevice;
 *       
 *           // 1) Allocate memory on the host for the variates.
 *           float resultsHost[NUM_VARIATES];
 *       
 *           // 2) Instantiate a stream creator (offset = 0 by default).
 *           StreamCreator<StreamMRG31> streamCreator(1234);
 *       
 *           // 3) Allocate memory on the device for the variates and the streams (see CUDA API).
 *           cudaMalloc((void**)&resultsDevice, NUM_VARIATES * sizeof(float));
 *           cudaMalloc((void**)&streamsDevice, NUM_STREAMS * sizeof(StreamMRG31));
 *       
 *           // 4) Create the streams using an adequate configuration of thread-blocks.
 *           streamCreator.create(streamsDevice, NUM_STREAMS, 80, 64); 
 *       
 *           // 5) Generate the variates using an adequate configuration of thread-blocks.
 *           generateUniform01(streamsDevice, resultsDevice, NUM_STREAMS, NUM_VARIATES, 80, 1024);
 *       
 *           // 6) Transfer the variates onto the host (see CUDA API).
 *           cudaMemcpy(resultsHost, resultsDevice, NUM_VARIATES * sizeof(float), cudaMemcpyDeviceToHost);
 *       
 *           // 7) Deallocate the arrays (see CUDA API).
 *           cudaFree(streamsDevice);
 *           cudaFree(resultsDevice);
 *       }
 *   \n
 *   
 *   The following CUDA kernel has a loop that generates *scnsPerThread* scenarios. This kernel also expects an array of streams of type *StreamMRG31* named *streamsDevice* (see the example above).\n More exactly, for every scenario, each thread retrieves a single stream from global memory, which corresponds to the step `1`, and generates two uniform variates of type *float* in `[0 ,1)`, which corresponds to the step `2`.
 *   
 *       #include "cuRNG.hpp"
 *       
 *       __global__ void runSimulation(StreamMRG31 streamsDevice[], int scnsPerThread)
 *       {
 *           int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
 *       
 *           for(int i = 0; i < scnsPerThread; i++)
 *           {
 *                // 1) Copy a stream from global memory to register memory.
 *                StreamMRG31 currentStream = streamsDevice[threadIndex * scnsPerThread + i]; 
 *                
 *                // 2) Generate the uniform variates.
 *                float resultA = currentStream.nextUniform01();
 *                float resultB = currentStream.nextUniform01();
 *                …
 *           }
 *       }
 *   \n
 *   
 *   # References
 *   [1] P. L’Ecuyer, D. Munger, and N. Kemerchou. 2015. “clRNG: A Random Number API with Multiple Streams for OpenCL”. Report. http://www.iro.umontreal.ca/~lecuyer/myftp/papers/clrng-api.pdf
 *   
 *   [2] P. L’Ecuyer, R. Simard, E. J. Chen, and W. D. Kelton. 2002. “An Object-Oriented Random-Number Package with Many Long Streams and Substreams”. *Operations Research* 50(6):1073–1075.
 *   
 *   [3] P. L’Ecuyer, O. Nadeau-Chamard, Y.-F. Chen, and J. Lebar. 2021. “Multiple Streams with Recurrence-Based, Counter-Based, and Splittable Random Number Generators”. In *Proceedings of the 2021 Winter Simulation Conference*.
 *   
 *   [4] D. Blackman and S. Vigna. 2021. “Scrambled Linear Pseudorandom Number Generators”. *ACM Transactions on Mathematical Software* 47(4):Article 36.
 */

/*! \file cuRNG.hpp
 *  \brief Header for cuRNG
 *   
 *   This file contains the classes that represent the streams of uniform random numbers and the functions to create arrays of streams.
 */

/*********************
 ****** Classes ******
 *********************/

/*! \brief Stream creator
 *   
 *   This class creates arrays of streams of type *T* in global memory of the device from the host.
 *   
 *  \note The streams are guaranteed to be unique for a given instance of this class.
 */
template<class T>
class StreamCreator
{
    struct State
    {
        uint32_t seed;
        uint32_t offset;
        uint32_t initialOffset;
    };
    
    std::shared_ptr<State> state;
    std::shared_ptr<std::mutex> mutex;
    
public:
    /*! \brief Constructor for the class StreamCreator
     *   
     *   This function instantiates a stream creator. The value of the parameter \p seed will 
     *   determine the starting point of the streams, and the first \p offset streams will be skipped.
     *   
     *  \param[in]       seed                  Initial value of the generator
     *  \param[in]       offset                Number of streams to skip
     */
    StreamCreator(uint32_t seed, uint32_t offset = 0)
    {
        state = std::shared_ptr<State>(new State);
        mutex = std::shared_ptr<std::mutex>(new std::mutex);
        
        state->seed = seed;
        state->offset = offset;
        state->initialOffset = offset;
    }
    
    /*! \brief Creates an array of streams in global memory of the device.
     *   
     *   This function creates new streams of type *T* in global memory of the device. It launches a CUDA 
     *   kernel using the configuration of thread-blocks that correspond to the parameters \p numBlocks and 
     *   \p threadsPerBlock. As a rule of thumb, the number of blocks should be equal to the number of streaming 
     *   multiprocessors on the device, and the number of threads per block should be between `64` and `256`.
     *   
     *  \param[out]      output                Array for the new streams
     *  \param[in]       numStreams            Number of streams to create
     *  \param[in]       numBlocks             Number of blocks of threads
     *  \param[in]       threadsPerBlock       Number of threads per block
     *   
     *  \note The output array must be allocated on the device before calling this function.
     */
    void create(T output[], int numStreams, int numBlocks, int threadsPerBlock)
    {
        int streamsPerThread = (numStreams - 1) / (numBlocks * threadsPerBlock) + 1;
        
        mutex->lock();
        
        cuRNG::createStreams <<< numBlocks, threadsPerBlock >>> (output, numStreams, streamsPerThread, state->seed, state->offset);
        
        state->offset += numStreams;
        
        mutex->unlock();
    }
    
    /*! \brief Resets the stream creator.
     *   
     *   This function reinitializes the stream creator. As a result, any subsequent call to the same
     *   stream creator will re-create the same streams in the same order as prior calls did.
     */
    void reset()
    {
        mutex->lock();
        
        state->offset = state->initialOffset;
        
        mutex->unlock();
    }
};

/*! \brief Stream of uniform random numbers based on the generator MRG31
 */
class StreamMRG31
{
    uint32_t initialStreamState[4];
    uint32_t initialSubstreamState[4];
    uint32_t substreamState[4];
    
    __host__ __device__ void jump(const uint32_t J[4])
    {
        uint64_t temp[] = {0, 0, 0, 0};
        uint32_t nextState[] = {0, 0, 0, 0};
        
        for(int i = 0; i < 4; i++)
        {
            temp[0] += (uint64_t)substreamState[0] * J[i];
            temp[1] += (uint64_t)substreamState[1] * J[i];
            temp[2] += (uint64_t)substreamState[2] * J[i];
            temp[3] += (uint64_t)substreamState[3] * J[i];
            
            if(i < 3)
            {
                nextInteger();
            }
        }
        
        // Applies the modulo operation to each scalar product
        for(int i = 0; i < 4; i++)
        {
            temp[i] = (temp[i] & 2147483647) + (temp[i] >> 31);
            
            nextState[i] = ((uint32_t)temp[i] & 2147483647) + (uint32_t)(temp[i] >> 31);
            
            if(nextState[i] >= 2147483647)
            {
                nextState[i] += 2147483649;
            }
        }
        
        cuRNG::copyArray(nextState, substreamState, 4);
    }
    
    __host__ __device__ StreamMRG31(uint32_t seed)
    {
        if(seed >= 2147483647)
        {
            seed += 2147483649;
        }
        
        cuRNG::fillArray(initialStreamState, seed, 4);
        cuRNG::fillArray(initialSubstreamState, seed, 4);
        cuRNG::fillArray(substreamState, seed, 4);
    }
    
    __device__ StreamMRG31(uint32_t seed, uint32_t streamIndex): StreamMRG31(seed)
    {
        for(int i = 0; i < 32; i++)
        {
            if(streamIndex & 1)
            {
                jump(cuRNG::MRG31_J92[i]);
            }
            
            streamIndex >>= 1;
        }
        
        cuRNG::copyArray(substreamState, initialSubstreamState, 4);
        cuRNG::copyArray(substreamState, initialStreamState, 4);
    }
    
    __host__ __device__ StreamMRG31& nextStream()
    {
        const uint32_t J92[4] = {0x030d40cb, 0x32875485, 0x18006110, 0x6cb29932};
        
        jump(J92);
        
        cuRNG::copyArray(substreamState, initialSubstreamState, 4);
        cuRNG::copyArray(substreamState, initialStreamState, 4);
        
        return *this;
    }
    
    __host__ __device__ static void createStreams(StreamMRG31 output[], StreamMRG31& baseStream, int numStreams)
    {
        output[0] = baseStream;
        
        for(int i = 1; i < numStreams; i++)
        {
            output[i] = baseStream.nextStream();
        }
    }
    
public:
    /*! \brief Default constructor for the class StreamMRG31
     *   
     *   This function instantiates a stream of type StreamMRG31, but it does not initialize the stream.
     *   This is required for the use of the operator "new".
     */
    __host__ __device__ StreamMRG31()
    {
    }
    
    /// \private
    __device__ static void createStreams(StreamMRG31 output[], int numStreams, uint32_t seed, uint32_t offset)
    {
        if(numStreams > 0)
        {
            StreamMRG31 baseStream(seed, offset);
            
            createStreams(output, baseStream, numStreams);
        }
    }
    
    /*! \brief Returns the next pseudo-random integer in `[0, 2^31 - 1)`.
     *   
     *  \return Next pseudo-random integer
     */
    __host__ __device__ uint32_t nextInteger()
    {
        uint32_t z[2];
        
    #if __CUDA_ARCH__ < 700 && __CUDA_ARCH__ > 0
        cuRNG::multiply_scalars(substreamState[0], cuRNG_MRG31_a3, z);
        
        cuRNG::multiplyAdd_scalars(substreamState[1], cuRNG_MRG31_a2, z);
        cuRNG::multiplyAdd_scalars(substreamState[2], cuRNG_MRG31_a1, z);
    #else
        uint64_t t = (uint64_t)substreamState[0] * cuRNG_MRG31_a3;
        
        t += (uint64_t)substreamState[1] * cuRNG_MRG31_a2;
        t += (uint64_t)substreamState[2] * cuRNG_MRG31_a1;
        
        z[0] = (uint32_t)t;
        z[1] = (uint32_t)(t >> 32);
    #endif
        
        uint32_t r = (z[0] >> 1) + z[1];
        
        if(r >= 2147483647)
        {
            r += 2147483649;
        }
        
        substreamState[0] = substreamState[1];
        substreamState[1] = substreamState[2];
        substreamState[2] = substreamState[3];
        substreamState[3] = r;
        
        return r;
    }
    
    /*! \brief Returns the next pseudo-random number of type *float* in `[0, 1)`.
     *   
     *  \return Next pseudo-random number of type *float*
     */
    __host__ __device__ float nextUniform01()
    {
        return (float)nextInteger() * 4.65661287308e-10f;
    }
    
    /*! \brief Returns the next pseudo-random number of type *double* in `[0, 1)`.
     *   
     *  \return Next pseudo-random number of type *double*
     */
    __host__ __device__ double nextUniform01_double()
    {
        return (double)nextInteger() * 4.65661287308e-10;
    }
    
    /*! \brief Advances the stream to the beginning of the next substream.
     */
    __host__ __device__ void nextSubstream()
    {
        const uint32_t J60[4] = {0x474833f, 0x5dee8356, 0x26de880d, 0x15e07d5d};
        
        cuRNG::copyArray(initialSubstreamState, substreamState, 4);
        
        jump(J60);
        
        cuRNG::copyArray(substreamState, initialSubstreamState, 4);
    }
    
    /*! \brief Resets the stream.
     *   
     *   This function reinitializes the stream. As a result, any subsequent call to the same
     *   stream will re-generate the same variates in the same order as prior calls did.
     */
    __host__ __device__ void reset()
    {
        cuRNG::copyArray(initialStreamState, initialSubstreamState, 4);
        cuRNG::copyArray(initialStreamState, substreamState, 4);
    }
};

/*! \brief Stream of uniform random numbers based on the generator MWC32
 */
class StreamMWC32
{
    uint32_t initialStreamState[4];
    uint32_t initialSubstreamState[4];
    uint32_t substreamState[4];
    
    __host__ __device__ void jump(const uint32_t J[4])
    {
        uint32_t temp[4];
        uint32_t nextState[4];
        
        uint32_t modulus[4] = {4294967295, cuRNG_MWC32_a1 - 1, cuRNG_MWC32_a2, cuRNG_MWC32_a3};
        
        // Concatenates the state
        cuRNG::copyArray(initialSubstreamState, nextState, 4);
        
        // Subtracts a1 * x_n-2 * 2^32
        temp[0] = 0;
        temp[3] = 0;
        
        cuRNG::multiply_scalars(initialSubstreamState[0], cuRNG_MWC32_a1, &temp[1]);
        cuRNG::subtract_vectors(nextState, temp, modulus, 4);
        
        // Subtracts a2 * x_n-2 * 2^64
        temp[1] = 0;
        
        cuRNG::multiply_scalars(initialSubstreamState[0], cuRNG_MWC32_a2, &temp[2]);
        cuRNG::subtract_vectors(nextState, temp, modulus, 4);
        
        // Subtracts a1 * x_n-1 * 2^64
        cuRNG::multiply_scalars(initialSubstreamState[1], cuRNG_MWC32_a1, &temp[2]);
        cuRNG::subtract_vectors(nextState, temp, modulus, 4);
        
        // Multiplies by J mod m
        cuRNG::multiply_vectors(nextState, J, modulus, initialSubstreamState, 4);
        
        // Adds a1 * x'_n-2 * 2^32
        temp[3] = 0;
        
        cuRNG::multiply_scalars(initialSubstreamState[0], cuRNG_MWC32_a1, &temp[1]);
        cuRNG::add_vectors(initialSubstreamState, temp, 4);
        
        // Adds a2 * x'_n-2 * 2^64
        temp[1] = 0;
        
        cuRNG::multiply_scalars(initialSubstreamState[0], cuRNG_MWC32_a2, &temp[2]);
        cuRNG::add_vectors(initialSubstreamState, temp, 4);
        
        // Adds a1 * x'_n-1 * 2^64
        cuRNG::multiply_scalars(initialSubstreamState[1], cuRNG_MWC32_a1, &temp[2]);
        cuRNG::add_vectors(initialSubstreamState, temp, 4);
    }
    
    __host__ __device__ StreamMWC32(uint32_t seed)
    {
        cuRNG::fillArray(initialStreamState, seed, 4);
        cuRNG::fillArray(initialSubstreamState, seed, 4);
        cuRNG::fillArray(substreamState, seed, 4);
    }
    
    __device__ StreamMWC32(uint32_t seed, uint32_t streamIndex): StreamMWC32(seed)
    {
        uint32_t temp[4];
        uint32_t coefficient[4] = {1, 0, 0, 0};
        
        uint32_t modulus[4] = {4294967295, cuRNG_MWC32_a1 - 1, cuRNG_MWC32_a2, cuRNG_MWC32_a3};
        
        // Aggregates the necessary vectors
        for(int i = 0; i < 32; i++)
        {
            if(streamIndex & 1)
            {
                cuRNG::multiply_vectors(coefficient, cuRNG::MWC32_J95[i], modulus, temp, 4);
                
                cuRNG::copyArray(temp, coefficient, 4);
            }
            
            streamIndex >>= 1;
        }
        
        jump(coefficient);
        
        cuRNG::copyArray(initialSubstreamState, initialStreamState, 4);
        cuRNG::copyArray(initialSubstreamState, substreamState, 4);
    }
    
    __host__ __device__ StreamMWC32& nextStream()
    {
        const uint32_t J95[4] = {0xb96ddfe3, 0x2ebdb71e, 0x20412e62, 0x07edad29};
        
        jump(J95);
        
        cuRNG::copyArray(initialSubstreamState, initialStreamState, 4);
        cuRNG::copyArray(initialSubstreamState, substreamState, 4);
        
        return *this;
    }
    
    __host__ __device__ static void createStreams(StreamMWC32 output[], StreamMWC32& baseStream, int numStreams)
    {
        output[0] = baseStream;
        
        for(int i = 1; i < numStreams; i++)
        {
            output[i] = baseStream.nextStream();
        }
    }
    
public:
    /*! \brief Default constructor for the class StreamMWC32
     *   
     *   This function instantiates a stream of type StreamMWC32, but it does not initialize the stream.
     *   This is required for the use of the operator "new".
     */
    __host__ __device__ StreamMWC32()
    {
    }
    
    /// \private
    __device__ static void createStreams(StreamMWC32 output[], int numStreams, uint32_t seed, uint32_t offset)
    {
        if(numStreams > 0)
        {
            StreamMWC32 baseStream(seed, offset);
        
            createStreams(output, baseStream, numStreams);
        }
    }
    
    /*! \brief Returns the next pseudo-random integer in `[0, 2^32)`.
     *   
     *  \return Next pseudo-random integer.
     */
    __host__ __device__ uint32_t nextInteger()
    {
        uint32_t z[2];
        
    #ifdef __CUDA_ARCH__
        z[1] = 0;
        z[0] = substreamState[3];
        
        cuRNG::multiplyAdd_scalars(substreamState[0], cuRNG_MWC32_a3, z);
        cuRNG::multiplyAdd_scalars(substreamState[1], cuRNG_MWC32_a2, z);
        cuRNG::multiplyAdd_scalars(substreamState[2], cuRNG_MWC32_a1, z);
    #else
        uint64_t t = substreamState[3];
        
        t += (uint64_t)substreamState[0] * cuRNG_MWC32_a3;
        t += (uint64_t)substreamState[1] * cuRNG_MWC32_a2;
        t += (uint64_t)substreamState[2] * cuRNG_MWC32_a1;
        
        z[0] = (uint32_t)t;
        z[1] = (uint32_t)(t >> 32);
    #endif
        
        substreamState[0] = substreamState[1];
        substreamState[1] = substreamState[2];
        substreamState[2] = z[0];
        substreamState[3] = z[1];
        
        return z[0];
    }
    
    /*! \brief Returns the next pseudo-random number of type *float* in `[0, 1)`.
     *   
     *  \return Next pseudo-random number of type *float*
     */
    __host__ __device__ float nextUniform01()
    {
        return (float)nextInteger() * 2.32830643654e-10f;
    }
    
    /*! \brief Returns the next pseudo-random number of type *double* in `[0, 1)`.
     *   
     *  \return Next pseudo-random number of type *double*
     */
    __host__ __device__ double nextUniform01_double()
    {
        return (double)nextInteger() * 2.32830643654e-10;
    }
    
    /*! \brief Advances the stream to the beginning of the next substream.
     */
    __host__ __device__ void nextSubstream()
    {
        const uint32_t J63[4] = {0xbb947030, 0xb8d5c8f0, 0xd8159e36, 0x7e5eae1};
        
        jump(J63);
        
        cuRNG::copyArray(initialSubstreamState, substreamState, 4);
    }
    
    /*! \brief Resets the stream.
     *   
     *   This function reinitializes the stream. As a result, any subsequent call to the same
     *   stream will re-generate the same variates in the same order as prior calls did.
     */
    __host__ __device__ void reset()
    {
        cuRNG::copyArray(initialStreamState, initialSubstreamState, 4);
        cuRNG::copyArray(initialStreamState, substreamState, 4);
    }
};

/*! \brief Stream of uniform random numbers based on the generator Xoshiro128++
 */
class StreamXoshiro128PlusPlus
{
    uint32_t initialStreamState[4];
    uint32_t initialSubstreamState[4];
    uint32_t substreamState[4];
    
    __host__ __device__ void jump(const uint32_t J[4])
    {
        uint32_t nextState[] = {0, 0, 0, 0};
        
        for(int i = 0; i < 4; i++)
        {
            uint32_t temp = J[i];
            
            for(int j = 0; j < 32; j++)
            {
                if(temp & 1)
                {
                    nextState[0] ^= substreamState[0];
                    nextState[1] ^= substreamState[1];
                    nextState[2] ^= substreamState[2];
                    nextState[3] ^= substreamState[3];
                }
                
                nextInteger();
                
                temp >>= 1;
            }
        }
        
        cuRNG::copyArray(nextState, substreamState, 4);
    }
    
    __host__ __device__ StreamXoshiro128PlusPlus(uint32_t seed)
    {
        cuRNG::fillArray(initialStreamState, seed, 4);
        cuRNG::fillArray(initialSubstreamState, seed, 4);
        cuRNG::fillArray(substreamState, seed, 4);
    }
    
    __device__ StreamXoshiro128PlusPlus(uint32_t seed, uint32_t streamIndex): StreamXoshiro128PlusPlus(seed)
    {
        for(int i = 0; i < 32; i++)
        {
            if(streamIndex & 1)
            {
                jump(cuRNG::Xoshiro128PlusPlus_J96[i]);
            }
            
            streamIndex >>= 1;
        }
        
        cuRNG::copyArray(substreamState, initialSubstreamState, 4);
        cuRNG::copyArray(substreamState, initialStreamState, 4);
    }
    
    __host__ __device__ StreamXoshiro128PlusPlus& nextStream()
    {
        const uint32_t J96[4] = {0xb523952e, 0x0b6f099f, 0xccf5a0ef, 0x1c580662};
        
        jump(J96);
        
        cuRNG::copyArray(substreamState, initialSubstreamState, 4);
        cuRNG::copyArray(substreamState, initialStreamState, 4);
        
        return *this;
    }
    
    __host__ __device__ static void createStreams(StreamXoshiro128PlusPlus output[], StreamXoshiro128PlusPlus& baseStream, int numStreams)
    {
        output[0] = baseStream;
        
        for(int i = 1; i < numStreams; i++)
        {
            output[i] = baseStream.nextStream();
        }
    }
    
public:
    /*! \brief Default constructor for the class StreamXoshiro128++
     *   
     *   This function instantiates a stream of type StreamXoshiro128++, but it does not initialize the stream.
     *   This is required for the use of the operator "new".
     */
    __host__ __device__ StreamXoshiro128PlusPlus()
    {
    }
    
    /// \private
    __device__ static void createStreams(StreamXoshiro128PlusPlus output[], int numStreams, uint32_t seed, uint32_t offset)
    {
        if(numStreams > 0)
        {
            StreamXoshiro128PlusPlus baseStream(seed, offset);
        
            createStreams(output, baseStream, numStreams);
        }
    }
    
    /*! \brief Returns the next pseudo-random integer in `[0, 2^32)`.
     *   
     *  \return Next pseudo-random integer.
     */
    __host__ __device__ uint32_t nextInteger()
    {
        uint32_t r, t;
        
        r = cuRNG::rotl(substreamState[0] + substreamState[3], 7) + substreamState[0];
        
        t = substreamState[1] << 9;
        
        substreamState[2] ^= substreamState[0];
        substreamState[3] ^= substreamState[1];
        substreamState[1] ^= substreamState[2];
        substreamState[0] ^= substreamState[3];
        
        substreamState[2] ^= t;
        
        substreamState[3] = cuRNG::rotl(substreamState[3], 11);
        
        return r;
    }
    
    /*! \brief Returns the next pseudo-random number of type *float* in `[0, 1)`.
     *   
     *  \return Next pseudo-random number of type *float*
     */
    __host__ __device__ float nextUniform01()
    {
        return (float)nextInteger() * 2.32830643654e-10f;
    }
    
    /*! \brief Returns the next pseudo-random number of type *double* in `[0, 1)`.
     *   
     *  \return Next pseudo-random number of type *double*
     */
    __host__ __device__ double nextUniform01_double()
    {
        return (double)nextInteger() * 2.32830643654e-10;
    }
    
    /*! \brief Advances the stream to the beginning of the next substream.
     */
    __host__ __device__ void nextSubstream()
    {
        const uint32_t J64[4] = {0x8764000b, 0xf542d2d3, 0x6fa035c3, 0x77f2db5b};
        
        cuRNG::copyArray(initialSubstreamState, substreamState, 4);
        
        jump(J64);
        
        cuRNG::copyArray(substreamState, initialSubstreamState, 4);
    }
    
    /*! \brief Resets the stream.
     *   
     *   This function reinitializes the stream. As a result, any subsequent call to the same
     *   stream will re-generate the same variates in the same order as prior calls did.
     */
    __host__ __device__ void reset()
    {
        cuRNG::copyArray(initialStreamState, initialSubstreamState, 4);
        cuRNG::copyArray(initialStreamState, substreamState, 4);
    }
};

/*********************
 ***** Functions *****
 *********************/

/*! \brief Creates an array of streams on the device from the device.
 *   
 *   This function creates new streams of type *T* on the device, where *T* is deduced based on the type of
 *   the parameter \p output. The parameters \p seed and \p offset have the same meaning as in the constructor
 *   of the class StreamCreator. It is also possible to create a single stream by passing a pointer to the
 *   stream instead of an array for the value of the parameter \p output.
 *   
 *  \param[out]      output                Array for the new streams
 *  \param[in]       numStreams            Number of streams to create
 *  \param[in]       seed                  Initial value of the generator
 *  \param[in]       offset                Number of streams to skip
 *   
 *  \note The output array must be allocated on the device before calling this function.
 */
template<class T>
__device__ void createStreams(T output[], int numStreams, uint32_t seed, uint32_t offset)
{
    T::createStreams(output, numStreams, seed, offset);
}

/*! \brief Generates an array of uniform variates of type *float* in `[0,1)` in global memory of the device.
 *   
 *   This function generates an array of uniform variates of type *float* in `[0,1)` in global memory of the
 *   device from the host. More exactly, it launches a CUDA kernel that generates a uniform variate from each
 *   stream in the parameter \p streams until enough variates have been generated; the variate at index `i`
 *   will be from the `n`-th stream where `n = i % numStreams`.
 *   
 *  \param[in]       streams               Array of streams
 *  \param[out]      output                Array for the uniform variates
 *  \param[in]       numStreams            Number of streams given
 *  \param[in]       numVariates           Number of variates to generate
 *  \param[in]       numBlocks             Number of blocks of threads
 *  \param[in]       threadsPerBlock       Number of threads per block
 *   
 *  \note The output array must be allocated on the device before calling this function.
 */
template<class T>
void generateUniform01(T streams[], float output[], int numStreams, int numVariates, int numBlocks, int threadsPerBlock)
{
    cuRNG::generateUniform01 <<< numBlocks, threadsPerBlock >>> (streams, output, numStreams, numVariates);
}

#endif
