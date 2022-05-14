#ifndef cuRNG_utilities
#define cuRNG_utilities

#include <cstdint>

using std::uint32_t;
using std::uint64_t;

namespace cuRNG
{
    /*********************
     ***** Functions *****
     *********************/
    
    template<class T>
    __global__ void createStreams(T output[], int numStreams, int streamsPerThread, uint32_t seed, uint32_t offset)
    {
        int outputOffset = (threadIdx.x + blockIdx.x * blockDim.x) * streamsPerThread;
        
        if(outputOffset < numStreams)
        {
            streamsPerThread = min(streamsPerThread, numStreams - outputOffset);
            
            createStreams(output + outputOffset, streamsPerThread, seed, offset + outputOffset);
        }
    }
    
    template<class T>
    __global__ void generateUniform01(T streams[], float output[], int numStreams, int numVariates)
    {
        int numThreads = gridDim.x * blockDim.x;
        int streamIndex = threadIdx.x + blockIdx.x * blockDim.x;
        
        while(streamIndex < numStreams)
        {
            int outputIndex = streamIndex;
            
            // Copies the stream in register memory
            T currentStream = streams[streamIndex];
            
            while(outputIndex < numVariates)
            {
                output[outputIndex] = currentStream.nextUniform01();
                outputIndex += numStreams;
            }
            
            streams[streamIndex] = currentStream;
            streamIndex += numThreads;
        }
    }
    
    __host__ __device__ void fillArray(uint32_t output[], int value, int length)
    {
        for (int i = 0; i < length; i++)
        {
            output[i] = value;
        }
    }
    
    __host__ __device__ void copyArray(const uint32_t source[], uint32_t output[], int length)
    {
        for (int i = 0; i < length; i++)
        {
            output[i] = source[i];
        }
    }
    
    __host__ __device__ int add_vectors(uint32_t output[], uint32_t b[], int length)
    {
        int carry = 0;
        
    #ifdef __CUDA_ARCH__
        asm volatile("add.cc.u32 %0, %0, %1;" : "+r"(output[0]) : "r"(b[0]));
        
        for (int i = 1; i < length; i++)
        {
            asm volatile("addc.cc.u32 %0, %0, %1;" : "+r"(output[i]) : "r"(b[i]));
        }
        
        asm volatile("addc.s32 %0, %0, 0;" : "+r"(carry));
    #else
        for (int i = 0; i < length; i++)
        {
            uint64_t temp = (uint64_t)output[i] + b[i] + carry;
            
            output[i] = (uint32_t)temp;
            
            carry = (uint32_t)(temp >> 32);
        }
    #endif
        
        return carry;
    }
    
    __host__ __device__ int subtract_vectors(uint32_t output[], uint32_t b[], int length)
    {
        int borrow = 0;
        
    #ifdef __CUDA_ARCH__
        asm volatile("sub.cc.u32 %0, %0, %1;" : "+r"(output[0]) : "r"(b[0]));
        
        for (int i = 1; i < length; i++)
        {
            asm volatile("subc.cc.u32 %0, %0, %1;" : "+r"(output[i]) : "r"(b[i]));
        }
        
        asm volatile("subc.s32 %0, %0, 0;" : "+r"(borrow));
    #else
        for (int i = 0; i < length; i++)
        {
            uint64_t temp = (uint64_t)output[i] - b[i] - borrow;
            
            output[i] = (uint32_t)temp;
            
            borrow = temp > 4294967295;
        }
    #endif
        
        return borrow;
    }
    
    __host__ __device__ void add_vectors(uint32_t output[], uint32_t b[], uint32_t modulus[], int length)
    {
        bool isGreater = add_vectors(output, b, length);
        
        // Applies the modulo operation
        bool isSmaller = false;
        
        for (int i = length - 1; i >= 0; i--)
        {
            isGreater |= output[i] > modulus[i] && !isSmaller;
            isSmaller |= output[i] < modulus[i];
        }
        
        if(isGreater)
        {
            subtract_vectors(output, modulus, length);
        }
    }
    
    __host__ __device__ void subtract_vectors(uint32_t output[], uint32_t b[], uint32_t modulus[], int length)
    {
        int borrow = subtract_vectors(output, b, length);
        
        if(borrow == 1)
        {
            add_vectors(output, modulus, length);
        }
    }
    
    __host__ __device__ void multiply_vectors(uint32_t a[], const uint32_t b[], uint32_t modulus[], uint32_t output[], int length)
    {
        uint32_t temp;
        
        fillArray(output, 0, length);
        
        for(int i = 0; i < length; i++)
        {
            temp = b[i];
            
            for(int j = 0; j < 32; j++)
            {
                if(temp & 1)
                {
                    add_vectors(output, a, modulus, length);
                }
                
                add_vectors(a, a, modulus, length);
                
                temp >>= 1;
            }
        }
    }
    
    __host__ __device__ void multiply_scalars(uint32_t a, uint32_t b, uint32_t output[])
    {
    #ifdef __CUDA_ARCH__
        output[0] = a * b;
        output[1] = __umulhi(a, b);
    #else
        uint64_t temp = (uint64_t)a * b;
        
        output[0] = (uint32_t)temp;
        output[1] = (uint32_t)(temp >> 32);
    #endif
    }
    
    __host__ __device__ void multiplyAdd_scalars(uint32_t a, uint32_t b, uint32_t output[])
    {
        uint32_t z[2];
        
        multiply_scalars(a, b, z);
        
    #ifdef __CUDA_ARCH__
        asm ("add.cc.u32 %0, %0, %2;\n\t"
             "addc.u32   %1, %1, %3;"
             : "+r"(output[0]), "+r"(output[1])
             : "r"(z[0]), "r"(z[1]));
    #else
        output[0] += z[0];
        output[1] += z[1] + (output[0] < z[0]);
    #endif
    }
    
    __host__ __device__ uint32_t rotl(uint32_t x, int k)
    {
        return (x << k) | (x >> (32 - k));
    }
    
    /*********************
     ***** Constants *****
     *********************/
    
    __device__ __constant__ const uint32_t MRG31_J92[32][4] = {{0x030d40cb, 0x32875485, 0x18006110, 0x6cb29932},
                                                               {0x45b71f79, 0x508533ed, 0x3335e41d, 0x7ff3698a},
                                                               {0x65fb683b, 0x38106671, 0x07edf0d2, 0x5adc7dbf},
                                                               {0x232ada7a, 0x4d514503, 0x28122c5a, 0x30cc491c},
                                                               {0x34b7d241, 0x1477c2ee, 0x0dcc3695, 0x338c142e},
                                                               {0x1902ffd6, 0x496593d7, 0x2032e07c, 0x363be544},
                                                               {0x7f3214c4, 0x77f0da66, 0x2f56ad1d, 0x378dd719},
                                                               {0x2d74cc7e, 0x5da1b918, 0x19c809fb, 0x7717318a},
                                                               {0x18af8750, 0x088b99bd, 0x5023f563, 0x1c4c1802},
                                                               {0x566fd46b, 0x3089f970, 0x43caa21e, 0x255647a0},
                                                               {0x46f4ba1a, 0x58ce9c27, 0x0598f0ce, 0x2171a643},
                                                               {0x0d642001, 0x136dc988, 0x473f459e, 0x0b5d87e0},
                                                               {0x179badf4, 0x1483aa25, 0x75705e01, 0x28d97aaa},
                                                               {0x72e88195, 0x4e2bde0c, 0x20f375a6, 0x3cd08ac5},
                                                               {0x1bf5b2cc, 0x70d248b3, 0x3c7213de, 0x2e2e1d27},
                                                               {0x3381d94b, 0x52ca89e9, 0x34c7f076, 0x084ef5c7},
                                                               {0x55dddca4, 0x11f8994a, 0x3edd8395, 0x648f8cfa},
                                                               {0x4ae73e3d, 0x576f053d, 0x202bcff6, 0x05da741a},
                                                               {0x717f53e0, 0x49dd01c9, 0x6a7bbcb5, 0x0ad12c52},
                                                               {0x75e27fdf, 0x2c094a00, 0x5feae4e7, 0x5979b93b},
                                                               {0x5e8f3638, 0x57a69b3c, 0x4d1784f4, 0x0e43dff3},
                                                               {0x2f536701, 0x2b4eaba5, 0x0a4dddc6, 0x6014133e},
                                                               {0x25f9557f, 0x549577ba, 0x023325c9, 0x5a6f12a2},
                                                               {0x4185e1de, 0x421a22a4, 0x656b65ac, 0x55e80029},
                                                               {0x6e139501, 0x74abc7aa, 0x3b611b89, 0x1d15399e},
                                                               {0x6a5cf915, 0x4940ff04, 0x03f1a58f, 0x0fb83970},
                                                               {0x02242eb3, 0x40dc11a9, 0x213ccf68, 0x7439659a},
                                                               {0x3f708ec0, 0x1a8608eb, 0x46bebe8b, 0x7f29156b},
                                                               {0x7d34cd0e, 0x3b8e7580, 0x6432ec69, 0x72bbb99b},
                                                               {0x247f1f64, 0x6ad059e7, 0x32f1b718, 0x54fa358d},
                                                               {0x6937e06d, 0x242bb010, 0x7db95f01, 0x7d8ee106},
                                                               {0x14cde955, 0x3dac3756, 0x1e476a2e, 0x689083b6}};
    
    
    __device__ __constant__ const uint32_t MWC32_J95[32][4] = {{0xb96ddfe3, 0x2ebdb71e, 0x20412e62, 0x07edad29},
                                                               {0x79568e75, 0xa28dcfa9, 0xd9a5a6a3, 0x07d52206},
                                                               {0xeac03a8a, 0x705a99b1, 0x9e61e285, 0x11459f10},
                                                               {0x85931057, 0x54551930, 0xfe509458, 0x0648e373},
                                                               {0x1b86a079, 0x87d301dc, 0xe8e09714, 0x02227d52},
                                                               {0x5626dc04, 0xc7910ee6, 0x047af013, 0x10f9b3cf},
                                                               {0xee2207c4, 0xa7868878, 0xa4f7ffee, 0x0e894454},
                                                               {0x4fd818c1, 0x3bd42ebb, 0xe1329c0c, 0x06a180e6},
                                                               {0x85df769d, 0x6ba49b6c, 0xa5dfcece, 0x064dba9b},
                                                               {0x21990278, 0x64b23c01, 0xfad655f7, 0x0b268fe2},
                                                               {0x009f22de, 0xe991c11e, 0xa0c38455, 0x08a49ddf},
                                                               {0xeeaba826, 0xa9a6f998, 0xdc3a8644, 0x03056b14},
                                                               {0x81c578ce, 0x70d5bc1e, 0xcab88a5d, 0x012980ec},
                                                               {0x15ba40e7, 0x1b4be268, 0xfc3811b9, 0x0236ccd6},
                                                               {0xc40a022a, 0x4cbc6346, 0x7cadbeec, 0x036d3790},
                                                               {0x929e5a10, 0x8f658eb7, 0x9552a200, 0x08fba211},
                                                               {0x5e8ecb9d, 0x4da8dbfd, 0x3bcec0a1, 0x0739ce68},
                                                               {0xb155d325, 0xd94397f2, 0x64fce906, 0x000574e5},
                                                               {0x9d48fdc2, 0x39cb0403, 0x4fed3b44, 0x06a79998},
                                                               {0x8aa0ed19, 0xe4597737, 0xf5df3125, 0x11984bed},
                                                               {0x61aef6ec, 0xbe123130, 0xb01f3a82, 0x02d5a30c},
                                                               {0x19858a64, 0x8fb04023, 0x9f5d2ecb, 0x029da87b},
                                                               {0xa7d15759, 0x0b3ff5bd, 0xb497a071, 0x05bcafc6},
                                                               {0xf63b0152, 0xa4618a61, 0xa6fe8642, 0x0c8125b0},
                                                               {0xde005f92, 0xe6dfc6f9, 0x5e628747, 0x050dfaa2},
                                                               {0x656d010a, 0x39ef39b2, 0xe72c57eb, 0x0987a0a3},
                                                               {0xaa81aef1, 0xdcc52768, 0xdce79469, 0x08c53257},
                                                               {0x06401cb1, 0x7c1d1dc1, 0xa37e8406, 0x0f1456d8},
                                                               {0x6ed7c99c, 0x15dadacc, 0x55880c04, 0x0af06d27},
                                                               {0x1611730a, 0x81e13432, 0xb01fc9e6, 0x072bfab1},
                                                               {0xda50c0d7, 0x257fb03b, 0x12737585, 0x0f3741ec},
                                                               {0xa579f5ff, 0x081dca6b, 0x60609c75, 0x114cb649}};
    
    
    __device__ __constant__ const uint32_t Xoshiro128PlusPlus_J96[32][4] = {{0xb523952e, 0x0b6f099f, 0xccf5a0ef, 0x1c580662},
                                                                            {0xeeb0e0a4, 0x77133e23, 0xdc596025, 0x97f55fe2},
                                                                            {0x9e9b45ac, 0x6d495900, 0x69ac41e5, 0x0356e935},
                                                                            {0x407883f3, 0x547d4854, 0x9065599b, 0x662b6ac9},
                                                                            {0x667ee2de, 0x8a954d8b, 0x6551c593, 0x2fcdf7e4},
                                                                            {0xfb5707aa, 0xdaa2886a, 0xb233cd67, 0x0f4183ca},
                                                                            {0x40dbcd63, 0x8e131a4f, 0x224fc251, 0xc64784ee},
                                                                            {0x4f4db4ff, 0x7b6ea15f, 0xb29e13b7, 0x563b1ea7},
                                                                            {0xbbd3ae5a, 0xebf544e9, 0xd28ec540, 0x5ce3332f},
                                                                            {0xd39c61eb, 0x1f4dd02e, 0x95a4e90f, 0xa9ac90e8},
                                                                            {0x790c846c, 0xd428b915, 0xd2660f23, 0x725dcd70},
                                                                            {0x08eff263, 0xf39ff6c1, 0x513d8ba0, 0xca4404ca},
                                                                            {0x26534b4d, 0xcf8db66b, 0x6102f64b, 0xf84f07e3},
                                                                            {0xa88724c5, 0x0870d7d7, 0x181f9787, 0xdc3d5d45},
                                                                            {0xdba73489, 0x0df0ec1f, 0x43005e2e, 0xd543edf1},
                                                                            {0x6d73a1e7, 0xfe43b2a7, 0xf9a46a20, 0x58859a86},
                                                                            {0xa683b6d0, 0xafc4a733, 0x1bf94979, 0xf904dd9f},
                                                                            {0x2ee03d84, 0x75c74e3d, 0x96efbfd6, 0x7d256f6c},
                                                                            {0x3ad0ebe7, 0x13f14f31, 0x796d291c, 0xa42bbfdd},
                                                                            {0xce04ddb0, 0x1fc44a96, 0xb6a00a91, 0x8a6c4326},
                                                                            {0x4e519967, 0x0d7a869e, 0x40012492, 0x6dc7c036},
                                                                            {0x9e4d0a48, 0x6a86db67, 0xae852b9b, 0x6cc51ceb},
                                                                            {0x5a52e97f, 0x77beacce, 0xb8030b6c, 0x5ead7c39},
                                                                            {0x022cefbe, 0x7d88e3d4, 0x858bbdfe, 0x6b644146},
                                                                            {0x90067a45, 0xb7ce03bc, 0xde4ac3e8, 0x99853a2c},
                                                                            {0xe3a7ccf3, 0x35c9b163, 0xbb5b8048, 0x31ac55d8},
                                                                            {0x8d4a33db, 0x169e96ef, 0x3788b4a3, 0x622cd32e},
                                                                            {0x0513f190, 0x06f60339, 0x93608184, 0x4576959d},
                                                                            {0x1a64167b, 0x05c745c5, 0xe2f50d3a, 0x8abc30fa},
                                                                            {0x1741bb62, 0x3afd4ba4, 0xb268faef, 0x18bf57c6},
                                                                            {0x39b7b7b9, 0x31bb1001, 0xd95f2dcc, 0x5686c6e7},
                                                                            {0x54d81f7e, 0x0453f0fe, 0x3bef4345, 0x9d5e1791}};
}

#endif
