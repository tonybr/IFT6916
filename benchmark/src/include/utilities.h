#ifndef UTILITIES
#define UTILITIES

/* For integers */

__host__ __device__ uint32_t addc(uint32_t x, uint32_t y)
{
#ifdef __CUDA_ARCH__
    asm("addc.u32 %0, %0, %1;" : "+r"(x) : "r"(y));
    
    return x;
#else
    return 0;
#endif
}

__host__ __device__ uint32_t addcc(uint32_t x, uint32_t y)
{
#ifdef __CUDA_ARCH__
    asm("add.cc.u32 %0, %0, %1;" : "+r"(x) : "r"(y));
    
    return x;
#else
    return 0;
#endif
}

__host__ __device__ uint32_t mulh(uint32_t x, uint32_t y)
{
#ifdef __CUDA_ARCH__
    return __umulhi(x, y);
#else
    return 0;
#endif
}

__host__ __device__ uint32_t madl(uint32_t x, uint32_t y, uint32_t z)
{
#ifdef __CUDA_ARCH__
    asm("mad.lo.u32 %0, %1, %2, %0;" : "+r"(z) : "r"(x), "r"(y));
    
    return z;
#else
    return 0;
#endif
}

__host__ __device__ uint64_t madw(uint32_t x, uint32_t y, uint64_t z)
{
#ifdef __CUDA_ARCH__
    asm("mad.wide.u32 %0, %1, %2, %0;" : "+l"(z) : "r"(x), "r"(y));
    
    return z;
#else
    return 0;
#endif
}

/* For data movement */

__host__ __device__ void unpack(uint64_t x, uint32_t& low, uint32_t& high)
{
#ifdef __CUDA_ARCH__
    asm("mov.b64  {%0, %1}, %2;" : "=r"(low), "=r"(high) : "l"(x));
#endif
}

/* For circular shifts */

__host__ __device__ uint32_t rotl(uint32_t x, int k)
{
    return (x << k) | (x >> (32 - k));
}

__host__ __device__ uint64_t rotl64(uint64_t x, int k)
{
    return (x << k) | (x >> (64 - k));
}

__host__ __device__ uint32_t shfl(uint32_t low, uint32_t high, int k)
{
#ifdef __CUDA_ARCH__
    return __funnelshift_l(low, high, k);
#else
    return 0;
#endif
}

__host__ __device__ uint32_t rotl_plus(uint32_t x, int k)
{
#ifdef __CUDA_ARCH__
    uint32_t xl = 0, xr = 0;
    
    asm("shl.b32 %1, %0, %3;\n\t"
        "shr.b32 %2, %0, %4;\n\t"
        "add.u32 %0, %1, %2;"
        : "+r"(x), "+r"(xl), "+r"(xr) : "r"(k), "r"(32 - k));
    
    return x;
#else
    return 0;
#endif
}

__host__ __device__ uint64_t rotl64_squares(uint64_t x)
{
#ifdef __CUDA_ARCH__
    uint64_t xl = 0, xr = 0;
    
    asm("shl.b64 %1, %0, 32;\n\t"
        "shr.b64 %2, %0, 32;\n\t"
        "xor.b64 %0, %1, %2;\n\t"
        : "+l"(x), "+l"(xl), "+l"(xr));
    
    return x;
#else
    return 0;
#endif
}

#endif
