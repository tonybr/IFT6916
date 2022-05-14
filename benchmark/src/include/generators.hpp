#ifndef GENERATORS
#define GENERATORS

#include <stdint.h>
#include <curand_kernel.h>

#include "utilities.h"

/*********************
 **** Base class *****
 *********************/

template<class T, int K>
class MRG
{
protected:
    T state[K];
    
    // Shifts the state to the left
    // and appends the next term
    __host__ __device__ T append(T next)
    {
        for (int i = 0; i < K - 1; i++)
        {
            state[i] = state[i + 1];
        }
        
        state[K - 1] = next;
        
        return next;
    }
    
public:
    // For reference in other templates
    typedef T ReturnType;
    
    __host__ __device__ MRG()
    {
    }
    
    __host__ __device__ MRG(T seed)
    {
        for(int i = 0; i < K; i++)
        {
            state[i] = seed;
        }
    }
    
    __host__ __device__ void nextStream()
    {
        for(int i = 0; i < K; i++)
        {
            state[i] += 1;
        }
    }
};

/*********************
 ******* MRGs ********
 *********************/

class MRGk5_93_A: public MRG<uint32_t, 5>
{
    using MRG::MRG;
    
public:
    __host__ __device__ uint32_t next()
    {
        // Using the modulo operator
        uint64_t z = madw(state[4], 107374182, (uint64_t)state[0] * 104480);
        
        uint32_t r = (uint32_t)(z % 2147483647);
        
        return append(r);
    }
    
    __host__ __device__ uint64_t mod()
    {
        return 2147483647;
    }
    
    __host__ __device__ float div()
    {
        return 4.65661287308e-10f;
    }
};

class MRGk5_93_B: public MRG<uint32_t, 5>
{
    using MRG::MRG;
    
public:
    __host__ __device__ uint32_t next()
    {
        // Using approximate factoring
        // and reciprocal multiplication
        
        uint32_t q5 = mulh(state[0], 3423603395) >> 14;
        uint32_t q1 = mulh(state[4], 3435973837) >> 4;
        
        int32_t z5 = madl(q5, 4294965569, madl(q5, 4294946742, state[0]) * 104480);
        int32_t z1 = madl(q1, 4294967289, madl(q1, 4294967276, state[4]) * 107374182);
        
        if(z5 < 0)
            z5 += 2147483647;
        if(z1 < 0)
            z1 += 2147483647;
        
        uint32_t r = z5 + z1;
        
        if(r >= 2147483647)
            r += 2147483649;
        
        return append(r);
    }
    
    __host__ __device__ uint64_t mod()
    {
        return 2147483647;
    }
    
    __host__ __device__ float div()
    {
        return 4.65661287308e-10f;
    }
};

class MRGk5_93_C: public MRG<uint32_t, 5>
{
    using MRG::MRG;
    
public:
    __host__ __device__ uint32_t next()
    {
        // Algorithm 3.3.1
        uint64_t z = (uint64_t)state[0] * 208960 + (uint64_t)state[4] * 214748364;
        
        uint32_t r = ((uint32_t)z >> 1) + (uint32_t)(z >> 32);
        
        if(r >= 2147483647)
            r += 2147483649;
        
        return append(r);
    }
    
    __host__ __device__ uint64_t mod()
    {
        return 2147483647;
    }
    
    __host__ __device__ float div()
    {
        return 4.65661287308e-10f;
    }
};

class MRGk5_93_D: public MRG<uint32_t, 5>
{
    using MRG::MRG;
    
public:
    __host__ __device__ uint32_t next()
    {
        // Algorithm 3.3.1, but using the ternary operator
        uint64_t z = (uint64_t)state[0] * 208960 + (uint64_t)state[4] * 214748364;
        
        uint32_t r = ((uint32_t)z >> 1) + (uint32_t)(z >> 32);
        
        r = (r >= 2147483647) ? r + 2147483649 : r;
        
        return append(r);
    }
    
    __host__ __device__ uint64_t mod()
    {
        return 2147483647;
    }
    
    __host__ __device__ float div()
    {
        return 4.65661287308e-10f;
    }
};

class MRGk5_93_E: public MRG<uint32_t, 5>
{
    using MRG::MRG;
    
public:
    __host__ __device__ uint32_t next()
    {
        // Algorithm 3.3.1, but using a multiplication
        uint64_t z = (uint64_t)state[0] * 208960 + (uint64_t)state[4] * 214748364;
        
        uint32_t r = ((uint32_t)z >> 1) + (uint32_t)(z >> 32);
        
        r = madl(r >= 2147483647, 2147483649, r);
        
        return append(r);
    }
    
    __host__ __device__ uint64_t mod()
    {
        return 2147483647;
    }
    
    __host__ __device__ float div()
    {
        return 4.65661287308e-10f;
    }
};

class MRGk5_93_F: public MRG<uint32_t, 5>
{
    using MRG::MRG;
    
public:
    __host__ __device__ uint32_t next()
    {
        // Algorithm 3.3.1, but "unpacking" the sum
        uint64_t z = (uint64_t)state[0] * 208960 + (uint64_t)state[4] * 214748364;
        
        uint32_t zl, zh; unpack(z, zl, zh);
        
        uint32_t r = (zl >> 1) + zh;
        
        if(r >= 2147483647)
            r += 2147483649;
        
        return append(r);
    }
    
    __host__ __device__ uint64_t mod()
    {
        return 2147483647;
    }
    
    __host__ __device__ float div()
    {
        return 4.65661287308e-10f;
    }
};

class MRGk5_93_G: public MRG<uint32_t, 5>
{
    using MRG::MRG;
    
public:
    __host__ __device__ uint32_t next()
    {
        // Algorithm 3.3.1, but "unpacking" each term
        uint64_t z5 = (uint64_t)state[0] * 208960;
        uint64_t z1 = (uint64_t)state[4] * 214748364;
        
        uint32_t zl5, zh5; unpack(z5, zl5, zh5);
        uint32_t zl1, zh1; unpack(z1, zl1, zh1);
        
        uint32_t zl = addcc(zl5, zl1);
        uint32_t zh = addc(zh5, zh1);
        
        uint32_t r = (zl >> 1) + zh;
        
        if(r >= 2147483647)
            r += 2147483649;
        
        return append(r);
    }
    
    __host__ __device__ uint64_t mod()
    {
        return 2147483647;
    }
    
    __host__ __device__ float div()
    {
        return 4.65661287308e-10f;
    }
};

class MRGk5_93_H: public MRG<uint32_t, 5>
{
    using MRG::MRG;
    
public:
    __host__ __device__ uint32_t next()
    {
        // Algorithm 3.3.1, but using mull and mulh
        uint32_t zl5 = state[0] * 208960;
        uint32_t zh5 = mulh(state[0], 208960);
        
        uint32_t zl1 = state[4] * 214748364;
        uint32_t zh1 = mulh(state[4], 214748364);
        
        uint32_t zl = addcc(zl5, zl1);
        uint32_t zh = addc(zh5, zh1);
        
        uint32_t r = (zl >> 1) + zh;
        
        if(r >= 2147483647)
            r += 2147483649;
        
        return append(r);
    }
    
    __host__ __device__ uint64_t mod()
    {
        return 2147483647;
    }
    
    __host__ __device__ float div()
    {
        return 4.65661287308e-10f;
    }
};

class MRGk5_93_I: public MRG<uint32_t, 5>
{
    using MRG::MRG;
    
public:
    
    __host__ __device__ uint32_t next()
    {
        // Algorithm 3.3.3
        uint64_t z = (uint64_t)state[0] * 104480 + (uint64_t)state[4] * 107374182;
        
        uint32_t r = (uint32_t)z + (uint32_t)(z >> 31) * 2147483649;
        
        if(r >= 2147483647)
            r += 2147483649;
        
        return append(r);
    }
    
    __host__ __device__ uint64_t mod()
    {
        return 2147483647;
    }
    
    __host__ __device__ float div()
    {
        return 4.65661287308e-10f;
    }
};

// Previous tests for MRGk5-93

// 1- Using the functions madw and/or madl
// 2- Algorithm 3.3.1, but using regular coefficients
// 3- Algorithm 3.3.1, but using (r + 1) >> 31 instead of r >= 2147483647
// 4- Algorithm 3.3.3, but using mull and mulh (see below)

// class MRGk5_93_J: public MRG<uint32_t, 5>
// {
    // using MRG::MRG;
    
// public:
    
    // __host__ __device__ uint32_t next()
    // {
        // // Algorithm 3.3.3, but using mull and mulh
        
        // // Coefficients are shifted by 32 - B'' in zh, and zh is shifted by B'' - B',
        // // where B'' = B - floor(lg K) for truncating each term + 1 for exact condition.
        
        // uint32_t zl = state[0] * 104480 + state[4] * 107374182;
        // uint32_t zh = mulh(state[0], 835840) + mulh(state[4], 858993456);
        
        // uint32_t r = zl + (zh >> 2) * 2147483649;
        
        // if(r >= 2147483647)
            // r += 2147483649;
        
        // return append(r);
    // }
    
    // __host__ __device__ uint64_t mod()
    // {
        // return 2147483647;
    // }
    
    // __host__ __device__ float div()
    // {
        // return 4.65661287308e-10f;
    // }
// };

// 5- Using an improved version of the modulo operator (see below)

// class MRGk5_93_K: public MRG<uint32_t, 5>
// {
    // using MRG::MRG;
    
// public:
    // __host__ __device__ uint32_t next()
    // {
        // // Using an improved version of the modulo operator
        // uint64_t z = (uint64_t)state[0] * 104480 + (uint64_t)state[4] * 107374182;
        
        // uint32_t q = (uint32_t)((mulh64(z, 8589934597) + z) >> 31);
        
        // uint32_t r = (uint32_t)z + q * 2147483649;
        
        // return append(r);
    // }
    
    // __host__ __device__ uint64_t mod()
    // {
        // return 2147483647;
    // }
    
    // __host__ __device__ float div()
    // {
        // return 4.65661287308e-10f;
    // }
// };

class MRG31a_C: public MRG<uint32_t, 3>
{
    using MRG::MRG;
    
public:
    __host__ __device__ uint32_t next()
    {
        // Algorithm 3.3.1
        uint64_t z = (uint64_t)state[0] * 208960 + (uint64_t)state[2] * 214748364;
        
        uint32_t r = ((uint32_t)z >> 1) + (uint32_t)(z >> 32);
        
        if(r >= 2147483647)
            r += 2147483649;
        
        return append(r);
    }
    
    __host__ __device__ uint64_t mod()
    {
        return 2147483647;
    }
    
    __host__ __device__ float div()
    {
        return 4.65661287308e-10f;
    }
};

class MRG31b_A: public MRG<uint32_t, 3>
{
    using MRG::MRG;
    
public:
    __host__ __device__ uint32_t next()
    {
        uint64_t z = (uint64_t)state[0] * 104480 + (uint64_t)state[1] * 107374182 + (uint64_t)state[2] * 1430957267;
        
        uint32_t r = (uint32_t)(z % 2147483647);
        
        return append(r);
    }
    
    __host__ __device__ uint64_t mod()
    {
        return 2147483647;
    }
    
    __host__ __device__ float div()
    {
        return 4.65661287308e-10f;
    }
};

class MRG31b_C: public MRG<uint32_t, 3>
{
    using MRG::MRG;
    
public:
    __host__ __device__ uint32_t next()
    {
        // Algorithm 3.3.1
        uint64_t z = (uint64_t)state[0] * 208960 + (uint64_t)state[1] * 214748364 + (uint64_t)state[2] * 2861914534;
        
        uint32_t r = ((uint32_t)z >> 1) + (uint32_t)(z >> 32);
        
        if(r >= 2147483647)
            r += 2147483649;
        
        return append(r);
    }
    
    __host__ __device__ uint64_t mod()
    {
        return 2147483647;
    }
    
    __host__ __device__ float div()
    {
        return 4.65661287308e-10f;
    }
};

class MRG31c_A: public MRG<uint32_t, 3>
{
    using MRG::MRG;
    
public:
    __host__ __device__ uint32_t next()
    {
        uint64_t z = (uint64_t)state[0] * 107374182 + (uint64_t)state[1] * 107374182 + (uint64_t)state[2] * 1430957267;
        
        uint32_t r = (uint32_t)(z % 2147483647);
        
        return append(r);
    }
    
    __host__ __device__ uint64_t mod()
    {
        return 2147483647;
    }
    
    __host__ __device__ float div()
    {
        return 4.65661287308e-10f;
    }
};

class MRG31c_C: public MRG<uint32_t, 3>
{
    using MRG::MRG;
    
public:
    __host__ __device__ uint32_t next()
    {
        // Algorithm 3.3.1
        uint64_t z = madw(state[0] + state[1], 214748364, (uint64_t)state[2] * 2861914534);
        
        uint32_t r = ((uint32_t)z >> 1) + (uint32_t)(z >> 32);
        
        if(r >= 2147483647)
            r += 2147483649;
        
        return append(r);
    }
    
    __host__ __device__ uint64_t mod()
    {
        return 2147483647;
    }
    
    __host__ __device__ float div()
    {
        return 4.65661287308e-10f;
    }
};

class MRG31d_C: public MRG<uint32_t, 2>
{
    using MRG::MRG;
    
public:
    __host__ __device__ uint32_t next()
    {
        // Algorithm 3.3.1
        uint64_t z = (uint64_t)state[0] * 208960 + (uint64_t)state[1] * 214748364 + 2861914534;
        
        uint32_t r = ((uint32_t)z >> 1) + (uint32_t)(z >> 32);
        
        if(r >= 2147483647)
            r += 2147483649;
        
        return append(r);
    }
    
    __host__ __device__ uint64_t mod()
    {
        return 2147483647;
    }
    
    __host__ __device__ float div()
    {
        return 4.65661287308e-10f;
    }
};

class MRG31e_A: public MRG<uint32_t, 3>
{
    using MRG::MRG;
    
public:
    __host__ __device__ uint32_t next()
    {
        uint64_t z = (uint64_t)state[0] * 128 + (uint64_t)state[1] * 32768 + (uint64_t)state[2] * 4194304;
        
        uint32_t r = (uint32_t)(z % 2147483647);
        
        return append(r);
    }
    
    __host__ __device__ uint64_t mod()
    {
        return 2147483647;
    }
    
    __host__ __device__ float div()
    {
        return 4.65661287308e-10f;
    }
};

class MRG31e_C: public MRG<uint32_t, 3>
{
    using MRG::MRG;
    
public:
    __host__ __device__ uint32_t next()
    {
        // Algorithm 3.3.1
        uint64_t z = (uint64_t)state[0] * 256 + (uint64_t)state[1] * 65536 + (uint64_t)state[2] * 8388608;
        
        uint32_t r = ((uint32_t)z >> 1) + (uint32_t)(z >> 32);
        
        if(r >= 2147483647)
            r += 2147483649;
        
        return append(r);
    }
    
    __host__ __device__ uint64_t mod()
    {
        return 2147483647;
    }
    
    __host__ __device__ float div()
    {
        return 4.65661287308e-10f;
    }
};

class MRG31e_J: public MRG<uint32_t, 3>
{
    using MRG::MRG;
    
public:
    __host__ __device__ uint32_t next()
    {
        // Using the power-of-two decomposition
        uint32_t z3 = (state[0] >> 24) | ((state[0] & 16777215) << 7);
        uint32_t z2 = (state[1] >> 16) | ((state[1] & 65535) << 15);
        uint32_t z1 = (state[2] >>  9) | ((state[2] & 511) << 22);
        
        uint32_t r = z3 + z2;
        
        if(r >= 2147483647)
            r += 2147483649;
        
        r += z1;
        
        if(r >= 2147483647)
            r += 2147483649;
        
        return append(r);
    }
    
    __host__ __device__ uint64_t mod()
    {
        return 2147483647;
    }
    
    __host__ __device__ float div()
    {
        return 4.65661287308e-10f;
    }
};

class MRG31e_K: public MRG<uint32_t, 3>
{
    using MRG::MRG;
    
public:
    __host__ __device__ uint32_t next()
    {
        // Same as MRG31e_J, but using a funnel shift
        uint32_t z3 = shfl(state[0] << 1, state[0],  7) & 2147483647;
        uint32_t z2 = shfl(state[1] << 1, state[1], 15) & 2147483647;
        uint32_t z1 = shfl(state[2] << 1, state[2], 22) & 2147483647;
        
        uint32_t r = z3 + z2;
        
        if(r >= 2147483647)
            r += 2147483649;
        
        r += z1;
        
        if(r >= 2147483647)
            r += 2147483649;
        
        return append(r);
    }
    
    __host__ __device__ uint64_t mod()
    {
        return 2147483647;
    }
    
    __host__ __device__ float div()
    {
        return 4.65661287308e-10f;
    }
};

class MRG31e_L: public MRG<double, 3>
{
    using MRG::MRG;
    
public:
    __host__ __device__ double next()
    {
        // Using floating-point arithmetic
        double z = state[0] * 128 + state[1] * 32768 + state[2] * 4194304;
        
        double r = z - (double)(uint32_t)(z * (1.0 / 2147483647)) * 2147483647;
        
        return append(r);
    }
    
    __host__ __device__ uint64_t mod()
    {
        return 2147483647;
    }
    
    __host__ __device__ float div()
    {
        return 4.65661287308e-10f;
    }
};

// Previous tests for MRG31e

// 1- Algorithm 3.3.1, but using mull and mulh

/*********************
 *** Combined MRGs ***
 *********************/

class MRG31k3p_A
{
    uint32_t state[2][3];
    
public:
    typedef uint32_t ReturnType;
    
    __host__ __device__ MRG31k3p_A()
    {
    }
    
    __host__ __device__ MRG31k3p_A(uint32_t seed)
    {
        state[0][0] = seed;
        state[0][1] = seed;
        state[0][2] = seed;
        state[1][0] = seed;
        state[1][1] = seed;
        state[1][2] = seed;
    }
    
    __host__ __device__ uint32_t next()
    {
        // Using the power-of-two decomposition
        
        // First component
        uint32_t r1 = ((state[0][0] >> 24) | ((state[0][0] & 16777215) << 7))
                    + ((state[0][1] >>  9) | ((state[0][1] & 511) << 22));
        
        if(r1 >= 2147483647)
            r1 += 2147483649;
        
        r1 += state[0][0];
        
        if(r1 >= 2147483647)
            r1 += 2147483649;
        
        state[0][0] = state[0][1];
        state[0][1] = state[0][2];
        state[0][2] = r1;
        
        // Second component
        uint32_t r2 = (state[1][2] >> 16) * 21069 + ((state[1][2] & 65535) << 15);
        uint32_t r3 = (state[1][0] >> 16) * 21069 + ((state[1][0] & 65535) << 15);
        
        if(r2 >= 2147462579)
            r2 += 2147504717;
        if(r3 >= 2147462579)
            r3 += 2147504717;
        
        r2 += r3;
        
        if(r2 >= 2147462579)
            r2 += 2147504717;
        
        r2 += state[1][0];
        
        if(r2 >= 2147462579)
            r2 += 2147504717;
        
        state[1][0] = state[1][1];
        state[1][1] = state[1][2];
        state[1][2] = r2;
        
        // Combining the components
        uint32_t r = r1 - r2;
        
        if(r1 <= r2)
            r += 2147483647;
        
        return r;
    }
    
    __host__ __device__ void nextStream()
    {
        state[0][0] += 1;
        state[0][1] += 1;
        state[0][2] += 1;
        state[1][0] += 1;
        state[1][1] += 1;
        state[1][2] += 1;
    }
    
    __host__ __device__ uint64_t mod()
    {
        return 2147483647;
    }
    
    __host__ __device__ float div()
    {
        return 4.65661287308e-10f;
    }
};

class MRG31k3p_B
{
    uint32_t state[2][3];
    
public:
    typedef uint32_t ReturnType;
    
    __host__ __device__ MRG31k3p_B()
    {
    }
    
    __host__ __device__ MRG31k3p_B(uint32_t seed)
    {
        state[0][0] = seed;
        state[0][1] = seed;
        state[0][2] = seed;
        state[1][0] = seed;
        state[1][1] = seed;
        state[1][2] = seed;
    }
    
    __host__ __device__ uint32_t next()
    {
        // Same as MRG31k3p_A, but combining the components first
        uint32_t r = state[0][2] - state[1][2];
        
        if(state[0][2] <= state[1][2])
            r += 2147483647;
        
        // First component
        uint32_t r1 = ((state[0][0] >> 24) | ((state[0][0] & 16777215) << 7))
                    + ((state[0][1] >>  9) | ((state[0][1] & 511) << 22));
        
        if(r1 >= 2147483647)
            r1 += 2147483649;
        
        r1 += state[0][0];
        
        if(r1 >= 2147483647)
            r1 += 2147483649;
        
        state[0][0] = state[0][1];
        state[0][1] = state[0][2];
        state[0][2] = r1;
        
        // Second component
        uint32_t r2 = (state[1][2] >> 16) * 21069 + ((state[1][2] & 65535) << 15);
        uint32_t r3 = (state[1][0] >> 16) * 21069 + ((state[1][0] & 65535) << 15);
        
        if(r2 >= 2147462579)
            r2 += 2147504717;
        if(r3 >= 2147462579)
            r3 += 2147504717;
        
        r2 += r3;
        
        if(r2 >= 2147462579)
            r2 += 2147504717;
        
        r2 += state[1][0];
        
        if(r2 >= 2147462579)
            r2 += 2147504717;
        
        state[1][0] = state[1][1];
        state[1][1] = state[1][2];
        state[1][2] = r2;
        
        return r;
    }
    
    __host__ __device__ void nextStream()
    {
        state[0][0] += 1;
        state[0][1] += 1;
        state[0][2] += 1;
        state[1][0] += 1;
        state[1][1] += 1;
        state[1][2] += 1;
    }
    
    __host__ __device__ uint64_t mod()
    {
        return 2147483647;
    }
    
    __host__ __device__ float div()
    {
        return 4.65661287308e-10f;
    }
};

class MRG31k3p_C
{
    uint32_t state[2][3];
    
public:
    typedef uint32_t ReturnType;
    
    __host__ __device__ MRG31k3p_C()
    {
    }
    
    __host__ __device__ MRG31k3p_C(uint32_t seed)
    {
        state[0][0] = seed;
        state[0][1] = seed;
        state[0][2] = seed;
        state[1][0] = seed;
        state[1][1] = seed;
        state[1][2] = seed;
    }
    
    __host__ __device__ uint32_t next()
    {
        // Algorithm 3.3.1
        uint64_t z1 = (uint64_t)state[0][0] * 258 + (uint64_t)state[0][1] * 8388608;
        
        uint32_t r1 = ((uint32_t)z1 >> 1) + (uint32_t)(z1 >> 32);
        
        if(r1 >= 2147483647)
            r1 += 2147483649;
        
        state[0][0] = state[0][1];
        state[0][1] = state[0][2];
        state[0][2] = r1;
        
        // Algorithm 3.3.1
        uint64_t z2 = (uint64_t)state[1][0] * 65538 + (uint64_t)state[1][2] * 65536;
        
        uint32_t r2 = ((uint32_t)z2 >> 1) + (uint32_t)(z2 >> 32) * 21069;
        
        if(r2 >= 2147462579)
            r2 += 2147504717;
        
        state[1][0] = state[1][1];
        state[1][1] = state[1][2];
        state[1][2] = r2;
        
        // Combining the components
        uint32_t r = r1 - r2;
        
        if(r1 <= r2)
            r += 2147483647;
        
        return r;
    }
    
    __host__ __device__ void nextStream()
    {
        state[0][0] += 1;
        state[0][1] += 1;
        state[0][2] += 1;
        state[1][0] += 1;
        state[1][1] += 1;
        state[1][2] += 1;
    }
    
    __host__ __device__ uint64_t mod()
    {
        return 2147483647;
    }
    
    __host__ __device__ float div()
    {
        return 4.65661287308e-10f;
    }
};

class MRG31k3p_D
{
    uint32_t state[2][3];
    
public:
    typedef uint32_t ReturnType;
    
    __host__ __device__ MRG31k3p_D()
    {
    }
    
    __host__ __device__ MRG31k3p_D(uint32_t seed)
    {
        state[0][0] = seed;
        state[0][1] = seed;
        state[0][2] = seed;
        state[1][0] = seed;
        state[1][1] = seed;
        state[1][2] = seed;
    }
    
    __host__ __device__ uint32_t next()
    {
        // Algorithm 3.3.1, but using 32-bit shifts
        uint32_t zl13 = state[0][0] * 258;
        uint32_t zh13 = mulh(state[0][0], 258);
        
        uint32_t zl1 = addcc(zl13, state[0][1] << 23);
        uint32_t zh1 = addc(zh13, state[0][1] >> 9);
        
        uint32_t r1 = (zl1 >> 1) + zh1;
        
        if(r1 >= 2147483647)
            r1 += 2147483649;
        
        state[0][0] = state[0][1];
        state[0][1] = state[0][2];
        state[0][2] = r1;
        
        // Algorithm 3.3.1, but using 32-bit shifts
        uint32_t zl23 = state[1][0] * 65538;
        uint32_t zh23 = mulh(state[1][0], 65538);
        
        uint32_t zl2 = addcc(zl23, state[1][2] << 16);
        uint32_t zh2 = addc(zh23, state[1][2] >> 16);
        
        uint32_t r2 = (zl2 >> 1) + zh2 * 21069;
        
        if(r2 >= 2147462579)
            r2 += 2147504717;
        
        state[1][0] = state[1][1];
        state[1][1] = state[1][2];
        state[1][2] = r2;
        
        // Combining the components
        uint32_t r = r1 - r2;
        
        if(r1 <= r2)
            r += 2147483647;
        
        return r;
    }
    
    __host__ __device__ void nextStream()
    {
        state[0][0] += 1;
        state[0][1] += 1;
        state[0][2] += 1;
        state[1][0] += 1;
        state[1][1] += 1;
        state[1][2] += 1;
    }
    
    __host__ __device__ uint64_t mod()
    {
        return 2147483647;
    }
    
    __host__ __device__ float div()
    {
        return 4.65661287308e-10f;
    }
};

// Previous tests for MRG31k3p

// 1- Using r <= 0 instead of r1 <= r2

class MRG32k3a_A
{
    curandStateMRG32k3a_t state;
    
public:
    typedef double ReturnType;
    
    __host__ __device__ MRG32k3a_A()
    {
    }
    
    __host__ __device__ MRG32k3a_A(uint32_t seed)
    {
        state.s1[0] = seed;
        state.s1[1] = seed;
        state.s1[2] = seed;
        state.s2[0] = seed;
        state.s2[1] = seed;
        state.s2[2] = seed;
    }
    
    __host__ __device__ double next()
    {
        // Using cuRAND
    #ifdef __CUDA_ARCH__
        return curand_MRG32k3a(&state);
    #else
        return 0;
    #endif
    }
    
    __host__ __device__ void nextStream()
    {
        state.s1[0] += 1;
        state.s1[1] += 1;
        state.s1[2] += 1;
        state.s2[0] += 1;
        state.s2[1] += 1;
        state.s2[2] += 1;
    }
    
    __host__ __device__ uint64_t mod()
    {
        return 4294967087;
    }
    
    __host__ __device__ float div()
    {
        return 2.32830643654e-10f;
    }
};

class MRG32k3a_B
{
    uint32_t state[2][3];
    
public:
    typedef uint32_t ReturnType;
    
    __host__ __device__ MRG32k3a_B()
    {
    }
    
    __host__ __device__ MRG32k3a_B(uint32_t seed)
    {
        state[0][0] = seed;
        state[0][1] = seed;
        state[0][2] = seed;
        state[1][0] = seed;
        state[1][1] = seed;
        state[1][2] = seed;
    }
    
    __host__ __device__ uint32_t next()
    {
        // Algorithm 3.3.1, but using a 64-bit condition
        uint64_t z1 = madw(4294967087 - state[0][0], 810728, (uint64_t)state[0][1] * 1403580);
        
        uint32_t r1 = (uint32_t)(z1 = madw((uint32_t)(z1 >> 32), 209, (uint32_t)z1));
        
        if(z1 >= 4294967087)
            r1 += 209;
        
        state[0][0] = state[0][1];
        state[0][1] = state[0][2];
        state[0][2] = r1;
        
        // Algorithm 3.3.1, but reducing twice
        uint64_t z2 = madw(4294944443 - state[1][0], 1370589, (uint64_t)state[1][2] * 527612);
        
        z2 = madw((uint32_t)(z2 >> 32), 22853, (uint32_t)z2);
        z2 = madw((uint32_t)(z2 >> 32), 22853, (uint32_t)z2);
        
        uint32_t r2 = (uint32_t)z2;
        
        if(z2 >= 4294944443)
            r2 += 22853;
        
        state[1][0] = state[1][1];
        state[1][1] = state[1][2];
        state[1][2] = r2;
        
        // Combining the components
        uint32_t r = r1 - r2;
        
        if(r1 <= r2)
            r += 4294967087;
        
        return r;
    }
    
    __host__ __device__ void nextStream()
    {
        state[0][0] += 1;
        state[0][1] += 1;
        state[0][2] += 1;
        state[1][0] += 1;
        state[1][1] += 1;
        state[1][2] += 1;
    }
    
    __host__ __device__ uint64_t mod()
    {
        return 4294967087;
    }
    
    __host__ __device__ float div()
    {
        return 2.32830643654e-10f;
    }
};

class MRG32k3a_C
{
    uint32_t state[2][3];
    
public:
    typedef uint32_t ReturnType;
    
    __host__ __device__ MRG32k3a_C()
    {
    }
    
    __host__ __device__ MRG32k3a_C(uint32_t seed)
    {
        state[0][0] = seed;
        state[0][1] = seed;
        state[0][2] = seed;
        state[1][0] = seed;
        state[1][1] = seed;
        state[1][2] = seed;
    }
    
    __host__ __device__ uint32_t next()
    {
        // Algorithm 3.3.1, but using a 64-bit condition
        uint64_t z1 = madw(4294967087 - state[0][0], 810728, (uint64_t)state[0][1] * 1403580);
        
        uint32_t r1 = (uint32_t)(z1 = madw((uint32_t)(z1 >> 32), 209, (uint32_t)z1));
        
        if(z1 >= 4294967087)
            r1 += 209;
        
        state[0][0] = state[0][1];
        state[0][1] = state[0][2];
        state[0][2] = r1;
        
        // Algorithm 3.3.2, but using a 64-bit condition
        uint64_t z2 = madw(4294944443 - state[1][0], 1370589, (uint64_t)state[1][2] * 527612);
        
        uint32_t k2 = mulh((uint32_t)(z2 >> 31), 2147494912);
        
        uint32_t r2 = (uint32_t)(z2 -= (uint64_t)k2 * 4294944443);
        
        if(z2 >= 4294944443)
            r2 += 22853;
        
        state[1][0] = state[1][1];
        state[1][1] = state[1][2];
        state[1][2] = r2;
        
        // Combining the components
        uint32_t r = r1 - r2;
        
        if(r1 <= r2)
            r += 4294967087;
        
        return r;
    }
    
    __host__ __device__ void nextStream()
    {
        state[0][0] += 1;
        state[0][1] += 1;
        state[0][2] += 1;
        state[1][0] += 1;
        state[1][1] += 1;
        state[1][2] += 1;
    }
    
    __host__ __device__ uint64_t mod()
    {
        return 4294967087;
    }
    
    __host__ __device__ float div()
    {
        return 2.32830643654e-10f;
    }
};

// Previous tests for MRG32k3a

// 1- Same as MRG32k3a_C, but using mull and mulh (see below)

// class MRG32k3a_D
// {
    // uint32_t state[2][3];
    
// public:
    // typedef uint32_t ReturnType;
    
    // __host__ __device__ MRG32k3a_D()
    // {
    // }
    
    // __host__ __device__ MRG32k3a_D(uint32_t seed)
    // {
        // state[0][0] = seed;
        // state[0][1] = seed;
        // state[0][2] = seed;
        // state[1][0] = seed;
        // state[1][1] = seed;
        // state[1][2] = seed;
    // }
    
    // __host__ __device__ uint32_t next()
    // {
        // // Same as MRG32k3a_C, but using mull and mulh
        // uint32_t x13 = 4294967087 - state[0][0];
        
        // uint32_t zl13 = x13 * 810728;
        // uint32_t zh13 = mulh(x13, 810728);
        
        // uint32_t zl12 = state[0][1] * 1403580;
        // uint32_t zh12 = mulh(state[0][1], 1403580);
        
        // uint64_t zl1 = addcc(zl13, zl12);
        // uint32_t zh1 = addc(zh13, zh12);
        
        // uint32_t r1 = (uint32_t)(zl1 += zh1 * 209);
        
        // if(zl1 >= 4294967087)
            // r1 += 209;
        
        // state[0][0] = state[0][1];
        // state[0][1] = state[0][2];
        // state[0][2] = r1;
        
        // // Algorithm 3.3.2, but using a 64-bit condition
        // uint64_t z2 = madw(4294944443 - state[1][0], 1370589, (uint64_t)state[1][2] * 527612);
        
        // uint32_t k2 = mulh((uint32_t)(z2 >> 31), 2147494912);
        
        // uint32_t r2 = (uint32_t)(z2 -= (uint64_t)k2 * 4294944443);
        
        // if(z2 >= 4294944443)
            // r2 += 22853;
        
        // state[1][0] = state[1][1];
        // state[1][1] = state[1][2];
        // state[1][2] = r2;
        
        // // Combining the components
        // uint32_t r = r1 - r2;
        
        // if(r1 <= r2)
            // r += 4294967087;
        
        // return r;
    // }
    
    // __host__ __device__ void nextStream()
    // {
        // state[0][0] += 1;
        // state[0][1] += 1;
        // state[0][2] += 1;
        // state[1][0] += 1;
        // state[1][1] += 1;
        // state[1][2] += 1;
    // }
    
    // __host__ __device__ uint64_t mod()
    // {
        // return 4294967087;
    // }
    
    // __host__ __device__ float div()
    // {
        // return 2.32830643654e-10f;
    // }
// };

// 2- Same as MRG32k3a_C, but using 32-bit conditions (see below)

// class MRG32k3a_D
// {
    // uint32_t state[2][3];
    
// public:
    // typedef uint32_t ReturnType;
    
    // __host__ __device__ MRG32k3a_D()
    // {
    // }
    
    // __host__ __device__ MRG32k3a_D(uint32_t seed)
    // {
        // state[0][0] = seed;
        // state[0][1] = seed;
        // state[0][2] = seed;
        // state[1][0] = seed;
        // state[1][1] = seed;
        // state[1][2] = seed;
    // }
    
    // __host__ __device__ uint32_t next()
    // {
        // // Same as MRG32k3a_C, but using 32-bit conditions
        // uint32_t x13 = 4294967087 - state[0][0];
        
        // uint32_t zl13 = x13 * 810728;
        // uint32_t zh13 = mulh(x13, 810728);
        
        // uint32_t zl12 = state[0][1] * 1403580;
        // uint32_t zh12 = mulh(state[0][1], 1403580);
        
        // uint32_t zl1 = addcc(zl13, zl12);
        // uint32_t zh1 = addc(zh13, zh12);
        
        // uint32_t r1 = addcc(zl1, zh1 * 209);
        // uint32_t c1 = addc(0, 0);
        
        // if(r1 >= 4294967087 | c1 == 1)
            // r1 += 209;
        
        // state[0][0] = state[0][1];
        // state[0][1] = state[0][2];
        // state[0][2] = r1;
        
        // // Algorithm 3.3.2, but using 64-bit condition
        // uint64_t z2 = madw(4294944443 - state[1][0], 1370589, (uint64_t)state[1][2] * 527612);
        
        // uint32_t k2 = mulh((uint32_t)(z2 >> 31), 2147494912);
        
        // uint32_t r2 = (uint32_t)(z2 -= (uint64_t)k2 * 4294944443);
        
        // if(z2 >= 4294944443)
            // r2 += 22853;
        
        // state[1][0] = state[1][1];
        // state[1][1] = state[1][2];
        // state[1][2] = r2;
        
        // // Combining the components
        // uint32_t r = r1 - r2;
        
        // if(r1 <= r2)
            // r += 4294967087;
        
        // return r;
    // }
    
    // __host__ __device__ void nextStream()
    // {
        // state[0][0] += 1;
        // state[0][1] += 1;
        // state[0][2] += 1;
        // state[1][0] += 1;
        // state[1][1] += 1;
        // state[1][2] += 1;
    // }
    
    // __host__ __device__ uint64_t mod()
    // {
        // return 4294967087;
    // }
    
    // __host__ __device__ float div()
    // {
        // return 2.32830643654e-10f;
    // }
// };

// 3- Using instructions similar to cuRAND (see below)

// class MRG32k3a_B
// {
    // uint32_t state[2][3];
    
// public:
    // typedef double ReturnType;
    
    // __host__ __device__ MRG32k3a_B()
    // {
    // }
    
    // __host__ __device__ MRG32k3a_B(uint32_t seed)
    // {
        // state[0][0] = seed;
        // state[0][1] = seed;
        // state[0][2] = seed;
        // state[1][0] = seed;
        // state[1][1] = seed;
        // state[1][2] = seed;
    // }
    
    // __host__ __device__ double next()
    // {
        // // Algorithm 3.3.1, but using a 64-bit condition
        // uint64_t z1 = madw(4294967087 - state[0][0], 810728, (uint64_t)state[0][1] * 1403580);
        
        // z1 = (z1 >> 32) * 209 + (uint32_t)z1;
        
        // if(z1 >= 4294967087)
            // z1 -= 4294967087;
        
        // uint32_t r1 = (uint32_t)z1;
        
        // state[0][0] = state[0][1];
        // state[0][1] = state[0][2];
        // state[0][2] = r1;
        
        // // Algorithm 3.3.1, but reducing twice
        // uint64_t z2 = madw(4294944443 - state[1][0], 1370589, (uint64_t)state[1][2] * 527612);
        
        // z2 = (z2 >> 32) * 22853 + (uint32_t)z2;
        // z2 = (z2 >> 32) * 22853 + (uint32_t)z2;
        
        // if(z2 >= 4294944443)
            // z2 -= 4294944443;
        
        // uint32_t r2 = (uint32_t)z2;
        
        // state[1][0] = state[1][1];
        // state[1][1] = state[1][2];
        // state[1][2] = r2;
        
        // // Combining the components
        // uint32_t r = r1 - r2;
        
        // if(r1 <= r2)
            // r += 4294967087;
        
        // return (double)r;
    // }
    
    // __host__ __device__ void nextStream()
    // {
        // state[0][0] += 1;
        // state[0][1] += 1;
        // state[0][2] += 1;
        // state[1][0] += 1;
        // state[1][1] += 1;
        // state[1][2] += 1;
    // }
    
    // __host__ __device__ uint64_t mod()
    // {
        // return 4294967087;
    // }
    
    // __host__ __device__ float div()
    // {
        // return 2.32830643654e-10f;
    // }
// };

/*********************
 ******* MWCs ********
 *********************/

class MWC32a_A: public MRG<uint32_t, 5>
{
    uint32_t carry;
    
public:
    __host__ __device__ MWC32a_A()
    {
    }
    
    __host__ __device__ MWC32a_A(uint32_t seed)
    {
        for(int i = 0; i < 5; i++)
        {
            state[i] = seed;
        }
        
        carry = seed;
    }
    
    __host__ __device__ uint32_t next()
    {
        // Naive implementation
        uint64_t z = (uint64_t)state[0] * 104480 + (uint64_t)state[4] * 107374182 + carry;
        
        uint32_t r = (uint32_t)z; carry = (uint32_t)(z >> 32);
        
        return append(r);
    }
    
    __host__ __device__ void nextStream()
    {
        for(int i = 0; i < 5; i++)
        {
            state[i] += 1;
        }
        
        carry += 1;
    }
    
    __host__ __device__ uint64_t mod()
    {
        return 4294967296L;
    }
    
    __host__ __device__ float div()
    {
        return 2.32830643654e-10f;
    }
};

class MWC32a_B: public MRG<uint32_t, 5>
{
    uint32_t carry;
    
public:
    __host__ __device__ MWC32a_B()
    {
    }
    
    __host__ __device__ MWC32a_B(uint32_t seed)
    {
        for(int i = 0; i < 5; i++)
        {
            state[i] = seed;
        }
        
        carry = seed;
    }
    
    __host__ __device__ uint32_t next()
    {
        // Same as MWC32a_A, but using mull and mulh
        uint32_t zl5 = state[0] * 104480;
        uint32_t zh5 = mulh(state[0], 104480);
        
        uint32_t zl1 = state[4] * 107374182;
        uint32_t zh1 = mulh(state[4], 107374182);
        
        uint32_t zl = addcc(zl5, zl1);
        uint32_t zh = addc(zh5, zh1);
        
        zl = addcc(zl, carry);
        carry = addc(zh, 0);
        
        return append(zl);
    }
    
    __host__ __device__ void nextStream()
    {
        for(int i = 0; i < 5; i++)
        {
            state[i] += 1;
        }
        
        carry += 1;
    }
    
    __host__ __device__ uint64_t mod()
    {
        return 4294967296L;
    }
    
    __host__ __device__ float div()
    {
        return 2.32830643654e-10f;
    }
};

class MWC32b_A: public MRG<uint32_t, 3>
{
    uint32_t carry;
    
public:
    __host__ __device__ MWC32b_A()
    {
    }
    
    __host__ __device__ MWC32b_A(uint32_t seed)
    {
        for(int i = 0; i < 3; i++)
        {
            state[i] = seed;
        }
        
        carry = seed;
    }
    
    __host__ __device__ uint32_t next()
    {
        // Naive implementation
        uint64_t z = (uint64_t)state[0] * 104480 + (uint64_t)state[1] * 107374182 + (uint64_t)state[2] * 1430957267 + carry;
        
        uint32_t r = (uint32_t)z; carry = (uint32_t)(z >> 32);
        
        return append(r);
    }
    
    __host__ __device__ void nextStream()
    {
        for(int i = 0; i < 3; i++)
        {
            state[i] += 1;
        }
        
        carry += 1;
    }
    
    __host__ __device__ uint64_t mod()
    {
        return 4294967296L;
    }
    
    __host__ __device__ float div()
    {
        return 2.32830643654e-10f;
    }
};

class MWC32b_B: public MRG<uint32_t, 3>
{
    uint32_t carry;
    
public:
    __host__ __device__ MWC32b_B()
    {
    }
    
    __host__ __device__ MWC32b_B(uint32_t seed)
    {
        for(int i = 0; i < 3; i++)
        {
            state[i] = seed;
        }
        
        carry = seed;
    }
    
    __host__ __device__ uint32_t next()
    {
        // Same as MWC32b_A, but using mull and mulh
        uint32_t zl3 = state[0] * 104480;
        uint32_t zh3 = mulh(state[0], 104480);
        
        uint32_t zl2 = state[1] * 107374182;
        uint32_t zh2 = mulh(state[1], 107374182);
        
        uint32_t zl1 = state[2] * 1430957267;
        uint32_t zh1 = mulh(state[2], 1430957267);
        
        uint32_t zl = addcc(zl3, zl2);
        uint32_t zh = addc(zh3, zh2);
        
        zl = addcc(zl, zl1);
        zh = addc(zh, zh1);
        
        zl = addcc(zl, carry);
        carry = addc(zh, 0);
        
        return append(zl);
    }
    
    __host__ __device__ void nextStream()
    {
        for(int i = 0; i < 3; i++)
        {
            state[i] += 1;
        }
        
        carry += 1;
    }
    
    __host__ __device__ uint64_t mod()
    {
        return 4294967296L;
    }
    
    __host__ __device__ float div()
    {
        return 2.32830643654e-10f;
    }
};

class MWC32c_A: public MRG<uint32_t, 3>
{
    uint32_t carry;
    
public:
    __host__ __device__ MWC32c_A()
    {
    }
    
    __host__ __device__ MWC32c_A(uint32_t seed)
    {
        for(int i = 0; i < 3; i++)
        {
            state[i] = seed;
        }
        
        carry = seed;
    }
    
    __host__ __device__ uint32_t next()
    {
        // Naive implementation
        uint64_t z = ((uint64_t)state[0] + state[1]) * 107374182 + (uint64_t)state[2] * 1430957267 + carry;
        
        uint32_t r = (uint32_t)z; carry = (uint32_t)(z >> 32);
        
        return append(r);
    }
    
    __host__ __device__ void nextStream()
    {
        for(int i = 0; i < 3; i++)
        {
            state[i] += 1;
        }
        
        carry += 1;
    }
    
    __host__ __device__ uint64_t mod()
    {
        return 4294967296L;
    }
    
    __host__ __device__ float div()
    {
        return 2.32830643654e-10f;
    }
};

class MWC31_A: public MRG<uint32_t, 3>
{
    uint32_t carry;
    
public:
    __host__ __device__ MWC31_A()
    {
    }
    
    __host__ __device__ MWC31_A(uint32_t seed)
    {
        for(int i = 0; i < 3; i++)
        {
            state[i] = seed;
        }
        
        carry = seed;
    }
    
    __host__ __device__ uint32_t next()
    {
        // Naive implementation
        uint64_t z = (uint64_t)(state[0] + state[1]) * 107374182 + (uint64_t)state[2] * 1430957267 + carry;
        
        uint32_t r = (uint32_t)z & 2147483647; carry = (uint32_t)(z >> 31);
        
        return append(r);
    }
    
    __host__ __device__ void nextStream()
    {
        for(int i = 0; i < 3; i++)
        {
            state[i] += 1;
        }
        
        carry += 1;
    }
    
    __host__ __device__ uint64_t mod()
    {
        return 2147483648;
    }
    
    __host__ __device__ float div()
    {
        return 4.65661287308e-10f;
    }
};

class MWC31_B: public MRG<uint32_t, 3>
{
    uint32_t carry;
    
public:
    __host__ __device__ MWC31_B()
    {
    }
    
    __host__ __device__ MWC31_B(uint32_t seed)
    {
        for(int i = 0; i < 3; i++)
        {
            state[i] = seed;
        }
        
        carry = seed;
    }
    
    __host__ __device__ uint32_t next()
    {
        // Same as MWC31_A, but using mull and mulh
        uint32_t x32 = state[0] + state[1];
        
        uint32_t zl3 = x32 * 214748364;
        uint32_t zh3 = mulh(x32, 214748364);
        
        uint32_t zl1 = state[2] * 2861914534;
        uint32_t zh1 = mulh(state[2], 2861914534);
        
        uint32_t zl = addcc(zl3, zl1);
        uint32_t zh = addc(zh3, zh1);
        
        zl = addcc(zl, carry << 1);
        carry = addc(zh, 0);
        
        uint32_t r = zl >> 1;
        
        return append(r);
    }
    
    __host__ __device__ void nextStream()
    {
        for(int i = 0; i < 3; i++)
        {
            state[i] += 1;
        }
        
        carry += 1;
    }
    
    __host__ __device__ uint64_t mod()
    {
        return 2147483648;
    }
    
    __host__ __device__ float div()
    {
        return 4.65661287308e-10f;
    }
};

/*********************
 ******* LFSRs *******
 *********************/

class LFSR88_A: public MRG<uint32_t, 3>
{
    using MRG::MRG;
    
public:
    __host__ __device__ uint32_t next()
    {
        // Same as TestU01
        
        uint32_t r, t;
        
        t = ((state[0] << 13) ^ state[0]) >> 19;
		state[0] = ((state[0] & 4294967294) << 12) ^ t;
        
		t = ((state[1] << 2) ^ state[1]) >> 25;
		state[1] = ((state[1] & 4294967288) << 4) ^ t;
        
		t = ((state[2] << 3) ^ state[2]) >> 11;
		state[2] = ((state[2] & 4294967280) << 17) ^ t;
        
        r = state[0] ^ state[1] ^ state[2];
        
        return r;
    }
    
    __host__ __device__ uint64_t mod()
    {
        return 4294967296L;
    }
    
    __host__ __device__ float div()
    {
        return 2.32830643654e-10f;
    }
};

class LFSR88_B: public MRG<uint32_t, 3>
{
    using MRG::MRG;
    
public:
    __host__ __device__ uint32_t next()
    {
        // Same as LFSR88_A, but combining the components first
        
        uint32_t r, t;
        
        r = state[0] ^ state[1] ^ state[2];
        
        t = ((state[0] << 13) ^ state[0]) >> 19;
		state[0] = ((state[0] & 4294967294) << 12) ^ t;
        
		t = ((state[1] << 2) ^ state[1]) >> 25;
		state[1] = ((state[1] & 4294967288) << 4) ^ t;
        
		t = ((state[2] << 3) ^ state[2]) >> 11;
		state[2] = ((state[2] & 4294967280) << 17) ^ t;
        
        return r;
    }
    
    __host__ __device__ uint64_t mod()
    {
        return 4294967296L;
    }
    
    __host__ __device__ float div()
    {
        return 2.32830643654e-10f;
    }
};

class LFSR88_C: public MRG<uint32_t, 3>
{
    using MRG::MRG;
    
public:
    __host__ __device__ uint32_t next()
    {
        // Same as LFSR88_A, but using some additions
        
        uint32_t r, t;
        
        t = ((state[0] << 13) ^ state[0]) >> 19;
		state[0] = ((state[0] & 4294967294) << 12) + t;
        
		t = ((state[1] << 2) ^ state[1]) >> 25;
		state[1] = ((state[1] & 4294967288) << 4) + t;
        
		t = ((state[2] << 3) ^ state[2]) >> 11;
		state[2] = ((state[2] & 4294967280) << 17) + t;
        
        r = state[0] ^ state[1] ^ state[2];
        
        return r;
    }
    
    __host__ __device__ uint64_t mod()
    {
        return 4294967296L;
    }
    
    __host__ __device__ float div()
    {
        return 2.32830643654e-10f;
    }
};

class LFSR88_D: public MRG<uint32_t, 3>
{
    using MRG::MRG;
    
public:
    __host__ __device__ uint32_t next()
    {
        // Same as LFSR88_A, but using distinct variables
        
        uint32_t r, t1, t2, t3;
        
        t1 = ((state[0] << 13) ^ state[0]) >> 19;
		state[0] = ((state[0] & 4294967294) << 12) ^ t1;
        
		t2 = ((state[1] << 2) ^ state[1]) >> 25;
		state[1] = ((state[1] & 4294967288) << 4) ^ t2;
        
		t3 = ((state[2] << 3) ^ state[2]) >> 11;
		state[2] = ((state[2] & 4294967280) << 17) ^ t3;
        
        r = state[0] ^ state[1] ^ state[2];
        
        return r;
    }
    
    __host__ __device__ uint64_t mod()
    {
        return 4294967296L;
    }
    
    __host__ __device__ float div()
    {
        return 2.32830643654e-10f;
    }
};

class LFSR113_A: public MRG<uint32_t, 4>
{
    using MRG::MRG;
    
public:
    __host__ __device__ uint32_t next()
    {
        uint32_t r, t;
        
        t = ((state[0] << 6) ^ state[0]) >> 13;
		state[0] = ((state[0] & 4294967294) << 18) ^ t;
        
		t = ((state[1] << 2) ^ state[1]) >> 27;
		state[1] = ((state[1] & 4294967288) << 2) ^ t;
        
		t = ((state[2] << 13) ^ state[2]) >> 21;
		state[2] = ((state[2] & 4294967280) << 7) ^ t;
        
        t = ((state[3] << 3) ^ state[3]) >> 12;
		state[3] = ((state[3] & 4294967168) << 13) ^ t;
        
        r = state[0] ^ state[1] ^ state[2] ^ state[3];
        
        return r;
    }
    
    __host__ __device__ uint64_t mod()
    {
        return 4294967296L;
    }
    
    __host__ __device__ float div()
    {
        return 2.32830643654e-10f;
    }
};

class LFSR113_B: public MRG<uint32_t, 4>
{
    using MRG::MRG;
    
public:
    __host__ __device__ uint32_t next()
    {
        uint32_t r, t;
        
        r = state[0] ^ state[1] ^ state[2] ^ state[3];
        
        t = ((state[0] << 6) ^ state[0]) >> 13;
		state[0] = ((state[0] & 4294967294) << 18) ^ t;
        
		t = ((state[1] << 2) ^ state[1]) >> 27;
		state[1] = ((state[1] & 4294967288) << 2) ^ t;
        
		t = ((state[2] << 13) ^ state[2]) >> 21;
		state[2] = ((state[2] & 4294967280) << 7) ^ t;
        
        t = ((state[3] << 3) ^ state[3]) >> 12;
		state[3] = ((state[3] & 4294967168) << 13) ^ t;
        
        return r;
    }
    
    __host__ __device__ uint64_t mod()
    {
        return 4294967296L;
    }
    
    __host__ __device__ float div()
    {
        return 2.32830643654e-10f;
    }
};

class LFSR113_C: public MRG<uint32_t, 4>
{
    using MRG::MRG;
    
public:
    __host__ __device__ uint32_t next()
    {
        uint32_t r, t;
        
        t = ((state[0] << 6) ^ state[0]) >> 13;
		state[0] = ((state[0] & 4294967294) << 18) + t;
        
		t = ((state[1] << 2) ^ state[1]) >> 27;
		state[1] = ((state[1] & 4294967288) << 2) + t;
        
		t = ((state[2] << 13) ^ state[2]) >> 21;
		state[2] = ((state[2] & 4294967280) << 7) + t;
        
        t = ((state[3] << 3) ^ state[3]) >> 12;
		state[3] = ((state[3] & 4294967168) << 13) + t;
        
        r = state[0] ^ state[1] ^ state[2] ^ state[3];
        
        return r;
    }
    
    __host__ __device__ uint64_t mod()
    {
        return 4294967296L;
    }
    
    __host__ __device__ float div()
    {
        return 2.32830643654e-10f;
    }
};

class LFSR113_D: public MRG<uint32_t, 4>
{
    using MRG::MRG;
    
public:
    __host__ __device__ uint32_t next()
    {
        uint32_t r, t1, t2, t3, t4;
        
        t1 = ((state[0] << 6) ^ state[0]) >> 13;
		state[0] = ((state[0] & 4294967294) << 18) ^ t1;
        
		t2 = ((state[1] << 2) ^ state[1]) >> 27;
		state[1] = ((state[1] & 4294967288) << 2) ^ t2;
        
		t3 = ((state[2] << 13) ^ state[2]) >> 21;
		state[2] = ((state[2] & 4294967280) << 7) ^ t3;
        
        t4 = ((state[3] << 3) ^ state[3]) >> 12;
		state[3] = ((state[3] & 4294967168) << 13) ^ t4;
        
        r = state[0] ^ state[1] ^ state[2] ^ state[3];
        
        return r;
    }
    
    __host__ __device__ uint64_t mod()
    {
        return 4294967296L;
    }
    
    __host__ __device__ float div()
    {
        return 2.32830643654e-10f;
    }
};

class LFSR113_E: public MRG<uint32_t, 4>
{
    using MRG::MRG;
    
public:
    __host__ __device__ uint32_t next()
    {
        // Same as LFSR113_A, but using a different formula for state[1]
        // The result of state[1] << 2 is reused.
        
        uint32_t r, t;
        
        t = ((state[0] << 6) ^ state[0]) >> 13;
		state[0] = ((state[0] & 4294967294) << 18) ^ t;
        
		t = ((state[1] << 2) ^ state[1]) >> 27;
		state[1] = ((state[1] << 2) & 4294967264) ^ t;
        
		t = ((state[2] << 13) ^ state[2]) >> 21;
		state[2] = ((state[2] & 4294967280) << 7) ^ t;
        
        t = ((state[3] << 3) ^ state[3]) >> 12;
		state[3] = ((state[3] & 4294967168) << 13) ^ t;
        
        r = state[0] ^ state[1] ^ state[2] ^ state[3];
        
        return r;
    }
    
    __host__ __device__ uint64_t mod()
    {
        return 4294967296L;
    }
    
    __host__ __device__ float div()
    {
        return 2.32830643654e-10f;
    }
};

class LFSR258_A: public MRG<uint64_t, 5>
{
    using MRG::MRG;
    
public:
    __host__ __device__ uint64_t next()
    {
        uint64_t r, t;
        
        t = ((state[0] << 1) ^ state[0]) >> 53;
        state[0] = ((state[0] & 18446744073709551614ULL) << 10) ^ t;

        t = ((state[1] << 24) ^ state[1]) >> 50;
        state[1] = ((state[1] & 18446744073709551104ULL) << 5) ^ t;

        t = ((state[2] << 3) ^ state[2]) >> 23;
        state[2] = ((state[2] & 18446744073709547520ULL) << 29) ^ t;

        t = ((state[3] << 5) ^ state[3]) >> 24;
        state[3] = ((state[3] & 18446744073709420544ULL) << 23) ^ t;

        t = ((state[4] << 3) ^ state[4]) >> 33;
        state[4] = ((state[4] & 18446744073701163008ULL) << 8) ^ t;
        
        r = state[0] ^ state[1] ^ state[2] ^ state[3] ^ state[4];
        
        return r;
    }
    
    __host__ __device__ uint64_t mod()
    {
        return UINT64_MAX;
    }
    
    __host__ __device__ float div()
    {
        return 5.42101086243e-20f;
    }
};

class LFSR258_B: public MRG<uint64_t, 5>
{
    using MRG::MRG;
    
public:
    __host__ __device__ uint64_t next()
    {
        uint64_t r, t;
        
        r = state[0] ^ state[1] ^ state[2] ^ state[3] ^ state[4];
        
        t = ((state[0] << 1) ^ state[0]) >> 53;
        state[0] = ((state[0] & 18446744073709551614ULL) << 10) ^ t;

        t = ((state[1] << 24) ^ state[1]) >> 50;
        state[1] = ((state[1] & 18446744073709551104ULL) << 5) ^ t;

        t = ((state[2] << 3) ^ state[2]) >> 23;
        state[2] = ((state[2] & 18446744073709547520ULL) << 29) ^ t;

        t = ((state[3] << 5) ^ state[3]) >> 24;
        state[3] = ((state[3] & 18446744073709420544ULL) << 23) ^ t;

        t = ((state[4] << 3) ^ state[4]) >> 33;
        state[4] = ((state[4] & 18446744073701163008ULL) << 8) ^ t;
        
        return r;
    }
    
    __host__ __device__ uint64_t mod()
    {
        return UINT64_MAX;
    }
    
    __host__ __device__ float div()
    {
        return 5.42101086243e-20f;
    }
};

class LFSR258_C: public MRG<uint64_t, 5>
{
    using MRG::MRG;
    
public:
    __host__ __device__ uint64_t next()
    {
        uint64_t r, t;
        
        t = ((state[0] << 1) ^ state[0]) >> 53;
        state[0] = ((state[0] & 18446744073709551614ULL) << 10) + t;

        t = ((state[1] << 24) ^ state[1]) >> 50;
        state[1] = ((state[1] & 18446744073709551104ULL) << 5) + t;

        t = ((state[2] << 3) ^ state[2]) >> 23;
        state[2] = ((state[2] & 18446744073709547520ULL) << 29) + t;

        t = ((state[3] << 5) ^ state[3]) >> 24;
        state[3] = ((state[3] & 18446744073709420544ULL) << 23) + t;

        t = ((state[4] << 3) ^ state[4]) >> 33;
        state[4] = ((state[4] & 18446744073701163008ULL) << 8) + t;
        
        r = state[0] ^ state[1] ^ state[2] ^ state[3] ^ state[4];
        
        return r;
    }
    
    __host__ __device__ uint64_t mod()
    {
        return UINT64_MAX;
    }
    
    __host__ __device__ float div()
    {
        return 5.42101086243e-20f;
    }
};

class LFSR258_D: public MRG<uint64_t, 5>
{
    using MRG::MRG;
    
public:
    __host__ __device__ uint64_t next()
    {
        uint64_t r, t1, t2, t3, t4, t5;
        
        t1 = ((state[0] << 1) ^ state[0]) >> 53;
        state[0] = ((state[0] & 18446744073709551614ULL) << 10) ^ t1;

        t2 = ((state[1] << 24) ^ state[1]) >> 50;
        state[1] = ((state[1] & 18446744073709551104ULL) << 5) ^ t2;

        t3 = ((state[2] << 3) ^ state[2]) >> 23;
        state[2] = ((state[2] & 18446744073709547520ULL) << 29) ^ t3;

        t4 = ((state[3] << 5) ^ state[3]) >> 24;
        state[3] = ((state[3] & 18446744073709420544ULL) << 23) ^ t4;

        t5 = ((state[4] << 3) ^ state[4]) >> 33;
        state[4] = ((state[4] & 18446744073701163008ULL) << 8) ^ t5;
        
        r = state[0] ^ state[1] ^ state[2] ^ state[3] ^ state[4];
        
        return r;
    }
    
    __host__ __device__ uint64_t mod()
    {
        return UINT64_MAX;
    }
    
    __host__ __device__ float div()
    {
        return 5.42101086243e-20f;
    }
};

/*********************
 **** Other F_2 ******
 *********************/

class Xorshift7_A: public MRG<uint32_t, 8>
{
    uint32_t index;
    
public:
    __host__ __device__ Xorshift7_A()
    {
    }
    
    __host__ __device__ Xorshift7_A(uint32_t seed)
    {
        for(int i = 0; i < 8; i++)
        {
            state[i] = seed;
        }
        
        index = 0;
    }
    
    __host__ __device__ uint32_t next()
    {
       // Same as TestU01
       
       uint32_t r, t;
       
       t = state[(index + 7) & 0x7U]; t ^= (t << 13);
       r = t ^ (t << 9);
       
       t = state[(index + 4) & 0x7U];
       r ^= t ^ (t << 7);
       
       t = state[(index + 3) & 0x7U];
       r ^= t ^ (t >> 3);
       
       t = state[(index + 1) & 0x7U];
       r ^= t ^ (t >> 10);
       
       t = state[index]; t ^= (t >> 7);
       r ^= t ^ (t << 24);
       
       state[index] = r;
       
       index = (index + 1) & 0x7U;
       
       return r;
    }
    
    __host__ __device__ uint64_t mod()
    {
        return 4294967296L;
    }
    
    __host__ __device__ float div()
    {
        return 2.32830643654e-10f;
    }
};

class Xorshift7_B: public MRG<uint32_t, 8>
{
    using MRG::MRG;
    
public:
    __host__ __device__ uint32_t next()
    {
       // Same as Xorshift7_A, but without the circular buffer
       uint32_t t1 = (state[7] << 13) ^ state[7]; t1 = (t1 << 9) ^ t1;
       
       uint32_t t2 = (state[4] <<  7) ^ state[4];
       uint32_t t3 = (state[3] >>  3) ^ state[3];
       uint32_t t4 = (state[1] >> 10) ^ state[1];
       
       uint32_t t5 = (state[0] >>  7) ^ state[0]; t5 = (t5 << 24) ^ t5;
       
       uint32_t r = t1 ^ t2 ^ t3 ^ t4 ^ t5;
       
       return append(r);
    }
    
    __host__ __device__ uint64_t mod()
    {
        return 4294967296L;
    }
    
    __host__ __device__ float div()
    {
        return 2.32830643654e-10f;
    }
};

class Xoshiro128Plus_A: public MRG<uint32_t, 4>
{
    using MRG::MRG;
    
public:
    __host__ __device__ uint32_t next()
    {
        uint32_t r = state[0] + state[3];
        
        uint32_t t = state[1] << 9;
        
        state[2] ^= state[0];
        state[3] ^= state[1];
        state[1] ^= state[2];
        state[0] ^= state[3];
        
        state[2] ^= t;
        
        state[3] = rotl(state[3], 11);
        
        return r;
    }
    
    __host__ __device__ uint64_t mod()
    {
        return 4294967296L;
    }
    
    __host__ __device__ float div()
    {
        return 2.32830643654e-10f;
    }
};

class Xoshiro128Plus_B: public MRG<uint32_t, 4>
{
    using MRG::MRG;
    
public:
    __host__ __device__ uint32_t next()
    {
        // Same as Xoroshiro128Plus_A, but using rotl_plus
        uint32_t r = state[0] + state[3];
        
        uint32_t t = state[1] << 9;
        
        state[2] ^= state[0];
        state[3] ^= state[1];
        state[1] ^= state[2];
        state[0] ^= state[3];
        
        state[2] ^= t;
        
        state[3] = rotl_plus(state[3], 11);
        
        return r;
    }
    
    __host__ __device__ uint64_t mod()
    {
        return 4294967296L;
    }
    
    __host__ __device__ float div()
    {
        return 2.32830643654e-10f;
    }
};

class Xoshiro128PlusPlus_A: public MRG<uint32_t, 4>
{
    using MRG::MRG;
    
public:
    __host__ __device__ uint32_t next()
    {
        uint32_t r = rotl(state[0] + state[3], 7) + state[0];
        
        uint32_t t = state[1] << 9;
        
        state[2] ^= state[0];
        state[3] ^= state[1];
        state[1] ^= state[2];
        state[0] ^= state[3];
        
        state[2] ^= t;
        
        state[3] = rotl(state[3], 11);
        
        return r;
    }
    
    __host__ __device__ uint64_t mod()
    {
        return 4294967296L;
    }
    
    __host__ __device__ float div()
    {
        return 2.32830643654e-10f;
    }
};

class Xoshiro128PlusPlus_B: public MRG<uint32_t, 4>
{
    using MRG::MRG;
    
public:
    __host__ __device__ uint32_t next()
    {
        // Same as Xoroshiro128PlusPlus_A, but using rotl_plus
        uint32_t r = rotl_plus(state[0] + state[3], 7) + state[0];
        
        uint32_t t = state[1] << 9;
        
        state[2] ^= state[0];
        state[3] ^= state[1];
        state[1] ^= state[2];
        state[0] ^= state[3];
        
        state[2] ^= t;
        
        state[3] = rotl_plus(state[3], 11);
        
        return r;
    }
    
    __host__ __device__ uint64_t mod()
    {
        return 4294967296L;
    }
    
    __host__ __device__ float div()
    {
        return 2.32830643654e-10f;
    }
};

class Xoroshiro128Plus: public MRG<uint64_t, 2>
{
    using MRG::MRG;
    
public:
    __host__ __device__ uint64_t next()
    {
        uint64_t r = state[0] + state[1];
        uint64_t t = state[0] ^ state[1];
        
        state[0] = rotl64(state[0], 24) ^ (t << 16) ^ t;
        state[1] = rotl64(t, 37);
        
        return r;
    }
    
    __host__ __device__ uint64_t mod()
    {
        return UINT64_MAX;
    }
    
    __host__ __device__ float div()
    {
        return 5.42101086243e-20f;
    }
};

class Xoshiro256Plus: public MRG<uint64_t, 4>
{
    using MRG::MRG;
    
public:
    __host__ __device__ uint64_t next()
    {
        uint64_t r = state[0] + state[3];
        
        uint64_t t = state[1] << 17;
        
        state[2] ^= state[0];
        state[3] ^= state[1];
        state[1] ^= state[2];
        state[0] ^= state[3];
        
        state[2] ^= t;
        
        state[3] = rotl64(state[3], 45);
        
        return r;
    }
    
    __host__ __device__ uint64_t mod()
    {
        return UINT64_MAX;
    }
    
    __host__ __device__ float div()
    {
        return 5.42101086243e-20f;
    }
};

class Xoshiro256PlusPlus: public MRG<uint64_t, 4>
{
    using MRG::MRG;
    
public:
    __host__ __device__ uint64_t next()
    {
        uint64_t r = rotl64(state[0] + state[3], 23) + state[0];
        
        uint64_t t = state[1] << 17;
        
        state[2] ^= state[0];
        state[3] ^= state[1];
        state[1] ^= state[2];
        state[0] ^= state[3];
        
        state[2] ^= t;
        
        state[3] = rotl64(state[3], 45);
        
        return r;
    }
    
    __host__ __device__ uint64_t mod()
    {
        return UINT64_MAX;
    }
    
    __host__ __device__ float div()
    {
        return 5.42101086243e-20f;
    }
};

// Previous tests for Xorshift

// 1- Using a funnel shift

// 2- Xoshiro128**

/*********************
 ****** CBRNGs *******
 *********************/

class Philox_4_32_10
{
    curandStatePhilox4_32_10_t state;
    
public:
    typedef uint32_t ReturnType;
    
    __host__ __device__ Philox_4_32_10()
    {
    }
    
    __host__ __device__ Philox_4_32_10(uint64_t key)
    {
        state.ctr.x = 0;
        state.ctr.y = 0;
        state.ctr.z = 0;
        state.ctr.w = 0;
        
        state.key.x = (uint32_t)key;
        state.key.y = (uint32_t)(key >> 32);
        
        state.output.x = 0;
        state.output.y = 0;
        state.output.z = 0;
        state.output.w = 0;
        
        state.STATE = 0;
    }
    
    __host__ __device__ uint32_t next()
    {
    #ifdef __CUDA_ARCH__
        return curand(&state);
    #else
        return 0;
    #endif
    }
    
    __host__ __device__ void nextStream()
    {
        state.key.x += 1;
        state.key.y += 1;
    }
    
    __host__ __device__ uint64_t mod()
    {
        return 4294967296L;
    }
    
    __host__ __device__ float div()
    {
        return 2.32830643654e-10f;
    }
};

class MSWS: public MRG<uint64_t, 3>
{
public:
    typedef uint32_t ReturnType;

    __host__ __device__ MSWS()
    {
    }
    
    __host__ __device__ MSWS(uint64_t key)
    {
        state[0] = 0;
        state[1] = 0;
        state[2] = key;
    }
    
    __host__ __device__ uint32_t next()
    {
        state[1] += state[2];
        
        state[0] = rotl64(state[0] * state[0] + state[1], 32);
        
        return (uint32_t)state[0];
    }
    
    __host__ __device__ void nextStream()
    {
        state[2] += 1;
    }
    
    __host__ __device__ uint64_t mod()
    {
        return 4294967296L;
    }
    
    __host__ __device__ float div()
    {
        return 2.32830643654e-10f;
    }
};

class Squares: public MRG<uint64_t, 2>
{
public:
    typedef uint32_t ReturnType;

    __host__ __device__ Squares()
    {
    }
    
    __host__ __device__ Squares(uint64_t key)
    {
        state[0] = 0;
        state[1] = key;
    }
    
    __host__ __device__ uint32_t next()
    {
        uint64_t r, s = state[0]; state[0] += state[1];
        
        r = rotl64_squares(s * s + s);                         /* round 1 */
        
        r = rotl64_squares(r * r + state[0]);                  /* round 2 */
        
        r = rotl64_squares(r * r + s);                         /* round 3 */
        
        r = (r * r + state[0]) >> 32;                          /* round 4 */
        
        return (uint32_t)r;
    }
    
    __host__ __device__ void nextStream()
    {
        state[1] += 1;
    }
    
    __host__ __device__ uint64_t mod()
    {
        return 4294967296L;
    }
    
    __host__ __device__ float div()
    {
        return 2.32830643654e-10f;
    }
};

#endif
