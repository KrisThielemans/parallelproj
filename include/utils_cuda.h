#ifndef __UTILS_CUDA_H__
#define __UTILS_CUDA_H__

extern "C" __global__ void add_to_first_kernel(float* a, float* b, unsigned long long n);

#endif

