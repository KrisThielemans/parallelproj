#ifndef __UTILS_CUDA_H__
#define __UTILS_CUDA_H__

extern "C" __global__ void add_to_first_kernel(float* a, float* b, unsigned long long n);
extern "C" void CreateTextureInterp(float* imagedata, int* img_dim, cudaArray** d_cuArrTex, cudaTextureObject_t *texImage, bool allocate, int num_devices);

#endif

