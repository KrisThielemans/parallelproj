#include<stdio.h>
#include<stdlib.h>
#include "utils_cuda.h"

// Simple transformation kernel
__global__ void transformKernel(cudaTextureObject_t texObj,
                                int n0, int n1, int n2,
                                float* output)
{
  // Calculate normalized texture coordinates
  unsigned int i0 = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int i1 = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int i2 = blockIdx.z * blockDim.z + threadIdx.z;

  float t0, t1, t2;

  if(i0 < n0)
  {
    if(i1 < n1)
    {
      if(i2 < n2)
      {
        // don't forget the 0.5 offset to address the "center"
        t0 = (float)i0 + 0.5f;
        t1 = (float)i1 + 0.5f;
        t2 = (float)i2 + 0.5f;
        // don't forget that axis are reverted in tex object
        // due to differences in underlying memory layout
        output[i0*n1*n2 + i1*n2 + i2] = tex3D<float>(texObj, t2, t1, t0);
      }
    }
  }
}


//--------------------------------------------------------------------


extern "C" void texture_test(float *h_img,
                             float *h_output,
                             int *h_img_dim)
{
  int deviceCount = 1;

  unsigned long long nvox = h_img_dim[0]*h_img_dim[1]*h_img_dim[2];

  // create output device array
  float *d_p;
  cudaMalloc(&d_p, nvox*sizeof(float));
  cudaMemsetAsync(&d_p, 0, nvox*sizeof(float));


  // create texture 3D cuda array and texture object
  cudaTextureObject_t *texImg = new cudaTextureObject_t[deviceCount];
  cudaArray **d_cuArrTex = new cudaArray*[deviceCount];
  CreateTextureInterp(h_img, h_img_dim, d_cuArrTex, texImg, true, deviceCount);

  // launch kernel
  dim3 dimBlock(8, 8, 8);
  dim3 dimGrid((h_img_dim[0] + dimBlock.x - 1) / dimBlock.x,
               (h_img_dim[1] + dimBlock.y - 1) / dimBlock.y,
               (h_img_dim[2] + dimBlock.z - 1) / dimBlock.z);

  transformKernel<<<dimGrid, dimBlock>>>(texImg[0], h_img_dim[0], h_img_dim[1], h_img_dim[2], d_p);

  // copy result back to host
  cudaMemcpyAsync(h_output, d_p, nvox*sizeof(float), cudaMemcpyDeviceToHost);

  // cuda arrays and texture object
  for (int dev = 0; dev < deviceCount; dev++){
    cudaSetDevice(dev);
    cudaDestroyTextureObject(texImg[dev]);
    cudaFreeArray(d_cuArrTex[dev]);
  }
}
