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

  float t0, t1, t2;

  if(i0 < n0)
  {
    if(i1 < n1)
    {
      //t0 = 0.708*i0 + 0.707*i1;
      //t1 = 0.708*i0 - 0.707*i1;

      t0 = (float)i0;
      t1 = (float)i1;

      for(unsigned int i2 = 0; i2 < n2; i2++)
      {
        t2 = (float)i2;
        //output[i0*n1*n2 + i1*n2 + i2] = tex3D<float>(texObj, t0, t1, t2);
        output[i0 + i1*n0 + i2*n0*n1] = tex3D<float>(texObj, t0, t1, t2);
      }
    }
  }
}


//--------------------------------------------------------------------


extern "C" void texture_test(float *h_img,
                             float *h_output,
                             int *h_img_dim, 
                             int deviceCount)
{
  unsigned long long nvox = h_img_dim[0]*h_img_dim[1]*h_img_dim[2];

  float *d_p;
  cudaMalloc(&d_p, nvox*sizeof(float));
  cudaMemsetAsync(&d_p, 0, nvox*sizeof(float));

  bool allocate = true;

  cudaTextureObject_t *texImg = new cudaTextureObject_t[deviceCount];
  cudaArray **d_cuArrTex = new cudaArray*[deviceCount];


  CreateTextureInterp(h_img, h_img_dim, d_cuArrTex, texImg, allocate, deviceCount);

  dim3 dimBlock(16, 16);
  dim3 dimGrid((h_img_dim[0] + dimBlock.x - 1) / dimBlock.x,
               (h_img_dim[1] + dimBlock.y - 1) / dimBlock.y);

  transformKernel<<<dimGrid, dimBlock>>>(texImg[0], h_img_dim[0], h_img_dim[1], h_img_dim[2], d_p);

  cudaMemcpyAsync(h_output, d_p, nvox*sizeof(float), cudaMemcpyDeviceToHost);

  for (int dev = 0; dev < deviceCount; dev++){
    cudaSetDevice(dev);
    cudaDestroyTextureObject(texImg[dev]);
    cudaFreeArray(d_cuArrTex[dev]);
  }
}
