/**
 * @file utils_cuda.cu
 */

#include<stdio.h>
#include<stdlib.h>

#include <cuda_runtime_api.h>
#include <cuda.h>



/** @brief CUDA kernel to add array b to array a
 * 
 *  @param a first array of length n
 *  @param b first array of length n
 *  @param n length of vectors
 *
*/ 
extern "C" __global__ void add_to_first_kernel(float* a, float* b, unsigned long long n)
{
// add a vector b onto a vector a both of length n

  unsigned long long i = blockDim.x * blockIdx.x + threadIdx.x;

  if(i < n)
  {
    a[i] += b[i];
  }
}




extern "C" void CreateTextureInterp(float* imagedata, int* img_dim, cudaArray** d_cuArrTex, cudaTextureObject_t *texImage, bool allocate, int num_devices)
{
    //const cudaExtent extent = make_cudaExtent(img_dim[0], img_dim[1], img_dim[2]);
    const cudaExtent extent = make_cudaExtent(img_dim[2], img_dim[1], img_dim[0]);
    if(allocate){
        
        for (int dev = 0; dev < num_devices; dev++){
            cudaSetDevice(dev);
            
            //cudaArray Descriptor
            
            cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
            //cuda Array
            cudaMalloc3DArray(&d_cuArrTex[dev], &channelDesc, extent);
            //cudaCheckErrors("Texture memory allocation fail");
        }
        
    }
    for (int dev = 0; dev < num_devices; dev++){
        cudaMemcpy3DParms copyParams = {0};
        cudaSetDevice(dev);
        //Array creation
        copyParams.srcPtr   = make_cudaPitchedPtr((void *)imagedata, extent.width*sizeof(float), extent.width, extent.height);
        copyParams.dstArray = d_cuArrTex[dev];
        copyParams.extent   = extent;
        copyParams.kind     = cudaMemcpyHostToDevice;
        cudaMemcpy3DAsync(&copyParams);
        //cudaCheckErrors("Texture memory data copy fail");
        //Array creation End
    }
    for (int dev = 0; dev < num_devices; dev++){
        cudaSetDevice(dev);
        cudaResourceDesc    texRes;
        memset(&texRes, 0, sizeof(cudaResourceDesc));
        texRes.resType = cudaResourceTypeArray;
        texRes.res.array.array  = d_cuArrTex[dev];
        cudaTextureDesc     texDescr;
        memset(&texDescr, 0, sizeof(cudaTextureDesc));
        texDescr.normalizedCoords = false;
        texDescr.filterMode = cudaFilterModeLinear;
        texDescr.addressMode[0] = cudaAddressModeBorder;
        texDescr.addressMode[1] = cudaAddressModeBorder;
        texDescr.addressMode[2] = cudaAddressModeBorder;
        texDescr.readMode = cudaReadModeElementType;
        cudaCreateTextureObject(&texImage[dev], &texRes, &texDescr, NULL);
        //cudaCheckErrors("Texture object creation fail");
    }
}
