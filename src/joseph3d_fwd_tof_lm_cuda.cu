/**
 * @file joseph3d_fwd_tof_lm_cuda.cu
 */

#include<stdio.h>
#include<stdlib.h>
#include<math.h>

#include "ray_cube_intersection_cuda.h"
#include "utils_cuda.h"

/** @brief 3D listmode tof joseph forward projector kernel
 *
 *  @param xstart array of shape [3*nlors] with the coordinates of the start points of the LORs.
 *                The start coordinates of the n-th LOR are at xstart[n*3 + i] with i = 0,1,2 
 *  @param xend   array of shape [3*nlors] with the coordinates of the end   points of the LORs.
 *                The start coordinates of the n-th LOR are at xstart[n*3 + i] with i = 0,1,2 
 *  @param img    cuda texture object containing the image
 *                for a c-contiguous array the elemet [i,j,k] is at [k+0.5,j+0.5,i+0.5]
 *  @param img_origin  array [x0_0,x0_1,x0_2] of coordinates of the center of the [0,0,0] voxel
 *  @param voxsize     array [vs0, vs1, vs2] of the voxel sizes
 *  @param p           array of length nlors (output) used to store the projections
 *  @param nlors       number of geomtrical LORs
 *  @param img_dim     array with dimensions of image [n0,n1,n2]
 *  @param tofbin_width     width of the TOF bins in spatial units (units of xstart and xend)
 *  @param sigma_tof        array of length nlors with the TOF resolution (sigma) for each LOR in
 *                          spatial units (units of xstart and xend) 
 *  @param tofcenter_offset array of length nlors with the offset of the central TOF bin from the 
 *                          midpoint of each LOR in spatial units (units of xstart and xend) 
 *  @param n_sigmas         number of sigmas to consider for calculation of TOF kernel
 *  @param tof_bin          array containing the TOF bin of each event
 */
__global__ void joseph3d_fwd_tof_lm_cuda_kernel(float *xstart, 
                                                float *xend, 
                                                cudaTextureObject_t img,
                                                float *img_origin, 
                                                float *voxsize, 
                                                float *p,
                                                long long nlors, 
                                                int *img_dim,
                                                float tofbin_width,
                                                float *sigma_tof,
                                                float *tofcenter_offset,
                                                float n_sigmas,
                                                short *tof_bin)
{
  long long i = blockDim.x * blockIdx.x + threadIdx.x;

  int n0 = img_dim[0];
  int n1 = img_dim[1];
  int n2 = img_dim[2];

  if(i < nlors)
  {
    float d0, d1, d2, d0_sq, d1_sq, d2_sq; 
    float lsq, cos0_sq, cos1_sq, cos2_sq;
    unsigned short direction; 
    int i0, i1, i2;
    float x_pr0, x_pr1, x_pr2;
   
    float u0, u1, u2, d_norm;
    float x_m0, x_m1, x_m2;    
    float x_v0, x_v1, x_v2;    

    int   it = tof_bin[i];
    float dtof, tw;

    // correction factor for cos(theta) and voxsize
    float cf;
    float toAdd;

    float sig_tof   = sigma_tof[i];
    float tc_offset = tofcenter_offset[i];

    float xstart0 = xstart[i*3 + 0];
    float xstart1 = xstart[i*3 + 1];
    float xstart2 = xstart[i*3 + 2];

    float xend0 = xend[i*3 + 0];
    float xend1 = xend[i*3 + 1];
    float xend2 = xend[i*3 + 2];

    float voxsize0 = voxsize[0];
    float voxsize1 = voxsize[1];
    float voxsize2 = voxsize[2];

    float img_origin0 = img_origin[0];
    float img_origin1 = img_origin[1];
    float img_origin2 = img_origin[2];

    unsigned char intersec;
    float t1, t2;
    float istart_f, iend_f, tmp;
    int   istart, iend;
    float istart_tof_f, iend_tof_f;
    int   istart_tof, iend_tof;

    // test whether the ray between the two detectors is most parallel
    // with the 0, 1, or 2 axis
    d0 = xend0 - xstart0;
    d1 = xend1 - xstart1;
    d2 = xend2 - xstart2;

    //-----------
    //--- test whether ray and cube intersect
    intersec = ray_cube_intersection_cuda(xstart0, xstart1, xstart2, 
                                          img_origin0 - 1*voxsize0, img_origin1 - 1*voxsize1, img_origin2 - 1*voxsize2,
                                          img_origin0 + n0*voxsize0, img_origin1 + n1*voxsize1, img_origin2 + n2*voxsize2,
                                          d0, d1, d2, &t1, &t2);

    if (intersec == 1)
    {
      d0_sq = d0*d0;
      d1_sq = d1*d1;
      d2_sq = d2*d2;

      lsq = d0_sq + d1_sq + d2_sq;

      cos0_sq = d0_sq / lsq;
      cos1_sq = d1_sq / lsq;
      cos2_sq = d2_sq / lsq;

      direction = 0;
      if ((cos1_sq >= cos0_sq) && (cos1_sq >= cos2_sq))
      {
        direction = 1;
      }
      else
      {
        if ((cos2_sq >= cos0_sq) && (cos2_sq >= cos1_sq))
        {
          direction = 2;
        }
      }
 
      //---------------------------------------------------------
      //--- calculate TOF related quantities
      
      // unit vector (u0,u1,u2) that points from xstart to end
      d_norm = sqrtf(lsq);
      u0 = d0 / d_norm; 
      u1 = d1 / d_norm; 
      u2 = d2 / d_norm; 

      // calculate mid point of LOR
      x_m0 = 0.5f*(xstart0 + xend0);
      x_m1 = 0.5f*(xstart1 + xend1);
      x_m2 = 0.5f*(xstart2 + xend2);

      //---------------------------------------------------------

      if (direction == 0)
      {
        cf = voxsize0 / sqrtf(cos0_sq);

        // case where ray is most parallel to the 0 axis
        // we step through the volume along the 0 direction

        //--- check where ray enters / leaves cube
        istart_f = (xstart0 + t1*d0 - img_origin0) / voxsize0;
        iend_f   = (xstart0 + t2*d0 - img_origin0) / voxsize0;

        if (istart_f > iend_f){
          tmp      = iend_f;
          iend_f   = istart_f;
          istart_f = tmp;
        }
    
        istart = (int)floor(istart_f);
        iend   = (int)ceil(iend_f);

        //-- check where we should start and stop according to the TOF kernel
        //-- the tof weights outside +- 3 sigma will be close to 0 so we can
        //-- ignore them         
        istart_tof_f = (x_m0 + (it*tofbin_width - n_sigmas*sig_tof)*u0 - img_origin0) / voxsize0;
        iend_tof_f   = (x_m0 + (it*tofbin_width + n_sigmas*sig_tof)*u0 - img_origin0) / voxsize0;
        
        if (istart_tof_f > iend_tof_f){
          tmp        = iend_tof_f;
          iend_tof_f = istart_tof_f;
          istart_tof_f = tmp;
        }

        istart_tof = (int)floor(istart_tof_f);
        iend_tof   = (int)ceil(iend_tof_f);

        if(istart_tof > istart){istart = istart_tof;}
        if(iend_tof   < iend){iend = iend_tof;}
        //-----------
        
        if (istart < 0){istart = 0;}
        if (iend >= n0){iend = n0;}

        for(i0 = istart; i0 < iend; i0++)
        {
          // get the indices where the ray intersects the image plane
          x_pr1 = xstart1 + (img_origin0 + i0*voxsize0 - xstart0)*d1 / d0;
          x_pr2 = xstart2 + (img_origin1 + i0*voxsize1 - xstart0)*d2 / d0;
  
          // texture memory to get interpolated value
          // don't forget to add the 0.5f voxel offset
          toAdd = tex3D<float>(img, ((x_pr2 - img_origin2)/voxsize2) + 0.5f, ((x_pr1 - img_origin1)/voxsize1) + 0.5f, float(i0) + 0.5f);

          //--------- TOF related quantities
          // calculate the voxel center needed for TOF weights
          x_v0 = img_origin0 + i0*voxsize0;
          x_v1 = x_pr1;
          x_v2 = x_pr2;

          if(toAdd != 0){
            // calculate distance of voxel to tof bin center
            dtof = sqrtf(powf((x_m0 + (it*tofbin_width + tc_offset)*u0 - x_v0), 2) + 
                         powf((x_m1 + (it*tofbin_width + tc_offset)*u1 - x_v1), 2) + 
                         powf((x_m2 + (it*tofbin_width + tc_offset)*u2 - x_v2), 2));

            //calculate the TOF weight
            tw = 0.5f*(erff((dtof + 0.5f*tofbin_width)/(sqrtf(2)*sig_tof)) - 
                      erff((dtof - 0.5f*tofbin_width)/(sqrtf(2)*sig_tof)));

            p[i] += (tw * cf * toAdd);
          }
        }
      }

      //--------------------------------------------------------------------------------- 
      if (direction == 1)
      {
        cf = voxsize1 / sqrtf(cos1_sq);

        // case where ray is most parallel to the 1 axis
        // we step through the volume along the 1 direction

        //--- check where ray enters / leaves cube
        istart_f = (xstart1 + t1*d1 - img_origin1) / voxsize1;
        iend_f   = (xstart1 + t2*d1 - img_origin1) / voxsize1;

        if (istart_f > iend_f){
          tmp      = iend_f;
          iend_f   = istart_f;
          istart_f = tmp;
        }
    
        istart = (int)floor(istart_f);
        iend   = (int)ceil(iend_f);

        //-- check where we should start and stop according to the TOF kernel
        //-- the tof weights outside +- 3 sigma will be close to 0 so we can
        //-- ignore them         
        istart_tof_f = (x_m1 + (it*tofbin_width - n_sigmas*sig_tof)*u1 - img_origin1) / voxsize1;
        iend_tof_f   = (x_m1 + (it*tofbin_width + n_sigmas*sig_tof)*u1 - img_origin1) / voxsize1;
        
        if (istart_tof_f > iend_tof_f){
          tmp        = iend_tof_f;
          iend_tof_f = istart_tof_f;
          istart_tof_f = tmp;
        }

        istart_tof = (int)floor(istart_tof_f);
        iend_tof   = (int)ceil(iend_tof_f);

        if(istart_tof > istart){istart = istart_tof;}
        if(iend_tof   < iend){iend = iend_tof;}
        //-----------

        if (istart < 0){istart = 0;}
        if (iend >= n1){iend = n1;}
        //---

        for (i1 = istart; i1 < iend; i1++)
        {
          // get the indices where the ray intersects the image plane
          x_pr0 = xstart0 + (img_origin1 + i1*voxsize1 - xstart1)*d0 / d1;
          x_pr2 = xstart2 + (img_origin1 + i1*voxsize1 - xstart1)*d2 / d1;
  
          // texture memory to get interpolated value
          // don't forget to add the 0.5f voxel offset
          toAdd = tex3D<float>(img, ((x_pr2 - img_origin2)/voxsize2) + 0.5f, (float)i1 + 0.5f, ((x_pr0 - img_origin0)/voxsize0) + 0.5f);

          //--------- TOF related quantities
          // calculate the voxel center needed for TOF weights
          x_v0 = x_pr0;
          x_v1 = img_origin1 + i1*voxsize1;
          x_v2 = x_pr2;

          if(toAdd != 0){
            // calculate distance of voxel to tof bin center
            dtof = sqrtf(powf((x_m0 + (it*tofbin_width + tc_offset)*u0 - x_v0), 2) + 
                         powf((x_m1 + (it*tofbin_width + tc_offset)*u1 - x_v1), 2) + 
                         powf((x_m2 + (it*tofbin_width + tc_offset)*u2 - x_v2), 2));

            //calculate the TOF weight
            tw = 0.5f*(erff((dtof + 0.5f*tofbin_width)/(sqrtf(2)*sig_tof)) - 
                      erff((dtof - 0.5f*tofbin_width)/(sqrtf(2)*sig_tof)));


            p[i] += (tw * cf * toAdd);
          }
        }
      }

      //--------------------------------------------------------------------------------- 
      if (direction == 2)
      {
        cf = voxsize2 / sqrtf(cos2_sq);

        // case where ray is most parallel to the 2 axis
        // we step through the volume along the 2 direction

        //--- check where ray enters / leaves cube
        istart_f = (xstart2 + t1*d2 - img_origin2) / voxsize2;
        iend_f   = (xstart2 + t2*d2 - img_origin2) / voxsize2;

        if (istart_f > iend_f){
          tmp      = iend_f;
          iend_f   = istart_f;
          istart_f = tmp;
        }
    
        istart = (int)floor(istart_f);
        iend   = (int)ceil(iend_f);

        //-- check where we should start and stop according to the TOF kernel
        //-- the tof weights outside +- 3 sigma will be close to 0 so we can
        //-- ignore them         
        istart_tof_f = (x_m2 + (it*tofbin_width - n_sigmas*sig_tof)*u2 - img_origin2) / voxsize2;
        iend_tof_f   = (x_m2 + (it*tofbin_width + n_sigmas*sig_tof)*u2 - img_origin2) / voxsize2;
        
        if (istart_tof_f > iend_tof_f){
          tmp        = iend_tof_f;
          iend_tof_f = istart_tof_f;
          istart_tof_f = tmp;
        }

        istart_tof = (int)floor(istart_tof_f);
        iend_tof   = (int)ceil(iend_tof_f);

        if(istart_tof > istart){istart = istart_tof;}
        if(iend_tof   < iend){iend = iend_tof;}
        //-----------

        if (istart < 0){istart = 0;}
        if (iend >= n2){iend = n2;}
        //---

        for(i2 = istart; i2 < iend; i2++)
        {
          // get the indices where the ray intersects the image plane
          x_pr0 = xstart0 + (img_origin2 + i2*voxsize2 - xstart2)*d0 / d2;
          x_pr1 = xstart1 + (img_origin2 + i2*voxsize2 - xstart2)*d1 / d2;
  
          // texture memory to get interpolated value
          // don't forget to add the 0.5f voxel offset
          toAdd = tex3D<float>(img, (float)i2 + 0.5f, ((x_pr1 - img_origin1)/voxsize1) + 0.5f, ((x_pr0 - img_origin0)/voxsize0) + 0.5f);

          //--------- TOF related quantities
          // calculate the voxel center needed for TOF weights
          x_v0 = x_pr0;
          x_v1 = x_pr1;
          x_v2 = img_origin2 + i2*voxsize2;

          if(toAdd != 0){
            // calculate distance of voxel to tof bin center
            dtof = sqrtf(powf((x_m0 + (it*tofbin_width + tc_offset)*u0 - x_v0), 2) + 
                         powf((x_m1 + (it*tofbin_width + tc_offset)*u1 - x_v1), 2) + 
                         powf((x_m2 + (it*tofbin_width + tc_offset)*u2 - x_v2), 2));

            //calculate the TOF weight
            tw = 0.5f*(erff((dtof + 0.5f*tofbin_width)/(sqrtf(2)*sig_tof)) - 
                      erff((dtof - 0.5f*tofbin_width)/(sqrtf(2)*sig_tof)));

            p[i] += (tw * cf * toAdd);
          }
        }
      }
    }
  }
}

//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------

/** @brief 3D listmode tof joseph forward projector CUDA wrapper
 *
 *  @param h_xstart array of shape [3*nlors] with the coordinates of the start points of the LORs.
 *                  The start coordinates of the n-th LOR are at xstart[n*3 + i] with i = 0,1,2 
 *  @param h_xend   array of shape [3*nlors] with the coordinates of the end   points of the LORs.
 *                  The start coordinates of the n-th LOR are at xstart[n*3 + i] with i = 0,1,2 
 *  @param h_img    array of shape [n0*n1*n2] containing the 3D image to be projected.
 *                  The pixel [i,j,k] ist stored at [n1*n2+i + n2*k + j].
 *  @param h_img_origin  array [x0_0,x0_1,x0_2] of coordinates of the center of the [0,0,0] voxel
 *  @param h_voxsize     array [vs0, vs1, vs2] of the voxel sizes
 *  @param h_p           array of length nlors (output) used to store the projections
 *  @param nlors         number of projections (length of p array)
 *  @param h_img_dim     array with dimensions of image [n0,n1,n2]
 *  @param tofbin_width     width of the TOF bins in spatial units (units of xstart and xend)
 *  @param h_sigma_tof      array of length nlors with the TOF resolution (sigma) for each LOR in
 *                          spatial units (units of xstart and xend) 
 *  @param h_tofcenter_offset array of length nlors with the offset of the central TOF bin from the 
 *                          midpoint of each LOR in spatial units (units of xstart and xend) 
 *  @param n_sigmas         number of sigmas to consider for calculation of TOF kernel
 *  @param h_tof_bin        array of length nlors with the tofbin of every event 
 *  @param threadsperblock number of threads per block
 *  @param num_devices     number of CUDA devices to use. if set to -1 cudaGetDeviceCount() is used
 */
extern "C" void joseph3d_fwd_tof_lm_cuda(float *h_xstart, 
                                         float *h_xend, 
                                         float *h_img,
                                         float *h_img_origin, 
                                         float *h_voxsize, 
                                         float *h_p,
                                         long long nlors, 
                                         int *h_img_dim,
                                         float tofbin_width,
                                         float *h_sigma_tof,
                                         float *h_tofcenter_offset,
                                         float n_sigmas,
                                         short *h_tof_bin,
                                         int threadsperblock,
                                         int num_devices)
{
  cudaError_t error;  
  int blockspergrid;

  dim3 block(threadsperblock);

  // offset for chunk of projections passed to a device 
  long long dev_offset;
  // number of projections to be calculated on a device
  long long dev_nlors;

  long long proj_bytes_dev;

  // get number of avilable CUDA devices specified as <=0 in input
  if(num_devices <= 0){cudaGetDeviceCount(&num_devices);}  

  // init the dynamic arrays of device arrays
  float **d_p              = new float * [num_devices];
  float **d_xstart         = new float * [num_devices];
  float **d_xend           = new float * [num_devices];
  float **d_img_origin     = new float * [num_devices];
  float **d_voxsize        = new float * [num_devices];
  int   **d_img_dim        = new int * [num_devices];

  // init the dynamic arrays of TOF device arrays
  float **d_sigma_tof        = new float * [num_devices];
  float **d_tofcenter_offset = new float * [num_devices];
  short **d_tof_bin          = new short * [num_devices];



  // create texture 3D cuda array and texture object
  cudaTextureObject_t *texImg = new cudaTextureObject_t[num_devices];
  cudaArray **d_cuArrTex = new cudaArray*[num_devices];
  CreateTextureInterp(h_img, h_img_dim, d_cuArrTex, texImg, true, num_devices);



  // we split the projections across all CUDA devices
  for (int i_dev = 0; i_dev < num_devices; i_dev++) 
  {
    cudaSetDevice(i_dev);
    // () are important in integer division!
    dev_offset = i_dev*(nlors/num_devices);
 
    // calculate the number of projections for a device (last chunck can be a bit bigger) 
    dev_nlors = i_dev == (num_devices - 1) ? (nlors - dev_offset) : (nlors/num_devices);

    // calculate the number of bytes for the projection array on the device
    proj_bytes_dev = dev_nlors*sizeof(float);

    // calculate the number of blocks needed for every device (chunk)
    blockspergrid = (int)ceil((float)dev_nlors / threadsperblock);
    dim3 grid(blockspergrid);

    // allocate the memory for the array containing the projection on the device
    error = cudaMalloc(&d_p[i_dev], proj_bytes_dev);
    if (error != cudaSuccess){
        printf("cudaMalloc returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);}
    cudaMemsetAsync(d_p[i_dev], 0, proj_bytes_dev);

    error = cudaMalloc(&d_xstart[i_dev], 3*proj_bytes_dev);
    if (error != cudaSuccess){
        printf("cudaMalloc returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);}
    cudaMemcpyAsync(d_xstart[i_dev], h_xstart + 3*dev_offset, 3*proj_bytes_dev, 
                    cudaMemcpyHostToDevice);

    error = cudaMalloc(&d_xend[i_dev], 3*proj_bytes_dev);
    if (error != cudaSuccess){
        printf("cudaMalloc returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);}
    cudaMemcpyAsync(d_xend[i_dev], h_xend + 3*dev_offset, 3*proj_bytes_dev, 
                    cudaMemcpyHostToDevice);
   
    error = cudaMalloc(&d_img_origin[i_dev], 3*sizeof(float));
    if (error != cudaSuccess){
        printf("cudaMalloc returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);}
    cudaMemcpyAsync(d_img_origin[i_dev], h_img_origin, 3*sizeof(float), 
                    cudaMemcpyHostToDevice);

    error = cudaMalloc(&d_voxsize[i_dev], 3*sizeof(float));
    if (error != cudaSuccess){
        printf("cudaMalloc returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);}
    cudaMemcpyAsync(d_voxsize[i_dev], h_voxsize, 3*sizeof(float), cudaMemcpyHostToDevice);

    error = cudaMalloc(&d_img_dim[i_dev], 3*sizeof(int));
    if (error != cudaSuccess){
        printf("cudaMalloc returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);}
    cudaMemcpyAsync(d_img_dim[i_dev], h_img_dim, 3*sizeof(int), cudaMemcpyHostToDevice);

    // send TOF arrays to device
    error = cudaMalloc(&d_sigma_tof[i_dev], proj_bytes_dev);
    if (error != cudaSuccess){
        printf("cudaMalloc returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);}
    cudaMemcpyAsync(d_sigma_tof[i_dev], h_sigma_tof + dev_offset, proj_bytes_dev, cudaMemcpyHostToDevice);

    error = cudaMalloc(&d_tofcenter_offset[i_dev], proj_bytes_dev);
    if (error != cudaSuccess){
        printf("cudaMalloc returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);}
    cudaMemcpyAsync(d_tofcenter_offset[i_dev], h_tofcenter_offset + dev_offset, proj_bytes_dev, 
                    cudaMemcpyHostToDevice);

    error = cudaMalloc(&d_tof_bin[i_dev], dev_nlors*sizeof(short));
    if (error != cudaSuccess){
        printf("cudaMalloc returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);}
    cudaMemcpyAsync(d_tof_bin[i_dev], h_tof_bin + dev_offset, dev_nlors*sizeof(short), cudaMemcpyHostToDevice);

    // call the kernel
    joseph3d_fwd_tof_lm_cuda_kernel<<<grid,block>>>(d_xstart[i_dev], d_xend[i_dev], texImg[i_dev], 
                                                    d_img_origin[i_dev], d_voxsize[i_dev], 
                                                    d_p[i_dev], dev_nlors, d_img_dim[i_dev],
                                                    tofbin_width, d_sigma_tof[i_dev],
                                                    d_tofcenter_offset[i_dev], n_sigmas,
                                                    d_tof_bin[i_dev]);

    // copy projection back from device to host
    cudaMemcpyAsync(h_p + dev_offset, d_p[i_dev], proj_bytes_dev, cudaMemcpyDeviceToHost);

    // deallocate memory on device
    cudaFree(d_p[i_dev]);
    cudaFree(d_xstart[i_dev]);
    cudaFree(d_xend[i_dev]);
    cudaFree(d_img_origin[i_dev]);
    cudaFree(d_voxsize[i_dev]);

    // deallocate TOF memory on device
    cudaFree(d_sigma_tof[i_dev]);
    cudaFree(d_tofcenter_offset[i_dev]);
    cudaFree(d_tof_bin[i_dev]);

    // deallocate texture object and 3d cuda image arrays
    cudaDestroyTextureObject(texImg[i_dev]);
    cudaFreeArray(d_cuArrTex[i_dev]);
  }

  // make sure that all devices are done before leaving
  for (int i_dev = 0; i_dev < num_devices; i_dev++){cudaDeviceSynchronize();}
}
