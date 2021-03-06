#ifndef __PARALLELPROJ_TOF_UTILS_H__
#define __PARALLELPROJ_TOF_UTILS_H__

float erff_as(float x);

void relevant_tof_bins(float x_m0,
		       float x_m1, 
		       float x_m2, 
		       float x_v0, 
		       float x_v1, 
		       float x_v2, 
		       float u0,
		       float u1,
		       float u2,
		       float tofbin_width,
		       float tofcenter_offset,
		       float sigma_tof,
		       float n_sigmas,
		       unsigned int n_half,
		       int *it1,
		       int *it2);
#endif
