import numpy as np
import numpy.ctypeslib as npct
import os
import sys
import ctypes
import platform

ar_1d_single = npct.ndpointer(dtype = ctypes.c_float, ndim = 1, flags = 'C')
ar_1d_int    = npct.ndpointer(dtype = ctypes.c_int,   ndim = 1, flags = 'C')
ar_1d_short  = npct.ndpointer(dtype = ctypes.c_short, ndim = 1, flags = 'C')

#---- find the compiled C / CUDA libraries

plt = platform.system()

if plt == 'Linux':
  fname_cuda = 'libparallelproj_cuda.so'
elif plt == 'Windows':
  fname_cuda = 'parallelproj_cuda.dll'
else:
  raise SystemError(f'{platform.system()} not supprted yet.')

lib_parallelproj_cuda_fname = os.path.abspath(os.path.join('..','pyparallelproj','lib',fname_cuda))

lib_parallelproj_cuda = npct.load_library(os.path.basename(lib_parallelproj_cuda_fname),
                                          os.path.dirname(lib_parallelproj_cuda_fname))

lib_parallelproj_cuda.texture_test.restype  = None
lib_parallelproj_cuda.texture_test.argtypes = [ar_1d_single, ar_1d_single, ar_1d_int]

#-------------------------------------------------------------------------------------------------------


#img = np.random.rand(61,117,219).astype(np.float32)
img = np.zeros((70,110,130), dtype = np.float32)
img[20:50,40:80,80:100] = 2.3

out   = np.zeros(img.shape, dtype = np.float32).flatten()
shape = np.array(img.shape)

lib_parallelproj_cuda.texture_test(img.flatten(), out, shape)

out = out.reshape(shape)
