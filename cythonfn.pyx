# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

import numpy as np
cimport numpy as np

def gauss_seidel(np.ndarray[np.float64_t, ndim=2] f, int num_iterations=1000):
    """
    Perform Gauss-Seidel iterations to solve the 2D Poisson equation.
    """
    cdef int n = f.shape[0]
    cdef int m = f.shape[1]
    cdef int i, j, k
    cdef np.ndarray[np.float64_t, ndim=2] newf = f.copy()

    for k in range(num_iterations):
        for i in range(1, n-1):
            for j in range(1, m-1):
                newf[i, j] = 0.25 * (newf[i, j+1] + newf[i, j-1] +
                                     newf[i+1, j] + newf[i-1, j])
    
    return newf  # Ensure function ends cleanly