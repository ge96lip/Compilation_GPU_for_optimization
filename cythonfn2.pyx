import numpy as np
cimport numpy as np

def mandelbrot(double complex c, max_iter=100):
    """Computes the number of iterations before divergence."""
    cdef double complex z = complex(0,0)
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z * z + c
    return max_iter