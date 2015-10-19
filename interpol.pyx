import cython
import numpy as np
cimport numpy as np

cdef extern:
    void c_interpol_2d(double * interpol, double tau, double * lookup, int * lookup_size)
    void c_interpol_3d(double * interpol, double tau, double * lookup, int * lookup_size)


@cython.boundscheck(False)
@cython.wraparound(False)

def interpol_2d(np.ndarray[double, ndim=1, mode = "c"] arr not None, \
                double tau, \
                np.ndarray[double, ndim = 2, mode="c"] lookup not None, \
                np.ndarray[int, ndim=1, mode = "c"] lookup_size):

    c_interpol_2d(&arr[0], tau, &lookup[0,0], &lookup_size[0])

def interpol_3d(np.ndarray[double,ndim=2, mode = "c"] arr not None, \
                double tau, \
                np.ndarray[double,ndim=3, mode = "c"] lookup not None, \
                np.ndarray[int, ndim=1, mode = "c"] lookup_size):

    c_interpol_3d(&arr[0,0], tau, &lookup[0,0,0], &lookup_size[0])

