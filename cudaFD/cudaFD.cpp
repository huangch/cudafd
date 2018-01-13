#define _CRT_SECURE_NO_WARNINGS

#include <float.h>
#include <math.h>

#include <cuda.h>


// #include <cudaCommon.h>
#include <cudaKernel.cuh>

#include "cudaFD.h"
#include "kernel.cuh"

int dbc(float * const result, int const * const image, int const height, int const width, int const s, int const G) {
	return dbcCore(result, image, height, width, s, G);
}

int ebfd(float * const result, int const * const image, int const height, int const width, int const s, int const G) {
	return ebfdCore(result, image, height, width, s, G);
}

#ifdef UNIX
	#define _isnan isnan
#endif
