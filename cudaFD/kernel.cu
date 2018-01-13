#include <float.h>
#include <math.h>
#include <cuda.h>
#include <cublas.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

#include <cudaCommon.h>
#include <cudaKernel.cuh>
#include "kernel.cuh"


__global__ void intSubKernel(int *dA, const int *dB, const int M, const int N) {
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int x = blockIdx.x * blockDim.x + threadIdx.x;

	if ((y < M) && (x < N)) {
		const int i = x * N + y;
		dA[i] -= dB[i];
	}

	__syncthreads();
}

int intSub(int *dA, const int *dB, const int M, const int N) {
	dim3 blockSize(BLOCKSIZE, BLOCKSIZE);
	dim3 gridSize((N - 1) / BLOCKSIZE + 1, (M - 1) / BLOCKSIZE + 1);
	intSubKernel << <gridSize, blockSize >> >(dA, dB, M, N);
	CUDA_GETLASTERROR();
	CUDA_CALL(cudaThreadSynchronize());
	return EXIT_SUCCESS;
}


__global__ void intAddKernel(int *dA, const int M, const int N, const int alpha) {
	const int y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
	const int x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if ((y < M) && (x < N)) {
		const int i = __umul24(x, M) + y;
		dA[i] += alpha;
	}

	__syncthreads();
}

int intAdd(int *dA, const int M, const int N, const int alpha) {
	dim3 blockSize(BLOCKSIZE, BLOCKSIZE);
	dim3 gridSize((N - 1) / BLOCKSIZE + 1, (M - 1) / BLOCKSIZE + 1);
	intAddKernel << <gridSize, blockSize >> >(dA, M, N, alpha);
	CUDA_GETLASTERROR();
	CUDA_CALL(cudaThreadSynchronize());
	return EXIT_SUCCESS;
}



__global__ void intMeanKernel(int *sumBuf, const int *dA, const int M, const int N) {
	extern __shared__ int shared[];
	const int i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

	shared[threadIdx.x] = (i<M*N) ? dA[i] : 0;
	__syncthreads();

	int offset = blockDim.x >> 1;

	while (offset > 0) {
		if (threadIdx.x < offset) {
			shared[threadIdx.x] += shared[threadIdx.x + offset];
		}
		offset >>= 1;
		__syncthreads();
	}

	if (threadIdx.x == 0) { sumBuf[blockIdx.x] = shared[0]; } __syncthreads();
}

int intSum(int *result, const int *dA, const int M, const int N) {
	const int blockSize = BLOCKSIZE*BLOCKSIZE;
	const int gridSize = ((M*N) - 1) / blockSize + 1;
	int *dSumBuf; CUDA_CALL(cudaMalloc((void**)&dSumBuf, gridSize*sizeof(int)));
	intMeanKernel << <gridSize, blockSize, blockSize*sizeof(int) >> >(dSumBuf, dA, M, N);
	CUDA_GETLASTERROR();
	CUDA_CALL(cudaThreadSynchronize());

	int *cSumBuf = (int*)malloc(gridSize*sizeof(int));
	CUDA_CALL(cudaMemcpy(cSumBuf, dSumBuf, gridSize*sizeof(int), cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaFree(dSumBuf));
	*result = 0;
	for (int i = 0; i < gridSize; i++) *result += cSumBuf[i];
	free(cSumBuf);
	return EXIT_SUCCESS;
}

__global__ void dbcKernel(int * const resultMax, int * const resultMin, int const * const image, int const height, int const width, int const s, int const h) {
	extern __shared__ int shared[];
	int const threadId = threadIdx.y*BLOCKSIZE + threadIdx.x;

	shared[threadId] = -1;
	shared[BLOCKSIZE2 + threadId] = -1;

	__syncthreads();

	for (int i = threadIdx.x; i < s; i += blockDim.x) {
		for (int j = threadIdx.y; j < s; j += blockDim.y) {
			int const pixel_value = image[(blockIdx.x*s + i)*height + blockIdx.y*s + j]/h;

			if (shared[threadId] == -1) shared[threadId] = pixel_value;
			else if (shared[threadId] < pixel_value) shared[threadId] = pixel_value;

			if (shared[BLOCKSIZE2 + threadId] == -1)  shared[BLOCKSIZE2 + threadId] = pixel_value;
			else if (shared[BLOCKSIZE2 + threadId] > pixel_value) shared[BLOCKSIZE2 + threadId] = pixel_value;
		}
	}

	// debug[BLOCKSIZE2*(blockIdx.x*gridDim.y + blockIdx.y) + threadId] = shared[threadId];

	__syncthreads();

	int offset = BLOCKSIZE2 >> 1;

	while (offset > 0) {
		if (threadId < offset) {
			if ((shared[threadId] == -1) && (shared[threadId + offset] != -1)) {
				shared[threadId] = shared[threadId + offset];
			}
			else if ((shared[threadId] != -1) && (shared[threadId + offset] != -1)) {
				if (shared[threadId] < shared[threadId + offset]) shared[threadId] = shared[threadId + offset];
			}

			if ((shared[BLOCKSIZE2 + threadId] == -1) && (shared[BLOCKSIZE2 + threadId + offset] != -1)) {
				shared[BLOCKSIZE2 + threadId] = shared[BLOCKSIZE2 + threadId + offset];
			}
			else if ((shared[BLOCKSIZE2 + threadId] != -1) && (shared[BLOCKSIZE2 + threadId + offset] != -1)) {
				if (shared[BLOCKSIZE2 + threadId] > shared[BLOCKSIZE2 + threadId + offset]) shared[BLOCKSIZE2 + threadId] = shared[BLOCKSIZE2 + threadId + offset];
			}
		}
		offset >>= 1;
		__syncthreads();
	}

	__syncthreads();

	if (threadIdx.x == 0) { 
		resultMax[blockIdx.x*gridDim.y + blockIdx.y] = shared[0]; 
		resultMin[blockIdx.x*gridDim.y + blockIdx.y] = shared[BLOCKSIZE2];
	} 
	
	__syncthreads();
}

int dbcCore(float * const result, int const * const image, int const height, int const width, int const s, int const G) {

	int const M = width < height ? width : height;
	int const r = M / s;
	int const h = G / r;
	
	dim3 const blockSize(BLOCKSIZE, BLOCKSIZE);
	dim3 const gridSize(r, r);


	int *image_dev;
	CUDA_CALL(cudaMalloc((void**)&image_dev, height*width*sizeof(int)));
	CUDA_CALL(cudaMemcpy(image_dev, image, height*width*sizeof(int), cudaMemcpyHostToDevice));

	int *resultMin_dev;
	CUDA_CALL(cudaMalloc((void**)&resultMin_dev, r*r*sizeof(int)));

	int *resultMax_dev;
	CUDA_CALL(cudaMalloc((void**)&resultMax_dev, r*r*sizeof(int)));

	dbcKernel << < gridSize, blockSize, 2 * BLOCKSIZE2*sizeof(int) >> >(resultMax_dev, resultMin_dev, image_dev, height, width, s, h);

	CUDA_GETLASTERROR();
	CUDA_CALL(cudaThreadSynchronize());
	CUDA_CALL(cudaFree(image_dev));

	int res = 0;
	CALL(intSub(resultMax_dev, resultMin_dev, r, r));
	CALL(intAdd(resultMax_dev, r, r, 1));
	CALL(intSum(&res, resultMax_dev, r, r));

	*result = (float)res;

	CUDA_CALL(cudaFree(resultMin_dev));
	CUDA_CALL(cudaFree(resultMax_dev));


	return EXIT_SUCCESS;
}







__global__ void ebfdKernel(float * const result, int const * const image, int const height, int const width, int const s, int const h) {
	extern __shared__ float sharedF[];

	sharedF[threadIdx.x] = 0.0F;

	__syncthreads();

	for (int i = 0; i < s; i ++) {
		for (int j = 0; j < s; j ++) {
			if (image[(blockIdx.x*s + i)*height + blockIdx.y*s + j] == threadIdx.x) sharedF[threadIdx.x] += 1.0F;
			__syncthreads();
		}
	}

	sharedF[threadIdx.x] /= (s*s);
	if (sharedF[threadIdx.x] != 0.0F) sharedF[threadIdx.x] = sharedF[threadIdx.x] * log2f(sharedF[threadIdx.x]);

	__syncthreads();

	int offset = blockDim.x >> 1;

	while (offset > 0) {
		if (threadIdx.x < offset) {
			sharedF[threadIdx.x] += sharedF[threadIdx.x + offset];
		}

		offset >>= 1;
		__syncthreads();
	}

	if (threadIdx.x == 0) {
		result[blockIdx.x*gridDim.y + blockIdx.y] = -sharedF[0];
	}

	__syncthreads();
}

int ebfdCore(float * const result, int const * const image, int const height, int const width, int const s, int const G) {

	int const M = width < height ? width : height;
	int const r = M / s;
	int const h = G / r;

	dim3 const blockSize(G);
	dim3 const gridSize(r, r);

	int *image_dev;
	CUDA_CALL(cudaMalloc((void**)&image_dev, height*width*sizeof(int)));
	CUDA_CALL(cudaMemcpy(image_dev, image, height*width*sizeof(int), cudaMemcpyHostToDevice));

	float *result_dev;
	CUDA_CALL(cudaMalloc((void**)&result_dev, r*r*sizeof(float)));

	ebfdKernel << < gridSize, blockSize, G*sizeof(float) >> >(result_dev, image_dev, height, width, s, h);

	CUDA_GETLASTERROR();
	CUDA_CALL(cudaThreadSynchronize());

	CUDA_CALL(cudaFree(image_dev));

	CALL(Sum(result, result_dev, r, r));

	CUDA_CALL(cudaFree(result_dev));

	return EXIT_SUCCESS;
}
