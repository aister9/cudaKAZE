#include "cudaKaze.h"
#include <iostream>
#include <cooperative_groups.h>

#define THREAD_LAYOUT_X 8
#define THREAD_LAYOUT_Y 8

namespace cg = cooperative_groups;

__device__ float validValues(float* input, int dx, int dy, int rows, int cols);

__global__
void GaussianConvolution(float *inputImage, float *outputImage, float *filter, int ksize, int rows, int cols) {
	__shared__ int kd;
	if (threadIdx.x == 0 && threadIdx.y == 0) kd = ksize / 2;
	int tidX = blockIdx.x * blockDim.x + threadIdx.x;
	int tidY = blockIdx.y * blockDim.y + threadIdx.y;

	if (tidX >= cols) return;
	if (tidY >= rows) return;

	//initialize
	outputImage[tidY * cols + tidX] = 0;

	for (int dy = 0; dy < ksize; dy++) {
		for (int dx = 0; dx < ksize; dx++) {
			outputImage[tidY * cols + tidX] += validValues(inputImage, tidX-kd+dx, tidY-kd+dy, rows, cols)*filter[dy * ksize + dx];
		}
	}//printf("%lf ", outImage[tidY * cols + tidX]);
	__syncthreads();

}

__global__
void scharr(float* inputImage, float* outputImage, int rows, int cols, int direction) {
	__shared__ float scharrf[9];
	if (threadIdx.x == 0 && threadIdx.y == 0) {
		if (direction == 0) {
			scharrf[0] = 47;
			scharrf[1] = 0;
			scharrf[2] = -47;
			scharrf[3] = 162;
			scharrf[4] = 0;
			scharrf[5] = -162;
			scharrf[6] = 47;
			scharrf[7] = 0;
			scharrf[8] = -47;
		}else {
			scharrf[0] = 47;
			scharrf[1] = 162;
			scharrf[2] = 47;
			scharrf[3] = 0;
			scharrf[4] = 0;
			scharrf[5] = 0;
			scharrf[6] = -47;
			scharrf[7] = -162;
			scharrf[8] = -47;
		}
	}
	int tidX = blockIdx.x * blockDim.x + threadIdx.x;
	int tidY = blockIdx.y * blockDim.y + threadIdx.y;

	if (tidX >= cols) return;
	if (tidY >= rows) return;

	//initialize
	float tmp = 0;
	for (int dy = 0; dy < 3; dy++) {
		for (int dx = 0; dx < 3; dx++) {
			tmp += validValues(inputImage, tidX - 1 + dx, tidY - 1 + dy, rows, cols) * scharrf[dy * 3 + dx];
		}
	}//printf("%lf ", outImage[tidY * cols + tidX]);
	outputImage[tidY * cols + tidX] = tmp;
	__syncthreads();
}

__global__ void nlss_calc_uber_kernel(float *Evolutions, int scaleLevel, float* filterA, float* filterB, int ksize, int rows, int cols) {
	cg::thread_block block = cg::this_thread_block();
	__shared__ int kd;
	int tidX = blockIdx.x * blockDim.x + threadIdx.x;
	int tidY = blockIdx.y * blockDim.y + threadIdx.y;

	if (tidX >= cols) return;
	if (tidY >= rows) return;

	float* inputImage = (Evolutions + (scaleLevel * rows * cols));
	float* outputImage = (Evolutions + ((scaleLevel+1)*rows*cols));

	float differentialX = 0, differentialY = 0;
	for (int dy = 0; dy < ksize; dy++) {
		for (int dx = 0; dx < ksize; dx++) {
			differentialX += validValues(inputImage, tidX - kd + dx, tidY - kd + dy, rows, cols) * filterA[dy * ksize + dx];
		}
	}//printf("%lf ", outImage[tidY * cols + tidX]);
	for (int dy = 0; dy < ksize; dy++) {
		for (int dx = 0; dx < ksize; dx++) {
			differentialY += validValues(inputImage, tidX - kd + dx, tidY - kd + dy, rows, cols) * filterB[dy * ksize + dx];
		}
	}//printf("%lf ", outImage[tidY * cols + tidX]);
	outputImage[tidY * cols + tidX] = 1 / (1 + (differentialX * differentialX + differentialY * differentialY) / (89.3f*89.3f));

	//outputImage[tidY * cols + tidX] = differentialY;

	__syncthreads();
}

void cudaKaze::Convolutions2D(int inputLayerNumber, int outputLayerNumber, float* gpuFilters, int ksize, int rows, int cols) {
	GaussianConvolution << <dim3(ceil(cols/16 + 1), ceil(rows/16 + 1)), dim3(16,16) >> > (getGPUMatrixPointer(inputLayerNumber), getGPUMatrixPointer(outputLayerNumber), gpuFilters, ksize, rows, cols); GaussianConvolution << <dim3(ceil(cols / 16 + 1), ceil(rows / 16 + 1)), dim3(16, 16) >> > (getGPUMatrixPointer(inputLayerNumber), getGPUMatrixPointer(outputLayerNumber), gpuFilters, ksize, rows, cols);
	//scharr << <dim3(ceil(cols / 16 + 1), ceil(rows / 16 + 1)), dim3(16, 16) >> > (getGPUMatrixPointer(outputLayerNumber), getGPUMatrixPointer(outputLayerNumber+1), rows, cols, 0);
	cudaDeviceSynchronize();
}

void cudaKaze::nlss_calc_uber(float* filterA, float* filterB, int ksize, int rows, int cols) {
	for (int i = 0; i < octave * sublevel - 1; i++) {
		nlss_calc_uber_kernel << <dim3(ceil(cols / 16 + 1), ceil(rows / 16 + 1)), dim3(16, 16) >> > (gpufloatMatrixs, i, filterA, filterB, ksize, rows, cols);
	}
	cudaDeviceSynchronize();
}

__device__
float validValues(float* input, int dx, int dy, int rows, int cols) {
	if (dx >= cols || dx < 0) return 0;
	if (dy >= rows || dy < 0) return 0;

	return input[dy * cols + dx];
}