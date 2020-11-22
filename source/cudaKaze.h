#pragma once
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


class cudaKaze {
	float* gpufloatMatrixs; //gpu float matrixs, octave*sublevel * Image
	int octave; //maximum octaves
	int sublevel; //maximum sublevel each octaves
	int imagesize; //input imagesize
	float* gpuFilter;
	float* taus;

public:
	cudaKaze(int octave, int sublevel, int imagesize) : octave(octave), sublevel(sublevel), imagesize(imagesize) {}

	void host_call_init_layers() {
		//gpufloatMatrix[octave, sublevel][imagesize];
		cudaMalloc(&gpufloatMatrixs, sizeof(float) * octave * sublevel * imagesize);
	}

	__host__ __device__ int getLayers(int o, int s)
	{
		return o * sublevel + s;
	}

	__host__ cudaError_t cudaMemcpyToOriImages(float* image) {
		return	cudaMemcpy(gpufloatMatrixs, image, sizeof(float) * imagesize, cudaMemcpyHostToDevice);
	}

	float* getGPUMatrixPointer() { return gpufloatMatrixs; }
	float* getGPUMatrixPointer(int layerNumber) { return (gpufloatMatrixs + (layerNumber * imagesize)); }

	__host__ cudaError_t getLayersImage(float* dst, int layerNumber) {
		return cudaMemcpy(dst, (gpufloatMatrixs + (layerNumber * imagesize)), sizeof(float) * imagesize, cudaMemcpyDeviceToHost);
	}

	void Convolutions2D(int inputLayerNumber, int outputLayerNumber, float* gpuFilters, int ksize, int rows, int cols);
	void nlss_calc_uber(float* filterA, float* filterB, int ksize, int rows, int cols);

	void cudaFreeAllMatrix() {
		cudaFree(gpufloatMatrixs);
	}
};