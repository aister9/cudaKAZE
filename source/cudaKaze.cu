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
	if (threadIdx.x == 0 && threadIdx.y == 0) {
		kd = ksize / 2;
	}
	__syncthreads();
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

__global__ void nlss_calc_uber_kernel(float *Evolutions,float *output, int scaleLevel, float* filterA, float* filterB, int ksize, int rows, int cols, float* kper) {
	cg::thread_block block = cg::this_thread_block();
	__shared__ int kd;
	int tidX = blockIdx.x * blockDim.x + threadIdx.x;
	int tidY = blockIdx.y * blockDim.y + threadIdx.y;

	if (tidX >= cols) return;
	if (tidY >= rows) return;

	float* inputImage = (Evolutions + (scaleLevel * rows * cols));
	//float* outputImage = (Evolutions + ((scaleLevel+1)*rows*cols));

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
	output[tidY * cols + tidX] = 1 / (1 + (differentialX * differentialX + differentialY * differentialY) / (kper[0]*kper[0]));

	//outputImage[tidY * cols + tidX] = differentialY;

	__syncthreads();
}

__global__ void nlss_calc_g2(float* a, float* b, float *c, int rows, int cols, float* kper) {
	int tidX = blockIdx.x * blockDim.x + threadIdx.x;
	int tidY = blockIdx.y * blockDim.y + threadIdx.y;

	if (tidX >= cols) return;
	if (tidY >= rows) return;

	c[tidY * cols + tidX] = 1 / (1 + (a[tidY*cols+tidX] * a[tidY * cols + tidX] + b[tidY * cols + tidX] * b[tidY * cols + tidX]) / (kper[0] * kper[0]));

	//outputImage[tidY * cols + tidX] = differentialY;

	__syncthreads();
}

__global__
void nlss_calc_divergence(float *image, float* normMat, int scaleLevel, int *nsteps_, float *tsteps_, int rows, int cols) {
	int tidX = blockIdx.x * blockDim.x + threadIdx.x;
	int tidY = blockIdx.y * blockDim.y + threadIdx.y;

	if (tidX >= cols) return;
	if (tidY >= rows) return;

	__shared__ int tstepPos;
	if (threadIdx.x == 0 && threadIdx.y == 0) {
		tstepPos = 0;
		for (int i = 1; i <= scaleLevel; i++) {
			tstepPos += nsteps_[i-1];
		}
	}
	__syncthreads();

	float* inputImage = image;
	float* outputImage = (image + ((scaleLevel + 1) * rows * cols));

	outputImage[tidY * cols + tidX] = inputImage[tidY * cols + tidX];

	if (tidX == cols - 1) return;
	if (tidX == 0) return;
	if (tidY == rows - 1) return;
	if (tidY == 0) return;

	for (int k = 0; k < nsteps_[scaleLevel]; k++) {
		float xpos = (-inputImage[tidY * cols + tidX] + inputImage[tidY * cols + tidX + 1]) * (normMat[tidY * cols + tidX+1] + normMat[tidY * cols + tidX]);
		float xneg = (-inputImage[tidY * cols + tidX-1] + inputImage[tidY * cols + tidX]) * (normMat[tidY * cols + tidX] + normMat[tidY * cols + tidX-1]);
		float ypos = (-inputImage[tidY * cols + tidX] + inputImage[(tidY+1) * cols + tidX]) * (normMat[(tidY+1) * cols + tidX] + normMat[tidY * cols + tidX]);
		float yneg = (-inputImage[(tidY - 1) * cols + tidX] + inputImage[tidY * cols + tidX]) * (normMat[tidY* cols + tidX] + normMat[(tidY - 1) * cols + tidX]);

		outputImage[tidY * cols + tidX] += 0.5f * tsteps_[tstepPos + k]*(xpos-xneg+ypos-yneg);
	}
	__syncthreads();
}

void cudaKaze::Convolutions2D(int inputLayerNumber, int outputLayerNumber, float* gpuFilters, int ksize, int rows, int cols) {
	GaussianConvolution << <dim3(ceil(cols/16 + 1), ceil(rows/16 + 1)), dim3(16,16) >> > (getGPUMatrixPointer(inputLayerNumber), getGPUMatrixPointer(outputLayerNumber), gpuFilters, ksize, rows, cols);
	//scharr << <dim3(ceil(cols / 16 + 1), ceil(rows / 16 + 1)), dim3(16, 16) >> > (getGPUMatrixPointer(outputLayerNumber), getGPUMatrixPointer(outputLayerNumber+1), rows, cols, 0);
	cudaDeviceSynchronize();
}

void cudaKaze::nlss_calc_uber(float* filterA, float* filterB, int ksize, int rows, int cols) {
	for (int i = 0; i < octave * sublevel - 1; i++) {
		float* gpuTempOut;
		cudaMalloc(&gpuTempOut, sizeof(float) * imagesize);
		nlss_calc_uber_kernel << <dim3(ceil(cols / 16 + 1), ceil(rows / 16 + 1)), dim3(16, 16) >> > (gpufloatMatrixs, gpuTempOut, i, filterA, filterB, ksize, rows, cols, kpercentile);
		nlss_calc_divergence << < dim3(ceil(cols / 16 + 1), ceil(rows / 16 + 1)), dim3(16, 16) >> > (gpufloatMatrixs, gpuTempOut, i, nsteps_, tsteps_, rows, cols);
		cudaFree(gpuTempOut);
	}

	//float* Lx, * Ly;
	//cudaMalloc(&Lx, sizeof(float) * rows * cols);
	//cudaMalloc(&Ly, sizeof(float) * rows * cols);
	//for (int i = 0; i < octave * sublevel - 1; i++) {
	//	float* gpuTempOut;
	//	cudaMalloc(&gpuTempOut, sizeof(float) * imagesize);

	//	GaussianConvolution << <dim3(ceil(cols / 16 + 1), ceil(rows / 16 + 1)), dim3(16, 16) >> > (getGPUMatrixPointer(i), Lx, filterA, ksize, rows, cols);
	//	GaussianConvolution << <dim3(ceil(cols / 16 + 1), ceil(rows / 16 + 1)), dim3(16, 16) >> > (getGPUMatrixPointer(i), Ly, filterB, ksize, rows, cols);
	//	nlss_calc_g2 << <dim3(ceil(cols / 16 + 1), ceil(rows / 16 + 1)), dim3(16, 16) >> > (Lx, Ly, gpuTempOut, rows, cols, kpercentile);
	//	nlss_calc_divergence << < dim3(ceil(cols / 16 + 1), ceil(rows / 16 + 1)), dim3(16, 16) >> > (gpufloatMatrixs, gpuTempOut, i, nsteps_, tsteps_, rows, cols);
	//	cudaFree(gpuTempOut);
	//}
	////divergence
	//cudaFree(Lx);
	//cudaFree(Ly);
	cudaDeviceSynchronize();
}

__global__ void printTaus(int* nsteps_, float* tsteps_, int size) {

	for (int i = 0; i < 11; i++) {
		int tstepPos = 0;
		for (int j = 1; j <= i; j++) {
			tstepPos += nsteps_[j - 1];
		}
		printf("%d : ", i);
		for (int k = 0; k < nsteps_[i]; k++) printf("%lf ", tsteps_[tstepPos + k]);
		printf("\n");
	}
}

__device__
float validValues(float* input, int dx, int dy, int rows, int cols) {
	if (dx >= cols || dx < 0) return 0;
	if (dy >= rows || dy < 0) return 0;

	return input[dy * cols + dx];
}

__global__ 
void calcLxLy(float* image, float* k, int rows, int cols, float* logFX, float* logFY, int ksize, float *lx, float *ly) {
	cg::thread_block block = cg::this_thread_block();
	__shared__ int kd;
	int tidX = blockIdx.x * blockDim.x + threadIdx.x;
	int tidY = blockIdx.y * blockDim.y + threadIdx.y;

	if (tidX >= cols) return;
	if (tidY >= rows) return;


	float differentialX = 0, differentialY = 0;
	//calculate LX
	for (int dy = 0; dy < ksize; dy++) {
		for (int dx = 0; dx < ksize; dx++) {
			differentialX += validValues(image, tidX - kd + dx, tidY - kd + dy, rows, cols) * logFX[dy * ksize + dx];
		}
	}//printf("%lf ", outImage[tidY * cols + tidX]);
	lx[tidY * cols + tidX] = differentialX;
	//calclulate LY
	for (int dy = 0; dy < ksize; dy++) {
		for (int dx = 0; dx < ksize; dx++) {
			differentialY += validValues(image, tidX - kd + dx, tidY - kd + dy, rows, cols) * logFY[dy * ksize + dx];
		}
	}//printf("%lf ", outImage[tidY * cols + tidX]);
	ly[tidY * cols + tidX] = differentialY;

	__syncthreads();
}
__global__ 
void calcHistoMax(float* image, float* lx, float* ly, int rows, int cols, float *max) {
	int tidX = threadIdx.x;
	int tidY = threadIdx.y;

	__shared__ float maxs[32][32];
	maxs[tidY][tidX] = 0.0f;
	__syncthreads();

	for (int i = 0; i < rows/32; i++) {
		for (int j = 0; j<cols/32; j++) {
			int ind = (i*32+tidY)*cols+j * 32 + tidX;
			if (j * 32 + tidX >= cols) continue;
			if (i * 32 + tidY >= rows) continue;

			float values = lx[ind] * lx[ind] + ly[ind] * ly[ind];

			maxs[tidY][tidX] = (maxs[tidY][tidX] < values) ? values : maxs[tidY][tidX];
		}
	}
	__syncthreads();

	if (threadIdx.x == 0 && threadIdx.y == 0) {
		max[0] = 0.0f;
		for (int i = 0; i < 32; i++) {
			for (int j = 0; j < 32; j++) {
				max[0] = (max[0] < maxs[i][j]) ? maxs[i][j] : max[0];
			}
		}
		max[0] = sqrtf(max[0]);
		printf("histomax = %lf \n", max[0]);
	}
	__syncthreads();
}
__global__ 
void calcHistogram(float* lx, float* ly, int rows, int cols, float* max, int* hist, float* kper) {
	int tidX = threadIdx.x;
	int tidY = threadIdx.y;

	__shared__ int npoints[16][16];
	npoints[tidY][tidX] = 0.0f;
	__syncthreads();

	for (int i = 0; i < rows / 16; i++) {
		for (int j = 0; j < cols / 16; j++) {
			int ind = (i * 16 + tidY) * cols + j * 16 + tidX;
			if (j * 16 + tidX >= cols) continue;
			if (i * 16 + tidY >= rows) continue;

			float values = lx[ind] * lx[ind] + ly[ind] * ly[ind];
			if (values != 0.0f) {
				int nbin = (int)floor(300 * (sqrtf(values) / max[0]));
				if (nbin == 300) {
					nbin--;
				}
				atomicAdd(&hist[nbin], 1);
				npoints[tidY][tidX]++;
			}
		}
	}
	__syncthreads();

	if (threadIdx.x == 0 && threadIdx.y == 0) {
		int npoint = 0;
		int nelements = 0;
		int k = 0;
		for (int i = 0; i < 16; i++) {
			for (int j = 0; j < 16; j++) {
				npoint += npoints[i][j];
			}
		}
		int nthreshold = (int)(npoint * 0.7f);

		for (k = 0; nelements < nthreshold && k < 300; k++) {
			nelements = nelements + hist[k];
		}

		if (nelements < nthreshold) kper[0] = 0.03f;
		else kper[0] = max[0] * ((float)(k) / (float)300);
		printf("kper = %lf \n", kper[0]);
	}
	__syncthreads();
}

__global__ //prefix Sum ver
void calcHistoMaxVer2(float* image, float* lx, float* ly, int rows, int cols, float* max) {
	int ind = blockIdx.x * blockDim.x + threadIdx.x;

	if (ind >= rows * cols) return;
	max[ind] = lx[ind] * lx[ind] + ly[ind] * ly[ind]; //myValue
	__syncthreads();
	float localMax = max[ind];

	for (int i = rows * cols / 2; i >= 1; i/=2) {
		if (ind < i) {
			int comInd = ind + i;
			if (comInd >= rows * cols) continue;
			float compareValue = max[comInd];
			localMax = (localMax < compareValue) ? compareValue : localMax;

			max[ind] = localMax;
		}
		__syncthreads();
	}
	__syncthreads();

	if (ind == 0) {
		max[0] = (max[0] < max[1]) ? max[1] : max[0];
		max[ind] = sqrtf(max[ind]);
		printf("max value : %lf ", max[ind]);
	}
}
__global__ //prefix Sum ver
void calcHistogramVer2(float* lx, float* ly, int rows, int cols, float* max, int* hist, float* kper) {
	int ind = blockIdx.x * blockDim.x + threadIdx.x;

	if (ind >= rows * cols) return;
	//npoints[tidY][tidX] = 0.0f;
	__syncthreads();
	float values = lx[ind] * lx[ind] + ly[ind] * ly[ind];

	if (values != 0.0f) {
		int nbin = (int)floor(300 * (sqrtf(values) / max[0]));
		if (nbin == 300) {
			nbin--;
		}
		atomicAdd(&hist[nbin], 1);
	}
	//npoints[tidY][tidX]++;

	
}
__global__ void calcKper(int* hist, float *max,float* kper) {
	int ind = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ int localHisto[300];
	localHisto[ind] = hist[ind];
	__syncthreads();

	for (int i = 300 / 2; i >= 1; i /= 2) {
		if (ind < i) {
			localHisto[ind] += localHisto[ind + i];
		}
		__syncthreads();
	}
	__syncthreads();

	if (ind == 0) {
		int npoint = localHisto[0];
		int nelements = 0;
		int k = 0;
		
		int nthreshold = (int)(npoint * 0.7f);

		for (k = 0; nelements < nthreshold && k < 300; k++) {
			nelements = nelements + hist[k];
		}

		if (nelements < nthreshold) kper[0] = 0.03f;
		else kper[0] = max[0] * ((float)(k) / (float)300);
		printf("kper = %lf \n", kper[0]);
	}
	__syncthreads();
}

void cudaKaze::calcKpercentile(int rows, int cols, float* logFX, float* logFY, int ksize) {
	float* lx, * ly;
	cudaMalloc(&lx, sizeof(float) * cols * rows);
	cudaMalloc(&ly, sizeof(float) * cols * rows);
	float* hmax;
	cudaMalloc(&hmax, sizeof(float) * cols * rows);
	int* gpuHist;
	cudaMalloc(&gpuHist, sizeof(int) * 300); // openCV histo bin size is 300, init k percep = 0.7f;
	cudaEvent_t handle[4];
	for (int i = 0; i < 4; i++) cudaEventCreate(&handle[i]);
	cudaEventRecord(handle[0]);
	calcLxLy << <dim3(ceil(cols / 16 + 1), ceil(rows / 16 + 1)), dim3(16, 16) >> > (gpufloatMatrixs, kpercentile, rows, cols, logFX, logFY, ksize, lx, ly); cudaEventRecord(handle[1]);
	//calcHistoMax <<<dim3(1,1), dim3(32, 32) >>>(gpufloatMatrixs, lx, ly, rows, cols, hmax); cudaEventRecord(handle[2]);
	calcHistoMaxVer2 << <ceil(rows * cols / 1024 + 1), 1024 >> > (gpufloatMatrixs, lx, ly, rows, cols, hmax); cudaEventRecord(handle[2]);
	//calcHistogram <<<dim3(1,1), dim3(16, 16) >>>(lx, ly, rows, cols, hmax, gpuHist, kpercentile); cudaEventRecord(handle[3]);
	calcHistogramVer2 << <ceil(rows * cols / 1024 + 1), 1024 >> > (lx, ly, rows, cols, hmax, gpuHist, kpercentile);
	calcKper << <1, 300 >> > (gpuHist, hmax, kpercentile); cudaEventRecord(handle[3]);
	cudaDeviceSynchronize();
	cudaFree(lx);
	cudaFree(ly);
	cudaFree(hmax);
	cudaFree(gpuHist);

	float time;
	cudaEventElapsedTime(&time, handle[0], handle[1]);
	std::cout << "calc LxLy : " << time << "ms" << std::endl;
	cudaEventElapsedTime(&time, handle[1], handle[2]);
	std::cout << "calc HistoMax : " << time << "ms" << std::endl;
	cudaEventElapsedTime(&time, handle[2], handle[3]);
	std::cout << "calc Kpercep : " << time << "ms" << std::endl;
	for (int i = 0; i < 4; i++) cudaEventDestroy(handle[i]);
}

void cudaKaze::calcTaus() {
	for (int i = 0; i < octave; i++) {
		for (int j = 0; j < sublevel; j++) {
			esigma[i * sublevel + j] = 1.6f * pow((float)2.f, (float)j / (float)(sublevel)+i);
			etime[i * sublevel + j] = 0.5f * (esigma[i * sublevel + j] * esigma[i * sublevel + j]);
		}
	}

	float etau[11];
	for (int i = 0; i < 11; i++) etau[i] = etime[i + 1] - etime[i];
	std::vector<float> taus;
	std::vector<int> nsteps;
	std::vector<std::vector<float>> tsteps;
	int tstepsize = 0;
	for (int i = 1; i < octave * sublevel; i++)
	{
		int naux = fed_tau_by_process_time(etau[i - 1], 1, 0.25f, true, taus);
		//printf("%d : ", naux);

		nsteps.push_back(naux);
		tsteps.push_back(taus);

		//for (float tauss : taus) {
		//	printf("%lf ", tauss);
		//}
		//printf("\n");
		tstepsize += taus.size();
	}

	cudaMalloc(&nsteps_, sizeof(int) * nsteps.size());
	cudaMemcpy(nsteps_, &nsteps[0], sizeof(int) * nsteps.size(), cudaMemcpyHostToDevice);
	cudaMalloc(&tsteps_, sizeof(float) * tstepsize);
	for (int i = 0, stepP = 0; i < tsteps.size(); i++) {
		cudaMemcpy(tsteps_ + stepP, &tsteps[i][0], sizeof(float) * tsteps[i].size(), cudaMemcpyHostToDevice);
		stepP += tsteps[i].size();
	}
	//printTaus<<<1,1>>>(nsteps_, tsteps_, 11);
}

__global__
void OLSConvolution(float* image, float* output, float* filter, int rows, int cols, int ksize) {
	int tidX = blockIdx.x * blockDim.x + threadIdx.x;
	int tidY = blockIdx.y * blockDim.y + threadIdx.y;
	if (tidX >= cols) return;
	if (tidY >= rows) return;
	__shared__ float localMap[18][18]; // 16x16 threadLay out, ksize =3 , so padding = 1 to 1

	if (threadIdx.x == 0 && threadIdx.y == 0) { // shared Memory Initalize
		for (int j = tidY - 1; j < tidY + 17; j++)
			for (int i = tidX - 1; i < tidX + 17; i++)
				localMap[j - tidY + 1][i - tidX + 1] = validValues(image, i, j, rows, cols);
	}
	__syncthreads();

	//normal convolution
	float convolvedData = 0.0f;
	for (int i = 0; i < ksize; i++) {
		for (int j = 0; j < ksize; j++) {
			convolvedData += localMap[threadIdx.y + i][threadIdx.x + j] * filter[i * ksize + j];
			__syncthreads();
		}
	}
	localMap[threadIdx.y + 1][threadIdx.x + 1] = convolvedData;
	__syncthreads();
	//

	output[tidY * cols + tidX] = localMap[threadIdx.y + 1][threadIdx.x + 1];
}

void cudaKaze::OLSConvTest(float* input, float* output, float* kernel, int rows, int cols, int ksize) {
	OLSConvolution << <dim3(ceil(cols / 16 + 1), ceil(rows / 16 + 1)), dim3(16, 16) >> > (input, output, kernel, rows, cols, ksize);
	cudaDeviceSynchronize();
}

////OLS Based Convolution
//__global__
//void calcDeterminants(float* image, float* deters, float* filterDx, float* filterDy, int rows, int cols, int k1, int k2) {
//	int tidX = blockIdx.x * blockDim.x + threadIdx.x;
//	int tidY = blockIdx.y * blockDim.y + threadIdx.y;
//	__shared__ float localMapX[20][20]; // 16x16 threadLay out, ksize =3 , so padding = 1 to 1
//	__shared__ float localMapY[20][20]; // 16x16 threadLay out, ksize =3 , so padding = 1 to 1
//	
//	if (threadIdx.x == 0 && threadIdx.y == 0) { // shared Memory Initalize
//		for(int j = tidY-2; j<tidY+18; j++)
//			for (int i = tidX - 2; i < tidX + 18; i++) {
//				localMapX[j - tidY + 2][i - tidX + 2] = validValues(image, j, i, rows, cols);
//				localMapY[j - tidY + 2][i - tidX + 2] = validValues(image, j, i, rows, cols);
//			}
//	}
//	__syncthreads();
//
//	//normal convolution
//	float convolvedData = 0.0f;
//	for (int i = 0; i < k2; i++) {
//		for (int j = 0; j < k2; j++) {
//			convolvedData += localMapX[threadIdx.y+i][threadIdx.x+j] * filterDx[i * k2 + j];
//			__syncthreads();
//		}
//	}
//	localMapX[threadIdx.y + 1][threadIdx.x + 1] = convolvedData;
//
//	float convolvedData = 0.0f;
//	for (int i = 0; i < k2; i++) {
//		for (int j = 0; j < k2; j++) {
//			convolvedData += localMapY[threadIdx.y + i][threadIdx.x + j] * filterDy[i * k2 + j];
//			__syncthreads();
//		}
//	}
//	localMapX[threadIdx.y + 1][threadIdx.x + 1] = convolvedData;
//	__syncthreads();
//	//
//
//	deters[tidY * cols + tidX] = localMap[threadIdx.y + 1][threadIdx.x + 1];
//}
__global__
void calcDeterminantsofHessian(float* Lxx, float* Lxy, float* Lyy, int esigma, float* det, int rows, int cols, int streamNumber) {
	int tidX = blockIdx.x * blockDim.x + threadIdx.x;
	int tidY = blockIdx.y * blockDim.y + threadIdx.y;

	if (tidX >= cols) return;
	if (tidY >= rows) return;

	int ind = tidY * cols + tidX;

	det[streamNumber * rows * cols + ind] = esigma * esigma * esigma * esigma * (Lxx[ind] * Lyy[ind] - Lxy[ind] * Lxy[ind]);
}


void genScharrKernel(float* filterDx, float* filterDy, int scale) {
	float f_scharr_x[3][3] = { {47, 0 ,-47}, {162, 0, -162}, {47, 0, -47} };
	float f_scharr_y[3][3] = { {47, 162 ,47}, {0, 0, 0}, {-47, -162, -47} };/*
	float* fx = new float[scale * scale];
	float* fy = new float[scale * scale];
	int mid = scale / 2;
	for (int i = 0; i < scale; i++) {
		for (int j = 0; j < scale; j++) {
			if ((mid - j) * (mid - j) + (mid - i) * (mid - i) == 0){
				fx[i * scale + j] = 0;
				fy[i * scale + j] = 0;
			}
			else {
				fx[i * scale + j] = (mid - j) / ((float)((mid - j) * (mid - j) + (mid - i) * (mid - i)));
				fy[i * scale + j] = (mid - i) / ((float)((mid - j) * (mid - j) + (mid - i) * (mid - i)));
			}
		}
	}*/
	cudaMemcpy(filterDx, f_scharr_x, sizeof(float) * scale * scale, cudaMemcpyHostToDevice);
	cudaMemcpy(filterDy, f_scharr_y, sizeof(float) * scale * scale, cudaMemcpyHostToDevice);
}

void cudaKaze::calcDeterminants(float* filterG, int k1, int rows, int cols) {
	cudaStream_t streams[12];
	for (int i = 0; i < 12; i++) {
		cudaStreamCreate(&streams[i]);
	}
#pragma omp parallel for
	for (int i = 0; i < 12; i++) {
		float* filterDx, * filterDy;
		int k2 = 3;
		//int k2 = round(esigma[i]);
		//k2 = (k2 % 2 == 0) ? k2 : k2 + 1;
		cudaMalloc(&filterDx, sizeof(float) * k2 * k2);
		cudaMalloc(&filterDy, sizeof(float) * k2 * k2);
		genScharrKernel(filterDx, filterDy, k2);
		float* Ls, * Lx, * Ly, * Lxx, * Lxy, * Lyy;
		cudaMalloc(&Ls, sizeof(float) * rows * cols); 
		cudaMalloc(&Lx, sizeof(float) * rows * cols);
		cudaMalloc(&Ly, sizeof(float) * rows * cols);
		cudaMalloc(&Lxx, sizeof(float) * rows * cols);
		cudaMalloc(&Lxy, sizeof(float) * rows * cols);
		cudaMalloc(&Lyy, sizeof(float) * rows * cols);


		GaussianConvolution << <dim3(ceil(cols/16 + 1), ceil(rows/16 + 1)), dim3(16,16), 0, streams[i] >> > (getGPUMatrixPointer(i), Ls, filterG, k1, rows, cols);
		GaussianConvolution << <dim3(ceil(cols/16 + 1), ceil(rows/16 + 1)), dim3(16,16), 0, streams[i] >> > (Ls, Lx, filterDx, k2, rows, cols); 
		GaussianConvolution << <dim3(ceil(cols/16 + 1), ceil(rows/16 + 1)), dim3(16,16), 0, streams[i] >> > (Ls, Ly, filterDy, k2, rows, cols); 
		GaussianConvolution << <dim3(ceil(cols/16 + 1), ceil(rows/16 + 1)), dim3(16,16), 0, streams[i] >> > (Lx, Lxx, filterDx, k2, rows, cols); 
		GaussianConvolution << <dim3(ceil(cols/16 + 1), ceil(rows/16 + 1)), dim3(16,16), 0, streams[i] >> > (Lx, Lxy, filterDy, k2, rows, cols); 
		GaussianConvolution << <dim3(ceil(cols/16 + 1), ceil(rows/16 + 1)), dim3(16,16), 0, streams[i] >> > (Ly, Lyy, filterDy, k2, rows, cols); 

		calcDeterminantsofHessian << <dim3(ceil(cols / 16 + 1), ceil(rows / 16 + 1)), dim3(16, 16), 0, streams[i] >> > (Lxx, Lxy, Lyy, k2, Imagedet, rows, cols, i);

		cudaFree(Ls);
		cudaFree(Lx);
		cudaFree(Ly);
		cudaFree(Lxx);
		cudaFree(Lxy);
		cudaFree(Lyy);
		cudaFree(filterDx);
		cudaFree(filterDy);
	}
	cudaDeviceSynchronize();

	for (int i = 0; i < 12; i++) {
		cudaStreamDestroy(streams[i]);
	}
}

void cudaKaze::findMaxima() {

}

