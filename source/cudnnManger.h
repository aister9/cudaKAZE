#pragma once
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cudnn.h"

#include <iostream>


class cudnnConvolutionManager_t {
    public:
    const int batch_count = 1; // 배치 개수
    const int in_channel = 1; // 채널 개수
    int padding_w = 1; // 패딩 width
    int padding_h = 1; // 패딩 height
    const int stride_horizontal = 1; // 스트라이드 가로
    const int stride_vertical = 1; // 스트라이드 세로
    const int filter_num = 1;
    float alpha = 1;
    float beta = 0;
    int in_height, in_width;
    int filter_height, filter_width;

    cudnnHandle_t cudnnHandle;
    cudnnTensorDescriptor_t inTensorDesc, outTensorDesc;
    cudnnFilterDescriptor_t filterDesc;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnConvolutionFwdAlgo_t algo;
    cudnnConvolutionFwdAlgoPerf_t algo1;
    void* workSpace = nullptr;
    size_t sizeInBytes = 0;

    cudnnConvolutionManager_t() {
        in_height = 0; in_width = 0;
        filter_height = 3; filter_width = 3;

        cudnnCreate(&cudnnHandle);
        cudnnCreateTensorDescriptor(&inTensorDesc);
        cudnnCreateTensorDescriptor(&outTensorDesc);
        cudnnCreateFilterDescriptor(&filterDesc);
        cudnnCreateConvolutionDescriptor(&convDesc);
        algo = CUDNN_CONVOLUTION_FWD_ALGO_FFT;
    }
    cudnnConvolutionManager_t(int ih, int iw, int fh, int fw) : in_height(ih), in_width(iw), filter_height(fh), filter_width(fw) {
        std::cout << "Input image size = " << iw << "*" << ih << std::endl;
        std::cout << "Input filter size = " << fw << "*" << fh << std::endl;
        cudnnCreate(&cudnnHandle);
        cudnnCreateTensorDescriptor(&inTensorDesc);
        cudnnCreateTensorDescriptor(&outTensorDesc);
        cudnnCreateFilterDescriptor(&filterDesc);
        cudnnCreateConvolutionDescriptor(&convDesc);
        algo = CUDNN_CONVOLUTION_FWD_ALGO_FFT;
    }

    void DestroyAll() {
        cudnnDestroyConvolutionDescriptor(convDesc);
        cudnnDestroyFilterDescriptor(filterDesc);
        cudnnDestroyTensorDescriptor(inTensorDesc);
        cudnnDestroyTensorDescriptor(outTensorDesc);
        cudnnDestroy(cudnnHandle);
    }

    void set() {
        cudnnErrorCheck(cudnnSetTensor4dDescriptor(inTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_count, in_channel, in_height, in_width));
        cudnnErrorCheck(cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, filter_num, in_channel, filter_height, filter_width));
        cudnnErrorCheck(cudnnSetConvolution2dDescriptor(convDesc, padding_h, padding_w, stride_vertical, stride_horizontal, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
        int out_n, out_c, out_h, out_w;
        cudnnErrorCheck(cudnnGetConvolution2dForwardOutputDim(convDesc, inTensorDesc, filterDesc, &out_n, &out_c, &out_h, &out_w));
        std::cout << out_n << "*" << out_c << "*" << out_h << "*" << out_w << std::endl;
        cudnnErrorCheck(cudnnSetTensor4dDescriptor(outTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, out_n, out_c, out_h, out_w));
        cudnnErrorCheck(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle, inTensorDesc, filterDesc, convDesc, outTensorDesc, CUDNN_CONVOLUTION_FWD_ALGO_FFT, &sizeInBytes));
        std::cerr << "Workspace size: " << (sizeInBytes / 1048576.0) << "MB"<< std::endl;
        if (sizeInBytes != 0) cudaMalloc(&workSpace, sizeInBytes);
    }

    /**
    *@param input gpuInputData Pointer
    *@param filter gpuFilterData Pointer
    *@param output gpuOutputData Pointer
    */
    void convolution(float* input, float* filter, float* output) {
        cudnnErrorCheck(cudnnConvolutionForward(cudnnHandle, &alpha, inTensorDesc, input, filterDesc, filter, convDesc, algo, workSpace, sizeInBytes, &beta, outTensorDesc, output));
    }

    void cudnnErrorCheck(cudnnStatus_t stat) {
        if (stat != CUDNN_STATUS_SUCCESS)
            std::cout << "CUDNN CONVOLUTION ERROR CODE : " << cudnnGetErrorString(stat) << std::endl;
        else
            std::cout << "CUDNN CONVOLUTION CODE : " << cudnnGetErrorString(stat) << std::endl;
    }
};