#pragma once
#include "cudnnManger.h"
#include "cudaKaze.h"
#include <opencv2/opencv.hpp>

float validValue(int dx, int dy, cv::Mat a);

//output matrix size = k+n-1
cv::Mat conv2d(cv::Mat iA, cv::Mat oA, cv::Mat fA) {
	//0-Padding
	int inputCols = iA.cols;
	int inputRows = iA.rows;

	//cv::Mat res(cv::Size(iA.cols, iA.rows), CV_32F);
	cv::Mat res(cv::Size(iA.cols+fA.cols - 1, iA.rows+fA.rows - 1), CV_32F);

	int ksize = fA.cols; // FilterSize x==y
	int kd = ksize / 2;


	for (int yy = 0; yy < res.rows; yy++) {
		for (int xx = 0; xx < res.cols; xx++) {
			res.at<float>(yy, xx) = 0;
			for (int dx = 0; dx < ksize; dx++) {
				for (int dy = 0; dy < ksize; dy++) {
					res.at<float>(yy, xx) += validValue(xx - ksize+ 1 + dx, yy - ksize + 1 + dy, iA)*(fA.at<float>(dy, dx));
				}
			}
			//printf("%lf ", *(res.ptr<float>(yy) + xx));
		}
		//printf("\n");
	}
	return res;
}

float validValue(int dx, int dy, cv::Mat a) {
	if (dx >= a.cols || dx < 0) return 0;
	if (dy >= a.rows || dy < 0) return 0;

	return *(a.ptr<float>(dy)+dx);
}