#include "cudnnManger.h"
#include "cudaKaze.h"
#include "MyConv.h"
#include "fed.h"
#include "DS_timer.h"
#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;

#define SCHARR_VALUE_A 47
#define SCHARR_VALUE_B 162

const float f_scharr_x[3][3] = { {-47, 0 ,47}, {-162, 0, 162}, {-47, 0, 47} };
const float f_scharr_y[3][3] = { {47, 162 ,47}, {0, 0, 0}, {-47, -162, -47} };


cv::Mat getGaussianKernel(int rows, int cols, double sigmax, double sigmay);

int main() {
	int octave = 4; int sublevel = 3;
	int imgSize = 1920 * 1080; //initiation to 1080p


	Mat img = imread("bird.jpg");
	imgSize = img.total();

	Mat grayImage;
	cvtColor(img, grayImage, COLOR_BGR2GRAY);
	grayImage.convertTo(grayImage, CV_32F);

	cudaKaze ck(octave, sublevel, imgSize);

	int a = ck.getLayers(2, 1);
	ck.host_call_init_layers();
	ck.cudaMemcpyToOriImages(grayImage.ptr<float>(0));

	float* forTest = new float[imgSize];

	int ksize = cvCeil(2.0f * (1.0f + (1.0f - 0.8f) / (0.3f)));
	if (ksize % 2 == 0) ksize += 1;
	Mat gaussianFilter = getGaussianKernel(ksize, ksize, 1.0f, 1.0f);
	Mat dogfilter;
	Mat dfx(Size(3, 3), CV_32F, (void*)f_scharr_x);
	Mat dfy(Size(3, 3), CV_32F, (void*)f_scharr_y);
	std::cout << "gaussian Filter Size " << gaussianFilter.cols << "*" << gaussianFilter.rows << "=" << gaussianFilter.total() << std::endl;
	std::cout << gaussianFilter << std::endl;
	float* gpuFilter;
	cudaMalloc(&gpuFilter, sizeof(float) * gaussianFilter.total());
	cudaMemcpy(gpuFilter, gaussianFilter.ptr<float>(0), sizeof(float) * gaussianFilter.total(), cudaMemcpyHostToDevice);

	float* dogFilterX, *dogFilterY;
	cudaMalloc(&dogFilterX, sizeof(float) * gaussianFilter.total());
	cudaMalloc(&dogFilterY, sizeof(float) * gaussianFilter.total());
	dogfilter = conv2d(gaussianFilter, gaussianFilter, dfx);
	cudaMalloc(&dogFilterX, sizeof(float) * gaussianFilter.total());
	cudaMemcpy(dogFilterX, dogfilter.ptr<float>(0), sizeof(float) * gaussianFilter.total(), cudaMemcpyHostToDevice);
	dogfilter = conv2d(gaussianFilter, gaussianFilter, dfy);
	cudaMalloc(&dogFilterY, sizeof(float) * gaussianFilter.total());
	cudaMemcpy(dogFilterY, dogfilter.ptr<float>(0), sizeof(float) * gaussianFilter.total(), cudaMemcpyHostToDevice);


	//cudnnConvolutionManager_t manager(img.rows, img.cols, ksize, ksize);
	//manager.set();
	//manager.convolution(ck.getGPUMatrixPointer(0), gpuFilter, ck.getGPUMatrixPointer(1));
	//manager.DestroyAll();
	ck.nlss_calc_uber(dogFilterX, dogFilterY, ksize, img.rows, img.cols);
	//ck.Convolutions2D(0, 1, gpuFilter, ksize, img.rows, img.cols);

	ck.getLayersImage(forTest, 1);


	Mat returnImg = Mat(img.rows, img.cols, CV_32F, forTest);
	//returnImg = conv2d(returnImg, returnImg, dfx);
	//std::cout << returnImg << std::endl;
	returnImg.convertTo(returnImg, CV_8UC1);

	while (waitKey(1) != 27) {
		imshow("test set", img);
		imshow("test set(conv)", returnImg);
	}
	gaussianFilter = getGaussianKernel(ksize, ksize, 1.0f, 1.0f);
	Mat dstx[2], dsty[2];
	dstx[0] = conv2d(gaussianFilter, dstx[0], dfx);
	filter2D(gaussianFilter, dsty[0], -1, dfy);
	dstx[1] = conv2d(dfx, dstx[1], gaussianFilter);
	filter2D(dfy, dsty[1], -1, gaussianFilter);

	Mat timg = imread("bird.jpg", IMREAD_GRAYSCALE);
	Mat convImg;
	timg.convertTo(convImg, CV_32F);

	//calc divergence
	float esigma[12];
	float etime[12];
	
	for (int i = 0; i < octave; i++) {
		for (int j = 0; j < sublevel; j++) {
			esigma[i*sublevel+j] = 1.6f * pow((float)2.f, (float)j / (float)(sublevel) + i);
			etime[i * sublevel + j] = 0.5f * (esigma[i * sublevel + j] * esigma[i * sublevel + j]);
		}
	}
	

	float etau[11];
	for(int i = 0 ; i<11; i++) etau[i] = etime[i+1] - etime[i];
	std::vector<float> taus;
	std::vector<int> nsteps;
	std::vector<std::vector<float>> tsteps;
	for(int i = 1 ; i<octave*sublevel; i++)
	{
		int naux = fed_tau_by_process_time(etau[i-1], 1, 0.25f, true, taus);
		printf("%d : ", naux);

		nsteps.push_back(naux);
		tsteps.push_back(taus);

		for (float tauss : taus) {
			printf("%lf ", tauss);
		}
		printf("\n");
	}
	Mat Lstep;
	Mat oimg[12];
	convImg.copyTo(oimg[0]);
	convImg.copyTo(Lstep);
	Mat gf1 = getGaussianKernel(9, 9, 1.6f, 1.6f);
	oimg[0] = conv2d(oimg[0], oimg[0], gf1);
	oimg[0] = conv2d(oimg[0], oimg[0], gaussianFilter);
	for (int t = 1; t < 12; t++) {
		oimg[t - 1].copyTo(oimg[t]);
		Mat dst1, dst2;
		//filter2D(convImg, dst1, -1, dstx[0]); // A * (B*C)
		dst1 = conv2d(oimg[t], dst1, dstx[0]);
		dst2 = conv2d(oimg[t], dst2, dsty[0]);
		//filter2D(convImg, dst2, -1, dfx);
		//filter2D(dst2, dst2, -1, gf);

		Mat dst3;
		dst2.copyTo(dst3);
		for (int i = 0; i < imgSize; i++) {
			float dx = *(dst1.ptr<float>(0) + i);
			float dy = *(dst2.ptr<float>(0) + i);

			float* vptr = dst3.ptr<float>(0) + i;
			vptr[0] = 1 / ((dx * dx + dy * dy) / (16.3f * 16.3f) + 1); // g2 norm
		}
		for (int k = 0; k < nsteps[t-1]; k++) {
			for (int i = 1; i < timg.rows - 1; i++) {
				const float* c_prev = dst3.ptr<float>(i - 1);
				const float* c_curr = dst3.ptr<float>(i);
				const float* c_next = dst3.ptr<float>(i + 1);
				const float* ld_prev = convImg.ptr<float>(i - 1);
				const float* ld_curr = convImg.ptr<float>(i);
				const float* ld_next = convImg.ptr<float>(i + 1);

				float* dst = Lstep.ptr<float>(i);

				for (int j = 1; j < timg.cols - 1; j++) {
					float xpos = (c_curr[j] + c_curr[j + 1]) * (ld_curr[j + 1] - ld_curr[j]);
					float xneg = (c_curr[j - 1] + c_curr[j]) * (ld_curr[j] - ld_curr[j - 1]);
					float ypos = (c_curr[j] + c_next[j]) * (ld_next[j] - ld_curr[j]);
					float yneg = (c_prev[j] + c_curr[j]) * (ld_curr[j] - ld_prev[j]);
					dst[j] = 0.5f * tsteps[t-1][k] * (xpos - xneg + ypos - yneg);
				}
			}
			oimg[t] += Lstep;
		}
		printf("%d calc ended\n", t);
		while (waitKey(1) != 27) {
			dst1.convertTo(dst1, CV_8UC1);
			dst2.convertTo(dst2, CV_8UC1);
			imshow("xdiff", dst1);
			imshow("ydiff", dst2);
		}
	}
	//find maxima
	std::vector<KeyPoint> kpts;
	for (int i = 1; i < 11; i++) {
		for (int yy = 0; yy < timg.rows; yy++) {
			for (int xx = 0; xx < timg.cols; xx++) {
				bool isMaxima = false;
				for (int c = -1; c <= 1; c++) {
					for (int ky = 0; ky < 3; ky++) {
						for (int kx = 0; kx < 3; kx++) {
							isMaxima = *(oimg[i].ptr<float>(yy) + xx) >= (validValue(xx - 1 + kx, yy - 1 + ky, oimg[i + c]));
							if (!isMaxima) break;
						}
						if (!isMaxima) break;
					}
					if (!isMaxima) break;
				}
				if (isMaxima) {
					KeyPoint kp;
					kp.pt.x = xx;
					kp.pt.y = yy;
					kp.response = fabs(*(oimg[i].ptr<float>(yy) + xx));
					kp.size = esigma[i];
					kp.octave = i / sublevel;
					kp.class_id = i;

					kpts.push_back(kp);
				}
			}
		}
	}

	Mat showImg;
	drawKeypoints(img, kpts, showImg, Scalar(0, 255, 0));

	//
	for (int i = 0; i < 12; i++) {
		oimg[i].convertTo(oimg[i], CV_8UC1);
	}
	//bitwise_not(dst2, dst2);

	for (int i = 0; i < 12; i++) {
		imwrite("images/evolutionImg_" + std::to_string(i) + ".jpg", oimg[i]);
	}

	while (waitKey(1) != 27) {
		imshow("kpts", showImg);
	}

	ck.cudaFreeAllMatrix();
	cudaFree(gpuFilter); 
}


cv::Mat getGaussianKernel(int rows, int cols, double sigmax, double sigmay)
{
	auto gauss_x = cv::getGaussianKernel(cols, sigmax, CV_32F);
	auto gauss_y = cv::getGaussianKernel(rows, sigmay, CV_32F);
	return gauss_x * gauss_y.t();
}