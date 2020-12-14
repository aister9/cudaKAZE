#include "cudnnManger.h"
#include "cudaKaze.h"
#include "MyConv.h"
#include "fed.h"
#include "DS_timer.h"
#include <omp.h>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <vector>

using namespace cv;

#define SCHARR_VALUE_A 47
#define SCHARR_VALUE_B 162

const float f_scharr_x[3][3] = { {47, 0 ,-47}, {162, 0, -162}, {47, 0, -47} };
const float f_scharr_y[3][3] = { {47, 162 ,47}, {0, 0, 0}, {-47, -162, -47} };


cv::Mat getGaussianKernel(int rows, int cols, double sigmax, double sigmay);

int main() {
	int octave = 4; int sublevel = 3;
	int imgSize = 1920 * 1080; //initiation to 1080p

	Mat img = imread("02.jpg");
	imgSize = img.total();

	Mat grayImage;
	cvtColor(img, grayImage, COLOR_BGR2GRAY);
	grayImage.convertTo(grayImage, CV_32F);
	Mat gf1 = getGaussianKernel(9, 9, 1.6f, 1.6f);
	int ksize = cvCeil(2.0f * (1.0f + (1.0f - 0.8f) / (0.3f)));
	if (ksize % 2 == 0) ksize += 1;
	Mat gaussianFilter = getGaussianKernel(ksize, ksize, 1.0f, 1.0f);
	filter2D(grayImage, grayImage, -1, gf1);
	filter2D(grayImage, grayImage, -1, gaussianFilter);

	cudaKaze ck(octave, sublevel, imgSize);

	int a = ck.getLayers(2, 1);
	ck.host_call_init_layers();
	ck.cudaMemcpyToOriImages(grayImage.ptr<float>(0));

	float* forTest = new float[imgSize];

	Mat dogfilter;
	Mat dfx(Size(3, 3), CV_32F, (void*)f_scharr_x);
	Mat dfy(Size(3, 3), CV_32F, (void*)f_scharr_y);
	std::cout << "gaussian Filter Size " << gaussianFilter.cols << "*" << gaussianFilter.rows << "=" << gaussianFilter.total() << std::endl;
	std::cout << gaussianFilter << std::endl;
	float* gpuGaussian;
	cudaMalloc(&gpuGaussian, sizeof(float) * gaussianFilter.total());
	cudaMemcpy(gpuGaussian, gaussianFilter.ptr<float>(0), sizeof(float) * gaussianFilter.total(), cudaMemcpyHostToDevice);

	float* gpuDx;
	cudaMalloc(&gpuDx, sizeof(float) * dfx.total());
	cudaMemcpy(gpuDx, dfx.ptr<float>(0), sizeof(float) * dfx.total(), cudaMemcpyHostToDevice);
 ck.OLSConvTest(ck.getGPUMatrixPointer(0), ck.getGPUMatrixPointer(1), gpuDx, img.rows, img.cols, 3);
	
		ck.getLayersImage(forTest, 1);

		Mat returnImg = Mat(img.rows, img.cols, CV_32F, forTest);
		//returnImg = conv2d(returnImg, returnImg, dfx);

		//std::cout << returnImg << std::endl;
		returnImg.convertTo(returnImg, CV_8UC1);

		while (waitKey(1) != 27) {
			imshow("test set", img);
			imshow("test set(conv)", returnImg);
		}
	

	float* dogFilterX, *dogFilterY;
	cudaMalloc(&dogFilterX, sizeof(float) * gaussianFilter.total());
	cudaMalloc(&dogFilterY, sizeof(float) * gaussianFilter.total());
	dogfilter = conv2d(dfx, gaussianFilter, gaussianFilter);
	//std::cout << dogfilter << std::endl;
	cudaMalloc(&dogFilterX, sizeof(float) * dogfilter.total());
	cudaMemcpy(dogFilterX, dogfilter.ptr<float>(0), sizeof(float) * dogfilter.total(), cudaMemcpyHostToDevice);
	dogfilter = conv2d(dfy, gaussianFilter, gaussianFilter);
	cudaMalloc(&dogFilterY, sizeof(float) * dogfilter.total());
	cudaMemcpy(dogFilterY, dogfilter.ptr<float>(0), sizeof(float) * dogfilter.total(), cudaMemcpyHostToDevice);

	//std::cout << dogfilter << std::endl;

	//cudnnConvolutionManager_t manager(img.rows, img.cols, ksize, ksize);
	//manager.set();
	//manager.convolution(ck.getGPUMatrixPointer(0), gpuFilter, ck.getGPUMatrixPointer(1));
	//manager.DestroyAll();
	DS_timer tt(4);
	tt.initTimers();
	tt.setTimerName(0, "GPU Calc nlss");
	tt.onTimer(0);
	ck.calcKpercentile(img.rows, img.cols, dogFilterX, dogFilterY, ksize + 2);
	ck.calcTaus();
	ck.nlss_calc_uber(dogFilterX, dogFilterY, ksize+2, img.rows, img.cols);
	tt.offTimer(0);
	DS_timer check(1); check.initTimers();
	check.onTimer(0); ck.calcDeterminants(gpuGaussian, ksize, img.rows, img.cols); check.offTimer(0);
	check.printTimer();
	//ck.Convolutions2D(0, 1, gpuFilter, ksize, img.rows, img.cols);

	//for (int i = 0; i < 12; i++) {
	//	ck.getLayersImage(forTest, i);

	//	Mat returnImg = Mat(img.rows, img.cols, CV_32F, forTest);
	//	//returnImg = conv2d(returnImg, returnImg, dfx);

	//	//std::cout << returnImg << std::endl;
	//	returnImg.convertTo(returnImg, CV_8UC1);

	//	while (waitKey(1) != 27) {
	//		imshow("test set", img);
	//		imshow("test set(conv)", returnImg);
	//	}
	//}
	gaussianFilter = getGaussianKernel(ksize, ksize, 1.0f, 1.0f);
	Mat dstx[2], dsty[2];
	dstx[0] = conv2d(gaussianFilter, dstx[0], dfx);
	filter2D(gaussianFilter, dsty[0], -1, dfy);
	dstx[1] = conv2d(dfx, dstx[1], gaussianFilter);
	filter2D(dfy, dsty[1], -1, gaussianFilter);
	//std::cout << dstx[0] << std::endl;
	//std::cout << dstx[1] << std::endl;

	Mat timg = imread("02.jpg", IMREAD_GRAYSCALE);
	Mat convImg;
	timg.convertTo(convImg, CV_32F);

	//Test Block
	//Mat forShow;
	//DS_timer timer(5);
	//timer.initTimers();
	//{
	//	timer.onTimer(0);
	//	//forShow = conv2d(convImg, forShow, dstx[0]);
	//	filter2D(convImg, forShow, -1, dstx[0]);
	//	timer.offTimer(0);
	//	while (waitKey(1) != 27) imshow("for show 1", forShow);
	//	timer.onTimer(1);
	//	//forShow = conv2d(convImg, forShow, dstx[1]);
	//	filter2D(convImg, forShow, -1, dstx[1]);
	//	timer.offTimer(1);
	//	while (waitKey(1) != 27) imshow("for show 2", forShow);
	//	timer.onTimer(2);
	//	/*
	//	forShow = conv2d(convImg, forShow, gaussianFilter);
	//	forShow = conv2d(forShow, forShow, dfx);*/
	//	filter2D(convImg, forShow, -1, gaussianFilter);
	//	filter2D(forShow, forShow, -1, dfx);
	//	timer.offTimer(2);
	//	while (waitKey(1) != 27) imshow("for show 3", forShow);
	//	timer.onTimer(3);
	//	ck.Convolutions2D(0, 1, dogFilterX, dogfilter.rows, img.rows, img.cols);
	//	timer.offTimer(3);
	//	ck.getLayersImage(forTest, 1);
	//	forShow = Mat(img.rows, img.cols, CV_32F, forTest);
	//	while (waitKey(1) != 27) imshow("for show 4", forShow);
	//	timer.onTimer(4);
	//	ck.Convolutions2D(0, 1, dogFilterX, 5, img.rows, img.cols);
	//	ck.Convolutions2D(0, 1, dogFilterX, 3, img.rows, img.cols);
	//	timer.offTimer(4);
	//	ck.getLayersImage(forTest, 1);
	//	forShow = Mat(img.rows, img.cols, CV_32F, forTest);
	//	while (waitKey(1) != 27) imshow("for show 4", forShow);
	//}
	//timer.printTimer();

	tt.setTimerName(1, "CPU NLSS TIME");
	tt.onTimer(1);
	//calc divergence
	float esigma[12];
	float etime[12];

	DS_timer details(4);
	details.initTimers();

	for (int i = 0; i < octave; i++) {
		for (int j = 0; j < sublevel; j++) {
			esigma[i*sublevel+j] = 1.6f * pow((float)2.f, (float)j / (float)(sublevel) + i);
			etime[i * sublevel + j] = 0.5f * (esigma[i * sublevel + j] * esigma[i * sublevel + j]);
		}
	}
	
	Mat Lx, Ly;
	int nbin = 0, nbins = 300;
	int histogram[300], npoints = 0;
	details.onTimer(0);
	filter2D(convImg, Lx, -1, dstx[1]);
	filter2D(convImg, Ly, -1, dsty[1]);
	details.offTimer(0);

	details.onTimer(1);
	float modg=0.0f, hmax = 0.0f;
	for (int i = 1; i < Lx.rows - 1; i++) {
		const float* lx = Lx.ptr<float>(i);
		const float* ly = Ly.ptr<float>(i);
		for (int j = 1; j < Lx.cols - 1; j++) {
			modg = lx[j] * lx[j] + ly[j] * ly[j];

			if (modg > hmax) {
				hmax = modg;
			}
		}
	}
	hmax = sqrt(hmax);
	details.offTimer(1);

	details.onTimer(3);
	for (int i = 1; i < Lx.rows - 1; i++) {
		const float* lx = Lx.ptr<float>(i);
		const float* ly = Ly.ptr<float>(i);
		for (int j = 1; j < Lx.cols - 1; j++) {
			modg = lx[j] * lx[j] + ly[j] * ly[j];

			if (modg!= 0.0) {
				nbin = (int)floor(nbins * (sqrt(modg) / hmax));

				if (nbin == nbins) nbin--;

				histogram[nbins]++;
				npoints++;
			}
		}
	}

	int nthreshold = (int)(npoints * 0.7f);
	int ks = 0, nelements = 0;
	for (ks = 0; nelements < nthreshold && ks < nbins; ks++) {
		nelements += histogram[ks];
	}
	float kperc;
	if (nelements < nthreshold) kperc = 0.03f;
	else kperc = hmax * ((float)(ks) / (float)nbins);
	details.offTimer(3);


	float* kpercep = new float[1];
	cudaMemcpy(kpercep, ck.getGPUKper(), sizeof(float), cudaMemcpyDeviceToHost);

	float etau[11];
	for(int i = 0 ; i<11; i++) etau[i] = etime[i+1] - etime[i];
	std::vector<float> taus;
	std::vector<int> nsteps;
	std::vector<std::vector<float>> tsteps;
	for(int i = 1 ; i<octave*sublevel; i++)
	{
		int naux = fed_tau_by_process_time(etau[i-1], 1, 0.25f, true, taus);

		nsteps.push_back(naux);
		tsteps.push_back(taus);
	}

	std::cout << "hmax : "<< hmax  <<" Kper " << kperc << std::endl;
	std::cout << "kper[0] " << kpercep[0] << std::endl;
	Mat Lstep(convImg.size(), CV_32F);
	Mat oimg[12];
	convImg.copyTo(oimg[0]);
	gf1 = getGaussianKernel(9, 9, 1.6f, 1.6f);
	filter2D(oimg[0], oimg[0],-1, gf1);
	filter2D(oimg[0], oimg[0], -1, gaussianFilter);
	for (int t = 1; t < 12; t++) {
		oimg[t - 1].copyTo(oimg[t]);
		Mat dst1, dst2;
		//filter2D(convImg, dst1, -1, dstx[0]); // A * (B*C)
		filter2D(oimg[t], dst1, -1, dstx[1]);
		filter2D(oimg[t], dst2, -1, dsty[1]);
		//filter2D(convImg, dst2, -1, dfx);
		//filter2D(dst2, dst2, -1, gf);

		Mat dst3;
		oimg[t].copyTo(dst3);
		for (int i = 0; i < imgSize; i++) {
			float *dx = (dst1.ptr<float>(0) + i);
			float *dy = (dst2.ptr<float>(0) + i);

			float* vptr = dst3.ptr<float>(0) + i;
			vptr[0] = 1 / (1 + (dx[0] * dx[0] + dy[0] * dy[0]) / (kpercep[0] * kpercep[0])); // g2 norm
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
					//oimg[t].at<float>(i,j) += 0.5f * tsteps[t-1][k] * (xpos - xneg + ypos - yneg);
					dst[j] = 0.5f * tsteps[t-1][k] * (xpos - xneg + ypos - yneg);
				}
			}
			oimg[t] += Lstep;
		}
		printf("%d calc ended\n", t);/*
		while (waitKey(1) != 27) {
			Mat show;
			oimg[t].convertTo(show, CV_8UC1);
			imshow("oimg", show);
		}*/
	}
	tt.offTimer(1);

	//for (int i = 0; i < 12; i++) {
	//	ck.getLayersImage(forTest, i);

	//	Mat returnImg = Mat(img.rows, img.cols, CV_32F, forTest);
	//	//returnImg = conv2d(returnImg, returnImg, dfx);

	//	//std::cout << returnImg << std::endl;
	//	returnImg.copyTo(oimg[i]);
	//}

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//                                                                   Feature Detection                                                                 //
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	//Calc LHessian
	Mat deter[12];
	details.onTimer(2);
#pragma omp parallel for
	for (int i = 0; i < 12; i++) {
		Mat Lxx, Lxy, Lyy;

		filter2D(oimg[i], Lxx, -1, gaussianFilter); //Ls
		filter2D(Lxx, Lxy, -1, dfx); //Lx
		filter2D(Lxx, Lyy, -1, dfy); //Ly
		filter2D(Lxy, Lxx, -1, dfx); //Lxx
		filter2D(Lxy, Lxy, -1, dfy); //Lxy
		filter2D(Lyy, Lyy, -1, dfy); //Lyy

		deter[i] = round(esigma[i])* round(esigma[i]) *(Lxx.mul(Lyy) - Lxy.mul(Lxy));
		
	}
	details.offTimer(2);
	details.printTimer();

	//for (int i = 0; i < 12; i++) {
	//	ck.getLayersImageDet(forTest, i);

	//	Mat returnImg = Mat(img.rows, img.cols, CV_32F, forTest);
	//	//returnImg = conv2d(returnImg, returnImg, dfx);

	//	//std::cout << returnImg << std::endl;
	//	/*while (waitKey(1) != 27) {
	//	imshow("deter", deter[i]);
	//	}*/
	//	returnImg.copyTo(deter[i]);
	//	
	//	returnImg.convertTo(returnImg, CV_8UC1);
	//	while (waitKey(1) != 27) {
	//		imshow("rimg", returnImg);
	//	}
	//}

	tt.setTimerName(3, "Feature Detection Time");
	tt.onTimer(3);
	//find maxima
	std::vector<std::vector<KeyPoint>> kpts_;
	for (int i = 1; i < 11; i++) {
		std::vector<KeyPoint> kpts_t;
		for (int yy = 1; yy < timg.rows-1; yy++) {
			for (int xx = 1; xx < timg.cols-1; xx++) {
				bool isMaxima = false;
				float value = *(deter[i].ptr<float>(yy) + xx);
				if (value <= 0.01f) continue;
				if (value >= *(deter[i].ptr<float>(yy) + xx - 1)) {
					for (int c = -1; c <= 1; c++) {
						for (int ky = 0; ky < 3; ky++) {
							for (int kx = 0; kx < 3; kx++) {
								isMaxima = (value >= (validValue(xx - 1 + kx, yy - 1 + ky, deter[i + c])));
								if (!isMaxima) break;
							}
							if (!isMaxima) break;
						}
						if (!isMaxima) break;
					}
				}
				if (isMaxima) {
					KeyPoint kp;
					kp.pt.x = xx;
					kp.pt.y = yy;
					kp.response = fabs(*(deter[i].ptr<float>(yy) + xx));
					kp.size = esigma[i];
					kp.octave = i / sublevel;
					kp.class_id = i;

					kp.angle = (float)(i % sublevel);
					kpts_t.push_back(kp);
				}
			}
		}
		kpts_.push_back(kpts_t);
	}

	//feature_detection_interpol
	std::vector<KeyPoint> kpts;
	for (int i = 0; i < (int)kpts_.size(); i++) {
		for (int j = 0; j < (int)kpts_[i].size(); j++) {
			int level = i + 1;
			bool is_extremum = true;
			bool is_repeated = false;
			bool is_out = false;
			int id_repeated = 0;

			for (int ik = 0; ik < kpts.size(); ik++) {
				if (kpts[ik].class_id == level || kpts[ik].class_id == level + 1 || kpts[ik].class_id == level - 1) {
					float dist = pow(kpts_[i][j].pt.x - kpts[ik].pt.x, 2) + pow(kpts_[i][j].pt.y - kpts[ik].pt.y, 2);

					//std::cout << dist << " > " << esigma[level] * esigma[level] << " " << esigma[level] << std::endl;

					if (dist < round(esigma[level]) * round(esigma[level])) {
						if (kpts_[i][j].response > kpts[ik].response) {
							id_repeated = ik;
							is_repeated = true;
						}
						else {
							is_extremum = false;
						}
						break;
					}
				}
			}

			if (is_extremum == true) {
				int leftx = round(kpts_[i][j].pt.x - 3 * kpts_[i][j].size);
				int rightx = round(kpts_[i][j].pt.x + 3 * kpts_[i][j].size);
				int upy = round(kpts_[i][j].pt.y - 3 * kpts_[i][j].size);
				int downy = round(kpts_[i][j].pt.y + 3 * kpts_[i][j].size);

				if (leftx < 0 || rightx >= deter[level].cols || upy < 0 || downy >= deter[level].rows) {
					is_out = true;
				}

				if (is_out == false) {
					if (is_repeated == false) {
						kpts.push_back(kpts_[i][j]);
					}
					else {
						kpts[id_repeated] = kpts_[i][j];
					}
				}
			}

		}
	}

	//Subpixel_Refinement
	Mat A = Mat::zeros(3, 3, CV_32F);
	Mat b = Mat::zeros(3, 1, CV_32F);
	Mat dst = Mat::zeros(3, 1, CV_32F);

	std::vector<KeyPoint> kpts_t(kpts);

	for (int i = 0; i < kpts.size(); i++) {
		int x = kpts[i].pt.x;
		int y = kpts[i].pt.y;

		float Dx = (1.0f / 2.0f) * (*(deter[kpts[i].class_id].ptr<float>(y) + x+1) - *(deter[kpts[i].class_id].ptr<float>(y) + x - 1));
		float Dy = (1.0f / 2.0f) * (*(deter[kpts[i].class_id].ptr<float>(y+1) + x) - *(deter[kpts[i].class_id].ptr<float>(y-1) + x));
		float Ds = (1.0f / 2.0f) * (*(deter[kpts[i].class_id + 1].ptr<float>(y) + x) - *(deter[kpts[i].class_id-1].ptr<float>(y) + x));
		
		float Dxx = (1.0f / 1.0f) * (*(deter[kpts[i].class_id].ptr<float>(y) + x + 1) + *(deter[kpts[i].class_id].ptr<float>(y) + x - 1) - 2.0f * *(deter[kpts[i].class_id].ptr<float>(y) + x));
		float Dyy = (1.0f / 1.0f) * (*(deter[kpts[i].class_id].ptr<float>(y + 1) + x) + *(deter[kpts[i].class_id].ptr<float>(y - 1) + x) - 2.0f * *(deter[kpts[i].class_id].ptr<float>(y) + x));
		float Dss = (1.0f / 1.0f) * (*(deter[kpts[i].class_id + 1].ptr<float>(y) + x) + *(deter[kpts[i].class_id - 1].ptr<float>(y) + x) - 2.0f * *(deter[kpts[i].class_id].ptr<float>(y) + x));
		
		float Dxy = (1.0f / 4.0f) * (*(deter[kpts[i].class_id].ptr<float>(y + 1) + x + 1) + *(deter[kpts[i].class_id].ptr<float>(y - 1) + x - 1))
			- (1.0f / 4.0f) * (*(deter[kpts[i].class_id].ptr<float>(y - 1) + x + 1) + *(deter[kpts[i].class_id].ptr<float>(y + 1) + x - 1));

		float Dxs = (1.0f / 4.0f) * (*(deter[kpts[i].class_id+1].ptr<float>(y) + x + 1) + *(deter[kpts[i].class_id-1].ptr<float>(y) + x - 1))
			- (1.0f / 4.0f) * (*(deter[kpts[i].class_id + 1].ptr<float>(y) + x - 1) + *(deter[kpts[i].class_id - 1].ptr<float>(y) + x + 1));

		float Dys = (1.0f / 4.0f) * (*(deter[kpts[i].class_id + 1].ptr<float>(y + 1) + x) + *(deter[kpts[i].class_id - 1].ptr<float>(y - 1) + x))
			- (1.0f / 4.0f) * (*(deter[kpts[i].class_id + 1].ptr<float>(y - 1) + x) + *(deter[kpts[i].class_id - 1].ptr<float>(y + 1) + x));
		
		*(A.ptr<float>(0)) = Dxx;
		*(A.ptr<float>(1) + 1) = Dyy;
		*(A.ptr<float>(2) + 2) = Dss;

		*(A.ptr<float>(0) + 1) = *(A.ptr<float>(1)) = Dxy;
		*(A.ptr<float>(0) + 2) = *(A.ptr<float>(2)) = Dxs;
		*(A.ptr<float>(1) + 2) = *(A.ptr<float>(2) + 1) = Dys;

		*(b.ptr<float>(0)) = -Dx;
		*(b.ptr<float>(1)) = -Dy;
		*(b.ptr<float>(2)) = -Ds;

		solve(A, b, dst, DECOMP_LU);

		if (fabs(*(dst.ptr<float>(0))) <= 1.0f && fabs(*(dst.ptr<float>(1))) <= 1.0f && fabs(*(dst.ptr<float>(2))) <= 1.0f) {
			kpts_t[i].pt.x += *(dst.ptr<float>(0));
			kpts_t[i].pt.y += *(dst.ptr<float>(1));

			float dsc = kpts_t[i].octave + (kpts_t[i].angle + *(dst.ptr<float>(2))) / ((float)(sublevel));

			kpts_t[i].size = 2.0f * 1.6f * pow((float)2.0f, dsc);
			kpts_t[i].angle = 0.0f;
		}
		else {
			kpts_t[i].response = -1;
		}
	}
	kpts.clear();
	for (int i = 0; i < kpts_t.size(); i++) {
		if (kpts_t[i].response != -1) {
			kpts.push_back(kpts_t[i]);
		}
	}
	tt.offTimer(3);

	Mat showImg;
	drawKeypoints(img, kpts, showImg, Scalar(0, 255, 0));

	//
	for (int i = 0; i < 12; i++) {
		oimg[i].convertTo(oimg[i], CV_8UC1);
	}
	//bitwise_not(dst2, dst2);

	for (int i = 0; i < 12; i++) {
		imwrite("images/evolutionImg_" + std::to_string(i) + ".jpg", oimg[i]);
		printf("evolution %d time = %lf\n", i, etime[i]);
	}
	

	tt.setTimerName(2, "OpenCV KAZE");
	Ptr<KAZE> kazeD = KAZE::create();
	std::vector<KeyPoint> kazep;
	tt.onTimer(2);
	kazeD->detect(img, kazep);
	tt.offTimer(2);
	drawKeypoints(showImg, kazep, showImg, Scalar(0, 0, 255));

	tt.printTimer();

	Mat hto2;
	float rep = 0.0; int cont = 0.0;

	while (waitKey(1) != 27) {
		imshow("kpts", showImg);
	}

	ck.cudaFreeAllMatrix();
	cudaFree(gpuGaussian); 
}


cv::Mat getGaussianKernel(int rows, int cols, double sigmax, double sigmay)
{
	auto gauss_x = cv::getGaussianKernel(cols, sigmax, CV_32F);
	auto gauss_y = cv::getGaussianKernel(rows, sigmay, CV_32F);
	return gauss_x * gauss_y.t();
}