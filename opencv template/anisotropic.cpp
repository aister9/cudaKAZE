//#include <opencv2/opencv.hpp>
//#include <opencv2/ximgproc.hpp>
//
//#include <iostream>
//#include <vector>
//#include <string>
//
//using namespace std;
//using namespace cv;
//
//int main() {
//	Mat img_ori = imread("bird.jpg");
//	//resize(img_ori, img_ori, Size(640, 480));
//
//	Mat diff;
//    Mat diff_gray;
//	ximgproc::anisotropicDiffusion(img_ori, diff, 0.1, 0.15, 4);
//
//    cvtColor(diff, diff_gray, COLOR_BGR2GRAY);
//
//    Mat gray_src;
//    cvtColor(img_ori, gray_src, COLOR_BGR2GRAY);
//    gray_src.convertTo(gray_src, CV_32FC1, 1.0 / 255.0);
//
//    Mat dst;
//    gray_src.copyTo(dst);
//    int number = 4;
//    for(int k = 0; k< number; k++)
//    for (int i = 1; i < img_ori.rows - 1; i++) {
//        for (int j = 1; j < img_ori.cols - 1; j++) {
//            float cN, cS, cE, cW;
//            float deltacN, deltacS, deltacE, deltacW;
//
//            deltacN = dst.at<float>(i, j - 1) - dst.at<float>(i, j);
//            deltacS = dst.at<float>(i, j + 1) - dst.at<float>(i, j);
//            deltacE = dst.at<float>(i + 1, j) - dst.at<float>(i, j);
//            deltacW = dst.at<float>(i - 1, j) - dst.at<float>(i, j);
//
//            cN = abs(exp(-1 * (deltacN * deltacN / (0.15 * 0.15))));
//            cS = abs(exp(-1 * (deltacS * deltacS / (0.15 * 0.15))));
//            cE = abs(exp(-1 * (deltacE * deltacE / (0.15 * 0.15))));
//            cW = abs(exp(-1 * (deltacW * deltacW / (0.15 * 0.15))));
//
//            dst.at<float>(i, j) = dst.at<float>(i, j) * (1 - 0.1 * (cN + cS + cE + cW)) +
//                0.1 * (cN * dst.at<float>(i, j - 1) + cS * dst.at<float>(i, j + 1)
//                    + cE * dst.at<float>(i + 1, j) + cW * dst.at<float>(i - 1, j));
//        }
//    }
//
//
//
//	while (waitKey(1) != 27) {
//		imshow("gray image1", gray_src);
//		imshow("gray image2", diff_gray);
//        imshow("gray image3", dst);
//	}
//}