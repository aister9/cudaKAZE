#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "DS_timer.h"

#include <iostream>
#include <vector>
#include <string>

using namespace std;
using namespace cv;

int compare(Vec3b a, Vec3b b) {
    int valA = a[0] * 0.0722 + a[1] * 0.7152 + a[2] * 0.2126, valB = b[0] * 0.0722 + b[1] * 0.7152 + b[2] * 0.2126;
    if (valA > valB)
        return 1;
    else
        if (valA == valB)
            return 0;
        else return -1;
}

int main() {
    Mat img_ori = imread("bird.jpg");
    //resize(img_ori, img_ori, Size(640, 480));
    
    vector<KeyPoint> kpts, kpts2, kpts3;
    Mat desc;
    Mat dst, dst2, dst3;

    DS_timer timer(6);
    timer.setTimerName(0, "CV KAZE Detector");
    timer.setTimerName(1, "CV SURF Detector");
    timer.setTimerName(2, "CV AKAZE Detector");
    timer.setTimerName(3, "AnisotropicDiffusion Method");
    timer.setTimerName(4, "GaussianBlur Method");
    timer.setTimerName(5, "make scale-space using cv non-linear filter method");
    timer.initTimers();

    Ptr<xfeatures2d::SURF> surfdetector = xfeatures2d::SURF::create();
    Ptr<KAZE> kazedetector = KAZE::create();
    Ptr<AKAZE> akazedetector = AKAZE::create();

    timer.onTimer(0); kazedetector->detect(img_ori, kpts); timer.offTimer(0);
    timer.onTimer(1); surfdetector->detect(img_ori, kpts2); timer.offTimer(1);
    timer.onTimer(2); akazedetector->detect(img_ori, kpts3); timer.offTimer(2);

    Mat diff;
    timer.onTimer(3); ximgproc::anisotropicDiffusion(img_ori, diff, 0.1, 0.15, 1); timer.offTimer(3);
    timer.onTimer(4); GaussianBlur(img_ori, diff, Size(3, 3), 0.15); timer.offTimer(4);

    timer.onTimer(5);
    //Make Scale-Space
    Mat scalespace[4][4]; // 4 ocataves, 3 layers

    //set octaves
    resize(img_ori, scalespace[0][0], Size(), 2, 2);
    img_ori.copyTo(scalespace[1][0]);    //resize(img_ori, scalespace[1][0], Size(), 1, 1); it is not needs
    resize(img_ori, scalespace[2][0], Size(), 0.5, 0.5);
    resize(img_ori, scalespace[3][0], Size(), 0.25, 0.25);

    //make layers
    ximgproc::anisotropicDiffusion(scalespace[0][0], scalespace[0][1], 0.1, 0.15, 1);
    ximgproc::anisotropicDiffusion(scalespace[0][1], scalespace[0][2], 0.1, 0.15, 1);
    ximgproc::anisotropicDiffusion(scalespace[0][2], scalespace[0][3], 0.1, 0.15, 1);

    ximgproc::anisotropicDiffusion(scalespace[1][0], scalespace[1][1], 0.1, 0.15, 1);
    ximgproc::anisotropicDiffusion(scalespace[1][1], scalespace[1][2], 0.1, 0.15, 1);
    ximgproc::anisotropicDiffusion(scalespace[1][2], scalespace[1][3], 0.1, 0.15, 1);

    ximgproc::anisotropicDiffusion(scalespace[2][0], scalespace[2][1], 0.1, 0.15, 1);
    ximgproc::anisotropicDiffusion(scalespace[2][1], scalespace[2][2], 0.1, 0.15, 1);
    ximgproc::anisotropicDiffusion(scalespace[2][2], scalespace[2][3], 0.1, 0.15, 1);

    ximgproc::anisotropicDiffusion(scalespace[3][0], scalespace[3][1], 0.1, 0.15, 1);
    ximgproc::anisotropicDiffusion(scalespace[3][1], scalespace[3][2], 0.1, 0.15, 1);
    ximgproc::anisotropicDiffusion(scalespace[3][2], scalespace[3][3], 0.1, 0.15, 1);

    Mat diffspace[4][3]; //4 octaves, 3layers
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 3; j++) {
            diffspace[i][j] = scalespace[i][j+1] - scalespace[i][j];
        }
    }

    //Extrema Detection
    vector<KeyPoint> extremas[4];
    for (int _octave = 0; _octave < 4; _octave++) {
        for (int yy = 1; yy < diffspace[_octave][0].rows - 1; yy++) {
            for (int xx = 1; xx < diffspace[_octave][0].cols - 1; xx++){
                int check = 0;
                Vec3b candidate = diffspace[_octave][1].at<Vec3b>(yy, xx);
                
                for (int _zz = -1; _zz <= 1; _zz++) {
                    for (int _yy = -1; _yy <= 1; _yy++) {
                        for (int _xx = -1; _xx <= 1; _xx++) {
                           if(compare(candidate,diffspace[_octave][1+_zz].at<Vec3b>(yy+_yy, xx+_xx)) == 1) check++;
                        }
                    }
                }
                if (check == 26/* || check == 0*/) {
                   // printf("%d, %d, %d\n", yy, xx, check);
                    KeyPoint extream(yy, xx, 3, 0, _octave);
                    extremas[_octave].push_back(extream);
                }
            }
        }
    }
    int times[] = { 0.5, 1, 2, 4 };
    vector<KeyPoint> MyExtrema;
    for (int _octave = 0; _octave < 4; _octave++) {
        for (int i = 0; i < extremas[_octave].size(); i++) {
            extremas[_octave][i].pt = Point2f(extremas[_octave][i].pt.x * times[_octave], extremas[_octave][i].pt.y * times[_octave]);
            MyExtrema.push_back(extremas[_octave][i]);
        }
    }
    //
    timer.offTimer(5);


    Mat dst4;

    drawKeypoints(img_ori, kpts, dst, Scalar(255, 0, 0));
    drawKeypoints(img_ori, kpts2, dst2, Scalar(255, 0, 0));
    drawKeypoints(img_ori, kpts3, dst3, Scalar(255, 0, 0));
    drawKeypoints(img_ori, MyExtrema, dst4, Scalar(255, 0, 0));

    //save scale space
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) { 
            string first = "scalespace/scale-space-octave-" + to_string(i) + "-layer-" + to_string(j)+".jpg";
            imwrite(first, scalespace[i][j]);
        }
    }
    //save dof space
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 3; j++) {
            string first = "scalespace/dof-space-octave-" + to_string(i) + "-layer-" + to_string(j) + ".jpg";
            imwrite(first, diffspace[i][j]);
        }
    }

    timer.printTimer();
    while (waitKey(1) != 27) {
        imshow("KAZE points",dst);
        imshow("SURF points", dst2);
        imshow("AKAZE points", dst3);
        imshow("extrema points", dst4);
    }
}