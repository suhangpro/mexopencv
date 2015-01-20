/**
 * @file BRISK.cpp
 * @brief mex interface for BRISK
 * @author Hang Su
 * @date 2015
 */
#include "mexopencv.hpp"
using namespace std;
using namespace cv;

/**
 * Main entry called from Matlab
 * @param nlhs number of left-hand-side arguments
 * @param plhs pointers to mxArrays in the left-hand-side
 * @param nrhs number of right-hand-side arguments
 * @param prhs pointers to mxArrays in the right-hand-side
 */
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[] )
{
    // Check the number of arguments
    if (nrhs<1 || ((nrhs%2)!=1) || nlhs>2)
        mexErrMsgIdAndTxt("mexopencv:error","Wrong number of arguments");

    // Argument vector
    vector<MxArray> rhs(prhs,prhs+nrhs);

    // Option processing
    int threshold = 30;
    int n_octaves = 3;
    float pattern_scale = 1.0f;
    Mat mask;
    for (int i=1; i<nrhs; i+=2) {
        string key = rhs[i].toString();
        if (key=="Threshold")
            threshold = rhs[i+1].toInt();
        else if (key=="NOctaves")
            n_octaves = rhs[i+1].toInt();
        else if (key=="PatternScale")
            pattern_scale = rhs[i+1].toDouble();
        else if (key=="Mask")
            mask = rhs[i+1].toMat(CV_8U);
        else
            mexErrMsgIdAndTxt("mexopencv:error","Unrecognized option");
    }

    // Process
    Ptr<BRISK> brisk = BRISK::create(threshold, n_octaves, pattern_scale);
    Mat image(rhs[0].toMat());
    vector<KeyPoint> keypoints;
    if (nlhs>1) {
        Mat descriptors;
	brisk->detectAndCompute(image,mask,keypoints,descriptors);
        plhs[1] = MxArray(descriptors);
    }
    else
	brisk->detect(image,keypoints,mask);
    plhs[0] = MxArray(keypoints);
}
