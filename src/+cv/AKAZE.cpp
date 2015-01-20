/**
 * @file AKAZE.cpp
 * @brief mex interface for AKAZE
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
    int descriptor_type = AKAZE::DESCRIPTOR_MLDB;
    int descriptor_size = 0;
    int descriptor_channels = 3;
    float threshold = 0.001f;
    int n_octaves = 4;
    int n_octave_layers = 4;
    int diffusivity = KAZE::DIFF_PM_G2;
    Mat mask;
    for (int i=1; i<nrhs; i+=2) {
        string key = rhs[i].toString();
        if (key=="DescriptorType")
            descriptor_type = rhs[i+1].toInt();
        else if (key=="DescriptorSize")
            descriptor_size = rhs[i+1].toInt();
        else if (key=="DescriptorChannels")
            descriptor_channels = rhs[i+1].toInt();
        else if (key=="Threshold")
            threshold = rhs[i+1].toDouble();
        else if (key=="NOctaves")
            n_octaves = rhs[i+1].toInt();
        else if (key=="NOctaveLayers")
            n_octave_layers = rhs[i+1].toInt();
        else if (key=="Diffusivity")
            diffusivity = rhs[i+1].toInt();
        else if (key=="Mask")
            mask = rhs[i+1].toMat(CV_8U);
        else
            mexErrMsgIdAndTxt("mexopencv:error","Unrecognized option");
    }

    // Process
    Ptr<AKAZE> akaze = AKAZE::create(descriptor_type, descriptor_size, descriptor_channels, threshold, 
        n_octaves, n_octave_layers, diffusivity);
    Mat image(rhs[0].toMat());
    vector<KeyPoint> keypoints;
    if (nlhs>1) {
        Mat descriptors;
	akaze->detectAndCompute(image,mask,keypoints,descriptors);
        plhs[1] = MxArray(descriptors);
    }
    else
	akaze->detect(image,keypoints,mask);
    plhs[0] = MxArray(keypoints);
}
