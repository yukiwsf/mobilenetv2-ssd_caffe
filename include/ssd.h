#ifndef SSD_H
#define SSD_H

#include <string>
#include <vector>
#include <cmath>
#include <iostream>
#include <boost/smart_ptr.hpp>
#include <boost/make_shared.hpp>
#include "opencv2/core.hpp"
// #include "opencv2/dnn.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "caffe/caffe.hpp"

constexpr int SSD_INPUT_SIZE = 320, NUM_CLASS = 21, MAX_DET = 100, NUM_STRIDES = 6;
const float CONFIDENCE_THRESHOLD = 0.01, SCORE_THRESHOLD = 0.5, NMS_THRESHOLD = 0.5;
const bool IS_NMS = true;

/* Maximum: int/float/double */
template<typename T>
inline T Maximum(T a, T b) {
    return a > b ? a : b;
} 

/* Minimum: int/float/double */
template<typename T>
inline T Minimum(T a, T b) {
    return a < b ? a : b;
} 

/* ssd class name */
extern const char *clsName[NUM_CLASS];

/* ssd priorbox params */
struct Prior {
    int feature_maps[NUM_STRIDES] = { 20, 10, 5, 3, 2, 1 };
    int strides[NUM_STRIDES] = { 16, 32, 64, 107, 160, 320 };
    int min_sizes[NUM_STRIDES] = { 60, 105, 150, 195, 240, 285 };
    int max_sizes[NUM_STRIDES] = { 105, 150, 195, 240, 285, 330 };
    int aspect_ratio[NUM_STRIDES][2] = { { 2, 3 }, { 2, 3 }, { 2, 3 }, { 2, 3 }, { 2, 3 }, { 2, 3 } };
    int prior_per_location[NUM_STRIDES] = { 6, 6, 6, 6, 6, 6 };
    float center_variance = 0.1;
    float size_variance = 0.2;
};

/* calculate ssd priorbox size */
void CalculatePriorBoxSize(Prior &priorBoxes, std::vector< std::vector<float> > &priorSizes);

/* store object-detection result information */
struct ObjInfo {
    int clsId;
    float score;
    cv::Rect bbox;
};

/* main ssd class */
class Detector {
public:
    Detector(std::string prototxt, std::string caffemodel);
    virtual ~Detector();
    void Detect(const cv::Mat &inputImage, std::vector<ObjInfo> &detResults);
private:
    std::shared_ptr< caffe::Net< float > > net;
    caffe::Blob<float> *inputBlob;
    std::vector< caffe::Blob< float >* > outputBlobs;
};

#endif  // #define SSD_H