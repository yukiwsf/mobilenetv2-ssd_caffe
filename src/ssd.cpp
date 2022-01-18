#include "ssd.h"

const char *clsName[NUM_CLASS] = {"background",
                                  "aeroplane", "bicycle", "bird", "boat",
                                  "bottle", "bus", "car", "cat", "chair",
                                  "cow", "diningtable", "dog", "horse",
                                  "motorbike", "person", "pottedplant",
                                  "sheep", "sofa", "train", "tvmonitor"};

template<typename T>
T ClampValue(T value, T upperLimit, T lowerLimit) {
    T ret = MAXIMUM(value, lowerLimit);
    ret = MINIMUM(ret, upperLimit);
    return ret;
}

/* softmax */
inline void Softmax(std::vector<float> &classes) {
    float sum = 0;
    std::transform(
                   classes.begin(), 
                   classes.end(), 
                   classes.begin(), 
                   [&sum](float score) -> float {
                       float expScore = exp(score);
                       sum += expScore;
                       return expScore;
                   }
                   );
    std::transform(
                   classes.begin(), 
                   classes.end(), 
                   classes.begin(),
                   [sum](float score) -> float { 
                       return score / sum; 
                   }
                   );
}

/* calculate sizes of ssd priorbox for each grid cell */
void CalculatePriorBoxSize(Prior &priorBoxes, std::vector< std::vector<float> > &priorSizes) {
    std::cout << "start calculate prior boxes." << std::endl;
    for(int k = 0; k < NUM_STRIDES; ++k) {
        int scale = priorBoxes.feature_maps[k];  // { 20, 10, 5, 3, 2, 1 }
        for(int i = 0; i < scale; ++i) {
            for(int j = 0; j < scale; ++j) {
                /* unit center x,y */
                float upperLimit = 1.0, lowerLimit = 0.0;
                float cx = ClampValue<float>((j + 0.5) / scale, upperLimit, lowerLimit);
                float cy = ClampValue<float>((i + 0.5) / scale, upperLimit, lowerLimit);
                /* small sized square box */
                int size_0 = priorBoxes.min_sizes[k];  // { 60, 105, 150, 195, 240, 285 }
                float h_0 = ClampValue<float>((float)size_0 / SSD_INPUT_SIZE, upperLimit, lowerLimit);
                float w_0 = h_0;
                std::vector<float> priorSize_0 = { cx, cy, w_0, h_0 };
                priorSizes.push_back(priorSize_0);
                /* big sized square box */
                float size_1 = std::sqrt(priorBoxes.min_sizes[k] * priorBoxes.max_sizes[k]);
                float h_1 = ClampValue<float>((float)size_1 / SSD_INPUT_SIZE, upperLimit, lowerLimit);
                float w_1 = h_1;
                std::vector<float> priorSize_1 = { cx, cy, w_1, h_1 };
                priorSizes.push_back(priorSize_1);
                /* change h/w ratio of the small sized box */
                int size_2 = size_0;
                float h_2 = (float)size_2 / SSD_INPUT_SIZE; 
                float w_2 = h_2;
                float ratio_2 = std::sqrt(priorBoxes.aspect_ratio[k][0]);
                std::vector<float> priorSize_2 = { cx, cy, ClampValue<float>(w_2 * ratio_2, upperLimit, lowerLimit), ClampValue<float>(h_2 * 1.0 / ratio_2, upperLimit, lowerLimit) };
                std::vector<float> priorSize_3 = { cx, cy, ClampValue<float>(w_2 * 1.0 / ratio_2, upperLimit, lowerLimit), ClampValue<float>(h_2 * ratio_2, upperLimit, lowerLimit) };
                priorSizes.push_back(priorSize_2);
                priorSizes.push_back(priorSize_3);
                float ratio_4 = std::sqrt(priorBoxes.aspect_ratio[k][1]);
                std::vector<float> priorSize_4 = { cx, cy, ClampValue<float>(w_2 * ratio_4, upperLimit, lowerLimit), ClampValue<float>(h_2 * 1.0 / ratio_4, upperLimit, lowerLimit) };
                std::vector<float> priorSize_5 = { cx, cy, ClampValue<float>(w_2 * 1.0 / ratio_4, upperLimit, lowerLimit), ClampValue<float>(h_2 * ratio_4, upperLimit, lowerLimit) };
                priorSizes.push_back(priorSize_4);
                priorSizes.push_back(priorSize_5);
            }
        }
    }
    std::cout << "end calculate prior boxes." << std::endl;
}

Detector::Detector(std::string prototxt, std::string caffemodel) {
    /* set device */
    caffe::Caffe::set_mode(caffe::Caffe::CPU);
    /* load and init network */
    this->net.reset(new caffe::Net<float>(prototxt, caffe::TEST));
    this->net->CopyTrainedLayersFrom(caffemodel);
    std::cout << "net inputs number is " << this->net->num_inputs();
    std::cout << "net outputs number is " << this->net->num_outputs();
    if(this->net->num_inputs() != 1) 
        std::cerr << "network should have exactly one input.";
    this->inputBlob = this->net->input_blobs()[0];
    std::cout << "input data layer channels is " << this->inputBlob->channels();
    std::cout << "input data layer width is " << this->inputBlob->width();
    std::cout << "input data layer height is " << this->inputBlob->height();
}

Detector::~Detector() {
    /* release memory */
    this->net.reset();
    std::cout << "release net sources." << std::endl;
}

/* pre-process: resize subtractmean hwc2chw bgr2rgb */
void PreProcess(const cv::Mat &src, cv::Mat &dst) {
    std::cout << "\nstart ssd pre-process." << std::endl;
    cv::resize(src, dst, cv::Size(SSD_INPUT_SIZE, SSD_INPUT_SIZE));  // resize
    cv::Mat dstCopy;
    dst.convertTo(dstCopy, CV_32F);  // dstCopy uint8 to float32
    int cnt = 0;
    int pixels = 0;
    float mean = 0;
    float *ptrCpy = (float*)dstCopy.data;
    for(int k = 0; k < dst.channels(); ++k) {
        for(int i = 0; i < dst.rows; ++i) {
            for(int j = 0; j < dst.cols; ++j) {            
                int idx = dst.channels() - 1 - k + j * dst.channels() + i * dst.step[0];  // hwc2chw, bgr2rgb
                // int idx = k + j * dst.channels() + i * dst.step[0];  // only hwc2chw
                int flag = pixels / (dst.rows * dst.cols);
                switch(flag) {
                    case 0: {
                        mean = 123.0; 
                        break;
                    }
                    case 1: {
                        mean = 117.0; 
                        break;
                    }
                    case 2: {
                        mean = 104.0; 
                        break;
                    }            
                }
                *ptrCpy++ = dst.data[idx] - mean;  // subtractmean
                ++pixels;
            }
        }
    }
    dstCopy.convertTo(dst, CV_32F);  // dst uint8 to float32
    // float *ptr = (float*)dst.data;
    // ptr += dst.cols * dst.rows * 2;
    // for(int i = 0; i < 10; ++i) {
    //     printf("%f\n", *ptr++);
    // }
    std::cout << "end ssd pre-process." << std::endl;
}

void DetectorAndClassifier(std::vector< caffe::Blob< float >* > &outputs, std::vector<int> &indexes, std::vector<cv::Rect> &boxes, std::vector<float> &scores) {
    /* for each output tensor */
    /*
      shape=1 126 20 20 (50400)
      shape=1 24 20 20 (9600)
      shape=1 126 10 10 (12600)
      shape=1 24 10 10 (2400)
      shape=1 126 5 5 (3150)
      shape=1 24 5 5 (600)
      shape=1 126 3 3 (1134)
      shape=1 24 3 3 (216)
      shape=1 126 2 2 (504)
      shape=1 24 2 2 (96)
      shape=1 126 1 1 (126)
      shape=1 24 1 1 (24)
    */
    Prior priorBoxes;
    std::vector< std::vector<float> > priorSizes;
    int cnt = 0;
    CalculatePriorBoxSize(priorBoxes, priorSizes);
    for(int stage = 0; stage < 2 * NUM_STRIDES; ++stage) {
        int stride = stage / 2;
        const int downScale = priorBoxes.strides[stride];  // 16, 32, 64, 107, 160, 320 
        caffe::Blob< float >* output = outputs[stage];  
        const float *data = (const float*)output->cpu_data();
        /* Classifier */
        if(stage % 2 == 0) {
            /* for every classification output tensor cell */
            for(int cy = 0; cy < output->height(); ++cy) {
                for(int cx = 0; cx < output->width(); ++cx) {
                    for(int np = 0; np < priorBoxes.prior_per_location[stride]; ++np) {
                        /* predicted box classification */
                        int channel = np * NUM_CLASS;  // 0 21 42 63 84 105
                        std::vector<float> classes(NUM_CLASS);
                        for (int i = 0; i < NUM_CLASS; ++i) {
                            int classIdx = output->width() * output->height() * (channel + i) + output->width() * cy + cx;
                            // std::cout << "class: " << classIdx << " " << "total: " << output->shape_string() << std:: endl;
                            classes[i] = data[classIdx];
                        }
                        Softmax(classes);
                        auto maxIterator = std::max_element(classes.begin(), classes.end());
                        int maxIndex = (int)(maxIterator - classes.begin());
                        indexes.push_back(maxIndex);
                        scores.push_back(classes[maxIndex]);
                    }
                }
            }
        }
        /* Detector */
        else {
            /* for every box regression output tensor cell */
            for(int cy = 0; cy < output->height(); ++cy) {
                for(int cx = 0; cx < output->width(); ++cx) {
                    for(int np = 0; np < priorBoxes.prior_per_location[stride]; ++np) {
                        /* predicted box regression */
                        int channel = np * 4;  // 0 21 42 63 84 105
                        int xIdx = output->width() * output->height() * (channel + 0) + output->width() * cy + cx;
                        float lcx = data[xIdx];
                        int yIdx = output->width() * output->height() * (channel + 1) + output->width() * cy + cx;
                        float lcy = data[yIdx];
                        int wIdx = output->width() * output->height() * (channel + 2) + output->width() * cy + cx;
                        float lw = data[wIdx];
                        int hIdx = output->width() * output->height() * (channel + 3) + output->width() * cy + cx;
                        float lh = data[hIdx];
                        float cx = lcx * priorBoxes.center_variance * priorSizes[cnt][2] + priorSizes[cnt][0];
                        float cy = lcy * priorBoxes.center_variance * priorSizes[cnt][3] + priorSizes[cnt][1];
                        float w = std::exp(lw * priorBoxes.size_variance) * priorSizes[cnt][2];
                        float h = std::exp(lh * priorBoxes.size_variance) * priorSizes[cnt][3];
                        float xmin = cx - w * 1.0 / 2;
                        // float xmax = cx + w * 1.0 / 2;
                        float ymin = cy - h * 1.0 / 2;
                        // float ymax = cy + h * 1.0 / 2;
                        int xmin_ssd_input_size = (int)(xmin * SSD_INPUT_SIZE);
                        int ymin_ssd_input_size = (int)(ymin * SSD_INPUT_SIZE);
                        int w_ssd_input_size = (int)(w * SSD_INPUT_SIZE);
                        int h_ssd_input_size = (int)(h * SSD_INPUT_SIZE);
                        boxes.push_back(cv::Rect(xmin_ssd_input_size, ymin_ssd_input_size, w_ssd_input_size, h_ssd_input_size));
                        ++cnt;
                    }
                }
            }
        }
    }
}

/* process box in case of crossing the image border*/
inline cv::Rect BoxBorderProcess(const cv::Rect &box, const int imgWidth, const int imgHeight) {
    int xmin = box.x;
    int ymin = box.y;
    int xmax = xmin + box.width;
    int ymax = ymin + box.height;
    xmin = MAXIMUM(0, xmin);
    xmin = MINIMUM(imgWidth, xmin);
    ymin = MAXIMUM(0, ymin);
    ymin = MINIMUM(imgHeight, ymin);
    xmax = MAXIMUM(0, xmax);
    xmax = MINIMUM(imgWidth, xmax);
    ymax = MAXIMUM(0, ymax);
    ymax = MINIMUM(imgHeight, ymax);
    int width = xmax - xmin;
    int height = ymax - ymin;  
    return cv::Rect(xmin, ymin, width, height);
};

void PostProcess(std::vector< caffe::Blob< float >* > &outputs, std::vector<ObjInfo> &detResults, int originalImageWidth, int originalImageHeight) {
    /* post-process */
    std::cout << "start ssd post-process." << std::endl;
    // indexes, boxes and confidences have the same element index
    std::vector<int> indexes;  // vector to store every maxIndex(index of clsName) of predicted box
    std::vector<cv::Rect> boxes;  // vector to store every coordinate(xmin ymin w h) of predicted box
    std::vector<float> scores;  // vector to store every score of predicted box
    DetectorAndClassifier(outputs, indexes, boxes, scores); 
    /* remove background class and low confidence predicted box */
    std::vector<int> indexesNew;  
    std::vector<cv::Rect> boxesNew;  
    std::vector<float> scoresNew;  
    for(int i = 0; i < indexes.size(); ++i) {
        if(indexes[i] != 0 && scores[i] > CONFIDENCE_THRESHOLD) {
            indexesNew.push_back(indexes[i]);
            boxesNew.push_back(boxes[i]);
            scoresNew.push_back(scores[i]);
        }
    }
    /* do nms */
    std::vector<int> indices;  // indices of indexs, boxes and scores after nms
    std::cout << "do nms..." << std::endl;
    if (IS_NMS) {
        cv::dnn::NMSBoxes(boxesNew, scoresNew, SCORE_THRESHOLD, NMS_THRESHOLD, indices);
    } 
    else {
        for(int i = 0; i < indexesNew.size(); ++i) indices.push_back(i);
    }
    /* limit number of detections */
    if(indices.size() > MAX_DET) indices.resize(MAX_DET);
    /* get ssd result */
    std::cout << "get ssd result..." << std::endl;
    float widthScale = (float)originalImageWidth / SSD_INPUT_SIZE;  // int divide int will cut-off to int, use int divide float or float divide int or float divide float
    float heightScale = (float)originalImageHeight / SSD_INPUT_SIZE;  // ((float)a) / b
    for(size_t i = 0; i < indices.size(); ++i) {  // for every predicted box
        int idx = indices[i];  // indexes/boxes/scores index 
        ObjInfo object;
        // remap box from input img size to orginal img size
        boxesNew[idx].x =(int)(boxesNew[idx].x * widthScale);
        boxesNew[idx].y = (int)(boxesNew[idx].y * heightScale);
        boxesNew[idx].width = (int)(boxesNew[idx].width * widthScale);
        boxesNew[idx].height = (int)(boxesNew[idx].height * heightScale);
        // process every box boundary according to input img size
        cv::Rect newBox = BoxBorderProcess(boxesNew[idx], originalImageWidth, originalImageHeight);
        object.bbox = newBox;
        object.score = scoresNew[idx];
        object.clsId = indexesNew[idx];
        detResults.push_back(object);
        printf("object: %s, xmin = %d, ymin = %d, width = %d, height = %d, score = %f\n", clsName[object.clsId], object.bbox.x, object.bbox.y, object.bbox.width, object.bbox.height, object.score);
    }
    std::cout << "end ssd post-process." << std::endl;
}

void Detector::Detect(const cv::Mat &originalImage, std::vector<ObjInfo> &detResults) {
    /* pre-process */
    cv::Mat imagePreprocess; 
    PreProcess(originalImage, imagePreprocess);  // hwc2chw, bgr2rgb
    /* copy data to input blob */
    memcpy((void*)this->inputBlob->mutable_cpu_data(), (void*)imagePreprocess.data, sizeof(float) * this->inputBlob->count());
    /* clean output blobs */
    this->outputBlobs.clear();     
    /* forward */
    std::cout << "start ssd forward." << std::endl;
    this->net->Forward();
    for(int i = 0; i < this->net->num_outputs(); ++i) {
        this->outputBlobs.push_back(this->net->output_blobs()[i]);
    }
    std::cout << "end ssd forward." << std::endl;

    /* post-process */
    int originalImageWidth = originalImage.cols;
    int originalImageHeight = originalImage.rows;
    // std::cout << this->outputBlobs.size() << std::endl;
    // caffe::Blob<float> *output = this->outputBlobs[0];
    // const float *p = (const float*)output->cpu_data();
    // for(int i = 0; i < 10; ++i) {
    //     printf("%f\n", *p++);
    // }
    PostProcess(outputBlobs, detResults, originalImageWidth, originalImageHeight);
}