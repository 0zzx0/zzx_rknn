#pragma once

#include <memory>

#include "../base/rknn_infer_base.h"
#include "../postprocess/rknn_postprocess.h"


// resize
static cv::Mat static_resize(cv::Mat& img, int input_w, int input_h);



class RknnYolox :public RknnInferBase{

public:
    RknnYolox(const std::string &model_path, const float nms_threshold, const float conf_threshold);
    ~RknnYolox() { }

    virtual void infer(cv::Mat &img) override; // 纯虚函数

private:
    std::shared_ptr<YoloxPostProcess> postprocess_;
};



std::shared_ptr<RknnInferBase> create_infer(const std::string &model_path, const float nms_threshold, const float conf_threshold);




