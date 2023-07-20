#pragma once

#include <memory>

#include "../base/rknn_infer_base.cpp"
#include "../postprocess/rknn_postprocess.h"


// resize



using RknnInferBaseObjBox = RknnInferBase<std::vector<ObjBox>>;
class RknnYolox :public RknnInferBaseObjBox{

public:
    RknnYolox(const std::string &model_path, const float nms_threshold, const float conf_threshold);
    ~RknnYolox() { }

    virtual void infer(cv::Mat &img, std::vector<ObjBox> &results) override; // 纯虚函数
    float static_resize(cv::Mat& img);

private:
    std::shared_ptr<YoloxPostProcess> postprocess_;
    // cv::Mat resize_img;
    cv::Mat resize_img; 
};



std::shared_ptr<RknnInferBaseObjBox> create_infer_yolox(const std::string &model_path, const float nms_threshold, const float conf_threshold);




