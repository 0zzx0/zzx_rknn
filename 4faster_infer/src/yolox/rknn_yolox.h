#pragma once

#include <memory>

#include "../base/rknn_infer_base.h" 
#include "../postprocess/rknn_postprocess.h"


// resize
static cv::Mat static_resize(cv::Mat& img, int input_w, int input_h);


using RknnInferBaseObjBox = RknnInferBase<std::vector<ObjBox>>;
class RknnYolox :public RknnInferBaseObjBox{

public:
    RknnYolox(const std::string &model_path, const float nms_threshold, const float conf_threshold);
    virtual ~RknnYolox();

    virtual std::vector<ObjBox> infer(const cv::Mat &img) override; // yolox推理

private:
    std::shared_ptr<YoloxPostProcess> postprocess_;
    std::vector<ObjBox> results_;
};



std::shared_ptr<RknnInferBaseObjBox> create_infer_yolox(const std::string &model_path, const float nms_threshold, const float conf_threshold);




