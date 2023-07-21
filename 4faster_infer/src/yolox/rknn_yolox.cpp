#include "rknn_yolox.h"


cv::Mat static_resize(cv::Mat& img, int input_w, int input_h) {
    float r = std::min(input_w / (img.cols*1.0), input_h / (img.rows*1.0));
    // r = std::min(r, 1.0f);
    int unpad_w = r * img.cols;
    int unpad_h = r * img.rows;
    cv::Mat re(unpad_h, unpad_w, CV_8UC3);
    cv::resize(img, re, re.size());
    cv::Mat out(input_h, input_w, CV_8UC3, cv::Scalar(114, 114, 114));
    re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));
    return out;
}

RknnYolox::~RknnYolox() {
    printf("begin release! \n");
}

RknnYolox::RknnYolox(const std::string &model_path, const float nms_threshold, const float conf_threshold) {

    Init(model_path, nms_threshold, conf_threshold);
    postprocess_ = std::make_shared<YoloxPostProcess>(input_h_, conf_threshold, nms_threshold, output_attrs_, out_zps_, out_scales_);
}


std::vector<ObjBox> RknnYolox::infer(cv::Mat &img) {

    auto img_out = static_resize(img, input_w_, input_h_);
    float img_scale =  std::min(input_w_ / (img.cols*1.0), input_h_ / (img.rows*1.0));
    memcpy(input_mems_[0]->virt_addr, img_out.data, input_attrs_[0].size_with_stride);
    CHECK_RKNN(rknn_run(ctx_, NULL));
    postprocess_->process((int8_t *)output_mems_[0]->virt_addr, results_, img_scale);

    return results_;
}




std::shared_ptr<RknnInferBaseObjBox> create_infer_yolox(const std::string &model_path, const float nms_threshold, const float conf_threshold){
    return std::make_shared<RknnYolox>(model_path, nms_threshold, conf_threshold);
}
