#include "rknn_yolox.h"


RknnYolox::RknnYolox(const std::string &model_path, const float nms_threshold, const float conf_threshold) {
    Init(model_path, nms_threshold, conf_threshold);
    postprocess_ = std::make_shared<YoloxPostProcess>(input_h_, conf_threshold, nms_threshold, output_attrs_);

    resize_img = cv::Mat::zeros(cv::Size(input_h_, input_w_), CV_8UC3); // 初始化
}


void RknnYolox::infer(cv::Mat &img, std::vector<ObjBox> &res_boxes) {
    res_boxes.clear();
    float img_scale = static_resize(img);
    memcpy(input_mems_[0]->virt_addr, resize_img.data, input_attrs_[0].size_with_stride);
    CHECK_RKNN(rknn_run(ctx_, NULL));
    postprocess_->process((int8_t *)output_mems_[0]->virt_addr, res_boxes, img_scale, out_zps_, out_scales_);
}


float RknnYolox::static_resize(cv::Mat &img) {
    resize_img = cv::Scalar(114, 114, 114);

    float r = std::min(input_w_ / (img.cols*1.0), input_h_ / (img.rows*1.0));
    int unpad_w = r * img.cols;
    int unpad_h = r * img.rows;
    cv::Mat re(unpad_h, unpad_w, CV_8UC3);
    cv::resize(img, re, re.size());
    re.copyTo(resize_img(cv::Rect(0, 0, re.cols, re.rows)));
    return r;
}



std::shared_ptr<RknnInferBaseObjBox> create_infer_yolox(const std::string &model_path, const float nms_threshold, const float conf_threshold){
    return std::make_shared<RknnYolox>(model_path, nms_threshold, conf_threshold);
}
