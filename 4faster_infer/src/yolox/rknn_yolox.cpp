/**
 * @file rknn_yolox.cpp
 * @author zzx
 * @brief yolox推理类的实现
 * @version 0.1
 * @date 2023-07-22
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "rknn_yolox.h"

RknnYolox::~RknnYolox() {
    printf("begin release! \n");
}

RknnYolox::RknnYolox(const std::string &model_path, const float nms_threshold,
                     const float conf_threshold) {
    Init(model_path);
    postprocess_ = std::make_shared<YoloxPostProcess>(input_h_, conf_threshold, nms_threshold,
                                                      output_attrs_, out_zps_, out_scales_);
}

std::vector<ObjBox> RknnYolox::infer(const cv::Mat &img) {
    auto img_out = static_resize(img, input_w_, input_h_);
    float img_scale = std::min(input_w_ / (img.cols * 1.0), input_h_ / (img.rows * 1.0));
    memcpy(input_mems_[0]->virt_addr, img_out.data, input_attrs_[0].size_with_stride);
    CHECK_RKNN(rknn_run(ctx_, NULL));
    postprocess_->process((int8_t *)output_mems_[0]->virt_addr, results_, img_scale);

    return results_;
}

std::shared_ptr<RknnInferBaseObjBox> create_infer_yolox(const std::string &model_path,
                                                        const float nms_threshold,
                                                        const float conf_threshold) {
    return std::make_shared<RknnYolox>(model_path, nms_threshold, conf_threshold);
}
